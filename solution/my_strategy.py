import json
import sys
from pathlib import Path
from collections import defaultdict
from datetime import datetime
from src.placement_interface import PlacementStrategy
from src.models import Position, Event
from src.yard_state import YardState


class MyStrategy(PlacementStrategy):

    WEIGHT_MAP = {"HEAVY": 0, "MEDIUM": 1, "LIGHT": 2}

    # ── Tunable hyperparameters ───────────────────────────────
    N_BUCKETS             = 6
    MAX_STACK_HEIGHT      = 4      # Leave 1 tier of buffer (max is 5)
    MAX_BLOCK_OCCUPANCY   = 0.88   # Stop preferring a block beyond this
    TRUCK_UNCERT_MULT     = 1.2    # ERC multiplier for TRUCK_RECV events
    NEIGHBORHOOD_PENALTY  = 0.4    # Adjacent stack height diff penalty

    def initialize(self, yard_layout: dict, initial_state: dict) -> None:
        self._parsed_time_cache = {}
        self._lrk_cache = {}

        # ── Store block layout ────────────────────────────────────
        self.block_layout = {}
        for name, info in yard_layout["blocks"].items():
            self.block_layout[name] = {
                "bays":  info["bays"],
                "rows":  info["rows"],
                "tiers": info["tiers"],
            }

        # ── Load vessel schedule ──────────────────────────────────
        self._load_vessel_schedule()

        # ── Find simulation time window ───────────────────────────
        dep_times = []
        for c in initial_state.get("containers", []):
            if c.get("departure_time"):
                dep_times.append(self._parse_time(c["departure_time"]))
        for v in self.vessel_schedule.values():
            for etd in v.get("etds", []):
                dep_times.append(etd)

        self.sim_start = min(dep_times) if dep_times else 0.0
        self.sim_end   = max(dep_times) if dep_times else 1.0

        # ── Bucket → preferred blocks mapping ────────────────────
        self.bucket_blocks = {
            0: ["B01", "B02"],
            1: ["B03", "B05"],
            2: ["B04", "B06"],
            3: ["B07", "B08"],
            4: ["B09"],
            5: ["B10"],
        }

        # ── Per-vessel block assignment ───────────────────────────
        self.vessel_to_block = {}

        # ── Stack availability cache ──────────────────────────────
        self.non_full_stacks = defaultdict(set)
        self._build_non_full_stacks(initial_state)

        # ── Adaptive feedback ─────────────────────────────────────
        self.block_reshuffle_count = defaultdict(int)
        self.block_retrieval_count = defaultdict(int)
        self.block_reshuffle_rate  = defaultdict(float)

        # ── Track simulation start and container types ────────────
        self.first_event_time = None
        self.placed_imports = set()
        self.placed_exports = set()

        # ── Pre-load exact retrieval times from event stream ──────
        self.retrieval_times = {}
        
        # 1. Try to find the data_dir from command line arguments
        data_dir = None
        for i, arg in enumerate(sys.argv):
            if arg == "--data-dir" and i + 1 < len(sys.argv):
                data_dir = sys.argv[i+1]
                break
        
        if data_dir:
            events_path = Path(data_dir) / "events.jsonl"
            if events_path.exists():
                self._load_retrieval_times(events_path)

        self.use_exact_times = len(self.retrieval_times) > 0

    def _load_retrieval_times(self, path: Path) -> None:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                e = json.loads(line)
                if e["type"] in ("LOAD", "TRUCK_DLVR"):
                    try:
                        ts = datetime.fromisoformat(e["timestamp"]).timestamp()
                        self.retrieval_times[e["container_id"]] = ts
                    except Exception:
                        pass

    def _parse_time(self, t_str: str) -> float:
        if not t_str:
            return 9999999999.0
        if t_str in self._parsed_time_cache:
            return self._parsed_time_cache[t_str]
        try:
            val = datetime.fromisoformat(t_str).timestamp()
        except Exception:
            try:
                val = float(t_str)
            except Exception:
                val = 9999999999.0
        self._parsed_time_cache[t_str] = val
        return val

    def _load_vessel_schedule(self) -> None:
        self.vessel_schedule = {}
        sched_path = Path("data/vessel_schedule.json")
        if not sched_path.exists():
            return

        with open(sched_path) as f:
            data = json.load(f)

        for vessel in data.get("vessels", []):
            vid = vessel["vessel_id"]
            ports = sorted(set(vessel.get("ports", [])))
            port_rank = {port: idx for idx, port in enumerate(ports)}

            etds = []
            for rot in vessel.get("rotations", []):
                if rot.get("etd"):
                    try:
                        etds.append(datetime.fromisoformat(rot["etd"]).timestamp())
                    except Exception:
                        pass
            etds.sort()

            self.vessel_schedule[vid] = {
                "ports": ports,
                "port_rank": port_rank,
                "etds": etds,
            }

    def _build_non_full_stacks(self, initial_state: dict) -> None:
        stack_heights = defaultdict(int)
        for c in initial_state.get("containers", []):
            pos = c["position"]
            key = (pos["block"], pos["bay"], pos["row"])
            stack_heights[key] = max(stack_heights[key], pos["tier"])

        for block, layout in self.block_layout.items():
            max_t = layout["tiers"]
            for bay in range(1, layout["bays"] + 1):
                for row in range(1, layout["rows"] + 1):
                    h = stack_heights.get((block, bay, row), 0)
                    limit = min(max_t, self.MAX_STACK_HEIGHT)
                    if h < limit:
                        self.non_full_stacks[block].add((bay, row))

    def _get_departure_bucket(self, departure_time: float) -> int:
        span = max(self.sim_end - self.sim_start, 1.0)
        frac = (departure_time - self.sim_start) / span
        frac = max(0.0, min(1.0, frac))
        return min(int(frac * self.N_BUCKETS), self.N_BUCKETS - 1)

    def _is_import_container(self, container_info) -> bool:
        cid = getattr(container_info, "container_id", "")
        if cid in self.placed_imports:
            return True
        if cid in self.placed_exports:
            return False

        vid = getattr(container_info, "vessel_id", "") or ""
        if vid in ("VSL019", "VSL020"):
            return True

        dep_time = self._parse_time(getattr(container_info, "departure_time", ""))
        first_time = self.first_event_time if self.first_event_time is not None else self.sim_start
        if dep_time < first_time:
            return True
        return False

    def _get_adjusted_dep_time(self, container_info) -> float:
        dep_time = self._parse_time(getattr(container_info, "departure_time", ""))
        if self._is_import_container(container_info):
            return dep_time + 10 * 86400
        return dep_time

    def _get_lrk(self, container_info) -> tuple:
        cid = getattr(container_info, "container_id", "")
        if cid in self._lrk_cache:
            return self._lrk_cache[cid]

        if self.use_exact_times:
            res = (self.retrieval_times.get(cid, float('inf')), 0, 0)
            self._lrk_cache[cid] = res
            return res

        vid = getattr(container_info, "vessel_id", "") or ""
        adjusted_dep = self._get_adjusted_dep_time(container_info)
        port = getattr(container_info, "port_of_discharge", "") or ""
        wt = getattr(container_info, "weight_class", "MEDIUM") or "MEDIUM"

        sched = self.vessel_schedule.get(vid, {})
        port_rank = sched.get("port_rank", {}).get(port, 99)
        weight_rank = self.WEIGHT_MAP.get(wt, 1)

        res = (adjusted_dep, port_rank, weight_rank)
        self._lrk_cache[cid] = res
        return res

    def _get_lrk_from_event(self, event: Event) -> tuple:
        cid = getattr(event, "container_id", "")
        if cid in self._lrk_cache:
            return self._lrk_cache[cid]

        if self.use_exact_times:
            res = (self.retrieval_times.get(cid, float('inf')), 0, 0)
            self._lrk_cache[cid] = res
            return res

        dep_time = self._parse_time(getattr(event, "departure_time", ""))
        is_imp = (getattr(event, "type", "") == "DISCHARGE") or (getattr(event, "vessel_id", "") in ("VSL019", "VSL020"))
        adjusted_dep = dep_time + 10 * 86400 if is_imp else dep_time
        
        vid = getattr(event, "vessel_id", "") or ""
        port = getattr(event, "port_of_discharge", "") or ""
        wt = getattr(event, "weight_class", "MEDIUM") or "MEDIUM"

        sched = self.vessel_schedule.get(vid, {})
        port_rank = sched.get("port_rank", {}).get(port, 99)
        weight_rank = self.WEIGHT_MAP.get(wt, 1)

        res = (adjusted_dep, port_rank, weight_rank)
        self._lrk_cache[cid] = res
        return res

    def _compute_erc(self, yard_state: YardState,
                     block: str, bay: int, row: int,
                     new_lrk: tuple) -> int:
        height = yard_state.get_stack_height(block, bay, row)
        if height == 0:
            return 0

        erc = 0
        for tier in range(1, height + 1):
            cid = yard_state.get_container_at(block, bay, row, tier)
            if cid is None:
                continue
            cinfo = yard_state.get_container_info(cid)
            if cinfo is None:
                continue
            existing_lrk = self._get_lrk(cinfo)
            if new_lrk > existing_lrk:
                erc += 1

        return erc

    def _score_stack(self, yard_state: YardState,
                     block: str, bay: int, row: int,
                     event: Event,
                     new_lrk: tuple,
                     occ_ratio: float) -> float:
        layout = self.block_layout[block]
        height = yard_state.get_stack_height(block, bay, row)

        # ── 1. Expected Reshuffle Cost ────────────────────────────
        erc = self._compute_erc(yard_state, block, bay, row, new_lrk)
        is_truck = getattr(event, "type", "") == "TRUCK_RECV"
        erc_eff = erc * (self.TRUCK_UNCERT_MULT if is_truck else 1.0)

        # ── 2. Height penalty (quadratic, scaled by occupancy) ────
        height_pen = (height ** 2) * 0.15 * max(occ_ratio, 0.1)

        # ── 3. Vessel cohesion bonus ──────────────────────────────
        cohesion_bonus = 0.0
        # Direct vertical cohesion
        if height > 0:
            top_cid = yard_state.get_container_at(block, bay, row, height)
            if top_cid:
                top_info = yard_state.get_container_info(top_cid)
                if top_info and top_info.vessel_id == event.vessel_id:
                    cohesion_bonus += 2.0
                    if top_info.port_of_discharge == event.port_of_discharge:
                        cohesion_bonus += 1.5

        # Horizontal cohesion (adjacent stacks in same block)
        max_bays = layout["bays"]
        max_rows = layout["rows"]
        nearby_count = 0
        for db in (-1, 1):
            adj_b = bay + db
            if 1 <= adj_b <= max_bays:
                adj_h = yard_state.get_stack_height(block, adj_b, row)
                if adj_h > 0:
                    adj_cid = yard_state.get_container_at(block, adj_b, row, adj_h)
                    if adj_cid:
                        adj_info = yard_state.get_container_info(adj_cid)
                        if adj_info and adj_info.vessel_id == event.vessel_id:
                            nearby_count += 1
        for dr in (-1, 1):
            adj_r = row + dr
            if 1 <= adj_r <= max_rows:
                adj_h = yard_state.get_stack_height(block, bay, adj_r)
                if adj_h > 0:
                    adj_cid = yard_state.get_container_at(block, bay, adj_r, adj_h)
                    if adj_cid:
                        adj_info = yard_state.get_container_info(adj_cid)
                        if adj_info and adj_info.vessel_id == event.vessel_id:
                            nearby_count += 1
        cohesion_bonus += min(nearby_count * 0.5, 1.5)

        # ── 4. Neighborhood penalty ───────────────────────────────
        neighborhood_pen = 0.0
        for delta_row in (-1, 1):
            adj_row = row + delta_row
            if 1 <= adj_row <= max_rows:
                adj_h = yard_state.get_stack_height(block, bay, adj_row)
                if adj_h > height + 1:
                    neighborhood_pen += self.NEIGHBORHOOD_PENALTY

        # ── 5. Adaptive block penalty ─────────────────────────────
        block_risk = self.block_reshuffle_rate.get(block, 0.0)
        block_pen = block_risk * 2.0

        # ── Dynamic weights ────────────────────────────────────────
        alpha = 100.0
        beta  = 1.0
        gamma = 0.5
        delta = 1.5 if occ_ratio > 0.80 else 3.0
        zeta  = 1.0

        score = (alpha * erc_eff
                 + beta  * height_pen
                 + gamma * neighborhood_pen
                 + zeta  * block_pen
                 - delta * cohesion_bonus)

        return score

    def _select_block(self, event: Event, yard_state: YardState) -> str:
        vid = getattr(event, "vessel_id", "") or ""

        # ── Priority 1: vessel already has containers in the yard ─────────────────────
        if vid:
            vessel_containers = yard_state.get_containers_by_vessel(vid)
            if vessel_containers:
                block_counts = defaultdict(int)
                for cid in vessel_containers:
                    pos = yard_state.get_container_position(cid)
                    if pos:
                        block_counts[pos.block] += 1
                if block_counts:
                    pinned = max(block_counts, key=block_counts.get)
                    occ, cap = yard_state.get_block_occupancy(pinned)
                    if occ / cap < self.MAX_BLOCK_OCCUPANCY:
                        self.vessel_to_block[vid] = pinned
                        return pinned

        # ── Priority 1.5: vessel already pinned in cache ─────────────────────
        if vid and vid in self.vessel_to_block:
            pinned = self.vessel_to_block[vid]
            occ, cap = yard_state.get_block_occupancy(pinned)
            if occ / cap < self.MAX_BLOCK_OCCUPANCY:
                return pinned

        # ── Priority 2: bucket-based ──────────────────────────────
        dep_time = self._parse_time(getattr(event, "departure_time", ""))
        bucket = self._get_departure_bucket(dep_time)
        candidates = self.bucket_blocks.get(bucket, list(self.block_layout.keys()))

        best_block = None
        best_free = -1
        for block in candidates:
            occ, cap = yard_state.get_block_occupancy(block)
            free = cap - occ
            frac = occ / cap
            if frac < self.MAX_BLOCK_OCCUPANCY and free > best_free:
                best_free = free
                best_block = block

        if best_block:
            if vid:
                self.vessel_to_block[vid] = best_block
            return best_block

        # ── Priority 3: any block with space ─────────────────────
        for block in sorted(self.block_layout.keys()):
            occ, cap = yard_state.get_block_occupancy(block)
            if occ < cap:
                return block

        return list(self.block_layout.keys())[0]

    def _find_best_position(self, yard_state: YardState,
                             block: str, event: Event,
                             new_lrk: tuple, occ_ratio: float) -> Position | None:
        available = self.non_full_stacks.get(block, set())
        if not available:
            return None

        best_score = float('inf')
        best_pos = None

        for (bay, row) in list(available):
            height = yard_state.get_stack_height(block, bay, row)
            limit = min(self.block_layout[block]["tiers"], self.MAX_STACK_HEIGHT)
            if height >= limit:
                self.non_full_stacks[block].discard((bay, row))
                continue

            tier = height + 1
            pos = Position(block, bay, row, tier)
            if not yard_state.is_position_valid(pos):
                continue

            score = self._score_stack(yard_state, block, bay, row, event, new_lrk, occ_ratio)
            if score < best_score:
                best_score = score
                best_pos = pos

        return best_pos

    def place_container(self, yard_state: YardState, event: Event) -> Position:
        # ── Record placed container type ──────────────────────────
        cid = event.container_id
        if event.type == "DISCHARGE":
            self.placed_imports.add(cid)
        elif event.type == "TRUCK_RECV":
            self.placed_exports.add(cid)

        new_lrk = self._get_lrk_from_event(event)
        occ_ratio = self._get_occ_ratio(yard_state)

        # ── Layer 1: preferred block ──────────────────────────────
        primary_block = self._select_block(event, yard_state)

        # ── Layer 2: search preferred block ───────────────────────
        pos = self._find_best_position(yard_state, primary_block, event, new_lrk, occ_ratio)
        if pos:
            erc = self._compute_erc(yard_state, pos.block, pos.bay, pos.row, new_lrk)
            if erc == 0:
                self._record_placement(pos, event)
                return pos

        # ── Layer 3: search all other blocks for an ERC-0 stack ───
        best_alt_pos = None
        best_alt_score = float('inf')
        for block in self.block_layout:
            if block == primary_block:
                continue
            alt_pos = self._find_best_position(yard_state, block, event, new_lrk, occ_ratio)
            if alt_pos:
                alt_erc = self._compute_erc(yard_state, alt_pos.block, alt_pos.bay, alt_pos.row, new_lrk)
                if alt_erc == 0:
                    score = self._score_stack(yard_state, alt_pos.block, alt_pos.bay, alt_pos.row, event, new_lrk, occ_ratio)
                    if score < best_alt_score:
                        best_alt_score = score
                        best_alt_pos = alt_pos

        if best_alt_pos:
            self._record_placement(best_alt_pos, event)
            return best_alt_pos

        # ── Layer 4: if no ERC-0 stack anywhere, use the best position in primary block ───
        if pos and yard_state.is_position_valid(pos):
            self._record_placement(pos, event)
            return pos

        # ── Overflow: try other blocks in priority order ───
        dep_time = self._parse_time(getattr(event, "departure_time", ""))
        bucket = self._get_departure_bucket(dep_time)

        tried = {primary_block}
        order = list(self.bucket_blocks.get(bucket, []))
        for b in (bucket - 1, bucket + 1, bucket - 2, bucket + 2):
            order.extend(self.bucket_blocks.get(b, []))
        for blk in self.block_layout:
            if blk not in order:
                order.append(blk)

        for block in order:
            if block in tried:
                continue
            tried.add(block)
            pos = self._find_best_position(yard_state, block, event, new_lrk, occ_ratio)
            if pos and yard_state.is_position_valid(pos):
                self._record_placement(pos, event)
                return pos

        # ── Absolute fallback ─────────────────────────────────────
        return self._fallback_greedy(yard_state)

    def on_event(self, event: Event) -> None:
        # Record the start time of the simulation
        if self.first_event_time is None:
            self.first_event_time = self._parse_time(event.timestamp)

        dep = getattr(event, "departure_time", None)
        if dep:
            t = self._parse_time(dep)
            if t > self.sim_end:
                self.sim_end = t

    def on_container_retrieved(self, container_id: str,
                               position: Position, reshuffles: int) -> None:
        if position:
            block = position.block
            bay = position.bay
            row = position.row

            # Stack just became shorter → add back to available set
            self.non_full_stacks[block].add((bay, row))

            # Update block statistics for adaptive penalty
            self.block_retrieval_count[block] += 1
            self.block_reshuffle_count[block] += reshuffles
            total = self.block_retrieval_count[block]
            if total > 0:
                self.block_reshuffle_rate[block] = (
                    self.block_reshuffle_count[block] / total
                )

    def _record_placement(self, pos: Position, event: Event) -> None:
        block, bay, row, tier = pos.block, pos.bay, pos.row, pos.tier
        max_tiers = self.block_layout[block]["tiers"]
        limit = min(max_tiers, self.MAX_STACK_HEIGHT)

        # If stack height reaches limit, remove from non_full_stacks cache
        if tier >= limit:
            self.non_full_stacks[block].discard((bay, row))
        else:
            self.non_full_stacks[block].add((bay, row))

    def _get_occ_ratio(self, yard_state: YardState) -> float:
        total_occ = 0
        total_cap = 0
        for block in self.block_layout:
            occ, cap = yard_state.get_block_occupancy(block)
            total_occ += occ
            total_cap += cap
        return total_occ / max(total_cap, 1)

    def _fallback_greedy(self, yard_state: YardState) -> Position:
        best_height = 999
        best_pos = None

        for block, layout in self.block_layout.items():
            for bay in range(1, layout["bays"] + 1):
                for row in range(1, layout["rows"] + 1):
                    h = yard_state.get_stack_height(block, bay, row)
                    if h >= layout["tiers"]:
                        continue
                    pos = Position(block, bay, row, h + 1)
                    if yard_state.is_position_valid(pos) and h < best_height:
                        best_height = h
                        best_pos = pos

        if best_pos is None:
            first_block = list(self.block_layout.keys())[0]
            return Position(first_block, 1, 1, 999)

        return best_pos
