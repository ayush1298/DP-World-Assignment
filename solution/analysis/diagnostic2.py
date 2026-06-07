"""Deep diagnostic: trace reshuffle heights, cascade chains, and block disorder."""
import json, sys, os
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.yard_state import YardState
from src.event_reader import read_events
from src.models import Position, Event
from src.simulator import Simulator
from solution.my_strategy import MyStrategy


def run_diagnostic(data_dir):
    with open("data/yard_layout.json") as f:
        yard_layout = json.load(f)
    with open(f"{data_dir}/initial_state.json") as f:
        initial_state = json.load(f)
    events = read_events(f"{data_dir}/events.jsonl")

    yard = YardState(yard_layout)
    yard.load_initial_state(initial_state)

    strategy = MyStrategy()
    strategy.data_dir = data_dir
    strategy.initialize(yard_layout, initial_state)

    # Analyze initial state disorder per block
    print("Initial state disorder analysis:")
    block_inversions = defaultdict(int)
    block_containers = defaultdict(int)

    for block_name, bi in yard.blocks.items():
        for bay in range(1, bi.bays + 1):
            for row in range(1, bi.rows + 1):
                h = yard.get_stack_height(block_name, bay, row)
                if h <= 1:
                    continue
                lrks = []
                for tier in range(1, h + 1):
                    cid = yard.get_container_at(block_name, bay, row, tier)
                    if cid:
                        cinfo = yard.get_container_info(cid)
                        if cinfo:
                            lrks.append(strategy._get_lrk(cinfo))
                            block_containers[block_name] += 1
                for i in range(len(lrks) - 1):
                    if lrks[i] < lrks[i + 1]:
                        block_inversions[block_name] += 1

    for block in sorted(yard.blocks.keys()):
        occ, cap = yard.get_block_occupancy(block)
        inv = block_inversions.get(block, 0)
        print(f"  {block}: {occ}/{cap} containers, {inv} inversions")

    # Run simulation tracking reshuffle heights
    reshuffle_height_dist = Counter()
    reshuffle_per_block = Counter()
    containers_reshuffled_times = Counter()
    reshuffle_chain_lengths = []

    initial_cids = set(c["container_id"] for c in initial_state["containers"])

    for event in events:
        strategy.on_event(event)

        if event.type in ("DISCHARGE", "TRUCK_RECV"):
            pos = strategy.place_container(yard, event)
            container = event.to_container()
            yard.place_container(container, pos)

        elif event.type in ("LOAD", "TRUCK_DLVR"):
            cid = event.container_id
            cpos = yard.get_container_position(cid)
            if cpos is None:
                continue

            above = yard.get_containers_above(cid)
            reshuffles = len(above)

            if reshuffles > 0:
                stack_height = yard.get_stack_height(cpos.block, cpos.bay, cpos.row)
                reshuffle_height_dist[stack_height] += reshuffles
                reshuffle_per_block[cpos.block] += reshuffles
                for acid in above:
                    containers_reshuffled_times[acid] += 1

            # Do the actual retrieval (matching simulator logic)
            block = cpos.block
            temp = []
            for acid in reversed(above):
                info = yard.get_container_info(acid)
                rpos = yard.remove_container(acid)
                if rpos and info:
                    temp.append((acid, info))
            yard.remove_container(cid)
            for acid, cont in reversed(temp):
                bi = yard.blocks.get(block)
                if not bi:
                    continue
                best_h = bi.tiers + 1
                best_pos = None
                for b in range(1, bi.bays + 1):
                    for r in range(1, bi.rows + 1):
                        h = yard.get_stack_height(block, b, r)
                        if h < bi.tiers and h < best_h:
                            best_h = h
                            best_pos = Position(block, b, r, h + 1)
                if best_pos:
                    yard.place_container(cont, best_pos)

            strategy.on_container_retrieved(cid, cpos, reshuffles)

    total_reshuffles = sum(reshuffle_height_dist.values())
    print(f"\nReshuffles by stack height at retrieval time:")
    for h in sorted(reshuffle_height_dist.keys()):
        cnt = reshuffle_height_dist[h]
        print(f"  Height {h}: {cnt} reshuffles ({100*cnt/total_reshuffles:.1f}%)")

    print(f"\nReshuffles by block:")
    for block in sorted(reshuffle_per_block.keys()):
        cnt = reshuffle_per_block[block]
        print(f"  {block}: {cnt} reshuffles ({100*cnt/total_reshuffles:.1f}%)")

    multi_reshuffled = sum(1 for v in containers_reshuffled_times.values() if v > 1)
    max_times = max(containers_reshuffled_times.values()) if containers_reshuffled_times else 0
    print(f"\nContainers reshuffled multiple times: {multi_reshuffled}")
    print(f"Max times a single container was reshuffled: {max_times}")
    print(f"Reshuffle frequency distribution:")
    freq = Counter(containers_reshuffled_times.values())
    for times in sorted(freq.keys()):
        print(f"  {times} times: {freq[times]} containers")


if __name__ == "__main__":
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data/train"
    run_diagnostic(data_dir)
