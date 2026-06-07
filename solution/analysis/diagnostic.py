"""Diagnostic: trace the root causes of reshuffles."""
import json, sys, os
from collections import Counter, defaultdict
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.yard_state import YardState
from src.event_reader import read_events
from src.models import Position, Event
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

    # Track which containers are from initial state vs placed by us
    initial_cids = set(c["container_id"] for c in initial_state["containers"])
    our_placements = {}  # cid -> (position, erc_at_placement, is_permanent)

    # Track reshuffles
    reshuffles_from_initial = 0
    reshuffles_from_ours = 0
    reshuffles_from_reshuffle_placed = 0
    reshuffle_placed_cids = set()  # containers placed by simulator during reshuffles

    erc_distribution = Counter()
    permanent_erc_distribution = Counter()
    finite_erc_distribution = Counter()
    total_placements = 0
    permanent_placements = 0

    for event in events:
        strategy.on_event(event)

        if event.type in ("DISCHARGE", "TRUCK_RECV"):
            total_placements += 1
            new_lrk = strategy._get_lrk_from_event(event)
            is_perm = strategy.use_exact_times and new_lrk[0] == float('inf')
            if is_perm:
                permanent_placements += 1

            pos = strategy.place_container(yard, event)
            erc = strategy._compute_erc(yard, pos.block, pos.bay, pos.row, new_lrk)
            erc_distribution[erc] += 1
            if is_perm:
                permanent_erc_distribution[erc] += 1
            else:
                finite_erc_distribution[erc] += 1

            our_placements[event.container_id] = {
                "erc": erc, "permanent": is_perm,
                "block": pos.block, "bay": pos.bay, "row": pos.row
            }

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
                for above_cid in above:
                    if above_cid in initial_cids:
                        reshuffles_from_initial += 1
                    elif above_cid in our_placements:
                        reshuffles_from_ours += 1
                    elif above_cid in reshuffle_placed_cids:
                        reshuffles_from_reshuffle_placed += 1
                    else:
                        reshuffles_from_initial += 1

                # Simulate the reshuffle placement by simulator
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
                    for bay in range(1, bi.bays + 1):
                        for row in range(1, bi.rows + 1):
                            h = yard.get_stack_height(block, bay, row)
                            if h < bi.tiers and h < best_h:
                                best_h = h
                                best_pos = Position(block, bay, row, h + 1)
                    if best_pos:
                        yard.place_container(cont, best_pos)
                        reshuffle_placed_cids.add(acid)

                strategy.on_container_retrieved(cid, cpos, reshuffles)
            else:
                yard.remove_container(cid)
                strategy.on_container_retrieved(cid, cpos, 0)

    total_reshuffles = reshuffles_from_initial + reshuffles_from_ours + reshuffles_from_reshuffle_placed
    print(f"\n{'='*60}")
    print(f"DIAGNOSTIC RESULTS for {data_dir}")
    print(f"{'='*60}")
    print(f"Total reshuffles: {total_reshuffles}")
    print(f"  From initial-state containers: {reshuffles_from_initial} ({100*reshuffles_from_initial/max(total_reshuffles,1):.1f}%)")
    print(f"  From our placements: {reshuffles_from_ours} ({100*reshuffles_from_ours/max(total_reshuffles,1):.1f}%)")
    print(f"  From simulator reshuffle placement: {reshuffles_from_reshuffle_placed} ({100*reshuffles_from_reshuffle_placed/max(total_reshuffles,1):.1f}%)")
    print()
    print(f"ERC distribution at placement time:")
    for erc_val in sorted(erc_distribution.keys()):
        cnt = erc_distribution[erc_val]
        print(f"  ERC={erc_val}: {cnt:5d} ({100*cnt/total_placements:.1f}%)")
    print()
    print(f"Permanent containers: {permanent_placements}/{total_placements} ({100*permanent_placements/total_placements:.1f}%)")
    print(f"  Permanent ERC distribution:")
    for erc_val in sorted(permanent_erc_distribution.keys()):
        cnt = permanent_erc_distribution[erc_val]
        print(f"    ERC={erc_val}: {cnt:5d} ({100*cnt/max(permanent_placements,1):.1f}%)")
    print(f"  Finite ERC distribution:")
    for erc_val in sorted(finite_erc_distribution.keys()):
        cnt = finite_erc_distribution[erc_val]
        finite_total = total_placements - permanent_placements
        print(f"    ERC={erc_val}: {cnt:5d} ({100*cnt/max(finite_total,1):.1f}%)")


if __name__ == "__main__":
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data/train"
    run_diagnostic(data_dir)
