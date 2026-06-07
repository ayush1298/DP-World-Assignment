"""Sweep empty bonus values."""
import json, sys, os, types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.yard_state import YardState
from src.event_reader import read_events
from src.simulator import Simulator


def run_config(name, empty_val, data_dir):
    from solution.my_strategy import MyStrategy

    with open("data/yard_layout.json") as f:
        yard_layout = json.load(f)
    with open(f"{data_dir}/initial_state.json") as f:
        initial_state = json.load(f)
    events = read_events(f"{data_dir}/events.jsonl")

    yard = YardState(yard_layout)
    yard.load_initial_state(initial_state)

    strategy = MyStrategy()
    strategy.data_dir = data_dir
    strategy.EMPTY_BONUS = empty_val
    
    original = strategy._find_best_position_with_rollout.__func__

    def patched(self, yard_state, block, event, new_lrk, occ_ratio):
        available = self.non_full_stacks.get(block, set())
        if not available:
            return None

        from src.models import Position
        candidates = []
        zero_erc_candidates = []
        max_h = self._get_effective_max_height(occ_ratio) if self.ENABLE_FIX_D else self.MAX_STACK_HEIGHT

        for (bay, row) in list(available):
            height = yard_state.get_stack_height(block, bay, row)
            max_tiers = self.block_layout[block]["tiers"]
            if height >= max_tiers:
                self.non_full_stacks[block].discard((bay, row))
                continue
            effective_limit = min(max_tiers, max_h)
            if height >= effective_limit:
                continue
            tier = height + 1
            pos = Position(block, bay, row, tier)
            if not yard_state.is_position_valid(pos):
                continue
            erc = self._compute_erc(yard_state, block, bay, row, new_lrk)
            score = self._score_stack(yard_state, block, bay, row, event, new_lrk, occ_ratio)
            if erc == 0:
                zero_erc_candidates.append((height, bay, row, tier))
            elif score < float('inf'):
                candidates.append((score, bay, row, tier))

        if zero_erc_candidates:
            scored = []
            for (h, bay, row, tier) in zero_erc_candidates:
                hom = self._stack_homogeneity_score(yard_state, block, bay, row, event)
                eb = self.EMPTY_BONUS if h == 0 else 0.0
                sort_key = h * 2.0 - hom + eb
                scored.append((sort_key, bay, row, tier))
            scored.sort(key=lambda x: x[0])
            _, bay, row, tier = scored[0]
            return Position(block, bay, row, tier)

        if not candidates:
            return None
        candidates.sort(key=lambda x: x[0])

        run_rollout = (
            self.ROLLOUT_ENABLED and len(candidates) > 1
            and occ_ratio > 0.40 and candidates[0][0] > 0
        )
        if not run_rollout:
            _, bay, row, tier = candidates[0]
            return Position(block, bay, row, tier)

        top_score = candidates[0][0]
        rollout_pool = [c for c in candidates[:self.ROLLOUT_K] if c[0] <= top_score * 1.2 + 2.0]
        best_rollout_score = float('inf')
        best_pos = None
        for (greedy_score, bay, row, tier) in rollout_pool:
            erc_placement = self._compute_erc(yard_state, block, bay, row, new_lrk)
            block_erc_now = self._block_total_erc(yard_state, block)
            block_erc_after = block_erc_now + erc_placement
            rollout_score = greedy_score + self.ROLLOUT_FUTURE_WEIGHT * block_erc_after
            if rollout_score < best_rollout_score:
                best_rollout_score = rollout_score
                best_pos = Position(block, bay, row, tier)
        return best_pos

    strategy._find_best_position_with_rollout = types.MethodType(patched, strategy)
    strategy.initialize(yard_layout, initial_state)

    sim = Simulator(yard, strategy, verbose=False)
    stats = sim.run(events)
    print(f"  {name:30s}: reshuf={stats.total_reshuffles:5d} "
          f"ratio={stats.reshuffles_per_retrieval:.4f} "
          f"score={stats.quantitative_score():.1f}", flush=True)


def main():
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data/train"
    print(f"Empty bonus sweep on {data_dir}", flush=True)

    values = [0, -1.0, -2.0, -3.0, -4.0, -5.0, -7.0, -10.0, -15.0, -20.0]
    for v in values:
        name = f"empty_bonus={v:.1f}"
        run_config(name, v, data_dir)


if __name__ == "__main__":
    main()
