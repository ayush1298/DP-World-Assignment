"""Sweep different ERC-0 tiebreaker strategies."""
import json, sys, os, copy

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.yard_state import YardState
from src.event_reader import read_events
from src.simulator import Simulator


def run_with_tiebreaker(name, tiebreaker_fn, data_dir):
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

    # Monkey-patch the tiebreaker if provided
    if tiebreaker_fn:
        original_method = strategy._find_best_position_with_rollout.__func__

        def patched_rollout(self, yard_state, block, event, new_lrk, occ_ratio):
            available = self.non_full_stacks.get(block, set())
            if not available:
                return None

            from src.models import Position
            candidates = []
            zero_erc_candidates = []
            max_h = self.MAX_STACK_HEIGHT

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
                    sort_key = tiebreaker_fn(self, yard_state, block, bay, row, h, event)
                    scored.append((sort_key, bay, row, tier))
                scored.sort(key=lambda x: x[0])
                _, bay, row, tier = scored[0]
                return Position(block, bay, row, tier)

            if not candidates:
                return None
            candidates.sort(key=lambda x: x[0])
            _, bay, row, tier = candidates[0]
            return Position(block, bay, row, tier)

        import types
        strategy._find_best_position_with_rollout = types.MethodType(patched_rollout, strategy)

    strategy.initialize(yard_layout, initial_state)
    sim = Simulator(yard, strategy, verbose=False)
    stats = sim.run(events)
    print(f"  {name:35s}: reshuf={stats.total_reshuffles:5d} "
          f"ratio={stats.reshuffles_per_retrieval:.4f} "
          f"score={stats.quantitative_score():.1f}", flush=True)


def main():
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data/train"
    print(f"Testing ERC-0 tiebreaker strategies on {data_dir}", flush=True)

    # Original tiebreaker: h * 2.0 - hom
    def tb_original(self, ys, block, bay, row, h, event):
        hom = self._stack_homogeneity_score(ys, block, bay, row, event)
        return h * 2.0 - hom

    # Only homogeneity, no height
    def tb_hom_only(self, ys, block, bay, row, h, event):
        hom = self._stack_homogeneity_score(ys, block, bay, row, event)
        return -hom

    # Only height, no homogeneity  
    def tb_height_only(self, ys, block, bay, row, h, event):
        return h * 2.0

    # Stronger height preference
    def tb_strong_height(self, ys, block, bay, row, h, event):
        hom = self._stack_homogeneity_score(ys, block, bay, row, event)
        return h * 4.0 - hom

    # Prefer empty stacks explicitly
    def tb_empty_bonus(self, ys, block, bay, row, h, event):
        hom = self._stack_homogeneity_score(ys, block, bay, row, event)
        empty_bonus = -3.0 if h == 0 else 0.0
        return h * 2.0 - hom + empty_bonus

    # Full scoring function
    def tb_full_score(self, ys, block, bay, row, h, event):
        new_lrk = self._get_lrk_from_event(event)
        occ_ratio = self._get_occ_ratio(ys)
        return self._score_stack(ys, block, bay, row, event, new_lrk, occ_ratio)

    # Height-aware: prefer shorter stacks among same-vessel, taller among empty
    def tb_adaptive(self, ys, block, bay, row, h, event):
        hom = self._stack_homogeneity_score(ys, block, bay, row, event)
        if h == 0:
            return 1.0  # neutral for empty stacks
        elif hom >= 3.0:  # same vessel
            return h * 1.5 - hom  # slight height preference
        else:
            return h * 3.0 - hom  # strong height preference for mixed

    configs = [
        ("baseline (h*2-hom)", None),
        ("hom_only (-hom)", tb_hom_only),
        ("height_only (h*2)", tb_height_only),
        ("strong_height (h*4-hom)", tb_strong_height),
        ("empty_bonus (h*2-hom-3_empty)", tb_empty_bonus),
        ("full_score", tb_full_score),
        ("adaptive", tb_adaptive),
    ]

    for name, fn in configs:
        run_with_tiebreaker(name, fn, data_dir)


if __name__ == "__main__":
    main()
    
