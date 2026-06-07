"""Quick sweep of targeted improvements."""
import json, sys, os, types, copy

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.yard_state import YardState
from src.event_reader import read_events
from src.simulator import Simulator


def run_config(name, config_fn, data_dir):
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
    config_fn(strategy)
    strategy.initialize(yard_layout, initial_state)

    sim = Simulator(yard, strategy, verbose=False)
    stats = sim.run(events)
    print(f"  {name:40s}: reshuf={stats.total_reshuffles:5d} "
          f"ratio={stats.reshuffles_per_retrieval:.4f} "
          f"score={stats.quantitative_score():.1f}", flush=True)
    return stats.total_reshuffles


def main():
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data/train"
    print(f"Quick sweep on {data_dir}", flush=True)

    def baseline(s): pass

    def max_h4(s): s.MAX_STACK_HEIGHT = 4

    def max_h3(s): s.MAX_STACK_HEIGHT = 3

    def mix_pen_5(s): s.MIXING_PENALTY = 5.0

    def mix_pen_0(s): s.MIXING_PENALTY = 0.0

    def mix_pen_25(s): s.MIXING_PENALTY = 25.0

    def max_occ_80(s): s.MAX_BLOCK_OCCUPANCY = 0.80

    def max_occ_75(s): s.MAX_BLOCK_OCCUPANCY = 0.75

    def truck_unc_1(s): s.TRUCK_UNCERT_MULT = 1.0

    def truck_unc_2(s): s.TRUCK_UNCERT_MULT = 2.0

    def height_coeff_025(s): s.HEIGHT_PENALTY_COEFF = 0.25

    def height_coeff_050(s): s.HEIGHT_PENALTY_COEFF = 0.50

    def no_preassign(s): s.ENABLE_T2_PREASSIGN = False

    def no_rollout(s): s.ENABLE_T3_ROLLOUT = False

    def combo_1(s):
        s.MIXING_PENALTY = 5.0
        s.HEIGHT_PENALTY_COEFF = 0.25

    def combo_2(s):
        s.MIXING_PENALTY = 5.0
        s.MAX_BLOCK_OCCUPANCY = 0.80

    def combo_3(s):
        s.MIXING_PENALTY = 0.0
        s.HEIGHT_PENALTY_COEFF = 0.25
        s.MAX_BLOCK_OCCUPANCY = 0.80

    configs = [
        ("baseline", baseline),
        ("MAX_H=4", max_h4),
        ("MAX_H=3", max_h3),
        ("MIXING=5", mix_pen_5),
        ("MIXING=0", mix_pen_0),
        ("MIXING=25", mix_pen_25),
        ("MAX_OCC=0.80", max_occ_80),
        ("MAX_OCC=0.75", max_occ_75),
        ("TRUCK_UNC=1.0", truck_unc_1),
        ("TRUCK_UNC=2.0", truck_unc_2),
        ("HEIGHT_COEFF=0.25", height_coeff_025),
        ("HEIGHT_COEFF=0.50", height_coeff_050),
        ("NO_PREASSIGN", no_preassign),
        ("NO_ROLLOUT", no_rollout),
        ("MIXING=5+HEIGHT=0.25", combo_1),
        ("MIXING=5+OCC=0.80", combo_2),
        ("MIXING=0+HEIGHT=0.25+OCC=0.80", combo_3),
    ]

    for name, fn in configs:
        run_config(name, fn, data_dir)


if __name__ == "__main__":
    main()
