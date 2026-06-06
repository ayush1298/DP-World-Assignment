import json
from solution.my_strategy import MyStrategy

with open("data/yard_layout.json") as f:
    layout = json.load(f)
with open("data/train/initial_state.json") as f:
    init = json.load(f)

strategy = MyStrategy()
strategy.initialize(layout, init)

print("sim_start:", strategy.sim_start)
print("sim_end:", strategy.sim_end)
print("vessel VSL003 schedule:", strategy.vessel_schedule.get("VSL003"))
print("non_full_stacks block count:", {b: len(strategy.non_full_stacks[b]) for b in strategy.non_full_stacks})
