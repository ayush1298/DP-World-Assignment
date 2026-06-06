import json
from src.yard_state import YardState
from src.event_reader import read_events
from solution.my_strategy import MyStrategy

with open("data/yard_layout.json") as f:
    layout = json.load(f)
with open("data/train/initial_state.json") as f:
    init = json.load(f)

events = read_events("data/train/events.jsonl")

yard = YardState(layout)
yard.load_initial_state(init)

strategy = MyStrategy()
strategy.initialize(layout, init)

# Let us find the first 5 placement events (DISCHARGE or TRUCK_RECV)
placement_events = [e for e in events if e.type in ("DISCHARGE", "TRUCK_RECV")][:5]

for idx, e in enumerate(placement_events):
    print(f"\n--- Event {idx+1}: {e.type} for container {e.container_id} (Vessel: {e.vessel_id}, Port: {e.port_of_discharge}, Weight: {e.weight_class}, DepTime: {e.departure_time}) ---")
    
    # Let's see what block is chosen
    primary_block = strategy._select_block(e, yard)
    print(f"Chosen block: {primary_block}")
    
    # Let's inspect some of the scores in this block
    available = list(strategy.non_full_stacks[primary_block])[:10]
    print(f"Sample stacks evaluated in {primary_block}:")
    for bay, row in available:
        h = yard.get_stack_height(primary_block, bay, row)
        score = strategy._score_stack(yard, primary_block, bay, row, e)
        erc = strategy._compute_erc(yard, primary_block, bay, row, strategy._get_lrk_from_event(e))
        print(f"  (bay={bay}, row={row}, height={h}) -> Score={score:.2f}, ERC={erc}")
        
    # Let's get the final chosen position
    pos = strategy.place_container(yard, e)
    print(f"Placed at: {pos}")
    # Apply to yard state so next event sees it
    container = e.to_container()
    yard.place_container(container, pos)
