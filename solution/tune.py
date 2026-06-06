import json
import time
from pathlib import Path
from collections import defaultdict
from src.yard_state import YardState
from src.event_reader import read_events
from src.simulator import Simulator
from solution.my_strategy import MyStrategy

def evaluate_config(data_dir, yard_layout, initial_state, events, config):
    # Create strategy and inject config parameters
    strategy = MyStrategy()
    for param, value in config.items():
        setattr(strategy, param, value)

    # Reset yard
    yard = YardState(yard_layout)
    yard.load_initial_state(initial_state)

    # Initialize strategy
    strategy.initialize(yard_layout, initial_state)

    # Run simulation
    sim = Simulator(yard, strategy, verbose=False)
    stats = sim.run(events)

    return stats.total_reshuffles, stats.reshuffles_per_retrieval, stats.quantitative_score()

def tune():
    data_dir = "data/train"
    layout_path = "data/yard_layout.json"
    initial_state_path = "data/train/initial_state.json"
    events_path = "data/train/events.jsonl"

    print("Loading data...")
    with open(layout_path) as f:
        yard_layout = json.load(f)
    with open(initial_state_path) as f:
        initial_state = json.load(f)
    events = read_events(events_path)
    print(f"Loaded {len(events)} events.")

    # Define hyperparameter grid
    # We will tune one or two at a time to keep it fast, or define a small grid
    grid = [
        {"N_BUCKETS": 5, "MAX_STACK_HEIGHT": 4, "MAX_BLOCK_OCCUPANCY": 0.88, "TRUCK_UNCERT_MULT": 1.2, "NEIGHBORHOOD_PENALTY": 0.4}, # Default (36.4)
        {"N_BUCKETS": 4, "MAX_STACK_HEIGHT": 4, "MAX_BLOCK_OCCUPANCY": 0.88, "TRUCK_UNCERT_MULT": 1.2, "NEIGHBORHOOD_PENALTY": 0.4},
        {"N_BUCKETS": 6, "MAX_STACK_HEIGHT": 4, "MAX_BLOCK_OCCUPANCY": 0.88, "TRUCK_UNCERT_MULT": 1.2, "NEIGHBORHOOD_PENALTY": 0.4},
        {"N_BUCKETS": 5, "MAX_STACK_HEIGHT": 4, "MAX_BLOCK_OCCUPANCY": 0.85, "TRUCK_UNCERT_MULT": 1.2, "NEIGNBORHOOD_PENALTY": 0.4},
        {"N_BUCKETS": 5, "MAX_STACK_HEIGHT": 4, "MAX_BLOCK_OCCUPANCY": 0.92, "TRUCK_UNCERT_MULT": 1.2, "NEIGHBORHOOD_PENALTY": 0.4},
        {"N_BUCKETS": 5, "MAX_STACK_HEIGHT": 4, "MAX_BLOCK_OCCUPANCY": 0.88, "TRUCK_UNCERT_MULT": 1.0, "NEIGHBORHOOD_PENALTY": 0.4},
        {"N_BUCKETS": 5, "MAX_STACK_HEIGHT": 4, "MAX_BLOCK_OCCUPANCY": 0.88, "TRUCK_UNCERT_MULT": 1.4, "NEIGHBORHOOD_PENALTY": 0.4},
        {"N_BUCKETS": 5, "MAX_STACK_HEIGHT": 4, "MAX_BLOCK_OCCUPANCY": 0.88, "TRUCK_UNCERT_MULT": 1.2, "NEIGHBORHOOD_PENALTY": 0.2},
        {"N_BUCKETS": 5, "MAX_STACK_HEIGHT": 4, "MAX_BLOCK_OCCUPANCY": 0.88, "TRUCK_UNCERT_MULT": 1.2, "NEIGHBORHOOD_PENALTY": 0.6},
        # Try MAX_STACK_HEIGHT = 5
        {"N_BUCKETS": 5, "MAX_STACK_HEIGHT": 5, "MAX_BLOCK_OCCUPANCY": 0.88, "TRUCK_UNCERT_MULT": 1.2, "NEIGHBORHOOD_PENALTY": 0.4},
    ]

    print("\n--- Starting Tuning Grid Search ---")
    best_reshuffles = float('inf')
    best_config = None
    best_score = 0.0

    for idx, config in enumerate(grid):
        start_t = time.time()
        reshuffles, rate, score = evaluate_config(data_dir, yard_layout, initial_state, events, config)
        elapsed = time.time() - start_t
        print(f"Config {idx+1}/{len(grid)}: Reshuffles={reshuffles}, Rate={rate:.4f}, Score={score:.1f} | Config={config} | Took {elapsed:.1f}s")
        if reshuffles < best_reshuffles:
            best_reshuffles = reshuffles
            best_config = config
            best_score = score

    print("\n=== BEST CONFIGURATION ===")
    print("Reshuffles:", best_reshuffles)
    print("Score:", best_score)
    print("Config:", best_config)

if __name__ == "__main__":
    tune()
