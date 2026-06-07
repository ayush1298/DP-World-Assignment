import os
import sys
import json
import pickle
import numpy as np
from pathlib import Path
from collections import defaultdict
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score

# Ensure the root of the workspace is in the python path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.models import Event
from src.yard_state import YardState
from src.simulator import Simulator
from src.event_reader import read_events
from solution.my_strategy import MyStrategy

FEATURE_NAMES = [
    "erc_exact", "erc_0_or_not", "stack_height", "height_to_max_ratio",
    "vessel_purity_ratio", "port_purity_ratio", "lrk_compat_ratio",
    "days_to_departure", "departure_bucket", "is_urgent", "is_truck",
    "block_occ_ratio", "block_reshuffle_rate", "block_erc_total",
    "vessel_in_block_count", "erc_times_height", "urgent_and_dirty",
    "lrk_delta_top",
]

class LoggingStrategy(MyStrategy):
    """Subclass of MyStrategy that logs features during placement and links them to eventual reshuffles."""
    
    def initialize(self, yard_layout: dict, initial_state: dict) -> None:
        # Set data_dir explicitly so it can load schedule and events
        self.data_dir = "data/train"
        super().initialize(yard_layout, initial_state)
        self.placement_snapshot = {}
        self.placement_log = []

    def place_container(self, yard_state: YardState, event: Event):
        # Delegate to base placement logic (which is AnalyticalRolloutStrategy / MyStrategy)
        pos = super().place_container(yard_state, event)
        if pos and event.container_id:
            # Record the features of the chosen stack before the container is placed
            feats = self._compute_features_dict(yard_state, pos.block, pos.bay, pos.row, event)
            self.placement_snapshot[event.container_id] = feats
        return pos

    def on_container_retrieved(self, container_id: str, position, reshuffles: int) -> None:
        super().on_container_retrieved(container_id, position, reshuffles)
        if container_id in self.placement_snapshot:
            feats = self.placement_snapshot.pop(container_id)
            self.placement_log.append({
                "features": feats,
                "reshuffles": reshuffles
            })

    def save_log(self, path="data/train_placement_log.jsonl"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            for rec in self.placement_log:
                f.write(json.dumps(rec) + "\n")
        print(f"Saved {len(self.placement_log)} placement records to {path}")

def run_simulation_and_collect_data():
    layout_path = "data/yard_layout.json"
    initial_state_path = "data/train/initial_state.json"
    events_path = "data/train/events.jsonl"

    print("Loading layout and initial state...")
    with open(layout_path) as f:
        layout = json.load(f)
    with open(initial_state_path) as f:
        init = json.load(f)

    print("Loading events...")
    events = read_events(events_path)

    yard = YardState(layout)
    yard.load_initial_state(init)

    strategy = LoggingStrategy()
    strategy.initialize(layout, init)

    print("Running simulation to collect training data...")
    sim = Simulator(yard, strategy, verbose=False)
    sim.run(events)

    log_path = "data/train_placement_log.jsonl"
    strategy.save_log(log_path)
    return log_path

def train_model(log_path):
    print("Loading training records...")
    records = []
    with open(log_path) as f:
        for line in f:
            records.append(json.loads(line))

    print(f"Loaded {len(records)} placement records.")
    
    # Filter records that have complete features
    valid_records = []
    for r in records:
        if all(f in r["features"] for f in FEATURE_NAMES):
            valid_records.append(r)
            
    print(f"Valid records for training: {len(valid_records)}")
    
    X = np.array([[r["features"][f] for f in FEATURE_NAMES] for r in valid_records])
    y = np.array([int(r["reshuffles"] > 0) for r in valid_records])
    
    print(f"Class balance (reshuffle rate): {y.mean():.2%}")

    print("Training Gradient Boosting model...")
    model = GradientBoostingClassifier(
        n_estimators=150,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        min_samples_leaf=20,
        random_state=42
    )

    # Perform cross validation
    scores = cross_val_score(model, X, y, cv=5, scoring="roc_auc")
    print(f"5-fold cross-validation AUC: {scores.mean():.4f} ± {scores.std():.4f}")

    # Train final model
    model.fit(X, y)

    # Save model
    model_dir = "solution"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "reshuffle_scorer.pkl")
    with open(model_path, "wb") as f:
        pickle.dump({"model": model, "features": FEATURE_NAMES}, f)
    print(f"Saved trained ML model to {model_path}")

    # Print feature importance
    print("Feature importances:")
    for name, imp in sorted(zip(FEATURE_NAMES, model.feature_importances_), key=lambda x: -x[1]):
        print(f"  {name}: {imp:.4f}")

if __name__ == "__main__":
    log_path = "data/train_placement_log.jsonl"
    if not os.path.exists(log_path):
        log_path = run_simulation_and_collect_data()
    train_model(log_path)
