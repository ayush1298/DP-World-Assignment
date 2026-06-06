"""
solution/analysis.py  —  Run with:  python solution/analysis.py
"""
import json, collections, statistics
from pathlib import Path

DATA = Path("data")

def analyze():
    from datetime import datetime

    # Load vessel schedule
    with open(DATA / "vessel_schedule.json") as f:
        schedule_data = json.load(f)
        schedule = schedule_data["vessels"]

    # Load train events
    events = []
    with open(DATA / "train/events.jsonl") as f:
        for line in f:
            events.append(json.loads(line))

    # Load initial state
    with open(DATA / "train/initial_state.json") as f:
        init = json.load(f)

    # ── Q1: Event type distribution ──────────────────────────
    type_counts = collections.Counter(e["type"] for e in events)
    print("Event type distribution:", dict(type_counts))

    # ── Q2: Containers per vessel ─────────────────────────────
    vessel_counts = collections.Counter(
        e.get("vessel_id") for e in events
        if e["type"] in ("DISCHARGE","TRUCK_RECV")
    )
    print("Top 10 vessels by discharge volume:", vessel_counts.most_common(10))

    # ── Q3: Departure time range ──────────────────────────────
    dep_times = [datetime.fromisoformat(e["departure_time"]).timestamp() for e in events if e.get("departure_time")]
    print(f"Departure time range (timestamps): {min(dep_times):.0f} to {max(dep_times):.0f}")
    print(f"  span = {(max(dep_times)-min(dep_times))/86400:.1f} days")

    # ── Q4: Weight class distribution ────────────────────────
    wt = collections.Counter(
        e.get("weight_class") for e in events
        if e["type"] in ("DISCHARGE","TRUCK_RECV")
    )
    print("Weight class distribution:", dict(wt))

    # ── Q5: Ports per vessel ──────────────────────────────────
    vessel_ports = collections.defaultdict(set)
    for e in events:
        if e.get("vessel_id") and e.get("port_of_discharge"):
            vessel_ports[e["vessel_id"]].add(e["port_of_discharge"])
    port_counts = {v: len(p) for v, p in vessel_ports.items()}
    print("Avg ports per vessel:", statistics.mean(port_counts.values()) if port_counts else 0)
    print("Max ports per vessel:", max(port_counts.values()) if port_counts else 0)

    # ── Q6: Initial yard occupancy by block ──────────────────
    block_occ = collections.Counter()
    for c in init.get("containers", []):
        block_occ[c["position"]["block"]] += 1
    print("Initial block occupancy:", dict(sorted(block_occ.items())))

    # ── Q7: Are departure times continuous floats or day-level? ──
    sample = sorted(dep_times)[:20]
    print("Sample departure times (first 20):", sample)

if __name__ == "__main__":
    analyze()
