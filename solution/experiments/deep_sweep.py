"""Sweep deeper combinations on top of LRK proximity=80, h_weight=0.5."""
import sys, os, json, subprocess

strategy_path = os.path.join(os.path.dirname(__file__), "my_strategy.py")

with open(strategy_path) as f:
    original = f.read()

configs = [
    ("baseline", {}),
    # Adjust LRK gap for permanent containers
    ("perm_gap_2.0", {"lrk_gap = 1.0  # permanent below - moderate penalty": "lrk_gap = 2.0  # permanent below - higher penalty"}),
    ("perm_gap_0.5", {"lrk_gap = 1.0  # permanent below - moderate penalty": "lrk_gap = 0.5  # permanent below - lower penalty"}),
    ("perm_gap_0.0", {"lrk_gap = 1.0  # permanent below - moderate penalty": "lrk_gap = 0.0  # no penalty for permanent below"}),
    # Global ERC-0 search (always evaluate all blocks)
    ("global_erc0",
     {"if erc == 0:\n                self._record_placement(pos, event)\n                return pos":
      "if erc == 0:\n                global_best_pos = pos\n                global_best_score = self._score_stack(yard_state, pos.block, pos.bay, pos.row, event, new_lrk, occ_ratio)"}),
    # Higher alpha for ERC weight
    ("alpha_150", {"alpha = getattr(self, \"SCORE_ALPHA\", 100.0)": "alpha = 150.0"}),
    # Truck uncertainty
    ("truck_1.5", {"TRUCK_UNCERT_MULT     = 1.35": "TRUCK_UNCERT_MULT     = 1.50"}),
    ("truck_1.0", {"TRUCK_UNCERT_MULT     = 1.35": "TRUCK_UNCERT_MULT     = 1.00"}),
    # Block occupancy
    ("occ_0.95", {"MAX_BLOCK_OCCUPANCY   = 0.88": "MAX_BLOCK_OCCUPANCY   = 0.95"}),
    ("occ_0.80", {"MAX_BLOCK_OCCUPANCY   = 0.88": "MAX_BLOCK_OCCUPANCY   = 0.80"}),
    # Neighborhood penalty
    ("no_nbr_pen", {"NEIGHBORHOOD_PENALTY  = 0.4": "NEIGHBORHOOD_PENALTY  = 0.0"}),
    ("high_nbr_pen", {"NEIGHBORHOOD_PENALTY  = 0.4": "NEIGHBORHOOD_PENALTY  = 1.0"}),
    # Cohesion delta values
    ("delta_high_2.0", {"SCORE_DELTA_HIGH\", 1.5)": "SCORE_DELTA_HIGH\", 2.0)"}),
    ("delta_norm_5.0", {"SCORE_DELTA_NORMAL\", 3.0)": "SCORE_DELTA_NORMAL\", 5.0)"}),
]

results = []
for name, patches in configs:
    patched = original
    ok = True
    for old, new in patches.items():
        if old in patched:
            patched = patched.replace(old, new)
        else:
            print(f"  WARNING: patch '{old[:40]}...' not found for {name}", flush=True)
            ok = False
    
    if not ok:
        print(f"{name:25s} → SKIPPED (patch failed)", flush=True)
        continue
    
    with open(strategy_path, 'w') as f:
        f.write(patched)
    
    result = subprocess.run(
        [sys.executable, "-m", "src.run", "--strategy", "solution.my_strategy.MyStrategy", "--data-dir", "data/train"],
        capture_output=True, text=True, cwd=os.path.dirname(os.path.dirname(__file__))
    )
    
    reshuffles = -1
    for line in result.stdout.split('\n'):
        if 'Total reshuffles:' in line:
            reshuffles = int(line.split(':')[1].strip().replace(',', ''))
            break
    
    print(f"{name:25s} → reshuffles={reshuffles}", flush=True)
    results.append((name, reshuffles))

# Restore original
with open(strategy_path, 'w') as f:
    f.write(original)

print("\n=== RESULTS ===")
best_name, best_r = min(results, key=lambda x: x[1])
for name, r in sorted(results, key=lambda x: x[1]):
    marker = " ← BEST" if name == best_name else ""
    delta = r - results[0][1]
    print(f"  {name:25s}: reshuffles={r:5d} (Δ={delta:+4d}){marker}")
