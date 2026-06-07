"""Sweep various parameter combinations on top of LRK proximity=80."""
import sys, os, json, subprocess, itertools

strategy_path = os.path.join(os.path.dirname(__file__), "my_strategy.py")

with open(strategy_path) as f:
    original = f.read()

configs = [
    ("baseline_w80", {}),
    ("no_mixing_pen", {"MIXING_PENALTY        = 15.0": "MIXING_PENALTY        = 0.0"}),
    ("low_mixing_pen", {"MIXING_PENALTY        = 15.0": "MIXING_PENALTY        = 5.0"}),
    ("no_empty_bonus", {"empty_bonus = -3.0 if h == 0 else 0.0": "empty_bonus = 0.0"}),
    ("pos_empty_pen", {"empty_bonus = -3.0 if h == 0 else 0.0": "empty_bonus = 3.0 if h == 0 else 0.0"}),
    ("height_coeff_0.1", {"height_coeff = getattr(self, \"HEIGHT_PENALTY_COEFF\", 0.15)": "height_coeff = 0.10"}),
    ("height_coeff_0.05", {"height_coeff = getattr(self, \"HEIGHT_PENALTY_COEFF\", 0.15)": "height_coeff = 0.05"}),
    ("h_weight_1.0", {"sort_key = h * 2.0 - hom + empty_bonus + lrk_proximity_bonus": "sort_key = h * 1.0 - hom + empty_bonus + lrk_proximity_bonus"}),
    ("h_weight_3.0", {"sort_key = h * 2.0 - hom + empty_bonus + lrk_proximity_bonus": "sort_key = h * 3.0 - hom + empty_bonus + lrk_proximity_bonus"}),
    ("h_weight_0.5", {"sort_key = h * 2.0 - hom + empty_bonus + lrk_proximity_bonus": "sort_key = h * 0.5 - hom + empty_bonus + lrk_proximity_bonus"}),
    ("no_hom", {"sort_key = h * 2.0 - hom + empty_bonus + lrk_proximity_bonus": "sort_key = h * 2.0 + empty_bonus + lrk_proximity_bonus"}),
    ("max_height_4", {"MAX_STACK_HEIGHT      = 5": "MAX_STACK_HEIGHT      = 4"}),
]

results = []
for name, patches in configs:
    patched = original
    for old, new in patches.items():
        patched = patched.replace(old, new)
    
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
