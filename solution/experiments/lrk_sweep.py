"""Sweep LRK proximity weight for ERC-0 tiebreaker."""
import sys, os, json, subprocess

weights = [50.0, 60.0, 65.0, 70.0, 75.0, 80.0, 85.0, 90.0, 100.0, 125.0]
results = []

strategy_path = os.path.join(os.path.dirname(__file__), "my_strategy.py")

with open(strategy_path) as f:
    original = f.read()

for w in weights:
    # Patch the weight
    patched = original.replace("lrk_proximity_bonus = gap * 4.0", f"lrk_proximity_bonus = gap * {w}")
    with open(strategy_path, 'w') as f:
        f.write(patched)
    
    result = subprocess.run(
        [sys.executable, "-m", "src.run", "--strategy", "solution.my_strategy.MyStrategy", "--data-dir", "data/train"],
        capture_output=True, text=True, cwd=os.path.dirname(os.path.dirname(__file__))
    )
    
    for line in result.stdout.split('\n'):
        if 'Total reshuffles:' in line:
            reshuffles = int(line.split(':')[1].strip().replace(',', ''))
            break
    else:
        reshuffles = -1
    
    print(f"Weight={w:5.1f} → reshuffles={reshuffles}", flush=True)
    results.append((w, reshuffles))

# Restore original
with open(strategy_path, 'w') as f:
    f.write(original)

print("\n=== RESULTS ===")
best_w, best_r = min(results, key=lambda x: x[1])
for w, r in results:
    marker = " ← BEST" if w == best_w else ""
    print(f"  Weight={w:5.1f}: reshuffles={r}{marker}")
print(f"\nBest weight: {best_w} with {best_r} reshuffles")
