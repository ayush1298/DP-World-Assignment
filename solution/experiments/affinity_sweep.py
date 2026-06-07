"""Sweep block affinity values for global ERC-0 search."""
import sys, os, subprocess

strategy_path = os.path.join(os.path.dirname(__file__), "my_strategy.py")
with open(strategy_path) as f:
    original = f.read()

values = [0.0, 2.0, 5.0, 10.0, 15.0, 20.0, 30.0, 50.0, 100.0, 500.0]
results = []

for val in values:
    patched = original.replace(
        "affinity = 0.0 if block == primary_block else 5.0",
        f"affinity = 0.0 if block == primary_block else {val}")
    with open(strategy_path, 'w') as f:
        f.write(patched)
    
    result = subprocess.run(
        [sys.executable, "-m", "src.run", "--strategy", "solution.my_strategy.MyStrategy", "--data-dir", "data/train"],
        capture_output=True, text=True, cwd=os.path.dirname(os.path.dirname(__file__)))
    
    reshuffles = -1
    for line in result.stdout.split('\n'):
        if 'Total reshuffles:' in line:
            reshuffles = int(line.split(':')[1].strip().replace(',', ''))
            break
    
    print(f"Affinity={val:6.1f} → reshuffles={reshuffles}", flush=True)
    results.append((val, reshuffles))

with open(strategy_path, 'w') as f:
    f.write(original)

print("\n=== RESULTS ===")
best_v, best_r = min(results, key=lambda x: x[1])
for v, r in sorted(results, key=lambda x: x[1]):
    marker = " ← BEST" if v == best_v else ""
    print(f"  Affinity={v:6.1f}: reshuffles={r:5d}{marker}")
