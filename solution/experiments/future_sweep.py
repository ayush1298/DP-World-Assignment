"""Quick sweep of future-aware flags on train data."""
import sys, os, subprocess

strategy_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "my_strategy.py")
with open(strategy_path) as f:
    original = f.read()

configs = [
    ("baseline_no_future", {"ENABLE_FUTURE_AWARE = True": "ENABLE_FUTURE_AWARE = False"}),
    ("future_only", {}),
    ("future_risk_1.0", {"FUTURE_RISK_WEIGHT = 2.5": "FUTURE_RISK_WEIGHT = 1.0"}),
    ("future_risk_0.5", {"FUTURE_RISK_WEIGHT = 2.5": "FUTURE_RISK_WEIGHT = 0.5"}),
    ("ideal_only", {
        "FUTURE_RISK_WEIGHT = 2.5": "FUTURE_RISK_WEIGHT = 0.0",
        "IDEAL_TIER_WEIGHT = 1.5": "IDEAL_TIER_WEIGHT = 3.0",
    }),
    ("no_global", {"ENABLE_GLOBAL_ERC0 = True": "ENABLE_GLOBAL_ERC0 = False"}),
    ("no_global_no_future", {
        "ENABLE_GLOBAL_ERC0 = True": "ENABLE_GLOBAL_ERC0 = False",
        "ENABLE_FUTURE_AWARE = True": "ENABLE_FUTURE_AWARE = False",
    }),
]

root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
for name, patches in configs:
    patched = original
    for old, new in patches.items():
        patched = patched.replace(old, new)
    with open(strategy_path, "w") as f:
        f.write(patched)
    r = subprocess.run(
        [sys.executable, "-m", "src.run", "--strategy", "solution.my_strategy.MyStrategy", "--data-dir", "data/train"],
        capture_output=True, text=True, cwd=root)
    reshuffles = -1
    for line in r.stdout.split("\n"):
        if "Total reshuffles:" in line:
            reshuffles = int(line.split(":")[1].strip().replace(",", ""))
    print(f"{name:25s} -> {reshuffles}", flush=True)

with open(strategy_path, "w") as f:
    f.write(original)
