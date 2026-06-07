# Container Yard Placement — Technical Assessment

## Quick Start

```bash
# 1. Verify Python 3.8+
python --version

# 2. Explore the baselines
python -m src.run --strategy src.baseline_greedy.GreedyStrategy --data-dir data/train -v
python -m src.run --strategy src.baseline_random.RandomStrategy --data-dir data/train -v

# 3. Create your strategy
mkdir -p solution
# Implement PlacementStrategy in solution/my_strategy.py

# 4. Test on train data
python -m src.run --strategy solution.my_strategy.MyStrategy --data-dir data/train -v

# 5. Run on train + test and save results (recommended)
bash run_strategy.sh solution.my_strategy.MyStrategy

# 6. Validate submission
bash validate_submission.sh
```

## Run Script

`run_strategy.sh` runs a strategy on both datasets and writes results under `results/<StrategyName>/`:

```bash
bash run_strategy.sh solution.my_strategy.MyStrategy
```

1. Runs on `data/train`, saves to `results/MyStrategy/train_results.json`
2. Runs on `data/test`, saves to `results/MyStrategy/test_results.json`
3. Copies test output to `results/results.json` for submission validation

## Project Structure

```
PROBLEM.md                  # Problem statement
README.md                   # This file
run_strategy.sh             # Run a strategy on train + test
validate_submission.sh      # Validate submission layout and results

data/
  yard_layout.json          # Yard block dimensions
  vessel_schedule.json      # Vessel rotation schedule
  train/
    initial_state.json      # Starting yard state (day 0)
    events.jsonl            # Days 1–20 events
  test/
    initial_state.json      # Starting yard state (day 20)
    events.jsonl            # Days 21–40 events (scored)

src/
  models.py                 # Data models: Position, Container, Event
  yard_state.py             # Yard state manager (query this, don't modify)
  event_reader.py           # Event file reader
  simulator.py              # Simulation engine
  scoring.py                # Scoring and metrics
  placement_interface.py    # Abstract class to implement
  baseline_random.py        # Reference: random placement
  baseline_greedy.py        # Reference: lowest-stack placement
  run.py                    # CLI runner
  external_adapter.py       # Non-Python solver adapter

solution/
  my_strategy.py            # Main placement strategy (MyStrategy)
  analysis/                 # Diagnostic and data analysis scripts
  experiments/              # Parameter sweeps and ML experiment artifacts

docs/
  design.md                 # Algorithm design document

results/
  results.json              # Test results (used by validate_submission.sh)
  MyStrategy/               # Per-strategy train/test result JSON files

tests/                      # Optional unit tests
```

## Non-Python Solvers

If using a language other than Python, implement the stdin/stdout JSON protocol:

1. Your solver receives on stdin:
   - `{"type": "INIT", "yard_layout": {...}, "initial_state": {...}}`
   - Respond: `{"status": "ok"}`

2. For each placement:
   - `{"type": "PLACE", "event": {...}, "yard_summary": {...}}`
   - Respond: `{"block": "B01", "bay": 1, "row": 1, "tier": 1}`

3. After each retrieval:
   - `{"type": "RETRIEVE", "container_id": "...", "reshuffles": N}`
   - Respond: `{"status": "ok"}`

4. At end: `{"type": "END"}`

Run with: `python -m src.run --external ./my_solver --data-dir data/test`

## Key Tips

- Read `PROBLEM.md` carefully before starting
- Study the train data — patterns in vessel schedules and departure times are your biggest opportunity
- Use `yard_state.snapshot()`/`restore()` for lookahead and what-if analysis
- The `on_event()` and `on_container_retrieved()` callbacks are optional but useful for adaptive strategies
- Focus on minimizing reshuffles — that's 30 of 40 quantitative points
