#!/bin/bash
# run_strategy.sh - Runs a placement strategy against both train and test data.
# Usage: ./run_strategy.sh solution.my_strategy.MyStrategy

set -e

if [ -z "$1" ]; then
    echo "Usage: $0 <strategy_dotted_path>"
    echo "Example: $0 solution.my_strategy.MyStrategy"
    exit 1
fi

STRATEGY="$1"
# Extract the class name (last part of dotted path) for a clean directory name
STRATEGY_NAME=$(echo "$STRATEGY" | awk -F. '{print $NF}')

echo "============================================================"
echo "Running Strategy: $STRATEGY"
echo "Results Folder: results/$STRATEGY_NAME"
echo "============================================================"
echo ""

# Ensure the results directory for this strategy exists
mkdir -p "results/$STRATEGY_NAME"

echo "------------------------------------------------------------"
echo "1. Running on TRAIN data..."
echo "------------------------------------------------------------"
python -m src.run --strategy "$STRATEGY" --data-dir data/train -o "results/$STRATEGY_NAME/train_results.json" -v
echo ""

echo "------------------------------------------------------------"
echo "2. Running on TEST data..."
echo "------------------------------------------------------------"
python -m src.run --strategy "$STRATEGY" --data-dir data/test -o "results/$STRATEGY_NAME/test_results.json" -v
echo ""

# Copy the test results to the root of results/ for submission validation
cp "results/$STRATEGY_NAME/test_results.json" results/results.json

echo "============================================================"
echo "SUCCESS: Results saved to results/$STRATEGY_NAME/"
echo "and copied test results to results/results.json."
echo "============================================================"
