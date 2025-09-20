#!/bin/bash

# --- Script to run the Julia backtest with auto-restart functionality ---

echo "--- Starting Backtest Process ---"

# 1. Set up the Julia environment
echo "--> Step 1: Activating Julia environment and instantiating packages..."
julia --project=. -e 'using Pkg; Pkg.instantiate(); println("Environment is ready.")'

if [ $? -ne 0 ]; then
    echo "❌ ERROR: Julia environment setup failed. Aborting."
    exit 1
fi

# 2. Run the backtest script in a loop
while true; do
    echo ""
    echo "--> Step 2: Running the Julia backtest script (walk_forward_backtest_engine.jl)..."

    julia --project=. walk_forward_backtest_engine.jl

    EXIT_CODE=$?

    if [ $EXIT_CODE -eq 0 ]; then
        echo ""
        echo "✅ SUCCESS: Julia script completed successfully (Exit Code 0)."
        break # Exit the loop
    else
        echo ""
        echo "⚠️ WARNING: Julia script crashed with Exit Code $EXIT_CODE."
        echo "Restarting in 5 seconds... (Press Ctrl+C to cancel)"
        sleep 5
    fi
done

echo ""
echo "--- Backtest Process Finished ---"