#!/bin/bash

# --- Script to run Julia processes with auto-restart functionality ---

# --- CONFIGURATION ---
RESTART_DELAY_SECONDS=2
JULIA_PROJECT_PATH="."
BACKTEST_SCRIPT="walk_forward_backtest_engine.jl"
OPTIMIZATION_SCRIPT="Portfolio_Optimization_prod.jl"

# Exit code from Julia signaling a successful quarter completion and request for restart
# This code MUST match the one set in Portfolio_Optimization_prod.jl
RESTART_ON_QUARTER_EXIT_CODE=10

# --- FUNCTION: Display Usage ---
usage() {
    echo "Usage: $0 [mode]"
    echo "Modes:"
    echo "  backtest    : Runs the full walk-forward backtesting engine."
    echo "  optimize    : Runs only the standalone portfolio optimization pipeline."
    exit 1
}

# --- FUNCTION: Run the selected Julia process ---
run_julia_process() {
    local script_name=$1
    echo ""
    echo "--> Step 2: Running the Julia script ($script_name)..."

    # Execute Julia
    julia --project="$JULIA_PROJECT_PATH" "$script_name"

    # Capture exit code
    return $?
}


# --- SCRIPT START ---

# 1. Validate Input
if [ "$#" -ne 1 ]; then
    usage
fi

MODE=$1
TARGET_SCRIPT=""

case "$MODE" in
    backtest)
        TARGET_SCRIPT=$BACKTEST_SCRIPT
        echo "--- Starting Backtest Process ---"
        ;;
    optimize)
        TARGET_SCRIPT=$OPTIMIZATION_SCRIPT
        echo "--- Starting Portfolio Optimization Process ---"
        ;;
    *)
        echo "❌ ERROR: Invalid mode '$MODE'."
        usage
        ;;
esac


# 2. Set up the Julia environment (once at the start)
echo "--> Step 1: Activating Julia environment and instantiating packages..."
julia --project="$JULIA_PROJECT_PATH" -e 'using Pkg; Pkg.instantiate(); println("Environment is ready.")'

if [ $? -ne 0 ]; then
    echo "❌ ERROR: Julia environment setup failed. Aborting."
    exit 1
fi

# 3. Run the selected script in a loop with special restart logic
while true; do

    run_julia_process "$TARGET_SCRIPT"
    EXIT_CODE=$?

    if [ $EXIT_CODE -eq 0 ]; then
        echo ""
        echo "✅ SUCCESS: Julia script completed the entire process successfully (Exit Code 0)."
        break # Exit the loop, all work is done

    elif [ $EXIT_CODE -eq $RESTART_ON_QUARTER_EXIT_CODE ]; then
        echo ""
        echo "🔄 INFO: APSO quarter complete. Restarting for the next quarter in $RESTART_DELAY_SECONDS seconds..."
        sleep $RESTART_DELAY_SECONDS
        # Continue to the next iteration of the loop

    else
        echo ""
        echo "⚠️ WARNING: Julia script crashed with an unexpected Exit Code: $EXIT_CODE."
        echo "Restarting in 5 seconds... (Press Ctrl+C to cancel)"
        sleep 5
        # Continue to the next iteration of the loop
    fi
done

echo ""
echo "--- Process Finished ---"