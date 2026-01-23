#!/bin/bash

# 1. Get the absolute path of the directory where this script lives
# This ensures the script works even if you call it from elsewhere
PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# 2. Add the root and src directories to PYTHONPATH
# This fixes your "ModuleNotFoundError" and "ImportError" issues
export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/src:$PYTHONPATH"

# 3. Change directory to root so save paths (./results/...) resolve correctly
cd "$PROJECT_ROOT"

echo "Running simulation: sim_circular_config.py..."

# 4. Execute the script
# We use the full path to the experiment file
python experiments/plot_geometries.py