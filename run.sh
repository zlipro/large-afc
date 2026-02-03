#!/bin/bash

# --- 1. Environment Setup ---
PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
cd "$PROJECT_ROOT"

# --- 2. Helper Functions ---
function show_help {
    echo "Usage: ./run.sh [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  (No arguments) Defaults to -demo"
    echo "  -demo          Run the visualization demo"
    echo "  -benchmark     Run ALL benchmark cases (Default)"
    echo "  -benchmark random-5   Run ONLY random-5 case"
    echo "  -benchmark random-8   Run ONLY random-8 case"
    echo "  -benchmark random-50  Run ONLY random-50 case"
    echo "  -benchmark random-100  Run ONLY random-100 case"
    echo "  -evaluate      Run all the analysis and plotting scripts"
    echo "  -evaluate random-5       Run evaluation ONLY random-5 case"
    echo "  -evaluate random-8       Run evaluation ONLY random-8 case"
    echo "  -evaluate random-50      Run evaluation ONLY random-50 case"
    echo "  -evaluate random-100     Run evaluation ONLY random-100 case"
    echo "  -all           Run everything (All Benchmarks -> Evaluate)"
    echo "  -help          Show this message"
    echo ""
}

# --- 3. Argument Parsing ---

# Initialize flags
RUN_DEMO=false
# Benchmark specific flags
RUN_BENCH_RANDOM_5=false
RUN_BENCH_RANDOM_8=false
RUN_BENCH_RANDOM_50=false
RUN_BENCH_RANDOM_100=false
# Evaluation specific flags
RUN_EVAL_RANDOM_5=false
RUN_EVAL_RANDOM_8=false
RUN_EVAL_RANDOM_50=false
RUN_EVAL_RANDOM_100=false
# Check if no arguments were provided
if [ $# -eq 0 ]; then
    echo "No arguments provided. Defaulting to DEMO mode."
    RUN_DEMO=true
else
    while [[ "$#" -gt 0 ]]; do
        case $1 in
            -demo) RUN_DEMO=true ;;
            
            -benchmark) 
                # Check the NEXT argument ($2) to see if a specific case was requested
                case "$2" in
                    random-5)
                        RUN_BENCH_RANDOM_5=true
                        shift # Consume the specific argument "5"
                        ;;
                    random-8)
                        RUN_BENCH_RANDOM_8=true
                        shift # Consume the specific argument "8"
                        ;;
                    random-50)
                        RUN_BENCH_RANDOM_50=true
                        shift # Consume the specific argument "50"
                        ;;
                    random-100)
                        RUN_BENCH_RANDOM_100=true
                        shift # Consume the specific argument "100"
                        ;;
                    *)
                        # No specific case provided (or next arg is another flag like -evaluate)
                        # Default to ALL benchmarks
                        RUN_BENCH_RANDOM_5=true
                        RUN_BENCH_RANDOM_8=true
                        RUN_BENCH_RANDOM_50=true
                        ;;
                esac
                ;;

            -evaluate) 
                # Check the NEXT argument ($2) to see if a specific case was requested
                case "$2" in
                    random-5)
                        RUN_EVAL_RANDOM_5=true
                        shift # Consume the specific argument "5"
                        ;;
                    random-8)
                        RUN_EVAL_RANDOM_8=true
                        shift # Consume the specific argument "8"
                        ;;
                    random-50)
                        RUN_EVAL_RANDOM_50=true
                        shift # Consume the specific argument "50"
                        ;;
                    random-100)
                        RUN_EVAL_RANDOM_100=true
                        shift # Consume the specific argument "100"
                        ;;
                    *)
                        # No specific case provided (or next arg is another flag like -evaluate)
                        # Default to ALL benchmarks
                        RUN_EVAL_RANDOM_5=true
                        RUN_EVAL_RANDOM_8=true
                        RUN_EVAL_RANDOM_50=true
                        RUN_EVAL_RANDOM_100=true
                        ;;
                esac
                ;;
            
            
            -all) 
                RUN_BENCH_RANDOM_5=true
                RUN_BENCH_RANDOM_8=true
                RUN_BENCH_RANDOM_50=true
                RUN_BENCH_RANDOM_100=true
                RUN_EVAL_RANDOM_5=true
                RUN_EVAL_RANDOM_8=true
                RUN_EVAL_RANDOM_50=true
                RUN_EVAL_RANDOM_100=true
                ;;
                
            -help) show_help; exit 0 ;;
            *) echo "Unknown parameter passed: $1"; exit 1 ;;
        esac
        shift # Move to next argument
    done
fi

# --- Execution Logic ---

if [ "$RUN_DEMO" = true ]; then
    echo "================================================================"
    echo "Running DEMO: Plot Geometries"
    echo "================================================================"
    python experiments/plot_geometries.py
    echo ""
fi

# -- Benchmark Random-5 --
if [ "$RUN_BENCH_RANDOM_5" = true ]; then
    echo "================================================================"
    echo "Running BENCHMARK: Random-5"
    echo "================================================================"
    python exp/benchmarks/random-configs.py --config configs/benchmarks/random-5.yaml
    echo ""
fi

# -- Benchmark Random-8 --
if [ "$RUN_BENCH_RANDOM_8" = true ]; then
    echo "================================================================"
    echo "Running BENCHMARK: Random-8"
    echo "================================================================"
    python exp/benchmarks/random-configs.py --config configs/benchmarks/random-8.yaml
    echo ""
fi

# -- Benchmark Random-50 --
if [ "$RUN_BENCH_RANDOM_50" = true ]; then
    echo "================================================================"
    echo "Running BENCHMARK: Random-50"
    echo "================================================================"
    python exp/benchmarks/random-configs.py --config configs/benchmarks/random-50.yaml
    echo ""
fi

# -- Benchmark Random-100 --
if [ "$RUN_BENCH_RANDOM_100" = true ]; then
    echo "================================================================"
    echo "Running BENCHMARK: Random-100"
    echo "================================================================"
    python exp/benchmarks/random-configs.py --config configs/benchmarks/random-100.yaml
    echo ""
fi

if [ "$RUN_EVAL_RANDOM_5" = true ]; then
    echo "================================================================"
    echo "Running EVALUATION: Random-5 Stats & Plots"
    echo "================================================================"
    python exp/benchmarks/evaluate.py --config configs/benchmarks/random-5.yaml
    echo ""
fi

if [ "$RUN_EVAL_RANDOM_8" = true ]; then
    echo "================================================================"
    echo "Running EVALUATION: Random-8 Stats & Plots"
    echo "================================================================"
    python exp/benchmarks/evaluate.py --config configs/benchmarks/random-8.yaml
    echo ""
fi

if [ "$RUN_EVAL_RANDOM_50" = true ]; then
    echo "================================================================"
    echo "Running EVALUATION: Random-50 Stats & Plots"
    echo "================================================================"
    python exp/benchmarks/evaluate.py --config configs/benchmarks/random-50.yaml
    echo ""
fi

if [ "$RUN_EVAL_RANDOM_100" = true ]; then
    echo "================================================================"
    echo "Running EVALUATION: Random-100 Stats & Plots"
    echo "================================================================"
    python exp/benchmarks/evaluate.py --config configs/benchmarks/random-100.yaml
    echo ""
fi

echo "Done."