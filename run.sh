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
    echo "  -demo          Run the large-AFC demo using the letter-w geometry"
    echo "  -benchmark     Run ALL benchmark cases (Default)"
    echo "  -benchmark random-5   Run ONLY random-5 case"
    echo "  -benchmark random-8   Run ONLY random-8 case"
    echo "  -benchmark random-50  Run ONLY random-50 case"
    echo "  -benchmark random-100  Run ONLY random-100 case"
    echo "  -benchmark random-500  Run ONLY random-500 case"
    echo "  -benchmark circular-10  Run ONLY circular-10 case"
    echo "  -benchmark cuboctahedron  Run ONLY cuboctahedron case"
    echo "  -benchmark trunc-icosahedron  Run ONLY truncated icosahedron case"
    echo "  -benchmark trunc-icosahedron-linked  Run ONLY truncated icosahedron linked case"
    echo "  -benchmark letter-w  Run ONLY letter-w case"
    echo "  -evaluate      Run all the analysis and plotting scripts"
    echo "  -evaluate random-5       Run evaluation ONLY random-5 case"
    echo "  -evaluate random-8       Run evaluation ONLY random-8 case"
    echo "  -evaluate random-50      Run evaluation ONLY random-50 case"
    echo "  -evaluate random-100     Run evaluation ONLY random-100 case"
    echo "  -evaluate random-500     Run evaluation ONLY random-500 case"
    echo "  -evaluate circular-10     Run evaluation ONLY circular-10 case"
    echo "  -evaluate cuboctahedron     Run evaluation ONLY cuboctahedron case"
    echo "  -evaluate trunc-icosahedron     Run evaluation ONLY truncated icosahedron case"
    echo "  -evaluate trunc-icosahedron-linked     Run evaluation ONLY truncated icosahedron linked case"
    echo "  -evaluate letter-w     Run evaluation ONLY letter-w case"
    echo "  -all           Run everything (All Benchmarks -> Evaluate)"
    echo "  -edm           Run the EDM vs USI stress comparison script"
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
RUN_BENCH_RANDOM_500=false
RUN_BENCH_CIRCULAR_10=false
RUN_BENCH_CUBOCTAHEDRON=false
RUN_BENCH_TRUNC_ICOSAHEDRON=false
RUN_BENCH_TRUNC_ICOSAHEDRON_LINKED=false
RUN_BENCH_LETTER_W=false

RUN_EDM = false

# Evaluation specific flags
RUN_EVAL_RANDOM_5=false
RUN_EVAL_RANDOM_8=false
RUN_EVAL_RANDOM_50=false
RUN_EVAL_RANDOM_100=false
RUN_EVAL_RANDOM_500=false
RUN_EVAL_CIRCULAR_10=false
RUN_EVAL_CUBOCTAHEDRON=false
RUN_EVAL_TRUNC_ICOSAHEDRON=false
RUN_EVAL_TRUNC_ICOSAHEDRON_LINKED=false
RUN_EVAL_LETTER_W=false
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
                    random-500)
                        RUN_BENCH_RANDOM_500=true
                        shift # Consume the specific argument "500"
                        ;;  
                    circular-10)
                        RUN_BENCH_CIRCULAR_10=true
                        shift # Consume the specific argument "circular-10"
                        ;;
                    cuboctahedron)
                        RUN_BENCH_CUBOCTAHEDRON=true
                        shift # Consume the specific argument "cuboctahedron"
                        ;;
                    trunc-icosahedron)
                        RUN_BENCH_TRUNC_ICOSAHEDRON=true
                        shift # Consume the specific argument "trunc-icosahedron"
                        ;;  
                    trunc-icosahedron-linked)
                        RUN_BENCH_TRUNC_ICOSAHEDRON_LINKED=true
                        shift # Consume the specific argument "trunc-icosahedron-linked"
                        ;;  
                    letter-w)
                        RUN_BENCH_LETTER_W=true
                        shift # Consume the specific argument "letter-w"
                        ;;  
                    *)
                        # No specific case provided (or next arg is another flag like -evaluate)
                        # Default to ALL benchmarks
                        RUN_BENCH_RANDOM_5=true
                        RUN_BENCH_RANDOM_8=true
                        RUN_BENCH_RANDOM_50=true
                        RUN_BENCH_RANDOM_100=true
                        RUN_BENCH_RANDOM_500=true
                        RUN_BENCH_CIRCULAR_10=true
                        RUN_BENCH_CUBOCTAHEDRON=true
                        RUN_BENCH_TRUNC_ICOSAHEDRON=true
                        RUN_BENCH_TRUNC_ICOSAHEDRON_LINKED=true
                        RUN_BENCH_LETTER_W=true
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
                    random-500)
                        RUN_EVAL_RANDOM_500=true
                        shift # Consume the specific argument "500"
                        ;;
                    circular-10)
                        RUN_EVAL_CIRCULAR_10=true
                        shift # Consume the specific argument "circular-10"
                        ;;  
                    cuboctahedron)
                        RUN_EVAL_CUBOCTAHEDRON=true
                        shift # Consume the specific argument "cuboctahedron"
                        ;;  
                    trunc-icosahedron)
                        RUN_EVAL_TRUNC_ICOSAHEDRON=true
                        shift # Consume the specific argument "trunc-icosahedron"
                        ;;  
                    trunc-icosahedron-linked)
                        RUN_EVAL_TRUNC_ICOSAHEDRON_LINKED=true
                        shift # Consume the specific argument "trunc-icosahedron-linked"
                        ;;  
                    letter-w)
                        RUN_EVAL_LETTER_W=true
                        shift # Consume the specific argument "letter-w"
                        ;;  
                    *)
                        # No specific case provided (or next arg is another flag like -evaluate)
                        # Default to ALL benchmarks
                        RUN_EVAL_RANDOM_5=true
                        RUN_EVAL_RANDOM_8=true
                        RUN_EVAL_RANDOM_50=true
                        RUN_EVAL_RANDOM_100=true
                        RUN_EVAL_RANDOM_500=true
                        RUN_EVAL_CIRCULAR_10=true
                        RUN_EVAL_CUBOCTAHEDRON=true
                        RUN_EVAL_TRUNC_ICOSAHEDRON=true
                        RUN_EVAL_TRUNC_ICOSAHEDRON_LINKED=true
                        RUN_EVAL_LETTER_W=true  
                        ;;
                esac
                ;;
            
            
            -all) 
                RUN_BENCH_RANDOM_5=true
                RUN_BENCH_RANDOM_8=true
                RUN_BENCH_RANDOM_50=true
                RUN_BENCH_RANDOM_100=true
                RUN_BENCH_RANDOM_500=true
                RUN_BENCH_CIRCULAR_10=true
                RUN_BENCH_CUBOCTAHEDRON=true
                RUN_BENCH_TRUNC_ICOSAHEDRON=true
                RUN_BENCH_TRUNC_ICOSAHEDRON_LINKED=true
                RUN_BENCH_LETTER_W=true
                RUN_EVAL_RANDOM_5=true
                RUN_EVAL_RANDOM_8=true
                RUN_EVAL_RANDOM_50=true
                RUN_EVAL_RANDOM_100=true
                RUN_EVAL_RANDOM_500=true
                RUN_EVAL_CIRCULAR_10=true
                RUN_EVAL_CUBOCTAHEDRON=true
                RUN_EVAL_TRUNC_ICOSAHEDRON=true
                RUN_EVAL_TRUNC_ICOSAHEDRON_LINKED=true
                RUN_EVAL_LETTER_W=true
                ;;

            -edm)
                RUN_EDM=true
                shift # Consume the specific argument "edm"
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
    echo "Running DEMO: Multicluster AFC with Letter W Geometry"
    echo "================================================================"
    python demo.py --config configs/demo.yaml   
    echo ""
fi

# -- Benchmark Random-5 --
if [ "$RUN_BENCH_RANDOM_5" = true ]; then
    echo "================================================================"
    echo "Running BENCHMARK: Random-5"
    echo "================================================================"
    python exp/benchmarks/benchmark.py --config configs/benchmarks/random-5.yaml
    echo ""
    echo "Finished at: $(date)"  # Prints the current system date and time
    echo ""
fi

# -- Benchmark Random-8 --
if [ "$RUN_BENCH_RANDOM_8" = true ]; then
    echo "================================================================"
    echo "Running BENCHMARK: Random-8"
    echo "================================================================"
    python exp/benchmarks/benchmark.py --config configs/benchmarks/random-8.yaml
    echo ""
    echo "Finished at: $(date)"  # Prints the current system date and time
    echo ""
fi

# -- Benchmark Random-50 --
if [ "$RUN_BENCH_RANDOM_50" = true ]; then
    echo "================================================================"
    echo "Running BENCHMARK: Random-50"
    echo "================================================================"
    python exp/benchmarks/benchmark.py --config configs/benchmarks/random-50.yaml
    echo ""
    echo "Finished at: $(date)"  # Prints the current system date and time
    echo ""
fi

# -- Benchmark Random-100 --
if [ "$RUN_BENCH_RANDOM_100" = true ]; then
    echo "================================================================"
    echo "Running BENCHMARK: Random-100"
    echo "================================================================"
    python exp/benchmarks/benchmark.py --config configs/benchmarks/random-100.yaml
    echo ""
    echo "Finished at: $(date)"  # Prints the current system date and time
    echo ""
fi

# -- Benchmark Random-500 --
if [ "$RUN_BENCH_RANDOM_500" = true ]; then
    echo "================================================================"
    echo "Running BENCHMARK: Random-500"
    echo "================================================================"
    python exp/benchmarks/benchmark.py --config configs/benchmarks/random-500.yaml
    echo ""
    echo "Finished at: $(date)"  # Prints the current system date and time
    echo ""
fi

# -- Benchmark Circular-10 --
if [ "$RUN_BENCH_CIRCULAR_10" = true ]; then
    echo "================================================================"
    echo "Running BENCHMARK: Circular-10"
    echo "================================================================"
    python exp/benchmarks/benchmark.py --config configs/benchmarks/circular-10.yaml
    echo ""
    echo "Finished at: $(date)"  # Prints the current system date and time
    echo ""
fi

# -- Benchmark Cuboctahedron --
if [ "$RUN_BENCH_CUBOCTAHEDRON" = true ]; then
    echo "================================================================"
    echo "Running BENCHMARK: Cuboctahedron"
    echo "================================================================"
    python exp/benchmarks/benchmark.py --config configs/benchmarks/cuboctahedron.yaml
    echo ""
    echo "Finished at: $(date)"  # Prints the current system date and time
    echo ""
fi

# -- Benchmark Truncated Icosahedron --
if [ "$RUN_BENCH_TRUNC_ICOSAHEDRON" = true ]; then
    echo "================================================================"
    echo "Running BENCHMARK: Truncated Icosahedron"
    echo "================================================================"
    python exp/benchmarks/benchmark.py --config configs/benchmarks/trunc-icosahedron.yaml
    echo ""
    echo "Finished at: $(date)"  # Prints the current system date and time
    echo ""
fi

# -- Benchmark Truncated Icosahedron Linked --
if [ "$RUN_BENCH_TRUNC_ICOSAHEDRON_LINKED" = true ]; then
    echo "================================================================"
    echo "Running BENCHMARK: Truncated Icosahedron Linked"
    echo "================================================================"
    python exp/benchmarks/benchmark-linked-geometry.py --config configs/benchmarks/trunc-icosahedron-linked.yaml
    echo ""
    echo "Finished at: $(date)"  # Prints the current system date and time
    echo ""
fi 

# -- Benchmark Letter W --
if [ "$RUN_BENCH_LETTER_W" = true ]; then
    echo "================================================================"
    echo "Running BENCHMARK: Letter W"
    echo "================================================================"
    python exp/benchmarks/benchmark-linked-geometry.py --config configs/benchmarks/letter-w.yaml
    echo ""
    echo "Finished at: $(date)"  # Prints the current system date and time
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

if [ "$RUN_EVAL_RANDOM_500" = true ]; then
    echo "================================================================"
    echo "Running EVALUATION: Random-500 Stats & Plots"
    echo "================================================================"
    python exp/benchmarks/evaluate.py --config configs/benchmarks/random-500.yaml
    echo ""
fi  

if [ "$RUN_EVAL_CIRCULAR_10" = true ]; then
    echo "================================================================"
    echo "Running EVALUATION: Circular-10 Stats & Plots"
    echo "================================================================"
    python exp/benchmarks/evaluate.py --config configs/benchmarks/circular-10.yaml
    echo ""
fi

if [ "$RUN_EVAL_CUBOCTAHEDRON" = true ]; then
    echo "================================================================"
    echo "Running EVALUATION: Cuboctahedron Stats & Plots"
    echo "================================================================"
    python exp/benchmarks/evaluate.py --config configs/benchmarks/cuboctahedron.yaml
    echo ""
fi

if [ "$RUN_EVAL_TRUNC_ICOSAHEDRON" = true ]; then
    echo "================================================================"
    echo "Running EVALUATION: Truncated Icosahedron Stats & Plots"
    echo "================================================================"
    python exp/benchmarks/evaluate.py --config configs/benchmarks/trunc-icosahedron.yaml
    echo ""
fi  

if [ "$RUN_EVAL_TRUNC_ICOSAHEDRON_LINKED" = true ]; then
    echo "================================================================"
    echo "Running EVALUATION: Truncated Icosahedron Linked Stats & Plots"
    echo "================================================================"
    python exp/benchmarks/evaluate.py --config configs/benchmarks/trunc-icosahedron-linked.yaml
    echo ""
fi  

if [ "$RUN_EVAL_LETTER_W" = true ]; then
    echo "================================================================"
    echo "Running EVALUATION: Letter W Stats & Plots"
    echo "================================================================"
    python exp/benchmarks/evaluate.py --config configs/benchmarks/letter-w.yaml
    echo ""
fi


if [ "$RUN_EDM" = true ]; then
    echo "================================================================"
    echo "Running EDM vs USI Stress Comparison"
    echo "================================================================"
    python exp/sim_edm_usi.py
    echo ""
fi
echo "All scripts Executed."