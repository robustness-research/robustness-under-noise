#!/bin/bash

arguments=("mfeat-factors" "mfeat-karhunen" "ozone-level-8hr" "phoneme" "tic-tac-toe" "vowel" "waveform-5000" "wdbc" "wilt")

# Maximum number of parallel jobs
MAX_JOBS=5

# Function to check memory usage
check_memory() {
    MEMORY_USAGE=$(free | grep Mem | awk '{printf("%.0f", $3/$2 * 100.0)}')
    if [ "$MEMORY_USAGE" -gt 85 ]; then
        echo "Memory usage is ${MEMORY_USAGE}%. Waiting..."
        return 1
    fi
    return 0
}

for arg in "${arguments[@]}"; do
    # Wait if we've reached the maximum number of jobs
    while [ $(jobs -r | wc -l) -ge $MAX_JOBS ] || ! check_memory; do
        sleep 1
    done

    echo "Starting job for dataset: $arg (Memory usage: $(free | grep Mem | awk '{printf("%.1f", $3/$2 * 100.0)}')%)"
    
    # Combine nice priority and CPU limiting
    nohup timeout 2h bash -c "ulimit -m 6291456; nice -n 5 cpulimit -l 85 -f Rscript Instances_Popular.R '$arg'" > output/instances-popular-out/"outputQ_$arg.log" 2>&1 &

    
done

wait

echo "All jobs completed"
