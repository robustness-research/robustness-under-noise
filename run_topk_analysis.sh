#!/bin/bash

# Output file
OUTPUT_FILE="output/spearman/topk_agreement_results.txt"
mkdir -p output/spearman

# Clear/create output file
echo "Top-K Feature Agreement Analysis" > "$OUTPUT_FILE"
echo "Generated: $(date)" >> "$OUTPUT_FILE"
echo "======================================" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

# Datasets
datasets=("analcatdata_authorship" "badges2" "banknote" "blood-transfusion-service-center" "breast-w" "cardiotocography" "climate-model-simulation-crashes" "cmc" "credit-g" "diabetes" "eucalyptus" "iris" "kc1" "liver-disorders" "mfeat-factors" "mfeat-karhunen" "mfeat-zernike" "ozone-level-8hr" "pc4" "phoneme" "qsar-biodeg" "tic-tac-toe" "vowel" "waveform-5000" "wdbc" "wilt")

# Models
models=("C5.0" "ctree" "fda" "gbm" "gcvEarth" "JRip" "lvq" "mlpML" "multinom" "naive_bayes" "PART" "rbfDDA" "rda" "rf" "rpart" "simpls" "svmLinear" "svmRadial" "rfRules" "knn" "bayesglm")

# Total combinations
total=$((${#datasets[@]} * ${#models[@]}))
current=0

echo "Starting analysis of $total combinations..."
echo "Running in parallel by dataset..."

# Function to process one dataset with all models
process_dataset() {
  local dataset=$1
  shift
  local models_list=("$@")
  local output_file="output/spearman/${dataset}_results.txt"
  
  echo "Processing dataset: $dataset"
  
  # Create output file for this dataset
  echo "Dataset: $dataset" > "$output_file"
  echo "Generated: $(date)" >> "$output_file"
  echo "======================================" >> "$output_file"
  echo "" >> "$output_file"
  
  for model in "${models_list[@]}"; do
    echo "  - Processing $dataset with $model"
    
    echo "" >> "$output_file"
    echo "========================================" >> "$output_file"
    echo "Model: $model" >> "$output_file"
    echo "========================================" >> "$output_file"
    
    # Run R script and capture output
    Rscript Feature_Ranking_TopK.R "$dataset" "$model" >> "$output_file" 2>&1
    
    if [ $? -eq 0 ]; then
      echo "  ✓ $dataset/$model Success"
    else
      echo "  ✗ $dataset/$model Failed"
      echo "FAILED - See error above" >> "$output_file"
    fi
    
    echo "" >> "$output_file"
  done
  
  echo "Completed dataset: $dataset"
}

export -f process_dataset

# Run datasets in parallel
for dataset in "${datasets[@]}"; do
  process_dataset "$dataset" "${models[@]}" &
done

# Wait for all background processes to complete
wait

# Combine all results into one file
echo "Combining results..."
for dataset in "${datasets[@]}"; do
  if [ -f "output/spearman/${dataset}_results.txt" ]; then
    cat "output/spearman/${dataset}_results.txt" >> "$OUTPUT_FILE"
    echo "" >> "$OUTPUT_FILE"
  fi
done

echo ""
echo "Analysis complete! Results saved to: $OUTPUT_FILE"
echo "Individual dataset results in: output/spearman/"
echo "Total combinations processed: $total"
