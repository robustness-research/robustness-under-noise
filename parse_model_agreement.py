#!/usr/bin/env python3
"""
Parse topk_agreement_results.txt to extract:
1. Average Top-3 Spearman scores per model (as pairwise table)
2. Top-1 feature agreement per model (as pairwise table)
"""

import re
from collections import defaultdict
import pandas as pd
import numpy as np

def parse_topk_results(filepath):
    """Parse the topk agreement results file."""
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Split by models
    model_sections = re.split(r'={40,}\nModel: (.+?)\n={40,}', content)
    
    # Storage for results
    top3_spearman_by_model = defaultdict(lambda: defaultdict(list))
    top1_agreement_by_model = defaultdict(lambda: defaultdict(list))
    
    # Process each model section
    for i in range(1, len(model_sections), 2):
        if i+1 >= len(model_sections):
            break
            
        model_name = model_sections[i].strip()
        model_content = model_sections[i+1]
        
        # Skip failed models
        if 'FAILED' in model_content or 'Error:' in model_content[:200]:
            continue
        
        # Extract Top-3 Spearman scores
        top3_match = re.search(
            r'=== Top-3 Spearman \(on common top-3 features only\) ===\s+' +
            r'Pair\s+Spearman\s+Common_Features\s+' +
            r'-+\s+-+\s+-+\s+' +
            r'((?:FEAT_IMP vs LIME\s+[^\n]+\n)?' +
            r'(?:FEAT_IMP vs SHAP\s+[^\n]+\n)?' +
            r'(?:LIME vs SHAP\s+[^\n]+\n)?)',
            model_content
        )
        
        if top3_match:
            pairs_text = top3_match.group(1)
            # Parse each pair line
            for line in pairs_text.strip().split('\n'):
                if not line.strip():
                    continue
                parts = line.split()
                if len(parts) >= 4:
                    pair = f"{parts[0]} vs {parts[2]}"
                    spearman_val = parts[3]
                    # Convert NA to NaN, otherwise to float
                    if spearman_val == 'NA':
                        spearman_val = np.nan
                    else:
                        try:
                            spearman_val = float(spearman_val)
                        except ValueError:
                            spearman_val = np.nan
                    
                    top3_spearman_by_model[model_name][pair].append(spearman_val)
        
        # Extract Top-1 Feature Agreement
        top1_match = re.search(
            r'=== Top-1 Feature Agreement \(pairwise\) ===\s*\n\s*\n' +
            r'Pair\s+Feature_1\s+Feature_2\s+Agreement\s*\n' +
            r'-+\s+-+\s+-+\s+-+\s*\n' +
            r'((?:.+\n)*?)' +
            r'\nAgreeing pairs:',
            model_content,
            re.MULTILINE
        )
        
        if top1_match:
            pairs_text = top1_match.group(1)
            # Parse each pair line - looking for pattern: "PAIR_NAME vs PAIR_NAME  feature1  feature2  TRUE/FALSE"
            for line in pairs_text.strip().split('\n'):
                if not line.strip():
                    continue
                # Split by whitespace
                parts = line.split()
                if len(parts) >= 5 and 'vs' in parts:
                    # Find 'vs' position
                    vs_idx = parts.index('vs')
                    pair = f"{parts[vs_idx-1]} vs {parts[vs_idx+1]}"
                    # Agreement is the last part
                    agreement = parts[-1]
                    # Convert TRUE to 1, FALSE to 0, NA to NaN
                    if agreement == 'TRUE':
                        agreement_val = 1.0
                    elif agreement == 'FALSE':
                        agreement_val = 0.0
                    else:
                        agreement_val = np.nan
                    
                    top1_agreement_by_model[model_name][pair].append(agreement_val)
    
    return top3_spearman_by_model, top1_agreement_by_model


def create_pairwise_tables(data_by_model):
    """Create pairwise tables with models as both rows and columns."""
    
    # Get all models and all pairs
    models = sorted(data_by_model.keys())
    all_pairs = set()
    for pairs_dict in data_by_model.values():
        all_pairs.update(pairs_dict.keys())
    pairs = sorted(all_pairs)
    
    # Calculate average for each model-pair combination
    results = {}
    for model in models:
        model_avgs = {}
        for pair in pairs:
            values = data_by_model[model].get(pair, [])
            if values:
                # Calculate mean, ignoring NaN values
                valid_values = [v for v in values if not np.isnan(v)]
                if valid_values:
                    model_avgs[pair] = np.mean(valid_values)
                else:
                    model_avgs[pair] = np.nan
            else:
                model_avgs[pair] = np.nan
        results[model] = model_avgs
    
    # Create DataFrame
    df = pd.DataFrame(results).T
    
    return df


def main():
    filepath = '/home/chris/github/robustness-under-noise/output/spearman/topk_agreement_results.txt'
    
    print("Parsing topk_agreement_results.txt...")
    top3_spearman, top1_agreement = parse_topk_results(filepath)
    
    print(f"\nFound {len(top3_spearman)} models with Top-3 Spearman data")
    print(f"Found {len(top1_agreement)} models with Top-1 Agreement data")
    
    # Create pairwise tables
    print("\n" + "="*80)
    print("TOP-3 SPEARMAN SCORES (Average across datasets)")
    print("="*80)
    top3_df = create_pairwise_tables(top3_spearman)
    print(top3_df.to_string(float_format='%.3f'))
    
    print("\n" + "="*80)
    print("TOP-1 FEATURE AGREEMENT (Average across datasets)")
    print("="*80)
    top1_df = create_pairwise_tables(top1_agreement)
    print(top1_df.to_string(float_format='%.3f'))
    
    # Save to CSV files
    output_dir = '/home/chris/github/robustness-under-noise/output/spearman'
    
    top3_csv = f'{output_dir}/top3_spearman_by_model.csv'
    top3_df.to_csv(top3_csv)
    print(f"\n\nTop-3 Spearman table saved to: {top3_csv}")
    
    top1_csv = f'{output_dir}/top1_agreement_by_model.csv'
    top1_df.to_csv(top1_csv)
    print(f"Top-1 Agreement table saved to: {top1_csv}")
    
    # Print summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    print("\nTop-3 Spearman (by pair):")
    for col in top3_df.columns:
        mean_val = top3_df[col].mean()
        print(f"  {col}: {mean_val:.3f}")
    
    print("\nTop-1 Agreement (by pair):")
    for col in top1_df.columns:
        mean_val = top1_df[col].mean()
        print(f"  {col}: {mean_val:.3f} ({mean_val*100:.1f}%)")


if __name__ == '__main__':
    main()
