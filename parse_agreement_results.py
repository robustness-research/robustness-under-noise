#!/usr/bin/env python3
"""
Parse ranking output files to extract and analyze agreement metrics.
"""

import os
import re
from pathlib import Path
import pandas as pd
import numpy as np

# Dataset and model names
DATASETS = [
    "analcatdata_authorship", "badges2", "banknote", "blood-transfusion-service-center", 
    "breast-w", "cardiotocography", "climate-model-simulation-crashes", "cmc", "credit-g", 
    "diabetes", "eucalyptus", "iris", "kc1", "liver-disorders", "mfeat-factors",
    "mfeat-karhunen", "mfeat-zernike", "ozone-level-8hr", "pc4", "phoneme",
    "qsar-biodeg", "tic-tac-toe", "vowel", "waveform-5000", "wdbc", "wilt"
]

MODELS = [
    "C5.0", "ctree", "fda", "gbm", "gcvEarth", "JRip", "lvq", "mlpML", "multinom", 
    "naive_bayes", "PART", "rbfDDA", "rda", "rf", "rpart", "simpls", "svmLinear", 
    "svmRadial", "rfRules", "knn", "bayesglm"
]


def parse_agreement_table(log_file):
    """
    Parse a single log file and extract the agreement metrics.
    
    Returns a list of dictionaries with the agreement data.
    """
    try:
        with open(log_file, 'r') as f:
            content = f.read()
        
        # Find the agreement section
        pattern = r'=== Agreement \(pairwise\) ===(.*?)=== Results recorded ==='
        match = re.search(pattern, content, re.DOTALL)
        
        if not match:
            return None
        
        agreement_section = match.group(1)
        
        # Parse the table rows
        results = []
        lines = agreement_section.strip().split('\n')
        
        for line in lines:
            # Skip header lines and separator lines
            if 'Pair' in line or '---' in line or not line.strip():
                continue
            
            # Parse data lines
            parts = line.split()
            if len(parts) >= 4:
                pair = ' '.join(parts[:-3])  # Handle "FEAT_IMP vs LIME" etc.
                try:
                    kendall_tau = float(parts[-3])
                    spearman = float(parts[-2])
                    iou_topk = float(parts[-1])
                    
                    results.append({
                        'pair': pair,
                        'kendall_tau': kendall_tau,
                        'spearman': spearman,
                        'iou_topk': iou_topk
                    })
                except (ValueError, IndexError):
                    continue
        
        return results
    
    except Exception as e:
        print(f"Error parsing {log_file}: {e}")
        return None


def main():
    """Main function to parse all log files and analyze results."""
    
    rankings_dir = Path("output/rankings")
    
    # Collect all results
    all_results = []
    
    for dataset in DATASETS:
        for model in MODELS:
            log_file = rankings_dir / f"ranking_output_{dataset}_{model}.log"
            
            if not log_file.exists():
                print(f"Missing: {log_file.name}")
                continue
            
            agreement_data = parse_agreement_table(log_file)
            
            if agreement_data:
                for entry in agreement_data:
                    all_results.append({
                        'dataset': dataset,
                        'model': model,
                        'pair': entry['pair'],
                        'kendall_tau': entry['kendall_tau'],
                        'spearman': entry['spearman'],
                        'iou_topk': entry['iou_topk']
                    })
    
    # Create DataFrame
    df = pd.DataFrame(all_results)
    
    if df.empty:
        print("No data found!")
        return
    
    print(f"\n{'='*80}")
    print(f"Total records parsed: {len(df)}")
    print(f"{'='*80}\n")
    
    # Analyze by pair
    print("="*80)
    print("SPEARMAN CORRELATION ANALYSIS BY PAIR")
    print("="*80)
    
    pair_stats = df.groupby('pair')['spearman'].agg([
        ('count', 'count'),
        ('mean', 'mean'),
        ('std', 'std'),
        ('min', 'min'),
        ('median', 'median'),
        ('max', 'max')
    ]).round(4)
    
    pair_stats = pair_stats.sort_values('mean', ascending=False)
    
    print("\nSpearman Correlation Statistics by Pair:")
    print(pair_stats.to_string())
    
    print(f"\n{'='*80}")
    print(f"BEST OVERALL AGREEMENT (by Spearman correlation):")
    print(f"{'='*80}")
    best_pair = pair_stats.index[0]
    best_mean = pair_stats.loc[best_pair, 'mean']
    print(f"\nPair: {best_pair}")
    print(f"Mean Spearman: {best_mean:.4f}")
    print(f"Std Dev: {pair_stats.loc[best_pair, 'std']:.4f}")
    print(f"Median: {pair_stats.loc[best_pair, 'median']:.4f}")
    print(f"Range: [{pair_stats.loc[best_pair, 'min']:.4f}, {pair_stats.loc[best_pair, 'max']:.4f}]")
    
    # Additional analyses
    print(f"\n{'='*80}")
    print("KENDALL TAU ANALYSIS BY PAIR")
    print("="*80)
    
    kendall_stats = df.groupby('pair')['kendall_tau'].agg([
        ('mean', 'mean'),
        ('std', 'std'),
        ('median', 'median')
    ]).round(4).sort_values('mean', ascending=False)
    
    print(kendall_stats.to_string())
    
    # Composite agreement score analysis
    print(f"\n{'='*80}")
    print("COMPOSITE AGREEMENT ANALYSIS (Tau + Spearman + IOU)")
    print("="*80)
    
    # Calculate composite score (equal weighting)
    df['composite_score'] = (df['kendall_tau'] + df['spearman'] + df['iou_topk']) / 3
    
    composite_stats = df.groupby('pair')['composite_score'].agg([
        ('count', 'count'),
        ('mean', 'mean'),
        ('std', 'std'),
        ('min', 'min'),
        ('median', 'median'),
        ('max', 'max')
    ]).round(4).sort_values('mean', ascending=False)
    
    print("\nComposite Agreement Score by Pair:")
    print(composite_stats.to_string())
    
    print(f"\n{'='*80}")
    print(f"BEST OVERALL AGREEMENT (by Composite Score):")
    print(f"{'='*80}")
    best_composite_pair = composite_stats.index[0]
    best_composite_mean = composite_stats.loc[best_composite_pair, 'mean']
    print(f"\nPair: {best_composite_pair}")
    print(f"Mean Composite Score: {best_composite_mean:.4f}")
    print(f"Std Dev: {composite_stats.loc[best_composite_pair, 'std']:.4f}")
    print(f"Median: {composite_stats.loc[best_composite_pair, 'median']:.4f}")
    print(f"Range: [{composite_stats.loc[best_composite_pair, 'min']:.4f}, {composite_stats.loc[best_composite_pair, 'max']:.4f}]")
    
    # Show detailed breakdown for best pair
    best_pair_data = df[df['pair'] == best_composite_pair]
    print(f"\nDetailed metrics for {best_composite_pair}:")
    print(f"  Kendall Tau - Mean: {best_pair_data['kendall_tau'].mean():.4f}, Std: {best_pair_data['kendall_tau'].std():.4f}")
    print(f"  Spearman    - Mean: {best_pair_data['spearman'].mean():.4f}, Std: {best_pair_data['spearman'].std():.4f}")
    print(f"  IOU@3       - Mean: {best_pair_data['iou_topk'].mean():.4f}, Std: {best_pair_data['iou_topk'].std():.4f}")

    # Analysis by dataset
    print(f"\n{'='*80}")
    print("SPEARMAN CORRELATION BY DATASET (averaged across all pairs and models)")
    print("="*80)
    
    dataset_stats = df.groupby('dataset')['spearman'].agg([
        ('mean', 'mean'),
        ('std', 'std'),
        ('count', 'count')
    ]).round(4).sort_values('mean', ascending=False)
    
    print(dataset_stats.head(10).to_string())
    
    # Analysis by model
    print(f"\n{'='*80}")
    print("SPEARMAN CORRELATION BY MODEL (averaged across all pairs and datasets)")
    print("="*80)
    
    model_stats = df.groupby('model')['spearman'].agg([
        ('mean', 'mean'),
        ('std', 'std'),
        ('count', 'count')
    ]).round(4).sort_values('mean', ascending=False)
    
    print(model_stats.head(10).to_string())
    
    # Save detailed results to CSV
    output_file = "results/agreement_analysis.csv"
    df.to_csv(output_file, index=False)
    print(f"\n{'='*80}")
    print(f"Detailed results saved to: {output_file}")
    print(f"{'='*80}\n")
    
    # Save summary statistics
    summary_file = "results/agreement_summary.csv"
    with open(summary_file, 'w') as f:
        f.write("=== COMPOSITE AGREEMENT SCORES ===\n")
        composite_stats.to_csv(f)
        f.write("\n=== SPEARMAN CORRELATION BY PAIR ===\n")
        pair_stats.to_csv(f)
        f.write("\n=== KENDALL TAU BY PAIR ===\n")
        kendall_stats.to_csv(f)
    
    print(f"Summary statistics saved to: {summary_file}\n")


if __name__ == "__main__":
    main()
