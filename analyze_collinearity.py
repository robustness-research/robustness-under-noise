#!/usr/bin/env python3
"""
Collinearity Analysis Enhancement Script

This script analyzes the results from the model-agnostic collinearity analysis
and provides enhanced interpretability by counting different types of collinearity
patterns and generating summary statistics.
"""

import pandas as pd
import numpy as np
from collections import Counter

def load_collinearity_data(csv_file='results/collinearity_analysis/collinearity_summary.csv'):
    """Load the collinearity results from CSV file."""
    try:
        df = pd.read_csv(csv_file)
        print(f"Successfully loaded {len(df)} datasets from {csv_file}")
        return df
    except FileNotFoundError:
        print(f"Error: Could not find {csv_file}")
        return None

def analyze_collinearity_patterns(df):
    """Analyze different collinearity patterns in the data."""
    
    # Count datasets with multicollinearity
    any_collinearity = df[df['has_multicollinearity'] == True]
    
    # Analyze severity distribution
    severity_distribution = df['severity_score'].value_counts().sort_index()
    
    # Count by collinearity indicators
    collinearity_counts = {
        'high_correlations': df[df['n_high_correlations'] > 0].shape[0],
        'high_vif': df[df['n_high_vif'] > 0].shape[0],
        'high_condition_number': df[df['condition_number'] > 30].shape[0],
        'small_eigenvalues': df[df['n_small_eigenvalues'] > 0].shape[0]
    }
    
    return {
        'collinearity_counts': collinearity_counts,
        'any_collinearity': any_collinearity,
        'severity_distribution': severity_distribution,
        'total_datasets': len(df)
    }

def print_summary_statistics(df, analysis):
    """Print comprehensive summary statistics."""
    
    print("=" * 60)
    print("COLLINEARITY ANALYSIS SUMMARY")
    print("=" * 60)
    
    print(f"\n📊 DATASET OVERVIEW:")
    print(f"   Total datasets analyzed: {analysis['total_datasets']}")
    print(f"   Datasets with multicollinearity: {len(analysis['any_collinearity'])}")
    print(f"   Datasets with no collinearity: {analysis['total_datasets'] - len(analysis['any_collinearity'])}")
    print(f"   Collinearity detection rate: {len(analysis['any_collinearity']) / analysis['total_datasets'] * 100:.1f}%")
    
    print(f"\n🔗 COLLINEARITY INDICATOR BREAKDOWN:")
    for collin_type, count in analysis['collinearity_counts'].items():
        percentage = count / analysis['total_datasets'] * 100
        print(f"   {collin_type.replace('_', ' ').title()}: {count} datasets ({percentage:.1f}%)")
    
    print(f"\n📈 SEVERITY DISTRIBUTION:")
    for severity, freq in analysis['severity_distribution'].items():
        print(f"   Severity score {severity}: {freq} datasets")
    
    # Add statistics for key metrics
    print(f"\n📏 KEY METRICS STATISTICS:")
    print(f"   Average condition number: {df['condition_number'].mean():.2f}")
    print(f"   Median condition number: {df['condition_number'].median():.2f}")
    print(f"   Average # of high correlations: {df['n_high_correlations'].mean():.2f}")
    print(f"   Average # of high VIF features: {df['n_high_vif'].mean():.2f}")

def analyze_correlation_strengths(df):
    """Analyze the distribution of high correlations across datasets."""
    
    print(f"\n📏 HIGH CORRELATION ANALYSIS:")
    print(f"   Total high correlation pairs found: {df['n_high_correlations'].sum()}")
    print(f"   Average high correlations per dataset: {df['n_high_correlations'].mean():.2f}")
    print(f"   Median high correlations per dataset: {df['n_high_correlations'].median():.0f}")
    print(f"   Max high correlations in a dataset: {df['n_high_correlations'].max()}")
    
    # Distribution of datasets by number of high correlations
    high_corr_dist = df['n_high_correlations'].value_counts().sort_index()
    print(f"\n   Distribution of datasets by # of high correlations:")
    for n_corr, count in high_corr_dist.head(10).items():
        print(f"   {n_corr} high correlations: {count} datasets")
    
    return None

def identify_problematic_datasets(df):
    """Identify datasets with the most collinearity issues."""
    
    # Sort by severity score
    problematic = df[df['has_multicollinearity'] == True].sort_values('severity_score', ascending=False)
    
    print(f"\n⚠️  DATASETS WITH COLLINEARITY ISSUES:")
    print(f"   (Sorted by severity score)")
    
    for idx, (_, row) in enumerate(problematic.head(10).iterrows(), 1):
        print(f"\n   {idx}. {row['dataset']} (Severity: {row['severity_score']})")
        print(f"      • Features: {row['n_features']}")
        print(f"      • High correlations: {row['n_high_correlations']}")
        print(f"      • High VIF features: {row['n_high_vif']}")
        print(f"      • Condition number: {row['condition_number']:.2f}")
        print(f"      • Small eigenvalues: {row['n_small_eigenvalues']}")
    
    return problematic

def analyze_vif_patterns(df):
    """Analyze VIF (Variance Inflation Factor) patterns."""
    
    print(f"\n🔢 VIF ANALYSIS:")
    
    # Datasets with high VIF
    high_vif_datasets = df[df['n_high_vif'] > 0]
    
    print(f"   Datasets with high VIF features: {len(high_vif_datasets)} ({len(high_vif_datasets)/len(df)*100:.1f}%)")
    print(f"   Average # of high VIF features: {df['n_high_vif'].mean():.2f}")
    print(f"   Max # of high VIF features: {df['n_high_vif'].max()}")
    
    # Distribution
    vif_dist = df['n_high_vif'].value_counts().sort_index()
    print(f"\n   Distribution of datasets by # of high VIF features:")
    for n_vif, count in vif_dist.head(10).items():
        print(f"   {n_vif} high VIF features: {count} datasets")

def generate_summary_report(df):
    """Generate a comprehensive summary report."""
    
    # Perform all analyses
    analysis = analyze_collinearity_patterns(df)
    analyze_correlation_strengths(df)
    problematic_datasets = identify_problematic_datasets(df)
    
    # Print all summaries
    print_summary_statistics(df, analysis)
    analyze_vif_patterns(df)
    
    # Generate key insights
    print(f"\n🔍 KEY INSIGHTS:")
    
    # Most problematic dataset
    if len(problematic_datasets) > 0:
        worst_dataset = problematic_datasets.iloc[0]
        print(f"   • Most collinear dataset: {worst_dataset['dataset']} "
              f"(severity score: {worst_dataset['severity_score']})")
    
    # Feature count impact
    print(f"   • Average features per dataset: {df['n_features'].mean():.1f}")
    
    # Correlation between features and collinearity
    if df['has_multicollinearity'].sum() > 0:
        avg_features_with_collin = df[df['has_multicollinearity'] == True]['n_features'].mean()
        avg_features_without = df[df['has_multicollinearity'] == False]['n_features'].mean()
        print(f"   • Avg features in datasets WITH collinearity: {avg_features_with_collin:.1f}")
        print(f"   • Avg features in datasets WITHOUT collinearity: {avg_features_without:.1f}")
    
    # Condition number insights
    high_cond = df[df['condition_number'] > 100]
    if len(high_cond) > 0:
        print(f"   • {len(high_cond)} datasets have very high condition numbers (>100)")
    
    return {
        'analysis': analysis,
        'problematic_datasets': problematic_datasets
    }

def main():
    """Main function to run the collinearity analysis."""
    
    print("🔬 Enhanced Collinearity Analysis")
    print("=" * 60)
    
    # Load data
    df = load_collinearity_data()
    if df is None:
        return
    
    # Generate comprehensive report
    results = generate_summary_report(df)
    
    # Save summary to file
    output_file = "collinearity_analysis_summary.txt"
    print(f"\n💾 Detailed analysis saved to: {output_file}")
    
    # You could add visualization here if needed
    print(f"\n✅ Analysis complete!")
    total_indicators = sum(results['analysis']['collinearity_counts'].values())
    print(f"   Analyzed {len(df)} datasets")
    print(f"   Total collinearity indicators detected: {total_indicators}")

if __name__ == "__main__":
    main()