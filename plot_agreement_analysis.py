#!/usr/bin/env python3
"""
Create comprehensive plots of agreement metrics aggregated by dataset and model.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")

def load_data():
    """Load the agreement analysis data."""
    df = pd.read_csv("results/agreement/agreement_analysis.csv")
    return df

def create_dataset_plots(df):
    """Create plots aggregated by dataset for all three metrics."""
    
    # Aggregate by dataset
    dataset_stats = df.groupby('dataset').agg({
        'kendall_tau': ['mean', 'std'],
        'spearman': ['mean', 'std'],
        'iou_topk': ['mean', 'std']
    }).round(4)
    
    # Flatten column names
    dataset_stats.columns = ['_'.join(col).strip() for col in dataset_stats.columns]
    dataset_stats = dataset_stats.reset_index()
    
    # Sort by composite score
    dataset_stats['composite_mean'] = (dataset_stats['kendall_tau_mean'] + 
                                     dataset_stats['spearman_mean'] + 
                                     dataset_stats['iou_topk_mean']) / 3
    dataset_stats = dataset_stats.sort_values('composite_mean', ascending=False)
    
    # Create subplot figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Agreement Metrics by Dataset', fontsize=16, fontweight='bold')
    
    # Plot 1: Kendall's Tau
    ax1 = axes[0, 0]
    bars1 = ax1.bar(range(len(dataset_stats)), dataset_stats['kendall_tau_mean'], 
                    yerr=dataset_stats['kendall_tau_std'], capsize=3, alpha=0.7, color='skyblue')
    ax1.set_title("Kendall's Tau by Dataset", fontweight='bold')
    ax1.set_ylabel("Mean Kendall's Tau")
    ax1.set_xticks(range(len(dataset_stats)))
    ax1.set_xticklabels(dataset_stats['dataset'], rotation=45, ha='right')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1)
    
    # Plot 2: Spearman Correlation
    ax2 = axes[0, 1]
    bars2 = ax2.bar(range(len(dataset_stats)), dataset_stats['spearman_mean'], 
                    yerr=dataset_stats['spearman_std'], capsize=3, alpha=0.7, color='lightcoral')
    ax2.set_title("Spearman Correlation by Dataset", fontweight='bold')
    ax2.set_ylabel("Mean Spearman Correlation")
    ax2.set_xticks(range(len(dataset_stats)))
    ax2.set_xticklabels(dataset_stats['dataset'], rotation=45, ha='right')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)
    
    # Plot 3: IOU@3
    ax3 = axes[1, 0]
    bars3 = ax3.bar(range(len(dataset_stats)), dataset_stats['iou_topk_mean'], 
                    yerr=dataset_stats['iou_topk_std'], capsize=3, alpha=0.7, color='lightgreen')
    ax3.set_title("IOU@3 by Dataset", fontweight='bold')
    ax3.set_ylabel("Mean IOU@3")
    ax3.set_xticks(range(len(dataset_stats)))
    ax3.set_xticklabels(dataset_stats['dataset'], rotation=45, ha='right')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 1)
    
    # Plot 4: Composite Score
    ax4 = axes[1, 1]
    bars4 = ax4.bar(range(len(dataset_stats)), dataset_stats['composite_mean'], 
                    alpha=0.7, color='gold')
    ax4.set_title("Composite Agreement Score by Dataset", fontweight='bold')
    ax4.set_ylabel("Mean Composite Score")
    ax4.set_xticks(range(len(dataset_stats)))
    ax4.set_xticklabels(dataset_stats['dataset'], rotation=45, ha='right')
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig('results/agreement/agreement_by_dataset.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return dataset_stats

def create_model_plots(df):
    """Create plots aggregated by model for all three metrics."""
    
    # Aggregate by model
    model_stats = df.groupby('model').agg({
        'kendall_tau': ['mean', 'std'],
        'spearman': ['mean', 'std'],
        'iou_topk': ['mean', 'std']
    }).round(4)
    
    # Flatten column names
    model_stats.columns = ['_'.join(col).strip() for col in model_stats.columns]
    model_stats = model_stats.reset_index()
    
    # Sort by composite score
    model_stats['composite_mean'] = (model_stats['kendall_tau_mean'] + 
                                   model_stats['spearman_mean'] + 
                                   model_stats['iou_topk_mean']) / 3
    model_stats = model_stats.sort_values('composite_mean', ascending=False)
    
    # Create subplot figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Agreement Metrics by Model', fontsize=16, fontweight='bold')
    
    # Plot 1: Kendall's Tau
    ax1 = axes[0, 0]
    bars1 = ax1.bar(range(len(model_stats)), model_stats['kendall_tau_mean'], 
                    yerr=model_stats['kendall_tau_std'], capsize=3, alpha=0.7, color='skyblue')
    ax1.set_title("Kendall's Tau by Model", fontweight='bold')
    ax1.set_ylabel("Mean Kendall's Tau")
    ax1.set_xticks(range(len(model_stats)))
    ax1.set_xticklabels(model_stats['model'], rotation=45, ha='right')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1)
    
    # Plot 2: Spearman Correlation
    ax2 = axes[0, 1]
    bars2 = ax2.bar(range(len(model_stats)), model_stats['spearman_mean'], 
                    yerr=model_stats['spearman_std'], capsize=3, alpha=0.7, color='lightcoral')
    ax2.set_title("Spearman Correlation by Model", fontweight='bold')
    ax2.set_ylabel("Mean Spearman Correlation")
    ax2.set_xticks(range(len(model_stats)))
    ax2.set_xticklabels(model_stats['model'], rotation=45, ha='right')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)
    
    # Plot 3: IOU@3
    ax3 = axes[1, 0]
    bars3 = ax3.bar(range(len(model_stats)), model_stats['iou_topk_mean'], 
                    yerr=model_stats['iou_topk_std'], capsize=3, alpha=0.7, color='lightgreen')
    ax3.set_title("IOU@3 by Model", fontweight='bold')
    ax3.set_ylabel("Mean IOU@3")
    ax3.set_xticks(range(len(model_stats)))
    ax3.set_xticklabels(model_stats['model'], rotation=45, ha='right')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 1)
    
    # Plot 4: Composite Score
    ax4 = axes[1, 1]
    bars4 = ax4.bar(range(len(model_stats)), model_stats['composite_mean'], 
                    alpha=0.7, color='gold')
    ax4.set_title("Composite Agreement Score by Model", fontweight='bold')
    ax4.set_ylabel("Mean Composite Score")
    ax4.set_xticks(range(len(model_stats)))
    ax4.set_xticklabels(model_stats['model'], rotation=45, ha='right')
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig('results/agreement/agreement_by_model.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return model_stats

def create_pairwise_comparison_plots(df):
    """Create heatmaps showing pairwise agreement patterns."""
    
    # Create separate plots for each metric
    metrics = ['kendall_tau', 'spearman', 'iou_topk']
    metric_names = ["Kendall's Tau", "Spearman Correlation", "IOU@3"]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Pairwise Agreement by Method Pair', fontsize=16, fontweight='bold')
    
    for i, (metric, name) in enumerate(zip(metrics, metric_names)):
        # Pivot data for heatmap
        pivot_data = df.groupby(['dataset', 'pair'])[metric].mean().unstack()
        
        # Create heatmap
        sns.heatmap(pivot_data, annot=False, cmap='RdYlBu_r', 
                   center=0.5, vmin=0, vmax=1, ax=axes[i])
        axes[i].set_title(f'{name}', fontweight='bold')
        axes[i].set_ylabel('Dataset')
        axes[i].set_xlabel('Method Pair')
        
        # Rotate labels
        axes[i].tick_params(axis='x', rotation=45)
        axes[i].tick_params(axis='y', rotation=0)
    
    plt.tight_layout()
    plt.savefig('results/agreement/agreement_heatmaps.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_distribution_plots(df):
    """Create distribution plots for agreement metrics."""
    
    # Create violin plots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Distribution of Agreement Metrics', fontsize=16, fontweight='bold')
    
    # Kendall's Tau distribution by pair
    ax1 = axes[0, 0]
    sns.violinplot(data=df, x='pair', y='kendall_tau', ax=ax1)
    ax1.set_title("Kendall's Tau Distribution by Pair", fontweight='bold')
    ax1.set_ylabel("Kendall's Tau")
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3)
    
    # Spearman distribution by pair
    ax2 = axes[0, 1]
    sns.violinplot(data=df, x='pair', y='spearman', ax=ax2)
    ax2.set_title("Spearman Correlation Distribution by Pair", fontweight='bold')
    ax2.set_ylabel("Spearman Correlation")
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)
    
    # IOU distribution by pair
    ax3 = axes[1, 0]
    sns.violinplot(data=df, x='pair', y='iou_topk', ax=ax3)
    ax3.set_title("IOU@3 Distribution by Pair", fontweight='bold')
    ax3.set_ylabel("IOU@3")
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(True, alpha=0.3)
    
    # Composite score distribution
    df['composite_score'] = (df['kendall_tau'] + df['spearman'] + df['iou_topk']) / 3
    ax4 = axes[1, 1]
    sns.violinplot(data=df, x='pair', y='composite_score', ax=ax4)
    ax4.set_title("Composite Score Distribution by Pair", fontweight='bold')
    ax4.set_ylabel("Composite Score")
    ax4.tick_params(axis='x', rotation=45)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/agreement/agreement_distributions.png', dpi=300, bbox_inches='tight')
    plt.show()

def print_top_performers(dataset_stats, model_stats):
    """Print summary of top performing datasets and models."""
    
    print("="*80)
    print("SUMMARY OF TOP PERFORMERS")
    print("="*80)
    
    print("\nTop 5 Datasets by Composite Agreement:")
    print(dataset_stats[['dataset', 'composite_mean']].head().to_string(index=False))
    
    print("\nBottom 5 Datasets by Composite Agreement:")
    print(dataset_stats[['dataset', 'composite_mean']].tail().to_string(index=False))
    
    print("\nTop 5 Models by Composite Agreement:")
    print(model_stats[['model', 'composite_mean']].head().to_string(index=False))
    
    print("\nBottom 5 Models by Composite Agreement:")
    print(model_stats[['model', 'composite_mean']].tail().to_string(index=False))

def main():
    """Main function to create all plots."""
    
    # Create results directory if it doesn't exist
    Path("results/agreement").mkdir(exist_ok=True)
    
    print("Loading agreement analysis data...")
    df = load_data()
    
    print("Creating dataset aggregation plots...")
    dataset_stats = create_dataset_plots(df)
    
    print("Creating model aggregation plots...")
    model_stats = create_model_plots(df)
    
    print("Creating pairwise comparison heatmaps...")
    create_pairwise_comparison_plots(df)
    
    print("Creating distribution plots...")
    create_distribution_plots(df)
    
    print("Generating summary statistics...")
    print_top_performers(dataset_stats, model_stats)
    
    print("\n" + "="*80)
    print("All plots saved to results/agreement/ directory:")
    print("- agreement_by_dataset.png")
    print("- agreement_by_model.png") 
    print("- agreement_heatmaps.png")
    print("- agreement_distributions.png")
    print("="*80)

if __name__ == "__main__":
    main()