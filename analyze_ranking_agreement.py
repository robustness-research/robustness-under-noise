"""
Analyze and interpret ranking agreement results across different feature importance methods.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')


class RankingAgreementAnalyzer:
    """Analyzer for ranking agreement results between feature importance methods."""
    
    def __init__(self, csv_path):
        """Initialize analyzer with CSV file."""
        self.df = pd.read_csv(csv_path)
        self.metrics = ['kendall_tau', 'spearman', 'iou_topk']
        self._prepare_data()
    
    def _prepare_data(self):
        """Prepare data for analysis."""
        # Separate method pairs
        self.df[['method1', 'method2']] = self.df['pair'].str.split(' vs ', expand=True)
        
        # Count non-null values per metric
        self.valid_counts = {metric: self.df[metric].notna().sum() for metric in self.metrics}
    
    def get_agreement_interpretation(self, value, metric):
        """Interpret agreement score based on metric type."""
        if pd.isna(value):
            return "No data"
        
        if metric in ['kendall_tau', 'spearman']:
            # Rank correlation: -1 to 1
            if value >= 0.8:
                return "Very High Agreement (>0.8)"
            elif value >= 0.6:
                return "High Agreement (0.6-0.8)"
            elif value >= 0.4:
                return "Moderate Agreement (0.4-0.6)"
            elif value >= 0.2:
                return "Weak Agreement (0.2-0.4)"
            elif value >= 0:
                return "Very Weak Agreement (0-0.2)"
            elif value >= -0.2:
                return "Very Weak Disagreement (-0.2-0)"
            elif value >= -0.4:
                return "Weak Disagreement (-0.4-(-0.2))"
            elif value >= -0.6:
                return "Moderate Disagreement (-0.6-(-0.4))"
            elif value >= -0.8:
                return "High Disagreement (-0.8-(-0.6))"
            else:
                return "Very High Disagreement (<-0.8)"
        
        elif metric == 'iou_topk':
            # IOU: 0 to 1
            if value >= 0.8:
                return "Very High Overlap (>0.8)"
            elif value >= 0.6:
                return "High Overlap (0.6-0.8)"
            elif value >= 0.4:
                return "Moderate Overlap (0.4-0.6)"
            elif value >= 0.2:
                return "Low Overlap (0.2-0.4)"
            else:
                return "Very Low Overlap (<0.2)"
    
    def summary_statistics(self):
        """Generate summary statistics for all metrics."""
        print("\n" + "="*70)
        print("RANKING AGREEMENT SUMMARY STATISTICS")
        print("="*70)
        
        for metric in self.metrics:
            data = self.df[metric].dropna()
            print(f"\n{metric.upper()}")
            print("-" * 70)
            print(f"  Valid Records:        {len(data)} / {len(self.df)}")
            print(f"  Mean:                 {data.mean():.4f}")
            print(f"  Median:               {data.median():.4f}")
            print(f"  Std Dev:              {data.std():.4f}")
            print(f"  Min:                  {data.min():.4f}")
            print(f"  Max:                  {data.max():.4f}")
            print(f"  Q1 (25th percentile): {data.quantile(0.25):.4f}")
            print(f"  Q3 (75th percentile): {data.quantile(0.75):.4f}")
    
    def agreement_distribution(self):
        """Analyze distribution of agreement levels."""
        print("\n" + "="*70)
        print("AGREEMENT LEVEL DISTRIBUTION")
        print("="*70)
        
        for metric in ['kendall_tau', 'spearman', 'iou_topk']:
            data = self.df[metric].dropna()
            
            print(f"\n{metric.upper()}")
            print("-" * 70)
            
            # Create bins for categorization
            if metric in ['kendall_tau', 'spearman']:
                bins = [-1.0, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1.0]
                labels = ['Very High Disagree', 'High Disagree', 'Moderate Disagree', 
                         'Weak Disagree', 'Very Weak Disagree', 'Very Weak Agree',
                         'Weak Agree', 'Moderate Agree', 'High Agree', 'Very High Agree']
            else:  # iou_topk
                bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
                labels = ['Very Low', 'Low', 'Moderate', 'High', 'Very High']
            
            categorized = pd.cut(data, bins=bins, labels=labels, include_lowest=True)
            counts = categorized.value_counts().sort_index()
            
            for level, count in counts.items():
                percentage = (count / len(data)) * 100
                print(f"  {level:.<40} {count:>4} ({percentage:>6.2f}%)")
    
    def best_worst_pairs(self, n=10):
        """Identify best and worst method pairs."""
        print("\n" + "="*70)
        print("BEST AND WORST METHOD PAIRS (by Kendall Tau)")
        print("="*70)
        
        valid_data = self.df[self.df['kendall_tau'].notna()].copy()
        valid_data['pair_display'] = (valid_data['dataset'] + " | " + 
                                      valid_data['method'] + " | " + 
                                      valid_data['pair'])
        
        print(f"\nTOP {n} BEST AGREEMENTS:")
        print("-" * 70)
        best = valid_data.nlargest(n, 'kendall_tau')[['pair_display', 'kendall_tau', 'spearman', 'iou_topk']]
        for idx, (_, row) in enumerate(best.iterrows(), 1):
            print(f"{idx:2d}. {row['pair_display']}")
            print(f"    Kendall Tau: {row['kendall_tau']:>7.4f} | Spearman: {row['spearman']:>7.4f} | IOU: {row['iou_topk']:>5.2f}")
        
        print(f"\nTOP {n} WORST AGREEMENTS:")
        print("-" * 70)
        worst = valid_data.nsmallest(n, 'kendall_tau')[['pair_display', 'kendall_tau', 'spearman', 'iou_topk']]
        for idx, (_, row) in enumerate(worst.iterrows(), 1):
            print(f"{idx:2d}. {row['pair_display']}")
            print(f"    Kendall Tau: {row['kendall_tau']:>7.4f} | Spearman: {row['spearman']:>7.4f} | IOU: {row['iou_topk']:>5.2f}")
    
    def method_pair_analysis(self):
        """Analyze agreement by method pair (method1 vs method2)."""
        print("\n" + "="*70)
        print("AGREEMENT BY METHOD PAIR")
        print("="*70)
        
        pair_stats = self.df.groupby('pair').agg({
            'kendall_tau': ['count', 'mean', 'std'],
            'spearman': ['mean', 'std'],
            'iou_topk': ['mean', 'std']
        }).round(4)
        
        # Flatten multi-level columns
        pair_stats.columns = ['_'.join(col).strip() for col in pair_stats.columns.values]
        pair_stats = pair_stats.rename(columns={
            'kendall_tau_count': 'n_obs',
            'kendall_tau_mean': 'tau_mean',
            'kendall_tau_std': 'tau_std',
            'spearman_mean': 'spear_mean',
            'spearman_std': 'spear_std',
            'iou_topk_mean': 'iou_mean',
            'iou_topk_std': 'iou_std'
        })
        
        # Sort by mean Kendall Tau
        pair_stats = pair_stats.sort_values('tau_mean', ascending=False)
        
        for pair, row in pair_stats.iterrows():
            print(f"\n{pair}")
            print(f"  N Observations:        {int(row['n_obs'])}")
            print(f"  Kendall Tau:   {row['tau_mean']:.4f} ± {row['tau_std']:.4f}")
            print(f"  Spearman:      {row['spear_mean']:.4f} ± {row['spear_std']:.4f}")
            print(f"  IOU Top-K:     {row['iou_mean']:.4f} ± {row['iou_std']:.4f}")
            agreement = self.get_agreement_interpretation(row['tau_mean'], 'kendall_tau')
            print(f"  Overall:       {agreement}")
    
    def method_analysis(self):
        """Analyze agreement by individual methods."""
        print("\n" + "="*70)
        print("AGREEMENT BY INDIVIDUAL METHOD")
        print("="*70)
        
        methods = set(self.df['method'].unique())
        method_stats = {}
        
        for method in sorted(methods):
            method_data = self.df[self.df['method'] == method]
            method_stats[method] = {
                'n_pairs': len(method_data),
                'kendall_mean': method_data['kendall_tau'].mean(),
                'spearman_mean': method_data['spearman'].mean(),
                'iou_mean': method_data['iou_topk'].mean(),
                'kendall_std': method_data['kendall_tau'].std(),
                'spearman_std': method_data['spearman'].std(),
                'iou_std': method_data['iou_topk'].std(),
            }
        
        # Sort by mean Kendall Tau
        sorted_methods = sorted(method_stats.items(), 
                               key=lambda x: x[1]['kendall_mean'], 
                               reverse=True)
        
        for method, stats in sorted_methods:
            print(f"\n{method}")
            print(f"  Total Comparisons:  {stats['n_pairs']}")
            print(f"  Kendall Tau:        {stats['kendall_mean']:.4f} ± {stats['kendall_std']:.4f}")
            print(f"  Spearman:           {stats['spearman_mean']:.4f} ± {stats['spearman_std']:.4f}")
            print(f"  IOU Top-K:          {stats['iou_mean']:.4f} ± {stats['iou_std']:.4f}")
    
    def dataset_analysis(self):
        """Analyze agreement by dataset."""
        print("\n" + "="*70)
        print("AGREEMENT BY DATASET")
        print("="*70)
        
        dataset_stats = self.df.groupby('dataset').agg({
            'kendall_tau': ['count', 'mean', 'std'],
            'spearman': ['mean', 'std'],
            'iou_topk': ['mean', 'std']
        }).round(4)
        
        # Flatten multi-level columns
        dataset_stats.columns = ['_'.join(col).strip() for col in dataset_stats.columns.values]
        dataset_stats = dataset_stats.rename(columns={
            'kendall_tau_count': 'n_obs',
            'kendall_tau_mean': 'tau_mean',
            'kendall_tau_std': 'tau_std',
            'spearman_mean': 'spear_mean',
            'spearman_std': 'spear_std',
            'iou_topk_mean': 'iou_mean',
            'iou_topk_std': 'iou_std'
        })
        
        # Sort by mean Kendall Tau
        dataset_stats = dataset_stats.sort_values('tau_mean', ascending=False)
        
        print("\nTOP 10 DATASETS WITH HIGHEST AGREEMENT:")
        print("-" * 70)
        for dataset, row in dataset_stats.head(10).iterrows():
            print(f"\n{dataset}")
            print(f"  N Observations:        {int(row['n_obs'])}")
            print(f"  Kendall Tau:   {row['tau_mean']:.4f} ± {row['tau_std']:.4f}")
            print(f"  Spearman:      {row['spear_mean']:.4f} ± {row['spear_std']:.4f}")
            print(f"  IOU Top-K:     {row['iou_mean']:.4f} ± {row['iou_std']:.4f}")
        
        print("\n\nTOP 10 DATASETS WITH LOWEST AGREEMENT:")
        print("-" * 70)
        for dataset, row in dataset_stats.tail(10).iterrows():
            print(f"\n{dataset}")
            print(f"  N Observations:        {int(row['n_obs'])}")
            print(f"  Kendall Tau:   {row['tau_mean']:.4f} ± {row['tau_std']:.4f}")
            print(f"  Spearman:      {row['spear_mean']:.4f} ± {row['spear_std']:.4f}")
            print(f"  IOU Top-K:     {row['iou_mean']:.4f} ± {row['iou_std']:.4f}")
    
    def overall_interpretation(self):
        """Generate overall interpretation of results."""
        print("\n" + "="*70)
        print("OVERALL INTERPRETATION")
        print("="*70)
        
        kendall_mean = self.df['kendall_tau'].mean()
        spearman_mean = self.df['spearman'].mean()
        iou_mean = self.df['iou_topk'].mean()
        
        print(f"\nGLOBAL AGREEMENT METRICS:")
        print("-" * 70)
        print(f"Mean Kendall Tau:   {kendall_mean:.4f}")
        print(f"Mean Spearman:      {spearman_mean:.4f}")
        print(f"Mean IOU Top-K:     {iou_mean:.4f}")
        
        print(f"\nKEY FINDINGS:")
        print("-" * 70)
        
        # Kendall Tau interpretation
        tau_interpretation = self.get_agreement_interpretation(kendall_mean, 'kendall_tau')
        print(f"\n1. Rank Correlation (Kendall Tau):")
        print(f"   Overall: {tau_interpretation}")
        print(f"   • This indicates that feature importance rankings from different")
        print(f"     methods show {tau_interpretation.lower()} patterns.")
        
        # IOU interpretation
        iou_interpretation = self.get_agreement_interpretation(iou_mean, 'iou_topk')
        print(f"\n2. Top-K Feature Overlap (IOU):")
        print(f"   Overall: {iou_interpretation}")
        print(f"   • Top-k important features show {iou_interpretation.lower()} overlap")
        print(f"     between different explanation methods.")
        
        # Count high vs low agreement
        high_agreement = (self.df['kendall_tau'] >= 0.6).sum()
        low_agreement = (self.df['kendall_tau'] < 0.2).sum()
        total_valid = self.df['kendall_tau'].notna().sum()
        
        print(f"\n3. Agreement Distribution:")
        print(f"   High Agreement (Kendall Tau ≥ 0.6):  {high_agreement:>4} ({high_agreement/total_valid*100:>6.1f}%)")
        print(f"   Low Agreement  (Kendall Tau < 0.2):  {low_agreement:>4} ({low_agreement/total_valid*100:>6.1f}%)")
        
        # Method pair insights
        print(f"\n4. Most/Least Reliable Method Pairs:")
        valid_data = self.df[self.df['kendall_tau'].notna()].copy()
        best_pair = valid_data.loc[valid_data['kendall_tau'].idxmax()]
        worst_pair = valid_data.loc[valid_data['kendall_tau'].idxmin()]
        
        print(f"   Best:   {best_pair['pair']} (Kendall Tau: {best_pair['kendall_tau']:.4f})")
        print(f"   Worst:  {worst_pair['pair']} (Kendall Tau: {worst_pair['kendall_tau']:.4f})")
        
        print(f"\nCONCLUSION:")
        print("-" * 70)
        if kendall_mean >= 0.6:
            conclusion = ("High agreement is observed across methods. Feature importance")
            conclusion += ("\nrankings are generally consistent, suggesting robust feature")
            conclusion += ("\nidentification patterns in the model.")
        elif kendall_mean >= 0.4:
            conclusion = ("Moderate agreement is observed. While there is some consistency")
            conclusion += ("\nin feature rankings, different methods may highlight different")
            conclusion += ("\naspects of feature importance.")
        elif kendall_mean >= 0.2:
            conclusion = ("Weak agreement suggests that feature importance methods provide")
            conclusion += ("\ndivergent perspectives. Results should be interpreted with")
            conclusion += ("\ncaution and consider multiple methods for robustness.")
        else:
            conclusion = ("Low agreement indicates significant disagreement between methods.")
            conclusion += ("\nEach method captures different aspects of feature importance.")
            conclusion += ("\nMulti-method approach is essential for comprehensive analysis.")
        
        print(conclusion)


def main():
    """Main function to run the analysis."""
    csv_path = Path(__file__).parent / 'ranking_agreement_results.csv'
    
    if not csv_path.exists():
        print(f"Error: File not found at {csv_path}")
        return
    
    analyzer = RankingAgreementAnalyzer(str(csv_path))
    
    # Run all analyses
    analyzer.summary_statistics()
    analyzer.agreement_distribution()
    analyzer.best_worst_pairs(n=5)
    analyzer.method_pair_analysis()
    analyzer.method_analysis()
    analyzer.dataset_analysis()
    analyzer.overall_interpretation()
    
    print("\n" + "="*70)
    print("Analysis Complete!")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
