#!/usr/bin/env python3

import os
import sys
import glob
import warnings
import json
from pathlib import Path

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.io import arff
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Try to import pyreadr for RDS file support
try:
    import pyreadr
    RDS_SUPPORT = True
except ImportError:
    print("Warning: pyreadr not installed. RDS file support disabled.")
    print("Install with: pip install pyreadr")
    RDS_SUPPORT = False


class CollinearityAnalyzer:
    """Analyzes multicollinearity in datasets using multiple methods."""
    
    def __init__(self, dataset_path, output_dir="results/collinearity_analysis"):
        """
        Initialize the analyzer.
        
        Args:
            dataset_path: Path to the dataset file
            output_dir: Directory to save output files
        """
        self.dataset_path = dataset_path
        self.dataset_name = Path(dataset_path).stem
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.df = None
        self.numeric_df = None
        self.correlation_matrix = None
        self.results = {
            'dataset': self.dataset_name,
            'file_path': str(dataset_path)
        }
    
    def load_data(self):
        """Load dataset from various formats (RDS, ARFF, CSV)."""
        print(f"\n{'='*60}")
        print(f"Loading dataset: {self.dataset_name}")
        print(f"{'='*60}")
        
        ext = Path(self.dataset_path).suffix.lower()
        
        try:
            if ext == '.rds':
                if not RDS_SUPPORT:
                    raise ImportError("pyreadr not installed")
                self._load_rds()
            elif ext == '.arff':
                self._load_arff()
            elif ext == '.csv':
                self._load_csv()
            else:
                raise ValueError(f"Unsupported file format: {ext}")
            
            print(f"✓ Successfully loaded dataset")
            print(f"  Shape: {self.df.shape}")
            print(f"  Columns: {list(self.df.columns)}")
            
            # Store basic info
            self.results['n_rows'] = len(self.df)
            self.results['n_columns'] = len(self.df.columns)
            
            return True
            
        except Exception as e:
            print(f"✗ Error loading dataset: {e}")
            self.results['error'] = str(e)
            return False
    
    def _load_rds(self):
        """Load RDS file using pyreadr."""
        result = pyreadr.read_r(self.dataset_path)
        # RDS files typically have one dataframe
        self.df = result[None] if None in result else list(result.values())[0]
    
    def _load_arff(self):
        """Load ARFF file."""
        data, meta = arff.loadarff(self.dataset_path)
        self.df = pd.DataFrame(data)
        
        # Convert byte strings to regular strings for categorical columns
        for col in self.df.columns:
            if self.df[col].dtype == object:
                try:
                    self.df[col] = self.df[col].str.decode('utf-8')
                except:
                    pass
    
    def _load_csv(self):
        """Load CSV file."""
        self.df = pd.read_csv(self.dataset_path)
    
    def prepare_numeric_data(self):
        """
        Prepare numeric data for collinearity analysis.
        - Select only numeric columns
        - Handle missing values
        - Exclude target variable if identifiable
        """
        print(f"\nPreparing numeric data...")
        
        # Get numeric columns
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) == 0:
            print("✗ No numeric columns found in dataset")
            self.results['error'] = "No numeric columns"
            return False
        
        # Create numeric dataframe
        self.numeric_df = self.df[numeric_cols].copy()
        
        # Handle missing values
        n_missing = self.numeric_df.isnull().sum().sum()
        if n_missing > 0:
            print(f"  Warning: {n_missing} missing values found, dropping rows")
            self.numeric_df = self.numeric_df.dropna()
        
        # Remove constant columns (zero variance)
        constant_cols = [col for col in self.numeric_df.columns 
                        if self.numeric_df[col].std() == 0]
        if constant_cols:
            print(f"  Removing {len(constant_cols)} constant columns: {constant_cols}")
            self.numeric_df = self.numeric_df.drop(columns=constant_cols)
        
        print(f"✓ Prepared {len(self.numeric_df.columns)} numeric features")
        print(f"  Features: {list(self.numeric_df.columns)}")
        
        self.results['n_numeric_features'] = len(self.numeric_df.columns)
        self.results['numeric_features'] = list(self.numeric_df.columns)
        
        return len(self.numeric_df.columns) >= 2
    
    def calculate_correlation_matrix(self, save_heatmap=True):
        """
        Calculate correlation matrix and identify high correlations.
        
        Args:
            save_heatmap: Whether to save correlation heatmap visualization
        """
        print(f"\n1. Correlation Matrix Analysis")
        print(f"-" * 60)
        
        if self.numeric_df is None or len(self.numeric_df.columns) < 2:
            print("✗ Insufficient numeric features for correlation analysis")
            return
        
        # Calculate correlation matrix
        self.correlation_matrix = self.numeric_df.corr()
        
        # Find high correlations (excluding diagonal)
        high_corr_threshold = 0.7
        high_correlations = []
        
        for i in range(len(self.correlation_matrix.columns)):
            for j in range(i+1, len(self.correlation_matrix.columns)):
                corr_value = self.correlation_matrix.iloc[i, j]
                if abs(corr_value) >= high_corr_threshold:
                    high_correlations.append({
                        'feature1': self.correlation_matrix.columns[i],
                        'feature2': self.correlation_matrix.columns[j],
                        'correlation': corr_value
                    })
        
        # Print results
        print(f"✓ Correlation matrix calculated ({self.correlation_matrix.shape})")
        print(f"  High correlations (|r| >= {high_corr_threshold}): {len(high_correlations)}")
        
        if high_correlations:
            print(f"\n  Top highly correlated pairs:")
            sorted_corrs = sorted(high_correlations, 
                                key=lambda x: abs(x['correlation']), 
                                reverse=True)[:5]
            for item in sorted_corrs:
                print(f"    • {item['feature1']} ↔ {item['feature2']}: {item['correlation']:.3f}")
        
        # Store results
        self.results['correlation_matrix_shape'] = self.correlation_matrix.shape
        self.results['high_correlations'] = high_correlations
        self.results['n_high_correlations'] = len(high_correlations)
        
        # Save heatmap
        if save_heatmap:
            self._save_correlation_heatmap()
    
    def _save_correlation_heatmap(self):
        """Save correlation matrix heatmap."""
        try:
            plt.figure(figsize=(12, 10))
            
            # Create heatmap
            sns.heatmap(self.correlation_matrix, 
                       annot=len(self.correlation_matrix.columns) <= 15,  # Annotate if not too many features
                       cmap='coolwarm', 
                       center=0,
                       vmin=-1, 
                       vmax=1,
                       square=True,
                       linewidths=0.5,
                       cbar_kws={"shrink": 0.8})
            
            plt.title(f'Correlation Matrix - {self.dataset_name}', 
                     fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            # Save
            output_path = self.output_dir / f"{self.dataset_name}_correlation_heatmap.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"  Heatmap saved: {output_path}")
            
        except Exception as e:
            print(f"  Warning: Could not save heatmap: {e}")
    
    def calculate_vif(self):
        """
        Calculate Variance Inflation Factor (VIF) for each feature.
        VIF > 10 indicates high multicollinearity.
        """
        print(f"\n2. Variance Inflation Factor (VIF) Analysis")
        print(f"-" * 60)
        
        if self.numeric_df is None or len(self.numeric_df.columns) < 2:
            print("✗ Insufficient numeric features for VIF analysis")
            return
        
        try:
            vif_data = pd.DataFrame()
            vif_data["feature"] = self.numeric_df.columns
            vif_data["VIF"] = [
                variance_inflation_factor(self.numeric_df.values, i) 
                for i in range(len(self.numeric_df.columns))
            ]
            
            # Sort by VIF value
            vif_data = vif_data.sort_values('VIF', ascending=False)
            
            # Identify high VIF features
            high_vif_threshold = 10
            high_vif = vif_data[vif_data['VIF'] > high_vif_threshold]
            
            print(f"✓ VIF calculated for {len(vif_data)} features")
            print(f"  Features with VIF > {high_vif_threshold}: {len(high_vif)}")
            
            print(f"\n  VIF Summary:")
            print(f"    Mean VIF: {vif_data['VIF'].mean():.2f}")
            print(f"    Median VIF: {vif_data['VIF'].median():.2f}")
            print(f"    Max VIF: {vif_data['VIF'].max():.2f}")
            
            if len(high_vif) > 0:
                print(f"\n  Features with high VIF (> {high_vif_threshold}):")
                for _, row in high_vif.iterrows():
                    vif_str = f"{row['VIF']:.2f}" if row['VIF'] < 1000 else "∞"
                    print(f"    • {row['feature']}: {vif_str}")
            
            # Store results
            self.results['vif_mean'] = float(vif_data['VIF'].mean())
            self.results['vif_median'] = float(vif_data['VIF'].median())
            self.results['vif_max'] = float(vif_data['VIF'].max())
            self.results['vif_values'] = vif_data.to_dict('records')
            self.results['n_high_vif'] = len(high_vif)
            
        except Exception as e:
            print(f"✗ Error calculating VIF: {e}")
            self.results['vif_error'] = str(e)
    
    def calculate_condition_number(self):
        """
        Calculate condition number of the feature matrix.
        Condition number > 30 indicates strong multicollinearity.
        """
        print(f"\n3. Condition Number Analysis")
        print(f"-" * 60)
        
        if self.numeric_df is None or len(self.numeric_df.columns) < 2:
            print("✗ Insufficient numeric features for condition number analysis")
            return
        
        try:
            # Standardize the data first for numerical stability
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(self.numeric_df)
            
            # Calculate condition number
            condition_number = np.linalg.cond(scaled_data)
            
            # Interpret result
            if condition_number < 10:
                interpretation = "Low - no multicollinearity"
            elif condition_number < 30:
                interpretation = "Moderate - some multicollinearity"
            else:
                interpretation = "High - strong multicollinearity"
            
            print(f"✓ Condition Number: {condition_number:.2f}")
            print(f"  Interpretation: {interpretation}")
            
            # Store results
            self.results['condition_number'] = float(condition_number)
            self.results['condition_number_interpretation'] = interpretation
            
        except Exception as e:
            print(f"✗ Error calculating condition number: {e}")
            self.results['condition_number_error'] = str(e)
    
    def calculate_eigenvalues(self):
        """
        Calculate eigenvalues of correlation matrix.
        Small eigenvalues (close to 0) indicate multicollinearity.
        """
        print(f"\n4. Eigenvalue Analysis")
        print(f"-" * 60)
        
        if self.correlation_matrix is None:
            print("✗ Correlation matrix not available")
            return
        
        try:
            # Calculate eigenvalues and eigenvectors
            eigenvalues, eigenvectors = np.linalg.eig(self.correlation_matrix)
            
            # Sort eigenvalues in descending order
            eigenvalues = np.sort(eigenvalues)[::-1]
            
            # Count small eigenvalues (indicating multicollinearity)
            small_eigenvalue_threshold = 0.01
            small_eigenvalues = eigenvalues[eigenvalues < small_eigenvalue_threshold]
            
            print(f"✓ Eigenvalues calculated")
            print(f"  Eigenvalues: {', '.join([f'{ev:.4f}' for ev in eigenvalues])}")
            print(f"  Smallest eigenvalue: {eigenvalues.min():.6f}")
            print(f"  Number of small eigenvalues (< {small_eigenvalue_threshold}): {len(small_eigenvalues)}")
            
            # Calculate condition index (ratio of max to min eigenvalue)
            if eigenvalues.min() > 0:
                condition_index = np.sqrt(eigenvalues.max() / eigenvalues.min())
                print(f"  Condition Index: {condition_index:.2f}")
                self.results['condition_index'] = float(condition_index)
            
            # Store results
            self.results['eigenvalues'] = [float(ev) for ev in eigenvalues]
            self.results['min_eigenvalue'] = float(eigenvalues.min())
            self.results['max_eigenvalue'] = float(eigenvalues.max())
            self.results['n_small_eigenvalues'] = int(len(small_eigenvalues))
            
        except Exception as e:
            print(f"✗ Error calculating eigenvalues: {e}")
            self.results['eigenvalue_error'] = str(e)
    
    def generate_summary(self):
        """Generate overall collinearity summary."""
        print(f"\n{'='*60}")
        print(f"COLLINEARITY SUMMARY - {self.dataset_name}")
        print(f"{'='*60}")
        
        has_multicollinearity = False
        severity_score = 0
        
        # Check each indicator
        print(f"\nMulticollinearity Indicators:")
        
        # 1. Correlation-based
        if 'n_high_correlations' in self.results:
            n_high_corr = self.results['n_high_correlations']
            status = "✓ DETECTED" if n_high_corr > 0 else "✗ None"
            print(f"  1. High Correlations (|r| >= 0.7): {status}")
            if n_high_corr > 0:
                print(f"     Found {n_high_corr} highly correlated pairs")
                has_multicollinearity = True
                severity_score += min(n_high_corr, 5)
        
        # 2. VIF-based
        if 'n_high_vif' in self.results:
            n_high_vif = self.results['n_high_vif']
            status = "✓ DETECTED" if n_high_vif > 0 else "✗ None"
            print(f"  2. High VIF (> 10): {status}")
            if n_high_vif > 0:
                print(f"     Found {n_high_vif} features with high VIF")
                has_multicollinearity = True
                severity_score += min(n_high_vif, 5)
        
        # 3. Condition Number-based
        if 'condition_number' in self.results:
            cond_num = self.results['condition_number']
            status = "✓ DETECTED" if cond_num > 30 else "✗ None"
            print(f"  3. Condition Number (> 30): {status}")
            print(f"     Condition Number: {cond_num:.2f}")
            if cond_num > 30:
                has_multicollinearity = True
                severity_score += min(int((cond_num - 30) / 10), 5)
        
        # 4. Eigenvalue-based
        if 'n_small_eigenvalues' in self.results:
            n_small_ev = self.results['n_small_eigenvalues']
            status = "✓ DETECTED" if n_small_ev > 0 else "✗ None"
            print(f"  4. Small Eigenvalues (< 0.01): {status}")
            if n_small_ev > 0:
                print(f"     Found {n_small_ev} small eigenvalues")
                has_multicollinearity = True
                severity_score += n_small_ev
        
        # Overall assessment
        print(f"\n{'─'*60}")
        if has_multicollinearity:
            if severity_score <= 3:
                severity = "MILD"
            elif severity_score <= 7:
                severity = "MODERATE"
            else:
                severity = "SEVERE"
            
            print(f"Overall Assessment: ⚠️  MULTICOLLINEARITY DETECTED ({severity})")
            print(f"Severity Score: {severity_score}/20")
        else:
            print(f"Overall Assessment: ✓ NO SIGNIFICANT MULTICOLLINEARITY")
        
        self.results['has_multicollinearity'] = has_multicollinearity
        self.results['severity_score'] = severity_score
        
        print(f"{'='*60}\n")
    
    def save_results(self):
        """Save analysis results to JSON file."""
        output_path = self.output_dir / f"{self.dataset_name}_collinearity_results.json"
        
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"Results saved to: {output_path}")
        
        return output_path
    
    def analyze(self):
        """Run complete collinearity analysis pipeline."""
        # Load data
        if not self.load_data():
            return False
        
        # Prepare numeric data
        if not self.prepare_numeric_data():
            return False
        
        # Run all analyses
        self.calculate_correlation_matrix()
        self.calculate_vif()
        self.calculate_condition_number()
        self.calculate_eigenvalues()
        
        # Generate summary
        self.generate_summary()
        
        # Save results
        self.save_results()
        
        return True


def analyze_all_datasets(dataset_dir="datasets", file_pattern="*.rds", 
                         output_dir="results/collinearity_analysis"):
    """
    Analyze all datasets matching the pattern.
    
    Args:
        dataset_dir: Directory containing datasets
        file_pattern: File pattern to match (e.g., "*.rds", "*.arff")
        output_dir: Directory to save results
    """
    # Find all matching files
    search_path = os.path.join(dataset_dir, file_pattern)
    dataset_files = glob.glob(search_path)
    
    print(f"\n{'='*60}")
    print(f"BATCH COLLINEARITY ANALYSIS")
    print(f"{'='*60}")
    print(f"Dataset directory: {dataset_dir}")
    print(f"File pattern: {file_pattern}")
    print(f"Found {len(dataset_files)} datasets")
    print(f"Output directory: {output_dir}")
    print(f"{'='*60}\n")
    
    if len(dataset_files) == 0:
        print(f"No datasets found matching pattern: {search_path}")
        return
    
    # Analyze each dataset
    all_results = []
    successful = 0
    failed = 0
    
    for i, dataset_file in enumerate(dataset_files, 1):
        print(f"\n[{i}/{len(dataset_files)}] Processing: {os.path.basename(dataset_file)}")
        
        analyzer = CollinearityAnalyzer(dataset_file, output_dir)
        
        try:
            success = analyzer.analyze()
            if success:
                successful += 1
                all_results.append(analyzer.results)
            else:
                failed += 1
        except Exception as e:
            print(f"✗ Unexpected error: {e}")
            failed += 1
    
    # Save combined results
    print(f"\n{'='*60}")
    print(f"BATCH ANALYSIS COMPLETE")
    print(f"{'='*60}")
    print(f"Successfully analyzed: {successful}")
    print(f"Failed: {failed}")
    print(f"Total: {len(dataset_files)}")
    
    # Create summary dataframe
    if all_results:
        summary_df = pd.DataFrame([
            {
                'dataset': r['dataset'],
                'n_features': r.get('n_numeric_features', 0),
                'n_high_correlations': r.get('n_high_correlations', 0),
                'n_high_vif': r.get('n_high_vif', 0),
                'condition_number': r.get('condition_number', np.nan),
                'n_small_eigenvalues': r.get('n_small_eigenvalues', 0),
                'has_multicollinearity': r.get('has_multicollinearity', False),
                'severity_score': r.get('severity_score', 0)
            }
            for r in all_results
        ])
        
        # Save summary
        summary_path = os.path.join(output_dir, 'collinearity_summary.csv')
        summary_df.to_csv(summary_path, index=False)
        print(f"\nSummary saved to: {summary_path}")
        
        # Print summary statistics
        print(f"\nSummary Statistics:")
        print(f"  Datasets with multicollinearity: {summary_df['has_multicollinearity'].sum()}")
        print(f"  Average severity score: {summary_df['severity_score'].mean():.2f}")
        print(f"  Average condition number: {summary_df['condition_number'].mean():.2f}")
        
        return summary_df
    
    return None


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Analyze datasets for multicollinearity using multiple methods'
    )
    parser.add_argument(
        '--dataset', '-d',
        help='Path to a single dataset file'
    )
    parser.add_argument(
        '--dir', 
        default='datasets',
        help='Directory containing datasets (default: datasets)'
    )
    parser.add_argument(
        '--pattern', '-p',
        default='*.rds',
        help='File pattern to match (default: *.rds)'
    )
    parser.add_argument(
        '--output', '-o',
        default='results/collinearity_analysis',
        help='Output directory (default: results/collinearity_analysis)'
    )
    parser.add_argument(
        '--arff',
        action='store_true',
        help='Process ARFF files instead of RDS files'
    )
    
    args = parser.parse_args()
    
    # Adjust pattern for ARFF files if specified
    if args.arff:
        args.pattern = '*.arff'
        if args.dir == 'datasets':
            args.dir = 'datasets/arff'
    
    # Single dataset or batch processing
    if args.dataset:
        # Analyze single dataset
        analyzer = CollinearityAnalyzer(args.dataset, args.output)
        analyzer.analyze()
    else:
        # Batch analysis
        analyze_all_datasets(args.dir, args.pattern, args.output)


if __name__ == "__main__":
    main()
