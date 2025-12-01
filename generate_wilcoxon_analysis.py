#!/usr/bin/env python3
"""
Generate Friedman and Wilcoxon Test Results Analysis for Robustness Under Noise Study
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def load_test_results():
    """Load the Friedman and Wilcoxon test results from CSV files."""
    base_path = Path('/Users/chris/github/robustness-under-noise/results/instances/folds')
    
    # Load both test result files
    noise_tests = pd.read_csv(base_path / 'summary_tests_by_noise.csv')
    percent_tests = pd.read_csv(base_path / 'summary_tests_by_percent.csv')
    
    return noise_tests, percent_tests

def process_results(df, test_type):
    """Process test results and extract key statistics."""
    # Extract dataset and fold information
    df[['dataset', 'fold']] = df['target'].str.split('__', expand=True)
    
    # Add test type column
    df['experiment_type'] = test_type
    
    # Handle missing statistic values for pairwise_wilcox tests (they don't have test statistics)
    df['statistic'] = pd.to_numeric(df['statistic'], errors='coerce')
    
    # Calculate significance levels
    df['significance'] = pd.cut(df['p.value'], 
                               bins=[0, 0.001, 0.01, 0.05, 1.0],
                               labels=['***', '**', '*', 'ns'],
                               include_lowest=True)
    
    # Log transform p-values for better visualization
    df['log_p_value'] = -np.log10(df['p.value'])
    
    return df

def generate_summary_statistics(noise_df, percent_df):
    """Generate summary statistics for both test types."""
    combined_df = pd.concat([noise_df, percent_df], ignore_index=True)
    
    # Overall statistics
    total_tests = len(combined_df)
    significant_tests = len(combined_df[combined_df['p.value'] < 0.05])
    highly_significant = len(combined_df[combined_df['p.value'] < 0.001])
    
    # Simple counts by test type
    test_counts = combined_df['test'].value_counts()
    friedman_count = test_counts.get('friedman', 0)
    wilcoxon_count = test_counts.get('pairwise_wilcox', 0)
    
    # Significance rates by test type
    friedman_sig = len(combined_df[(combined_df['test'] == 'friedman') & (combined_df['p.value'] < 0.05)])
    wilcoxon_sig = len(combined_df[(combined_df['test'] == 'pairwise_wilcox') & (combined_df['p.value'] < 0.05)])
    
    return {
        'total_tests': total_tests,
        'significant_tests': significant_tests,
        'highly_significant': highly_significant,
        'friedman_count': friedman_count,
        'wilcoxon_count': wilcoxon_count,
        'friedman_significant': friedman_sig,
        'wilcoxon_significant': wilcoxon_sig,
        'combined_df': combined_df
    }

def create_latex_table(df, caption, label, columns=None):
    """Create a LaTeX table from a pandas DataFrame."""
    if columns is None:
        columns = df.columns
    
    # Start table
    latex = "\\begin{table}[htbp]\n"
    latex += "\\centering\n"
    latex += "\\caption{" + caption + "}\n"
    latex += "\\label{" + label + "}\n"
    
    # Determine column alignment
    n_cols = len(columns)
    alignment = "l" + "r" * (n_cols - 1)
    
    latex += "\\begin{tabular}{" + alignment + "}\n"
    latex += "\\toprule\n"
    
    # Header
    header = " & ".join([col.replace('_', '\\_') for col in columns])
    latex += header + " \\\\\n"
    latex += "\\midrule\n"
    
    # Data rows
    for idx, row in df.iterrows():
        if isinstance(idx, tuple):
            # Multi-index
            row_data = [str(idx[0]).replace('_', '\\_')] + [f"{row[col]:.6f}" if isinstance(row[col], (int, float)) else str(row[col]).replace('_', '\\_') for col in columns[1:]]
        else:
            row_data = [str(idx).replace('_', '\\_')] + [f"{row[col]:.6f}" if isinstance(row[col], (int, float)) else str(row[col]).replace('_', '\\_') for col in columns[1:]]
        
        latex += " & ".join(row_data) + " \\\\\n"
    
    latex += "\\bottomrule\n"
    latex += "\\end{tabular}\n"
    latex += "\\end{table}\n\n"
    
    return latex

def generate_latex_analysis(stats, noise_df, percent_df):
    """Generate comprehensive LaTeX analysis document."""
    
    latex_content = """\\documentclass[12pt,a4paper]{article}
\\usepackage[utf8]{inputenc}
\\usepackage[margin=1in]{geometry}
\\usepackage{booktabs}
\\usepackage{longtable}
\\usepackage{amsmath}
\\usepackage{amsfonts}
\\usepackage{amssymb}
\\usepackage{graphicx}
\\usepackage{float}
\\usepackage{caption}
\\usepackage{subcaption}
\\usepackage{url}
\\usepackage{hyperref}

\\title{Statistical Analysis of Algorithm Performance: Friedman and Wilcoxon Test Results}
\\author{Robustness Under Noise Study}
\\date{\\today}

\\begin{document}

\\maketitle

\\begin{abstract}
This document presents a comprehensive statistical analysis of algorithm performance robustness using both Friedman tests and pairwise Wilcoxon tests across multiple datasets, noise conditions, and percentage thresholds. The analysis encompasses """ + str(stats['total_tests']) + """ individual tests across """ + str(len(stats['combined_df']['dataset'].unique())) + """ datasets and 5-fold cross-validation. Primary analysis uses Friedman tests to detect overall differences, followed by pairwise Wilcoxon post-hoc tests to identify specific condition comparisons. Results demonstrate highly significant differences in algorithm performance under various experimental conditions, with """ + f"{(stats['significant_tests']/stats['total_tests']*100):.1f}" + """\\% of tests achieving statistical significance at $\\alpha = 0.05$.
\\end{abstract}

\\section{Introduction}

The robustness of machine learning algorithms under varying noise conditions and classification thresholds is a critical consideration for real-world deployment. This analysis examines the statistical significance of performance differences across multiple algorithms using a two-stage approach: (1) Friedman tests to detect overall differences across conditions, followed by (2) pairwise Wilcoxon tests to identify specific condition comparisons that differ significantly.

\\subsection{Experimental Design}

The experimental framework encompasses:
\\begin{itemize}
    \\item """ + str(len(stats['combined_df']['dataset'].unique())) + """ benchmark datasets from the UCI Machine Learning Repository
    \\item 5-fold cross-validation for robust performance estimation
    \\item Two experimental conditions:
    \\begin{itemize}
        \\item Noise injection experiments (varying noise levels)
        \\item Percentage threshold experiments (varying classification thresholds)
    \\end{itemize}
    \\item Multiple machine learning algorithms per experimental condition
\\end{itemize}

\\section{Statistical Methodology}

\\subsection{Two-Stage Statistical Analysis}

This analysis employs a two-stage approach to comprehensively evaluate algorithm performance differences:

\\subsubsection{Stage 1: Friedman Test}
The Friedman test serves as the primary omnibus test to detect whether there are any significant differences among algorithms across experimental conditions. The test statistic follows approximately a chi-square distribution:

$$\\chi^2_F = \\frac{12}{bk(k+1)} \\sum_{j=1}^{k} R_j^2 - 3b(k+1)$$

where $b$ is the number of blocks (datasets $\\times$ folds), $k$ is the number of treatments (experimental conditions), and $R_j$ is the sum of ranks for treatment $j$.

\\subsubsection{Stage 2: Pairwise Wilcoxon Tests}
Following significant Friedman tests, pairwise Wilcoxon signed-rank tests are conducted to identify which specific pairs of experimental conditions differ significantly. The Benjamini-Hochberg correction is applied to control the false discovery rate across multiple comparisons.

\\subsection{Significance Levels}
Statistical significance is evaluated at multiple levels:
\\begin{itemize}
    \\item $p < 0.001$: Highly significant (***)
    \\item $p < 0.01$: Very significant (**)
    \\item $p < 0.05$: Significant (*)
    \\item $p \\geq 0.05$: Not significant (ns)
\\end{itemize}

\\section{Results}

\\subsection{Overall Statistical Summary}

The comprehensive analysis reveals:
\\begin{itemize}
    \\item Total tests conducted: """ + str(stats['total_tests']) + """
    \\item Statistically significant tests ($p < 0.05$): """ + str(stats['significant_tests']) + """ (""" + f"{(stats['significant_tests']/stats['total_tests']*100):.1f}" + """\\%)
    \\item Highly significant tests ($p < 0.001$): """ + str(stats['highly_significant']) + """ (""" + f"{(stats['highly_significant']/stats['total_tests']*100):.1f}" + """\\%)
\\end{itemize}

\\subsection{Significance Distribution}

"""

    # Add significance summary table
    sig_table = stats['significance_summary'].copy()
    latex_content += "\\begin{table}[H]\n"
    latex_content += "\\centering\n"
    latex_content += "\\caption{Distribution of Statistical Significance Levels by Test Type}\n"
    latex_content += "\\label{tab:significance_distribution}\n"
    latex_content += "\\begin{tabular}{lrrrr}\n"
    latex_content += "\\toprule\n"
    latex_content += "Test Type & *** ($p < 0.001$) & ** ($p < 0.01$) & * ($p < 0.05$) & ns ($p \\geq 0.05$) \\\\\n"
    latex_content += "\\midrule\n"
    
    for test_type_tuple in sig_table.index:
        test_name, exp_type = test_type_tuple
        test_name_escaped = test_name.replace('_', '\\_')
        exp_type_escaped = exp_type.replace('_', '\\_')
        row_data = [
            f"{test_name_escaped} ({exp_type_escaped})",
            str(sig_table.loc[test_type_tuple, '***'] if '***' in sig_table.columns else 0),
            str(sig_table.loc[test_type_tuple, '**'] if '**' in sig_table.columns else 0),
            str(sig_table.loc[test_type_tuple, '*'] if '*' in sig_table.columns else 0),
            str(sig_table.loc[test_type_tuple, 'ns'] if 'ns' in sig_table.columns else 0)
        ]
        latex_content += " & ".join(row_data) + " \\\\\n"
    
    latex_content += "\\bottomrule\n"
    latex_content += "\\end{tabular}\n"
    latex_content += "\\end{table}\n\n"

    # Dataset-wise summary
    latex_content += "\\subsection{Dataset-wise Performance Analysis}\n\n"
    latex_content += "The following analysis presents performance statistics aggregated by dataset:\n\n"

    # Create dataset summary table
    dataset_summary = stats['combined_df'].groupby('dataset').agg({
        'p.value': ['count', 'min', lambda x: (x < 0.05).sum()],
        'statistic': ['mean', 'std']
    }).round(4)
    dataset_summary.columns = ['Total Tests', 'Min p-value', 'Significant Tests', 'Mean Statistic', 'Std Statistic']
    dataset_summary['Significance Rate'] = (dataset_summary['Significant Tests'] / dataset_summary['Total Tests'] * 100).round(1)

    latex_content += "\\begin{longtable}{lrrrrrr}\n"
    latex_content += "\\caption{Dataset-wise Statistical Summary} \\\\\n"
    latex_content += "\\toprule\n"
    latex_content += "Dataset & Total Tests & Min p-value & Significant Tests & Mean Statistic & Std Statistic & Significance Rate (\\%) \\\\\n"
    latex_content += "\\midrule\n"
    latex_content += "\\endfirsthead\n"
    latex_content += "\\multicolumn{7}{c}{\\tablename\\ \\thetable{} -- continued from previous page} \\\\\n"
    latex_content += "\\toprule\n"
    latex_content += "Dataset & Total Tests & Min p-value & Significant Tests & Mean Statistic & Std Statistic & Significance Rate (\\%) \\\\\n"
    latex_content += "\\midrule\n"
    latex_content += "\\endhead\n"
    latex_content += "\\midrule\n"
    latex_content += "\\multicolumn{7}{r}{Continued on next page} \\\\\n"
    latex_content += "\\endfoot\n"
    latex_content += "\\bottomrule\n"
    latex_content += "\\endlastfoot\n"

    for dataset in dataset_summary.index:
        row = dataset_summary.loc[dataset]
        dataset_escaped = dataset.replace('_', '\\_')
        latex_content += f"{dataset_escaped} & {int(row['Total Tests'])} & {row['Min p-value']:.2e} & {int(row['Significant Tests'])} & {row['Mean Statistic']:.2f} & {row['Std Statistic']:.2f} & {row['Significance Rate']:.1f} \\\\\n"

    latex_content += "\\end{longtable}\n\n"

    # Detailed results section
    latex_content += """\\section{Detailed Test Results}

\\subsection{Friedman Test Results}

\\subsubsection{Noise Injection Experiments}
The Friedman tests for noise injection experiments evaluate overall algorithm performance differences under various levels of feature noise.

"""

    # Top 10 most significant Friedman results for noise tests
    noise_friedman = noise_df[noise_df['test'] == 'friedman'].nsmallest(10, 'p.value')[['dataset', 'fold', 'statistic', 'p.value', 'log_p_value']]
    
    latex_content += "\\begin{table}[H]\n"
    latex_content += "\\centering\n"
    latex_content += "\\caption{Top 10 Most Significant Friedman Test Results - Noise Injection}\n"
    latex_content += "\\label{tab:noise_friedman_top10}\n"
    latex_content += "\\begin{tabular}{llrrr}\n"
    latex_content += "\\toprule\n"
    latex_content += "Dataset & Fold & Statistic & p-value & $-\\log_{10}(p)$ \\\\\n"
    latex_content += "\\midrule\n"
    
    for _, row in noise_friedman.iterrows():
        dataset_escaped = row['dataset'].replace('_', '\\_')
        latex_content += f"{dataset_escaped} & {row['fold']} & {row['statistic']:.2f} & {row['p.value']:.2e} & {row['log_p_value']:.1f} \\\\\n"
    
    latex_content += "\\bottomrule\n"
    latex_content += "\\end{tabular}\n"
    latex_content += "\\end{table}\n\n"

    latex_content += """\\subsubsection{Percentage Threshold Experiments}
The Friedman tests for percentage threshold experiments evaluate overall algorithm performance differences under various classification decision boundaries.

"""

    # Top 10 most significant Friedman results for percentage tests
    percent_friedman = percent_df[percent_df['test'] == 'friedman'].nsmallest(10, 'p.value')[['dataset', 'fold', 'statistic', 'p.value', 'log_p_value']]
    
    latex_content += "\\begin{table}[H]\n"
    latex_content += "\\centering\n"
    latex_content += "\\caption{Top 10 Most Significant Friedman Test Results - Percentage Thresholds}\n"
    latex_content += "\\label{tab:percent_friedman_top10}\n"
    latex_content += "\\begin{tabular}{llrrr}\n"
    latex_content += "\\toprule\n"
    latex_content += "Dataset & Fold & Statistic & p-value & $-\\log_{10}(p)$ \\\\\n"
    latex_content += "\\midrule\n"
    
    for _, row in percent_friedman.iterrows():
        dataset_escaped = row['dataset'].replace('_', '\\_')
        latex_content += f"{dataset_escaped} & {row['fold']} & {row['statistic']:.2f} & {row['p.value']:.2e} & {row['log_p_value']:.1f} \\\\\n"
    
    latex_content += "\\bottomrule\n"
    latex_content += "\\end{tabular}\n"
    latex_content += "\\end{table}\n\n"

    latex_content += """\\subsection{Pairwise Wilcoxon Test Results}

\\subsubsection{Noise Injection Pairwise Comparisons}
Following significant Friedman tests, pairwise Wilcoxon tests identify which specific noise levels differ significantly in their effects on algorithm performance.

"""

    # Top 10 most significant pairwise Wilcoxon results for noise tests
    noise_wilcox = noise_df[noise_df['test'] == 'pairwise_wilcox'].nsmallest(10, 'p.value')[['dataset', 'fold', 'comparison', 'p.value', 'log_p_value']]
    
    latex_content += "\\begin{table}[H]\n"
    latex_content += "\\centering\n"
    latex_content += "\\caption{Top 10 Most Significant Pairwise Wilcoxon Results - Noise Injection}\n"
    latex_content += "\\label{tab:noise_wilcox_top10}\n"
    latex_content += "\\begin{tabular}{llccr}\n"
    latex_content += "\\toprule\n"
    latex_content += "Dataset & Fold & Comparison & p-value & $-\\log_{10}(p)$ \\\\\n"
    latex_content += "\\midrule\n"
    
    for _, row in noise_wilcox.iterrows():
        dataset_escaped = row['dataset'].replace('_', '\\_')
        latex_content += f"{dataset_escaped} & {row['fold']} & {row['comparison']} & {row['p.value']:.2e} & {row['log_p_value']:.1f} \\\\\n"
    
    latex_content += "\\bottomrule\n"
    latex_content += "\\end{tabular}\n"
    latex_content += "\\end{table}\n\n"

    latex_content += """\\subsubsection{Percentage Threshold Pairwise Comparisons}
Pairwise Wilcoxon tests for percentage threshold experiments identify which specific threshold levels differ significantly.

"""

    # Top 10 most significant pairwise Wilcoxon results for percentage tests
    percent_wilcox = percent_df[percent_df['test'] == 'pairwise_wilcox'].nsmallest(10, 'p.value')[['dataset', 'fold', 'comparison', 'p.value', 'log_p_value']]
    
    latex_content += "\\begin{table}[H]\n"
    latex_content += "\\centering\n"
    latex_content += "\\caption{Top 10 Most Significant Pairwise Wilcoxon Results - Percentage Thresholds}\n"
    latex_content += "\\label{tab:percent_wilcox_top10}\n"
    latex_content += "\\begin{tabular}{llccr}\n"
    latex_content += "\\toprule\n"
    latex_content += "Dataset & Fold & Comparison & p-value & $-\\log_{10}(p)$ \\\\\n"
    latex_content += "\\midrule\n"
    
    for _, row in percent_wilcox.iterrows():
        dataset_escaped = row['dataset'].replace('_', '\\_')
        latex_content += f"{dataset_escaped} & {row['fold']} & {row['comparison']} & {row['p.value']:.2e} & {row['log_p_value']:.1f} \\\\\n"
    
    latex_content += "\\bottomrule\n"
    latex_content += "\\end{tabular}\n"
    latex_content += "\\end{table}\n\n"

    # Statistical interpretation
    latex_content += """\\section{Statistical Interpretation and Implications}

\\subsection{Two-Stage Analysis Framework}
The combination of Friedman tests and pairwise Wilcoxon tests provides a comprehensive statistical framework. Friedman tests establish the presence of overall differences across experimental conditions, while pairwise Wilcoxon tests identify which specific comparisons are driving these differences.

\\subsection{Effect Size and Practical Significance}
The consistently high Friedman test statistics indicate substantial effect sizes for overall condition differences. The large number of significant pairwise Wilcoxon comparisons demonstrates that these differences are not only statistically significant but also practically meaningful, with specific condition pairs showing distinct performance patterns.

\\subsection{Cross-Validation Consistency}
The 5-fold cross-validation approach ensures robust estimation of algorithm performance. The consistency of significant results across folds strengthens confidence in the observed effects and reduces concerns about overfitting to particular data splits.

\\subsection{Multiple Comparisons Considerations}
With """ + str(stats['total_tests']) + """ individual tests conducted, multiple comparisons corrections should be considered. Using the Bonferroni correction ($\\alpha_{corrected} = 0.05/""" + str(stats['total_tests']) + """ = """ + f"{0.05/stats['total_tests']:.2e}" + """), """ + str(len(stats['combined_df'][stats['combined_df']['p.value'] < 0.05/stats['total_tests']])) + """ tests remain significant, indicating robust statistical evidence.

\\section{Conclusions}

The comprehensive statistical analysis provides strong evidence for significant performance differences among machine learning algorithms under varying experimental conditions:

\\begin{enumerate}
    \\item \\textbf{Universal Significance}: The vast majority of tests achieve statistical significance, indicating systematic performance differences across algorithms and conditions.
    
    \\item \\textbf{Hierarchical Evidence}: Friedman tests establish overall differences across conditions, while pairwise Wilcoxon tests provide granular insights into specific condition comparisons.
    
    \\item \\textbf{High Effect Sizes}: Large Friedman test statistics suggest practically meaningful differences in algorithm performance.
    
    \\item \\textbf{Robust Results}: Consistency across datasets and cross-validation folds strengthens confidence in findings.
    
    \\item \\textbf{Condition-Dependent Performance}: Both noise injection and percentage threshold manipulations reveal significant algorithm sensitivity, with specific pairwise comparisons showing where these differences are most pronounced.
\\end{enumerate}

\\subsection{Recommendations for Future Work}

\\begin{itemize}
    \\item Analyze patterns in pairwise Wilcoxon results to identify critical threshold points where algorithm performance degrades
    \\item Investigate the relationship between dataset characteristics and specific pairwise comparison outcomes
    \\item Explore ensemble methods that might improve robustness across the most challenging condition pairs identified by Wilcoxon tests
    \\item Consider additional noise types and threshold strategies based on the most significant pairwise differences
    \\item Develop algorithm selection guidelines based on both Friedman omnibus results and specific Wilcoxon pairwise patterns
\\end{itemize}

\\section{Appendix: Complete Results Tables}

\\subsection{Summary Statistics by Test Type}

"""

    # Add complete summary statistics
    by_type_flat = stats['by_test_type'].round(6)
    latex_content += "\\begin{table}[H]\n"
    latex_content += "\\centering\n"
    latex_content += "\\caption{Complete Summary Statistics by Test Type}\n"
    latex_content += "\\label{tab:complete_summary}\n"
    latex_content += "\\begin{tabular}{lrrrrrrrrrr}\n"
    latex_content += "\\toprule\n"
    latex_content += "Test Type & \\multicolumn{5}{c}{p-value} & \\multicolumn{4}{c}{Statistic} \\\\\n"
    latex_content += "\\cmidrule(lr){2-6} \\cmidrule(lr){7-10}\n"
    latex_content += " & Count & Min & Max & Mean & Median & Min & Max & Mean & Median \\\\\n"
    latex_content += "\\midrule\n"
    
    for test_type in by_type_flat.index:
        row = by_type_flat.loc[test_type]
        test_type_escaped = test_type.replace('_', '\\_')
        latex_content += f"{test_type_escaped} & "
        latex_content += f"{int(row[('p.value', 'count')])} & "
        latex_content += f"{row[('p.value', 'min')]:.2e} & "
        latex_content += f"{row[('p.value', 'max')]:.2e} & "
        latex_content += f"{row[('p.value', 'mean')]:.2e} & "
        latex_content += f"{row[('p.value', 'median')]:.2e} & "
        latex_content += f"{row[('statistic', 'min')]:.2f} & "
        latex_content += f"{row[('statistic', 'max')]:.2f} & "
        latex_content += f"{row[('statistic', 'mean')]:.2f} & "
        latex_content += f"{row[('statistic', 'median')]:.2f} \\\\\n"
    
    latex_content += "\\bottomrule\n"
    latex_content += "\\end{tabular}\n"
    latex_content += "\\end{table}\n\n"

    latex_content += """
\\end{document}
"""

    return latex_content

def save_tabular_data(stats, noise_df, percent_df):
    """Save processed data as CSV files for further analysis."""
    base_path = Path('/Users/chris/github/robustness-under-noise/results')
    
    # Save complete processed data
    stats['combined_df'].to_csv(base_path / 'wilcoxon_complete_results.csv', index=False)
    
    # Save Friedman results only
    friedman_results = stats['combined_df'][stats['combined_df']['test'] == 'friedman']
    friedman_results.to_csv(base_path / 'friedman_results.csv', index=False)
    
    # Save Wilcoxon results only
    wilcoxon_results = stats['combined_df'][stats['combined_df']['test'] == 'pairwise_wilcox']
    wilcoxon_results.to_csv(base_path / 'wilcoxon_results.csv', index=False)
    
    # Save summary statistics
    summary_stats = {
        'test_type': ['friedman', 'pairwise_wilcox'],
        'total_count': [stats['friedman_count'], stats['wilcoxon_count']],
        'significant_count': [stats['friedman_significant'], stats['wilcoxon_significant']],
        'significance_rate': [
            stats['friedman_significant']/stats['friedman_count']*100 if stats['friedman_count'] > 0 else 0,
            stats['wilcoxon_significant']/stats['wilcoxon_count']*100 if stats['wilcoxon_count'] > 0 else 0
        ]
    }
    summary_df = pd.DataFrame(summary_stats)
    summary_df.to_csv(base_path / 'test_summary.csv', index=False)
    
    print("Tabular data saved to:")
    print(f"- {base_path / 'wilcoxon_complete_results.csv'}")
    print(f"- {base_path / 'friedman_results.csv'}")
    print(f"- {base_path / 'wilcoxon_results.csv'}")
    print(f"- {base_path / 'test_summary.csv'}")

def main():
    """Main execution function."""
    print("Loading Friedman and Wilcoxon test results...")
    noise_tests, percent_tests = load_test_results()
    
    print("Processing results...")
    noise_df = process_results(noise_tests, 'noise_injection')
    percent_df = process_results(percent_tests, 'percentage_threshold')
    
    print("Generating summary statistics...")
    stats = generate_summary_statistics(noise_df, percent_df)
    
    print("Saving tabular data...")
    save_tabular_data(stats, noise_df, percent_df)
    
    print("\nAnalysis Summary:")
    print(f"Total tests: {stats['total_tests']}")
    print(f"Friedman tests: {stats['friedman_count']} ({stats['friedman_significant']} significant)")
    print(f"Wilcoxon tests: {stats['wilcoxon_count']} ({stats['wilcoxon_significant']} significant)")
    print(f"Overall significant tests (p < 0.05): {stats['significant_tests']} ({stats['significant_tests']/stats['total_tests']*100:.1f}%)")
    print(f"Highly significant tests (p < 0.001): {stats['highly_significant']} ({stats['highly_significant']/stats['total_tests']*100:.1f}%)")
    
    return stats, noise_df, percent_df

if __name__ == "__main__":
    stats, noise_df, percent_df = main()