# Model Agreement Summary

Generated from: `topk_agreement_results.txt`  
Date: November 25, 2025

## Overview

This document presents pairwise agreement metrics for different machine learning models across multiple datasets. For each model, we compute:

1. **Top-3 Spearman Correlation**: Average Spearman rank correlation for the top-3 most important features across feature importance methods (FEAT_IMP, LIME, SHAP)
2. **Top-1 Feature Agreement**: Proportion of datasets where the top-1 feature matches between different feature importance methods

---

## Top-3 Spearman Correlation (Average)

Values range from -1 to 1, where 1 indicates perfect agreement in ranking.

| Model      | FEAT_IMP vs LIME | FEAT_IMP vs SHAP | LIME vs SHAP |
|------------|------------------|------------------|--------------|
| JRip       | 0.550            | 0.705            | 0.587        |
| PART       | 0.500            | 0.250            | 0.306        |
| bayesglm   | 0.765            | 0.447            | 0.528        |
| ctree      | 0.262            | 0.684            | 0.432        |
| fda        | 1.000            | 0.857            | 1.000        |
| gbm        | 0.667            | 0.400            | 0.596        |
| gcvEarth   | 0.227            | 0.479            | 0.261        |
| knn        | 0.038            | 0.500            | 0.306        |
| lvq        | N/A              | N/A              | N/A          |
| mlpML      | 0.553            | 0.531            | 0.553        |
| multinom   | 0.500            | 0.375            | 0.432        |
| rbfDDA     | 0.545            | 0.583            | 0.577        |
| rda        | 0.625            | 0.556            | 0.500        |
| rf         | 0.750            | 0.361            | 0.646        |
| rfRules    | N/A              | N/A              | N/A          |
| rpart      | N/A              | 0.500            | N/A          |
| simpls     | 0.423            | 0.786            | 0.316        |
| svmLinear  | N/A              | N/A              | N/A          |
| svmRadial  | N/A              | N/A              | N/A          |

**Average across all models:**
- FEAT_IMP vs LIME: 0.529
- FEAT_IMP vs SHAP: 0.534
- LIME vs SHAP: 0.503

---

## Top-1 Feature Agreement (Proportion)

Values range from 0 to 1, where 1 indicates the top feature always matches.

| Model      | FEAT_IMP vs LIME | FEAT_IMP vs SHAP | LIME vs SHAP |
|------------|------------------|------------------|--------------|
| JRip       | 0.680            | 0.731            | 0.680        |
| PART       | 0.538            | 0.577            | 0.500        |
| bayesglm   | 0.727            | 0.538            | 0.591        |
| ctree      | 0.600            | 0.692            | 0.760        |
| fda        | 0.909            | 0.909            | 1.000        |
| gbm        | 0.731            | 0.577            | 0.731        |
| gcvEarth   | 0.520            | 0.600            | 0.615        |
| knn        | 0.346            | 0.385            | 0.385        |
| lvq        | N/A              | 0.000            | N/A          |
| mlpML      | 0.577            | 0.577            | 0.500        |
| multinom   | 0.423            | 0.500            | 0.423        |
| rbfDDA     | 0.154            | 0.346            | 0.346        |
| rda        | 0.455            | 0.538            | 0.273        |
| rf         | 0.538            | 0.462            | 0.654        |
| rfRules    | N/A              | 0.000            | N/A          |
| rpart      | N/A              | 0.538            | N/A          |
| simpls     | 0.346            | 0.423            | 0.423        |
| svmLinear  | N/A              | 0.000            | N/A          |
| svmRadial  | N/A              | 0.000            | N/A          |

**Average across all models:**
- FEAT_IMP vs LIME: 0.539 (53.9%)
- FEAT_IMP vs SHAP: 0.442 (44.2%)
- LIME vs SHAP: 0.563 (56.3%)

---

## Data Files

- Top-3 Spearman CSV: `top3_spearman_by_model.csv`
- Top-1 Agreement CSV: `top1_agreement_by_model.csv`
- Source data: `topk_agreement_results.txt`
