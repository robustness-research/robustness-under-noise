library(iml)
library(caret)

# Datasets and methods
args <- commandArgs(trailingOnly = TRUE)
datasets <- args

## ----colinearity-------------------------------------------------------------------------------------
# Function to check collinearity (dataset-level, not dataset-method level)
# By default it evaluates collinearity among the top_n numeric predictors
# selected by variance (top_n = 3).
check_collinearity <- function(data, corr_threshold = 0.8, vif_threshold = 5) {

  # Select predictor columns (exclude target 'class')
  if (!"class" %in% names(data)) {
    stop("Data must contain a 'class' column.")
  }

  predictors <- data[, setdiff(names(data), "class"), drop = FALSE]

  # Keep only numeric predictors for correlation / VIF checks
  numeric_preds <- predictors[, sapply(predictors, is.numeric), drop = FALSE]
  if (ncol(numeric_preds) == 0) {
    cat("\n--- Collinearity diagnostics ---\n")
    cat("No numeric predictors available for collinearity checks.\n")
    return(FALSE)
  }

  top_n = 3

  # Choose top_n numeric predictors by variance (dataset-level choice)
  vars <- apply(numeric_preds, 2, var, na.rm = TRUE)
  top_n <- min(top_n, length(vars))
  top_attributes <- names(sort(vars, decreasing = TRUE))[seq_len(top_n)]
  subset_data <- numeric_preds[, top_attributes, drop = FALSE]

  cat("\n--- Collinearity diagnostics ---\n")
  cat(sprintf("Selected predictors for collinearity check (top %d by variance): %s\n",

  top_n, paste(top_attributes, collapse = ", ")))

  # --- Pairwise correlation (Pearson)
  cor_matrix <- try(suppressWarnings(cor(subset_data, method = "pearson", use = "pairwise.complete.obs")), silent = TRUE)
  if (inherits(cor_matrix, "try-error") || ncol(subset_data) < 2) {
    max_abs_corr <- NA_real_
    high_corr <- FALSE
  } else {
    upper_vals <- abs(cor_matrix[upper.tri(cor_matrix)])
    max_abs_corr <- if (length(upper_vals)) max(upper_vals, na.rm = TRUE) else NA_real_
    high_corr <- any(upper_vals > corr_threshold, na.rm = TRUE)
  }

  # --- Variance Inflation Factor (VIF)
  vifs <- rep(NA_real_, ncol(subset_data))
  names(vifs) <- colnames(subset_data)

  if (ncol(subset_data) >= 2) {
    for (j in seq_len(ncol(subset_data))) {
      xj <- subset_data[[j]]
      others <- subset_data[, -j, drop = FALSE]
      fit <- try(lm(xj ~ ., data = as.data.frame(others)), silent = TRUE)
      if (inherits(fit, "try-error")) {
        vifs[j] <- Inf
      } 
      else {
        r2 <- try(summary(fit)$r.squared, silent = TRUE)
        if (inherits(r2, "try-error") || is.na(r2) || r2 >= 1) {
          vifs[j] <- Inf
        } else {
          vifs[j] <- 1 / (1 - r2)
        }
      }
    }
  }

  any_vif_high <- any(vifs > vif_threshold, na.rm = TRUE)
  max_vif <- if (any(is.infinite(vifs))) Inf else suppressWarnings(max(vifs, na.rm = TRUE))

  # --- Diagnostics output
  cat(sprintf("Correlation: thr=%.2f | any>thr=%s | max|r|=%s\n",
  corr_threshold,
  ifelse(isTRUE(high_corr), "TRUE", "FALSE"),
  ifelse(is.na(max_abs_corr), "NA", sprintf("%.3f", max_abs_corr))))
  cat(sprintf("VIF: thr=%.1f | any>thr=%s | max VIF=%s\n",
  vif_threshold,
  ifelse(isTRUE(any_vif_high), "TRUE", "FALSE"),
  ifelse(is.infinite(max_vif), "Inf", sprintf("%.3f", max_vif))))

  # Return boolean indicating whether correlation threshold was exceeded
  return(high_corr)
}

## ----main--------------------------------------------------------------------------------------------
set.seed(1)

# Main
#for (dataset in datasets) {
  # Load dataset
  dataset <- args
  filename = paste0("datasets/", dataset, ".rds")
  df <- readRDS(filename)

  # Check colinearity among top 3 attributes from FEAT_IMP
  #top_attributes <- head(feat_imp, 3)
  is_colinear <- check_collinearity(df)
  cat("\n=== Colinearity Check for Top 3 FEAT_IMP Attributes ===\n")
  
  # IF TRUE ELSE logic
  if (is_colinear) {
    print("Colinear")
  } else {
    print("Not colinear")
  }
  cat("\n=== Results recorded ===\n")
#}

