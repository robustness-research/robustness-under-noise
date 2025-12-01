library(iml)
library(lime)
library(caret)
library(knitr)

# Command-line arguments
args <- commandArgs(trailingOnly = TRUE)

if (length(args) < 2) {
  stop("Usage: Rscript Feature_Ranking_TopK.R <dataset> <method>")
}

## ----topk-spearman-----------------------------------------------------------------------------------
# Function to calculate Spearman correlation on top-k features only
calculate_topk_spearman <- function(rankings_list, k) {
  # Convert ranking to named rank vector
  as_rank_vector <- function(x) {
    if (is.null(x)) return(setNames(numeric(0), character(0)))
    
    # Character vector already ordered best->worst
    if (is.character(x)) {
      return(setNames(seq_along(x), x))
    }
    
    # Data frame already sorted by importance
    if (is.data.frame(x) && "Feature" %in% names(x)) {
      return(setNames(seq_len(nrow(x)), as.character(x$Feature)))
    }
    
    stop("Unsupported ranking format.")
  }
  
  # Filter out NULL rankings
  valid_rankings <- rankings_list[!sapply(rankings_list, is.null)]
  
  if (length(valid_rankings) < 2) {
    cat(sprintf("\n=== Top-%d Spearman ===\n", k))
    cat("Not enough valid rankings (need at least 2, got", 
        length(valid_rankings), ")\n")
    return(data.frame())
  }
  
  R <- lapply(valid_rankings, as_rank_vector)
  
  # Compute pairwise Spearman on top-k only
  pairs <- combn(names(R), 2, simplify = FALSE)
  results <- lapply(pairs, function(p) {
    r1 <- R[[p[1]]]
    r2 <- R[[p[2]]]
    
    # Get top-k features from each ranking
    top_k_1 <- head(names(sort(r1)), k)
    top_k_2 <- head(names(sort(r2)), k)
    
    # Use only features in top-k of BOTH rankings
    common_topk <- intersect(top_k_1, top_k_2)
    
    if (length(common_topk) < 2) {
      return(data.frame(
        Pair        = paste(p, collapse = " vs "),
        Spearman    = NA,
        Common_Features = length(common_topk),
        stringsAsFactors = FALSE
      ))
    }
    
    # Calculate Spearman on these common top-k features
    r1_topk <- r1[common_topk]
    r2_topk <- r2[common_topk]
    
    sp <- cor(r1_topk, r2_topk, method = "spearman", use = "complete.obs")
    
    data.frame(
      Pair        = paste(p, collapse = " vs "),
      Spearman    = unname(sp),
      Common_Features = length(common_topk),
      stringsAsFactors = FALSE
    )
  })
  
  spearman_tbl <- do.call(rbind, results)
  cat(sprintf("\n=== Top-%d Spearman (on common top-%d features only) ===\n", k, k))
  print(knitr::kable(spearman_tbl, format = "simple", row.names = FALSE))
  
  return(spearman_tbl)
}


## ----top1-agreement----------------------------------------------------------------------------------
# Function to count pairwise agreement on the top-1 feature
count_top1_agreement <- function(rankings_list) {
  # Get top-1 feature from each ranking
  get_top1 <- function(x) {
    if (is.null(x)) return(NA_character_)
    
    # Character vector
    if (is.character(x) && length(x) > 0) {
      return(x[1])
    }
    
    # Data frame
    if (is.data.frame(x) && "Feature" %in% names(x) && nrow(x) > 0) {
      return(as.character(x$Feature[1]))
    }
    
    return(NA_character_)
  }
  
  # Filter out NULL rankings
  valid_rankings <- rankings_list[!sapply(rankings_list, is.null)]
  
  if (length(valid_rankings) < 2) {
    cat("\n=== Top-1 Feature Agreement Count ===\n")
    cat("Not enough valid rankings (need at least 2, got", 
        length(valid_rankings), ")\n")
    return(data.frame())
  }
  
  # Get top-1 from each method
  top1_features <- sapply(valid_rankings, get_top1)
  
  cat("\n=== Top-1 Features ===\n")
  for (method in names(top1_features)) {
    cat(sprintf("%s: %s\n", method, top1_features[method]))
  }
  
  # Count pairwise agreements
  pairs <- combn(names(top1_features), 2, simplify = FALSE)
  results <- lapply(pairs, function(p) {
    feat1 <- top1_features[p[1]]
    feat2 <- top1_features[p[2]]
    
    # Check if both are valid and equal
    agree <- !is.na(feat1) && !is.na(feat2) && feat1 == feat2
    
    data.frame(
      Pair        = paste(p, collapse = " vs "),
      Feature_1   = feat1,
      Feature_2   = feat2,
      Agreement   = agree,
      stringsAsFactors = FALSE
    )
  })
  
  agreement_tbl <- do.call(rbind, results)
  
  cat("\n=== Top-1 Feature Agreement (pairwise) ===\n")
  print(knitr::kable(agreement_tbl, format = "simple", row.names = FALSE))
  
  # Summary
  total_pairs <- nrow(agreement_tbl)
  agreeing_pairs <- sum(agreement_tbl$Agreement, na.rm = TRUE)
  
  cat(sprintf("\nAgreeing pairs: %d / %d (%.1f%%)\n", 
              agreeing_pairs, total_pairs, 
              100 * agreeing_pairs / total_pairs))
  
  return(agreement_tbl)
}


## ----train-------------------------------------------------------------------------------------------
train_model <- function(data, method) {
  # Define methods that benefit from scaling
  scale_methods <- c("svmLinear", "svmRadial", "knn", "bayesglm", "multinom", "lvq", 
                     "mlpML", "simpls", "rda", "rbfDDA")
  
  # Set up training control and preprocessing parameters
  control <- trainControl(method = "none", number = 1)
  
  if (method %in% scale_methods) {
    cat(sprintf("Applying data scaling for method: %s\n", method))
    preprocess_params <- c("center", "scale")
  } else {
    cat(sprintf("No scaling applied for method: %s\n", method))
    preprocess_params <- NULL
  }
  
  # Train the model with appropriate preprocessing
  if (method == "multinom") {
    # Special handling for multinom with additional parameters
    trained_model <- caret::train(class ~ ., data = data, method = method, 
                                 trControl = control,
                                 preProcess = preprocess_params,
                                 tuneGrid = expand.grid(decay = c(0)), 
                                 MaxNWts = 10000)
  } else if (method == "knn") {
    # Special handling for knn with k parameter
    trained_model <- caret::train(class ~ ., data = data, method = method, 
                                 trControl = control,
                                 preProcess = preprocess_params,
                                 tuneGrid = expand.grid(k = 5:5))
  } else {
    # Standard training for other methods
    trained_model <- caret::train(class ~ ., data = data, method = method, 
                                 trControl = control,
                                 preProcess = preprocess_params)
  }
  
  cat("\n\n=== Trained model ===\n")
  cat(sprintf("Model class: %s\n", class(trained_model)[1]))
  cat("Is trained_model NULL?:", is.null(trained_model), "\n")
  return(trained_model)
}


## ----get-featimp-------------------------------------------------------------------------------------
get_featimp <- function(data, trained_model) {
  tryCatch({
    predictor <- Predictor$new(
      trained_model
    )
    feature_importance <- FeatureImp$new(predictor, loss = "ce")
    
    # Create ranking table
    featimp_results <- feature_importance$results[order(-feature_importance$results$importance), ]
    
    # Print table
    cat("\n=== IML Rankings ===\n")
    print(knitr::kable(head(featimp_results, 5), format = "simple", row.names = FALSE))  

    return(featimp_results$feature)
  }, error = function(e) {
    cat("\n=== IML Rankings ===\n")
    cat("Error in get_featimp:", conditionMessage(e), "\n")
    return(NULL)
  })
}


## ----get-lime----------------------------------------------------------------------------------------
get_lime <- function(data, trained_model) {
  tryCatch({
    # Separate features and target
    x_train <- data[, -which(names(data) == "class")]
    
    # Create explainer
    explainer <- lime(
      x_train, 
      trained_model
    )
    
    # Get predictions for explanation (use first 5 instances)
    explanation <- lime::explain(
      x = x_train[1:5, ], 
      explainer = explainer, 
      n_features = ncol(x_train),
      n_labels = 1  # Explain the top predicted class
    )
    
    # Aggregate feature weights
    lime_importance <- aggregate(
      abs(explanation$feature_weight), 
      by = list(explanation$feature), 
      FUN = mean
    )
    colnames(lime_importance) <- c("Feature", "Importance")
    lime_results <- lime_importance[order(-lime_importance$Importance), ]
    
    # Print table
    cat("\n=== LIME Rankings ===\n")
    print(knitr::kable(head(lime_results, 5), format = "simple", row.names = FALSE))  
    
    return(lime_results)
  }, error = function(e) {
    cat("\n=== LIME Rankings ===\n")
    cat("Error in get_lime:", conditionMessage(e), "\n")
    return(NULL)
  })
}


## ----get-shap----------------------------------------------------------------------------------------
get_shap <- function(data, trained_model, sample.size = 100) {
  tryCatch({
    # Separate features and target
    x_train <- data[, -which(names(data) == "class"), drop = FALSE]
    y <- data[["class"]]
    n_instances <- min(5, nrow(x_train))

    collected <- list()

    if (is.factor(y)) {
      # Classification: explain the top predicted class per instance
      for (i in seq_len(n_instances)) {
        top_class <- as.character(predict(trained_model, x_train[i, , drop = FALSE]))

        predictor <- Predictor$new(
          model = trained_model,
          data  = x_train,
          y     = y,
          type  = "prob",
          class = top_class
        )

        sh <- try(Shapley$new(
          predictor   = predictor,
          x.interest  = x_train[i, , drop = FALSE],
          sample.size = sample.size
        ), silent = TRUE)
        if (inherits(sh, "try-error")) next

        res <- sh$results
        if (!("feature" %in% names(res))) next
        res <- res[!is.na(res$feature) & res$feature != "", , drop = FALSE]
        contrib <- if ("phi" %in% names(res)) res$phi else if ("effect" %in% names(res)) res$effect else NA_real_
        df <- data.frame(Feature = as.character(res$feature),
                         Importance = abs(contrib),
                         stringsAsFactors = FALSE)
        collected[[length(collected) + 1]] <- df
      }
    } else {
      # Regression
      predictor <- Predictor$new(
        model = trained_model,
        data  = x_train,
        y     = y,
        type  = "response"
      )

      for (i in seq_len(n_instances)) {
        sh <- try(Shapley$new(
          predictor   = predictor,
          x.interest  = x_train[i, , drop = FALSE],
          sample.size = sample.size
        ), silent = TRUE)
        if (inherits(sh, "try-error")) next

        res <- sh$results
        if (!("feature" %in% names(res))) next
        res <- res[!is.na(res$feature) & res$feature != "", , drop = FALSE]
        contrib <- if ("phi" %in% names(res)) res$phi else if ("effect" %in% names(res)) res$effect else NA_real_
        df <- data.frame(Feature = as.character(res$feature),
                         Importance = abs(contrib),
                         stringsAsFactors = FALSE)
        collected[[length(collected) + 1]] <- df
      }
    }

    if (length(collected) == 0) return(data.frame(Feature = character(), Importance = numeric()))

    # Aggregate mean absolute SHAP value per feature and sort
    combined <- do.call(rbind, collected)
    shap_importance <- aggregate(Importance ~ Feature, data = combined, FUN = mean)
    shap_results <- shap_importance[order(-shap_importance$Importance), ]

    cat("\n=== SHAP (IML Shapley) Rankings ===\n")
    print(knitr::kable(head(shap_results, 5), format = "simple", row.names = FALSE))

    return(shap_results)
  }, error = function(e) {
    cat("\n=== SHAP (IML Shapley) Rankings ===\n")
    cat("Error in get_shap:", conditionMessage(e), "\n")
    return(NULL)
  })
}


## ----main--------------------------------------------------------------------------------------------
set.seed(1)

# Load dataset
dataset <- args[1]
method <- args[2]
filename <- paste0("datasets/", dataset, ".rds")
df <- readRDS(filename)

cat(sprintf("\n\n=== Processing: Dataset=%s, Method=%s ===\n", dataset, method))

# Train the model for the dataset
trained_model <- train_model(df, method)

# Get ranking of attributes by importance (all already sorted best->worst)
feat_imp      <- get_featimp(df, trained_model)  # character vector
lime_ranking  <- get_lime(df, trained_model)      # data.frame sorted by Importance
shap_ranking  <- get_shap(df, trained_model)      # data.frame sorted by Importance

rankings_list <- list(
  FEAT_IMP = feat_imp,
  LIME     = lime_ranking,
  SHAP     = shap_ranking
)

# 1. Measure Spearman on top-3 attributes
top3_spearman <- calculate_topk_spearman(rankings_list, k = 3)

# 2. Measure Spearman on top-1 attribute
top1_spearman <- calculate_topk_spearman(rankings_list, k = 1)

# 3. Count top-1 feature agreement
top1_agreement <- count_top1_agreement(rankings_list)

cat("\n=== All Results Recorded ===\n")
