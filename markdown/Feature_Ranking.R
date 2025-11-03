library(iml)
library(lime)
library(caret)

# Datasets and methods
args <- commandArgs(trailingOnly = TRUE)
datasets <- args[1]  # First argument is the dataset
methods <- args[2]   # Second argument is the method

if (length(args) < 2) {
  stop("Usage: Rscript Ranking.R <dataset> <method>")
}

## ----agreement---------------------------------------------------------------------------------------
# Function to calculate agreement statistic (pairwise Kendall, Spearman, IOU@k)
calculate_agreement <- function(rankings_list, k = 3) {
  # Convert any ranking object to a named rank vector: names=features, values=rank (1=best)
  as_rank_vector <- function(x) {
    if (is.null(x)) return(setNames(numeric(0), character(0)))
    
    # Character vector already ordered best->worst (from get_featimp)
    if (is.character(x)) {
      return(setNames(seq_along(x), x))
    }
    
    # Data frame already sorted by importance (from get_local, get_lime, get_shap)
    if (is.data.frame(x) && "Feature" %in% names(x)) {
      # Assume already sorted descending by importance
      return(setNames(seq_len(nrow(x)), as.character(x$Feature)))
    }
    
    stop("Unsupported ranking format for agreement.")
  }
  
  # Filter out NULL rankings and build named rank vectors
  valid_rankings <- rankings_list[!sapply(rankings_list, is.null)]
  
  if (length(valid_rankings) < 2) {
    cat("\n=== Agreement (pairwise) ===\n")
    cat("Not enough valid rankings to calculate agreement (need at least 2, got", 
        length(valid_rankings), ")\n")
    return(data.frame())
  }
  
  R <- lapply(valid_rankings, as_rank_vector)
  
  # Compute pairwise stats
  pairs <- combn(names(R), 2, simplify = FALSE)
  results <- lapply(pairs, function(p) {
    r1 <- R[[p[1]]]
    r2 <- R[[p[2]]]
    
    # Use only features present in both rankings
    common_feats <- intersect(names(r1), names(r2))
    
    if (length(common_feats) < 2) {
      return(data.frame(
        Pair        = paste(p, collapse = " vs "),
        Kendall_Tau = NA,
        Spearman    = NA,
        IOU_TopK    = NA,
        stringsAsFactors = FALSE
      ))
    }
    
    r1_aligned <- r1[common_feats]
    r2_aligned <- r2[common_feats]
    
    # Kendall's Tau
    kt <- cor(r1_aligned, r2_aligned, method = "kendall", use = "complete.obs")
    
    # Spearman's Rank Correlation
    sp <- cor(r1_aligned, r2_aligned, method = "spearman", use = "complete.obs")
    
    # IOU for top-k
    top_k_1 <- head(names(sort(r1)), k)
    top_k_2 <- head(names(sort(r2)), k)
    iou <- length(intersect(top_k_1, top_k_2)) / length(union(top_k_1, top_k_2))
    
    data.frame(
      Pair        = paste(p, collapse = " vs "),
      Kendall_Tau = unname(kt),
      Spearman    = unname(sp),
      IOU_TopK    = iou,
      stringsAsFactors = FALSE
    )
  })
  
  agreement_tbl <- do.call(rbind, results)
  cat("\n=== Agreement (pairwise) ===\n")
  print(knitr::kable(agreement_tbl, format = "simple", row.names = FALSE))
  
  return(agreement_tbl)
}


## ----train-------------------------------------------------------------------------------------------
train_model <- function(data, method) {
  # Train the model
  trained_model <- caret::train(class ~ ., data = data, method = method)
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


## ----get-local---------------------------------------------------------------------------------------
get_local <- function(data, trained_model) {
  tryCatch({
    # Separate features and target
    x_train <- data[, -which(names(data) == "class"), drop = FALSE]
    y <- data[["class"]]
    n_instances <- min(5, nrow(x_train))
    k_features <- ncol(x_train)

    # Aggregate absolute local effects across instances
    collected <- list()

    if (is.factor(y)) {
      # Classification: explain the top predicted class per instance
      for (i in seq_len(n_instances)) {
        # Top predicted class for this instance
        top_class <- as.character(predict(trained_model, x_train[i, , drop = FALSE]))

        predictor <- Predictor$new(
          model = trained_model,
          data  = x_train,
          y     = y,
          type  = "prob",
          class = top_class
        )

        lm <- try(LocalModel$new(
          predictor   = predictor,
          x.interest  = x_train[i, , drop = FALSE],
          k           = k_features
        ), silent = TRUE)
        if (inherits(lm, "try-error")) next

        res <- lm$results
        if (!all(c("feature") %in% names(res))) next
        contrib <- if ("effect" %in% names(res)) res$effect else if ("phi" %in% names(res)) res$phi else NA_real_
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
        lm <- try(LocalModel$new(
          predictor   = predictor,
          x.interest  = x_train[i, , drop = FALSE],
          k           = k_features
        ), silent = TRUE)
        if (inherits(lm, "try-error")) next

        res <- lm$results
        if (!all(c("feature") %in% names(res))) next
        contrib <- if ("effect" %in% names(res)) res$effect else if ("phi" %in% names(res)) res$phi else NA_real_
        df <- data.frame(Feature = as.character(res$feature),
                         Importance = abs(contrib),
                         stringsAsFactors = FALSE)
        collected[[length(collected) + 1]] <- df
      }
    }

    if (length(collected) == 0) return(data.frame(Feature = character(), Importance = numeric()))

    # Aggregate mean absolute effect per feature and sort
    combined <- do.call(rbind, collected)
    local_importance <- aggregate(Importance ~ Feature, data = combined, FUN = mean)
    local_results <- local_importance[order(-local_importance$Importance), ]

    cat("\n=== LocalModel (IML) Rankings ===\n")
    print(knitr::kable(head(local_results, 5), format = "simple", row.names = FALSE))

    return(local_results)
  }, error = function(e) {
    cat("\n=== LocalModel (IML) Rankings ===\n")
    cat("Error in get_local:", conditionMessage(e), "\n")
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

# Test values (iris dataset with random forest)
#dataset <- "iris"
#method <- "rf"

# Main
#for (dataset in datasets) {
  # Load dataset
  dataset <- args[1]  # Use directly, don't loop
  method <- args[2]
  filename = paste0("datasets/", dataset, ".rds")
  df <- readRDS(filename)

  #for (method in methods) {
    cat(sprintf("\n\n=== Processing: Dataset=%s, Method=%s ===\n", dataset, method))
    
    # Train the model for the dataset
    trained_model <- train_model(df, method)

    # Get ranking of attributes by importance (all already sorted best->worst)
    feat_imp      <- get_featimp(df, trained_model)  # character vector
    local_ranking <- get_local(df, trained_model)     # data.frame sorted by Importance
    lime_ranking  <- get_lime(df, trained_model)      # data.frame sorted by Importance
    shap_ranking  <- get_shap(df, trained_model)      # data.frame sorted by Importance

    rankings_list <- list(
      FEAT_IMP   = feat_imp,
      LOCAL = local_ranking,
      LIME  = lime_ranking,
      SHAP  = shap_ranking
    )
    
    # Calculate agreement statistic (pairwise)
    agreement_stat <- calculate_agreement(rankings_list, k = 3)
    cat("\n=== Results recorded ===\n")
  #}
#}

