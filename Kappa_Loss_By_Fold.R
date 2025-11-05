#!/usr/bin/env Rscript
# Kappa loss per fold (by noise level and by percent altered)
# Converted from Kappa_Loss_By_Fold.Rmd for terminal execution

library(dplyr)
library(purrr)
library(tidyr)
library(broom)

# ============================================================================
# Setup and helper functions
# ============================================================================

# Helper to coerce names like "noise_10" -> 0.10 and percent names like "25" -> 0.25
parse_noise <- function(nm) {
  # nm expected like "noise_0" or "noise_10" or "0"
  if (is.numeric(nm)) return(as.numeric(nm))
  if (grepl("^noise_", nm)) {
    as.numeric(sub("^noise_", "", nm))/100
  } else {
    as.numeric(nm)
  }
}

parse_percent <- function(pct_nm) {
  # pct_nm likely strings like "0", "25", "50", etc. If they are fractional already, return as-is
  if (is.numeric(pct_nm)) return(as.numeric(pct_nm))
  val <- suppressWarnings(as.numeric(pct_nm))
  if (is.na(val)) return(NA_real_)
  # If the value looks like an integer percentage (0..100) convert to fraction if >1
  if (val > 1) return(val/100)
  val
}

safe_kappa <- function(x) {
  # try to extract Kappa from a confusionMatrix-like object
  if (inherits(x, "confusionMatrix")) {
    as.numeric(x$overall["Kappa"])
  } else if (is.list(x) && !is.null(x$overall) && !is.null(x$overall["Kappa"])) {
    as.numeric(x$overall["Kappa"])
  } else if (is.numeric(x)) {
    # in case the stored value is already numeric
    as.numeric(x)
  } else {
    NA_real_
  }
}

instances_file_candidates <- c(
  "results/instances/instances_list.rds",
  "results/instances/instancesCM_list.rds",
  "results/instances/instancesCM.rds"
)
instances_file <- instances_file_candidates[file.exists(instances_file_candidates)][1]
if (is.na(instances_file) || is.null(instances_file)) {
  stop("No instances file found in results/instances; expected instances_list.rds or instancesCM_list.rds")
}

message("Using instances file: ", instances_file)

instances_all <- readRDS(instances_file)

# ============================================================================
# Extract Kappa values into a tidy table
# ============================================================================

message("Extracting Kappa values from nested structure...")

rows <- list()
ri <- 1

for (dataset in names(instances_all)) {
  folds <- instances_all[[dataset]]
  for (fold_name in names(folds)) {
    methods <- folds[[fold_name]]
    for (method in names(methods)) {
      noise_list <- methods[[method]]
      for (noise_name in names(noise_list)) {
        percent_list <- noise_list[[noise_name]]
        for (pct_name in names(percent_list)) {
          value <- percent_list[[pct_name]]
          kappa_val <- safe_kappa(value)

          rows[[ri]] <- data.frame(
            dataset = dataset,
            fold = fold_name,
            method = method,
            noise_name = noise_name,
            percent_name = pct_name,
            noise = parse_noise(noise_name),
            percent = parse_percent(pct_name),
            kappa = kappa_val,
            stringsAsFactors = FALSE
          )
          ri <- ri + 1
        }
      }
    }
  }
}

df_kappa <- bind_rows(rows)

if (nrow(df_kappa) == 0) {
  stop("No kappa values could be extracted from the file; check structure")
}

message("Extracted ", nrow(df_kappa), " kappa values")
message("First few rows:")
print(head(df_kappa))

# ============================================================================
# Compute baseline Kappa and Kappa loss
# ============================================================================

message("\nComputing baseline Kappa and Kappa loss...")

# compute baseline per dataset/fold/method
baseline <- df_kappa %>%
  group_by(dataset, fold, method) %>%
  summarise(
    baseline_kappa = {
      x <- kappa[which(noise == 0)]
      if (length(x) == 0 || all(is.na(x))) x <- max(kappa, na.rm = TRUE)
      # if still NA (all missing), set NA
      if (!is.finite(x)) NA_real_ else mean(x, na.rm = TRUE)
    },
    .groups = "drop"
  )

df_kappa2 <- df_kappa %>% left_join(baseline, by = c("dataset","fold","method")) %>%
  mutate(kappa_loss = baseline_kappa - kappa)

# Quick sanity check: noise==0 rows should have loss approx 0
message("Sanity check (noise==0 should have ~0 loss):")
print(df_kappa2 %>%
  filter(noise == 0) %>%
  select(dataset, fold, method, percent, kappa, baseline_kappa, kappa_loss) %>%
  head())

# ============================================================================
# Aggregate: 1) Kappa loss by noise level
# ============================================================================

message("\nAggregating kappa loss by noise level...")

agg_by_noise <- df_kappa2 %>%
  group_by(dataset, fold, noise) %>%
  summarise(mean_kappa_loss = mean(kappa_loss, na.rm = TRUE),
            sd_kappa_loss = sd(kappa_loss, na.rm = TRUE),
            n = sum(!is.na(kappa_loss)),
            .groups = "drop") %>%
  arrange(dataset, fold, noise)

message("Aggregated by noise (first few rows):")
print(head(agg_by_noise))

# Save
dir.create("results/instances/aggregated", recursive = TRUE, showWarnings = FALSE)
saveRDS(agg_by_noise, file = "results/instances/aggregated/agg_kappa_loss_by_noise_per_fold.rds")
write.csv(agg_by_noise, file = "results/instances/aggregated/agg_kappa_loss_by_noise_per_fold.csv", row.names = FALSE)
message("Saved: results/instances/aggregated/agg_kappa_loss_by_noise_per_fold.rds and .csv")

# ============================================================================
# Aggregate: 2) Kappa loss by percentage of altered instances
# ============================================================================

message("\nAggregating kappa loss by percentage...")

agg_by_percent <- df_kappa2 %>%
  group_by(dataset, fold, percent) %>%
  summarise(mean_kappa_loss = mean(kappa_loss, na.rm = TRUE),
            sd_kappa_loss = sd(kappa_loss, na.rm = TRUE),
            n = sum(!is.na(kappa_loss)),
            .groups = "drop") %>%
  arrange(dataset, fold, percent)

message("Aggregated by percent (first few rows):")
print(head(agg_by_percent))

saveRDS(agg_by_percent, file = "results/instances/aggregated/agg_kappa_loss_by_percent_per_fold.rds")
write.csv(agg_by_percent, file = "results/instances/aggregated/agg_kappa_loss_by_percent_per_fold.csv", row.names = FALSE)
message("Saved: results/instances/aggregated/agg_kappa_loss_by_percent_per_fold.rds and .csv")

# ============================================================================
# Statistical tests (ANOVA / Friedman / Wilcoxon)
# ============================================================================

message("\nPerforming statistical tests...")

dir.create("results/instances/aggregated/tests", recursive = TRUE, showWarnings = FALSE)

tests_by_noise <- list()
tests_by_percent <- list()

# Prepare per-method per-noise means (average across percent)
per_method_noise <- df_kappa2 %>%
  group_by(dataset, fold, method, noise) %>%
  summarise(mean_kappa_loss = mean(kappa_loss, na.rm = TRUE), .groups = "drop")

datasets_folds <- per_method_noise %>% select(dataset, fold) %>% distinct()

message("Running tests by noise level across ", nrow(datasets_folds), " dataset-fold combinations...")

for (r in seq_len(nrow(datasets_folds))) {
  ds <- datasets_folds$dataset[r]
  fd <- datasets_folds$fold[r]

  dat_sub <- per_method_noise %>% filter(dataset == ds, fold == fd)
  # ensure noise is factor ordered
  dat_sub <- dat_sub %>% mutate(noise_f = factor(noise))

  res <- list()

  try({
    n_levels <- length(unique(dat_sub$noise_f))
    if (n_levels > 2) {
      # Friedman test: mean_kappa_loss ~ noise | method
      fr <- tryCatch(
        friedman.test(mean_kappa_loss ~ noise_f | method, data = dat_sub),
        error = function(e) e
      )
      res$friedman <- fr

      # Pairwise Wilcoxon (paired across methods)
      pw <- tryCatch(
        pairwise.wilcox.test(dat_sub$mean_kappa_loss, dat_sub$noise_f, paired = TRUE, p.adjust.method = "BH"),
        error = function(e) e
      )
      res$pairwise_wilcox <- pw

      # Try repeated measures ANOVA (may fail if assumptions not met)
      aov_res <- tryCatch({
        aov(mean_kappa_loss ~ noise_f + Error(factor(method)), data = dat_sub)
      }, error = function(e) e)
      res$aov <- aov_res
    } else if (n_levels == 2) {
      # Paired Wilcoxon across methods
      w <- tryCatch({
        # reshape to wide: rows methods, cols two noise levels
        wide <- dat_sub %>% select(method, noise_f, mean_kappa_loss) %>% pivot_wider(names_from = noise_f, values_from = mean_kappa_loss)
        # paired wilcox.test on the two columns
        cols <- names(wide)[-1]
        wilcox.test(wide[[cols[1]]], wide[[cols[2]]], paired = TRUE)
      }, error = function(e) e)
      res$wilcox <- w
    }
  }, silent = TRUE)

  tests_by_noise[[paste(ds, fd, sep = "__")]] <- res
}

saveRDS(tests_by_noise, file = "results/instances/aggregated/tests/tests_by_noise_per_fold.rds")
message("Saved: results/instances/aggregated/tests/tests_by_noise_per_fold.rds")

# Now do the analogous analysis across percent levels (average across noise)
per_method_percent <- df_kappa2 %>%
  group_by(dataset, fold, method, percent) %>%
  summarise(mean_kappa_loss = mean(kappa_loss, na.rm = TRUE), .groups = "drop")

datasets_folds2 <- per_method_percent %>% select(dataset, fold) %>% distinct()

message("Running tests by percent level across ", nrow(datasets_folds2), " dataset-fold combinations...")

for (r in seq_len(nrow(datasets_folds2))) {
  ds <- datasets_folds2$dataset[r]
  fd <- datasets_folds2$fold[r]

  dat_sub <- per_method_percent %>% filter(dataset == ds, fold == fd)
  dat_sub <- dat_sub %>% mutate(percent_f = factor(percent))

  res <- list()

  try({
    n_levels <- length(unique(dat_sub$percent_f))
    if (n_levels > 2) {
      fr <- tryCatch(
        friedman.test(mean_kappa_loss ~ percent_f | method, data = dat_sub),
        error = function(e) e
      )
      res$friedman <- fr

      pw <- tryCatch(
        pairwise.wilcox.test(dat_sub$mean_kappa_loss, dat_sub$percent_f, paired = TRUE, p.adjust.method = "BH"),
        error = function(e) e
      )
      res$pairwise_wilcox <- pw

      aov_res <- tryCatch({
        aov(mean_kappa_loss ~ percent_f + Error(factor(method)), data = dat_sub)
      }, error = function(e) e)
      res$aov <- aov_res
    } else if (n_levels == 2) {
      w <- tryCatch({
        wide <- dat_sub %>% select(method, percent_f, mean_kappa_loss) %>% pivot_wider(names_from = percent_f, values_from = mean_kappa_loss)
        cols <- names(wide)[-1]
        wilcox.test(wide[[cols[1]]], wide[[cols[2]]], paired = TRUE)
      }, error = function(e) e)
      res$wilcox <- w
    }
  }, silent = TRUE)

  tests_by_percent[[paste(ds, fd, sep = "__")]] <- res
}

saveRDS(tests_by_percent, file = "results/instances/aggregated/tests/tests_by_percent_per_fold.rds")
message("Saved: results/instances/aggregated/tests/tests_by_percent_per_fold.rds")

# ============================================================================
# Write brief summaries to CSVs for quick inspection
# ============================================================================

message("\nGenerating test summaries...")

summarize_test <- function(tlist) {
  rows <- list(); ri <- 1
  for (nm in names(tlist)) {
    res <- tlist[[nm]]
    # collect available test names
    if (!is.null(res$friedman) && inherits(res$friedman, "htest")) {
      rows[[ri]] <- data.frame(target = nm, test = "friedman", statistic = res$friedman$statistic, p.value = res$friedman$p.value, stringsAsFactors = FALSE); ri <- ri+1
    }
    if (!is.null(res$wilcox) && inherits(res$wilcox, "htest")) {
      rows[[ri]] <- data.frame(target = nm, test = "wilcox", statistic = res$wilcox$statistic, p.value = res$wilcox$p.value, stringsAsFactors = FALSE); ri <- ri+1
    }
    if (!is.null(res$pairwise_wilcox) && inherits(res$pairwise_wilcox, "list")) {
      # pairwise.wilcox.test returns list with p.value matrix
      pmat <- res$pairwise_wilcox$p.value
      # flatten
      if (!is.null(pmat)) {
        for (i in seq_len(nrow(pmat))) for (j in seq_len(ncol(pmat))) {
          pv <- pmat[i,j]
          if (!is.na(pv)) {
            rows[[ri]] <- data.frame(target = nm, test = "pairwise_wilcox", comparison = paste(rownames(pmat)[i], colnames(pmat)[j], sep = " vs "), p.value = pv, stringsAsFactors = FALSE); ri <- ri+1
          }
        }
      }
    }
  }
  if (length(rows) == 0) return(tibble())
  bind_rows(rows)
}

sum_noise <- summarize_test(tests_by_noise)
sum_percent <- summarize_test(tests_by_percent)
if (nrow(sum_noise)>0) write.csv(sum_noise, file = "results/instances/aggregated/tests/summary_tests_by_noise.csv", row.names = FALSE)
if (nrow(sum_percent)>0) write.csv(sum_percent, file = "results/instances/aggregated/tests/summary_tests_by_percent.csv", row.names = FALSE)

# ============================================================================
# Final summary
# ============================================================================

cat("\n=============================================================================\n")
cat("Statistical tests completed. Results saved to results/instances/aggregated/tests/\n")
cat("\nOutput files created:\n")
cat("  - results/instances/aggregated/agg_kappa_loss_by_noise_per_fold.rds (and .csv)\n")
cat("  - results/instances/aggregated/agg_kappa_loss_by_percent_per_fold.rds (and .csv)\n")
cat("  - results/instances/aggregated/tests/tests_by_noise_per_fold.rds\n")
cat("  - results/instances/aggregated/tests/tests_by_percent_per_fold.rds\n")
if (nrow(sum_noise)>0) cat("  - results/instances/aggregated/tests/summary_tests_by_noise.csv\n")
if (nrow(sum_percent)>0) cat("  - results/instances/aggregated/tests/summary_tests_by_percent.csv\n")
cat("=============================================================================\n\n")
