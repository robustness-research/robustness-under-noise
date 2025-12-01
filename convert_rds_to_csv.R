#!/usr/bin/env Rscript
# Convert RDS files to CSV format for Python processing
# Usage: Rscript convert_rds_to_csv.R [input_dir] [output_dir]

# Load required libraries
suppressPackageStartupMessages({
  library(methods)
})

# Parse command line arguments
args <- commandArgs(trailingOnly = TRUE)
input_dir <- if(length(args) >= 1) args[1] else "datasets"
output_dir <- if(length(args) >= 2) args[2] else "datasets/csv"

# Create output directory if it doesn't exist
if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
  cat("Created output directory:", output_dir, "\n")
}

# Find all RDS files
rds_files <- list.files(input_dir, pattern = "\\.rds$", full.names = TRUE)

if (length(rds_files) == 0) {
  cat("No RDS files found in", input_dir, "\n")
  quit(status = 1)
}

cat("Found", length(rds_files), "RDS files\n")
cat(sprintf("Converting RDS files from '%s' to CSV in '%s'\n", input_dir, output_dir))
cat(paste(rep("=", 60), collapse = ""), "\n\n")

# Convert each RDS file
successful <- 0
failed <- 0

for (rds_file in rds_files) {
  # Get base filename without extension
  base_name <- tools::file_path_sans_ext(basename(rds_file))
  output_file <- file.path(output_dir, paste0(base_name, ".csv"))
  
  cat(sprintf("[%d/%d] Converting: %s\n", 
              which(rds_files == rds_file), 
              length(rds_files), 
              base_name))
  
  tryCatch({
    # Read RDS file
    data <- readRDS(rds_file)
    
    # Convert to data frame if needed
    if (!is.data.frame(data)) {
      if (is.list(data)) {
        # If it's a list, try to get the first data frame
        data <- data[[1]]
      } else {
        data <- as.data.frame(data)
      }
    }
    
    # Write to CSV
    write.csv(data, output_file, row.names = FALSE)
    
    cat(sprintf("  ✓ Success: %d rows × %d columns → %s\n", 
                nrow(data), ncol(data), output_file))
    successful <- successful + 1
    
  }, error = function(e) {
    cat(sprintf("  ✗ Error: %s\n", e$message))
    failed <- failed + 1
  })
}

# Print summary
cat("\n", paste(rep("=", 60), collapse = ""), "\n")
cat("CONVERSION COMPLETE\n")
cat(paste(rep("=", 60), collapse = ""), "\n")
cat(sprintf("Successfully converted: %d\n", successful))
cat(sprintf("Failed: %d\n", failed))
cat(sprintf("Total: %d\n", length(rds_files)))
cat(sprintf("\nCSV files saved to: %s\n", output_dir))
