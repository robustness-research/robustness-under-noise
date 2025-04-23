# BDMA-2025 README

## About Missing Files

This repository is intentionally missing certain file types for the following reasons:

### .log Files
- All `.log` files have been excluded from this repository.
- These files were simply the output from `nohup` commands used during execution.
- They were generated to ensure proper function monitoring during long-running processes.
- The `.log` files are not essential for the project and can be regenerated if needed.

### .rds Files
- All `.rds` files have been excluded due to their large file size.
- These files exceed GitHub's file size limits for repository uploads.
- You can generate all necessary `.rds` files by running the R scripts in the proper order.

## Script Execution Order
To regenerate all necessary files, please run the scripts in the following order:

1. `Filenames.R`
2. `DatasetCleaner.R`
3. `Calculator_MIA.R`
4. `NoiseInjector.R`
5. `Instances.R` (and `Instances_Popular.R`)
6. `Calculator_CM.R`

Running these scripts in sequence will generate all required data files for the project.

## Plots
To obtain the resulting plots... TODO
