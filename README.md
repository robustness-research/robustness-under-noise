# Robustness Under Noise README

## About Missing Files

This repository is intentionally missing certain file types for the following reasons:

### .log Files

- All `.log` files have been excluded from this repository.
- These files were simply the output from `nohup` commands used during execution.
- They were generated to ensure proper function monitoring during long-running processes.
- The `.log` files are not essential for the project and can be regenerated if needed.
- The `.log` files were stored in corresponding subfolders within the `output/` directory.

### .rds Files

- All `.rds` files have been excluded due to their large file size.
- These files exceed GitHub's file size limits for repository uploads.
- You can generate all necessary `.rds` files by running the R scripts in the proper order.

## About the Data

The datasets used are stored in `datasets/` as `.arff` files. The script `DatasetCleaner.R` turns them into usable `.rds` files.

## About the Scripts

The bash scripts, as well as the R scripts found in `unifiers/` are present for ease of use. Since execution of the scripts with all datasets, models,
levels of noise and percentages of instances takes a very long time and unexpectedly hault execution (errors, memory leaks, etc) if done in a linear order,
the `.sh` scripts are used to execute each script by dataset and can later be put together using the corresponding unifier script present in `unifiers/` into a single file.

## Script Execution Order

To regenerate all necessary files, please run the scripts in the following order:

1.  `Filenames.R`
2.  `DatasetCleaner.R`
3.  `Calculator_MIA.R`
4.  `NoiseInjector.R`
5.  `Instances.R` (and `Instances_Popular.R`)
6.  `Calculator_CM.R`

Running these scripts in sequence will generate all required data files for the project (using the bash and 'unifier' scripts as needed).

## Plots

To obtain the resulting plots and all other results, please run the files found in the `markdown/` directory in the following order:

1.  `KLC_Plots.Rmd`
2.  `Aggregate_Curves.Rmd`
3.  `Model_Heirarchy.Rmd`
4.  `Plots.Rmd`
5.  `Dataset_Clustering.Rmd`

For the results pertaining to the most voted attribute (popular), the subfolder `popular/` contains all the files necessary to obtain the data present in the article. For the analysis that compares the original clustering compared with the "popular attribute" clustering, the files `Plot_Comparison.Rmd` and `Cluster_Stability.Rmd` is used.

Additionally, the data generated to be plotted as an example for the introduction can be found in `Example_Data.Rmd`.

Lastly, the file `Attribute_Histogram.Rmd` contains the code used to generate the histograms of the importance of each attribute in a dataset. This was to corroborate further results.
