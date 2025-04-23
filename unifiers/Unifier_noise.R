# /usr/bin/env/Rscript

# Packages that need to be loaded
pacman::p_load(dplyr, data.table)

# Set the seed to make the experiment reproducible
set.seed(1)

# Load files
datasets <- readRDS("../files/datasets.rds")

# Merge all dataset's noise lists in one .rds file
print("Unifying all noise lists")
  
noiseMIA_list <- list()
c = 1
  
for(dataset in datasets) {
  # Load dataset
  filename = paste0("../results/noise/by_dataset/", dataset, "_noiseMIA.rds")
  df <- readRDS(filename)
  noiseMIA_list[[c]] <- df
  c = c + 1
  print(paste0("Added dataset: ", dataset))
}
  
names(noiseMIA_list) <- datasets
saveRDS(noiseMIA_list, file = "../results/noise/noise_list.rds")
print("Results recorded")
print("-----")

