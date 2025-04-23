# /usr/bin/env/Rscript

# Packages that need to be loaded
pacman::p_load(dplyr, data.table)

# Set the seed to make the experiment reproducible
set.seed(1)

# Load files
datasets <- readRDS("../files/datasets.rds")

# Merge all dataset's instances lists in one .rds file
print("Unifying all matrices lists")
  
matrices_list <- list()
c = 1
  
for(dataset in datasets) {
  # Load dataset
  #filename1 = paste0("../results/conf_matrices/by_dataset/", dataset, "_cm.rds")
  filename1 = paste0("../results/conf_matrices/by_dataset/", dataset, "_cm_popular.rds")
  df1 <- readRDS(filename1)
  matrices_list[[c]] <- df1
  c = c + 1
  print(paste0("Added dataset: ", dataset))
}
  
names(matrices_list) <- datasets
#saveRDS(matrices_list, file = "../results/conf_matrices/confusion_matrices.rds")
saveRDS(matrices_list, file = "../results/conf_matrices/confusion_matrices_popular.rds")
print("Results recorded")
print("-----")


