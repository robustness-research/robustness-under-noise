# /usr/bin/env/Rscript

# Packages that need to be loaded
pacman::p_load(dplyr, data.table)

# Load files
datasets <- readRDS("../files/datasets.rds")

# Merge all dataset's instances lists in one .rds file
print("Unifying all instances lists")
  
instances_list <- list()
instancesCM_list <- list()
c = 1
  
for(dataset in datasets) {
  # Load dataset
  #filename1 = paste0("../results/instances/by_dataset/", dataset, "_instances.rds")
  #filename2 = paste0("../results/instances/by_dataset/", dataset, "_instancesCM.rds")
  filename1 = paste0("../results/instances/by_dataset_popular/", dataset, "_instances_popular.rds")
  filename2 = paste0("../results/instances/by_dataset_popular/", dataset, "_instancesCM_popular.rds")
  df1 <- readRDS(filename1)
  df2 <- readRDS(filename2)
  instances_list[[c]] <- df1
  instancesCM_list[[c]] <- df2
  c = c + 1
  print(paste0("Added dataset: ", dataset))
}
  
names(instances_list) <- datasets
names(instancesCM_list) <- datasets
#saveRDS(instances_list, file = "../results/instances/instances_list.rds")
#saveRDS(instancesCM_list, file = "../results/instances/instancesCM_list.rds")
saveRDS(instances_list, file = "../results/instances_list_popular.rds")
saveRDS(instancesCM_list, file = "../results/instancesCM_list_popular.rds")
print("Results recorded")
print("-----")


