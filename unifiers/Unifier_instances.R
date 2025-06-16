# /usr/bin/env/Rscript

# Packages that need to be loaded
pacman::p_load(dplyr, data.table)

# Load files
datasets <- readRDS("../files/datasets.rds")

if(FALSE) {
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
}

if(TRUE) {
  fold_names <- readRDS("../files/folds.rds")
  #####
  # Add only KNN
  # Load dataframes
  instances_list <- readRDS("../results/instances/instances_list.rds")
  instances_list_popular <- readRDS("../results/instances/instances_list_popular.rds")
  instancesCM_list <- readRDS("../results/instances/instancesCM_list.rds")
  instancesCM_list_popular <- readRDS("../results/instances/instancesCM_list_popular.rds")
  print("Files loaded")

  # Remove KNN entries in lists
  for(dataset in datasets) {
    # Load new KNN dataset
    knn_filename1 = paste0("../results/instances/by_dataset/", dataset, "_instances_KNN.rds") 
    knn_filename2 = paste0("../results/instances/by_dataset_popular/", dataset, "_instances_popular_knn.rds")
    knn_filename3 = paste0("../results/instances/by_dataset/", dataset, "_instancesCM_KNN.rds")
    knn_filename4 = paste0("../results/instances/by_dataset_popular/", dataset, "_instancesCM_popular_knn.rds")
    print("KNN removed from lists")
    
    knn_list1 <- readRDS(knn_filename1)
    knn_list2 <- readRDS(knn_filename2)
    knn_list3 <- readRDS(knn_filename3)
    knn_list4 <- readRDS(knn_filename4)

    for(fold in fold_names) {
      # Remove KNN from instances_list
      instances_list[[dataset]][[fold]][["knn"]] <- NULL
      instances_list_popular[[dataset]][[fold]][["knn"]] <- NULL
      instancesCM_list[[dataset]][[fold]][["knn"]] <- NULL
      instancesCM_list_popular[[dataset]][[fold]][["knn"]] <- NULL

      # Re-add KNN (and svmLinear) to lists
      # Add to main lists
      instances_list[[dataset]][[fold]][["knn"]] <- knn_list1[[fold]]
      instances_list_popular[[dataset]][[fold]][["knn"]] <- knn_list2[[fold]]
      instances_list_popular[[dataset]][[fold]][["svmLinear"]] <- knn_list2[[fold]]
      instancesCM_list[[dataset]][[fold]][["knn"]] <- knn_list3[[fold]]
      instancesCM_list_popular[[dataset]][[fold]][["knn"]] <- knn_list4[[fold]]
      instancesCM_list_popular[[dataset]][[fold]][["svmLinear"]] <- knn_list4[[fold]]
    
      print(paste0("Added model results for dataset: ", dataset, "and fold", fold))
    }
    print(paste0("Added model results for dataset: ", dataset))
  }
  
  # Reorder lists by dataset name
  instances_list <- instances_list[order(names(instances_list))]
  instances_list_popular <- instances_list_popular[order(names(instances_list_popular))]
  instancesCM_list <- instancesCM_list[order(names(instancesCM_list))]
  instancesCM_list_popular <- instancesCM_list_popular[order(names(instancesCM_list_popular))]

  # Save the updated dataframes
  saveRDS(instances_list, file = "../results/instances/instances_list.rds")
  saveRDS(instancesCM_list, file = "../results/instances/instancesCM_list.rds")
  saveRDS(instances_list_popular, file = "../results/instances/instances_list_popular.rds")
  saveRDS(instancesCM_list_popular, file = "../results/instances/instancesCM_list_popular.rds")
  print("Updated results saved")
  print("-----")
}
