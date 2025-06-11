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

if(FALSE) {
  #####
  # Add only KNN
  # Load dataframes
  datasets <- readRDS("../files/datasets.rds")
  mia_df <- readRDS("../results/most_important_attr/mia_df.rds")
  miaUnique_df <- readRDS("../results/most_important_attr/miaUnique_df.rds")
  
  # Remove KNN entries in df
  mia_df <- mia_df %>% filter(technique != "knn")
  
  # Re-add KNN to dataframe
  for(dataset in datasets) {
    # Load new KNN dataset
    knn_filename1 = paste0("../results/most_important_attr/by_dataset/", dataset, "_mia_KNN.rds")
    knn_filename2 = paste0("../results/most_important_attr/by_dataset/", dataset, "_miaUnique_KNN.rds")
    
    knn_df1 <- readRDS(knn_filename1)
    knn_df2 <- readRDS(knn_filename2)
    
    # Add to main dataframes
    mia_df <- rbind(mia_df, knn_df1)
    miaUnique_df <- rbind(miaUnique_df, knn_df2)
    miaUnique_df <- distinct(miaUnique_df, dataset_name, most_important_attribute)
    print(paste0("Added KNN results for dataset: ", dataset))
    
  }
  
  mia_df <- mia_df %>% arrange(dataset_name, technique)
  miaUnique_df <- miaUnique_df %>% arrange(dataset_name)
  
  # Save the updated dataframes
  saveRDS(mia_df, file = "../results/most_important_attr/mia_df.rds")
  saveRDS(miaUnique_df, file = "../results/most_important_attr/miaUnique_df.rds")
  print("Updated results saved")
  print("-----")
}
