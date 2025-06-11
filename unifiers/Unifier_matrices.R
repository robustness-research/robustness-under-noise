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
