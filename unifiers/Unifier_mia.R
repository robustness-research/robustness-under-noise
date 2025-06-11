# /usr/bin/env/Rscript

# Packages that need to be loaded
pacman::p_load(dplyr, data.table)

# Load files
datasets <- readRDS("../files/datasets.rds")

# Merge all dataset's mia dataframes in one .rds file
print("Merging all MIA datframes")
mia_df = data.frame(matrix(ncol = 3, nrow = 0))
colnames(mia_df) = c("dataset_name", "technique", "most_important_attribute")
miaUnique_df = data.frame(matrix(ncol = 3, nrow = 0))
colnames(miaUnique_df) = c("dataset_name", "technique", "most_important_attribute")
  
for(dataset in datasets) {
  # Load dataset
  filename1 = paste0("../results/most_important_attr/by_dataset/", dataset, "_mia.rds")
  filename2 = paste0("../results/most_important_attr/by_dataset/", dataset, "_miaUnique.rds")
  df1 <- readRDS(filename1)
  df2 <- readRDS(filename2)
  mia_df <- rbind(mia_df, df1)
  miaUnique_df <- rbind(miaUnique_df, df2)
  print(paste0("Added dataset: ", dataset))
}

mia_df <- mia_df %>% arrange(dataset_name, technique)
miaUnique_df <- miaUnique_df %>% arrange(dataset_name)
  
saveRDS(mia_df, file = "../results/most_important_attr/mia_df.rds")
saveRDS(miaUnique_df, file = "../results/most_important_attr/miaUnique_df.rds")
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