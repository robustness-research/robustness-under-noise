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
  
saveRDS(mia_df, file = "../results/most_important_attr/mia_df.rds")
saveRDS(miaUnique_df, file = "../results/most_important_attr/miaUnique_df.rds")
print("Results recorded")
print("-----")



