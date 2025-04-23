# /usr/bin/env/Rscript
setwd("~/rscripts/code")

# Packages that need to be loaded
library("pacman")
pacman::p_load(dplyr, farff)

# Load data
datasets <- readRDS("files/datasets.rds")

# Read and clean all datasets
for(dataset in datasets) {
  
  # Obtain the dataset's path
  path = paste0(paste0("datasets/arff/", dataset), ".arff")
  
  # Read clean DF
  df = readARFF(path)
  
  # Set the class name
  if(dataset == "analcatdata_authorship"){
    names(df)[names(df)=="binaryClass"] <- "class"
    # if , for and in
  } else if(dataset == "badges2") {
    levels(df$class) = c(0, 1)
    df <- df %>% select(-ID)
  } else if(dataset == "blood-transfusion-service-center"){
    names(df)[names(df)=="Class"] <- "class"
  } else if(dataset == "breast-w"){
    names(df)[names(df)=="Class"] <- "class"
  } else if(dataset == "cardiotocography"){
    names(df)[names(df)=="Class"] <- "class"
  } else if(dataset == "climate-model-simulation-crashes"){
    names(df)[names(df)=="outcome"] <- "class"
    df <- df %>% select(-Study)
    df <- df %>% select(-Run)
  } else if(dataset == "cmc"){
    names(df)[names(df)=="binaryClass"] <- "class"
  } else if(dataset == "eucalyptus"){
    names(df)[names(df)=="binaryClass"] <- "class"
  } else if(dataset == "first-order-theorem-proving"){
    names(df)[names(df)=="Class"] <- "class"
  } else if(dataset == "jm1"){
    names(df)[names(df)=="defects"] <- "class"
  } else if(dataset == "kc1"){
    names(df)[names(df)=="defects"] <- "class"
  } else if(dataset == "liver-disorders"){
    names(df)[names(df)=="selector"] <- "class"
  } else if(dataset == "mfeat-zernike"){
    names(df)[names(df)=="binaryClass"] <- "class"
  } else if(dataset == "ozone-level-8hr"){
    names(df)[names(df)=="Class"] <- "class"
  } else if(dataset == "pc4"){
    names(df)[names(df)=="c"] <- "class"
  } else if(dataset == "pendigits"){
    names(df)[names(df)=="binaryClass"] <- "class"
  } else if(dataset == "phoneme"){
    names(df)[names(df)=="Class"] <- "class"
  } else if(dataset == "qsar-biodeg"){
    names(df)[names(df)=="Class"] <- "class"
  } else if(dataset == "synthetic_control"){
    names(df)[names(df)=="binaryClass"] <- "class"
  } else if(dataset == "tic-tac-toe"){
    names(df)[names(df)=="Class"] <- "class"
  } else if(dataset == "vowel"){
    names(df)[names(df)=="binaryClass"] <- "class"
    df <- df %>% select(-Train_or_Test)
    df <- df %>% select(-Speaker_Number)
    df <- df %>% select(-Sex)
  } else if(dataset == "wdbc"){
    names(df)[names(df)=="Class"] <- "class"
  } else if(dataset == "waveform-5000"){
    names(df)[names(df)=="binaryClass"] <- "class"
  }
  
  # Clean dataframe from any NAs
  df <- na.omit(df)
  
  # Set the class as a factor
  df$class = as.factor(df$class)
  
  # Save clean file
  filename = paste0("datasets/", dataset, ".rds")
  saveRDS(df, file = filename)
  print(paste0("Dataset: ", dataset))
  
  print("----------------")
}

print("****************")
print("RESULTS RECORDED")
print("****************")
