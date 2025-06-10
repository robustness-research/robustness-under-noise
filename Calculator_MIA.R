# /usr/bin/env/Rscript

# Packages that need to be loaded
pacman::p_load(caret, iml, citation, dplyr, earth, lime)

# Set the seed to make the experiment reproducible
set.seed(1)

# Load data
args <- commandArgs(trailingOnly = TRUE)
datasets <- args
methods <- readRDS("files/methods.rds")
control <- readRDS("files/control.rds")

# Create a resulting dataframe
mia_df = data.frame(matrix(ncol = 3, nrow = 0))
colnames(mia_df) = c("dataset_name", "technique", "most_important_attribute")


# Calculate most important attributes for each dataset,
# depending on technique used
for (dataset in datasets) {

  # Load dataset
  ds_filename = paste0("datasets/", dataset, ".rds")
  df = readRDS(ds_filename)

  # Auxiliary dataframe to save per dataset
  aux_df = data.frame(matrix(ncol = 3, nrow = 0))
  colnames(aux_df) = c("dataset_name", "technique", "most_important_attribute")
  if(FALSE) {
    # For each method that is not knn or bayesglm
    for(method in methods) {
        
      # Create a frequency table for the class in the 'Training' dataset
      # Calculate the frequency table as a proportion of all values
      print(paste("Dataset:", dataset))
      print("Probability table:")
      prop.table(table(df$class))
      
      #TRAINING#
      print(paste("Dataset:", dataset))
      print(paste("Technique:", method))
      print("BEGIN TRAINING")
      if(method == "C5.0"){
        fit = caret::train(class ~ ., data = df, method = "C5.0")
      }else if(method == "ctree"){
        fit = caret::train(class~., data = df, method = "ctree")
      }else if(method == "fda"){
        fit = caret::train(class~., data = df, method = "fda")
      }else if(method == "gbm"){
        fit = caret::train(class~., data = df, method = "gbm")
      }else if(method == "gcvEarth"){
        fit = caret::train(class ~ ., data = df, method = "gcvEarth")
      }else if(method == "JRip"){
        fit = caret::train(class~., data = df, method = "JRip")
      }else if(method == "lvq"){
        fit = caret::train(class~., data = df, method = "lvq")
      }else if(method == "mlpML"){
        fit = caret::train(class ~ ., data = df, method = "mlpML")
      }else if(method == "multinom"){
        fit = caret::train(class ~ ., data = df, method = "multinom", trControl = control, tuneGrid = expand.grid(decay = c(0)), MaxNWts = 10000)
      }else if(method == "naive_bayes"){
        fit = caret::train(class ~ ., data = df, method = "naive_bayes")
      }else if(method == "PART"){
        fit = caret::train(class ~ ., data = df, method = "PART")
      }else if(method == "rbfDDA"){
        fit = caret::train(class ~ ., data = df, method = "rbfDDA", trControl = control)
      }else if(method == "rda"){
        fit = caret::train(class~., data = df, method = "rda")
      }else if(method == "rf"){
        fit = caret::train(class ~ ., data = df, method = "rf")
      }else if(method == "rpart"){
        fit = caret::train(class ~ ., data = df, method = "rpart")
      }else if(method == "simpls"){
        fit = caret::train(class ~ ., data = df, method = "simpls")
      }else if(method == "svmLinear"){
        fit = caret::train(class ~ ., data = df, method = "svmLinear")
      }else if(method == "svmRadial"){
        fit = caret::train(class ~ ., data = df, method = "svmRadial")
      }else if(method == "rfRules"){
        fit = caret::train(class ~ ., data = df, method = "rfRules")
      }
      print("TRAINING SUCCESSFULL")

      # Create a predictor object from the trained data and
      # Compute feature importance for prediction models with 'classification' loss
      featImp = FeatureImp$new(Predictor$new(fit), loss = "ce")
      
      #f = paste0("results/mia_results/importance/", dataset, "_", method, "_featImp.rds")
      #saveRDS(featImp, file = f)
      
      # In the list of results of the computed feature importance:
      ## Find max value from the importance column in the results list
      ## Return the associated feature -> The most important attribute
      mia = featImp$results[which.max(featImp$results$importance),'feature']
      print(paste0("The most important attribute: ", mia, " (for technique: ", method, ")"))
      
      # Record the results
      mia_df[nrow(mia_df) + 1,] = c(dataset, method, mia)
      aux_df[nrow(aux_df) + 1,] = c(dataset, method, mia)
      print("________________")
    }
  }
  
  # Standardize DF for KNN and GLM
  df =  df %>% mutate_if(is.numeric, ~(scale(.) %>% as.vector))
  
  #########
  ## KNN ##
  #########

  if(TRUE) {
    method = "knn"
      
    # Create a frequency table for the class in the 'Training' dataset
    # Calculate the frequency table as a proportion of all values
    print(paste("Dataset:", dataset))
    print("Probability table:")
    prop.table(table(df$class))
    
    #TRAINING#
    print(paste("Dataset:", dataset))
    print(paste("Technique:", method))
    print("BEGIN TRAINING")
    fit = caret::train(class ~ ., data = df, method="knn", tuneGrid = expand.grid(k = 5:5), trControl = trainControl(method = "none", number = 1, preProcess = c("center", "scale")))
    print("TRAINING SUCCESSFULL")
      
    # Create a predictor object from the trained data and
    # Compute feature importance for prediction models with 'classification' loss
    featImp = FeatureImp$new(Predictor$new(fit), loss = "ce")
    
    #f = paste0("results/mia_results/importance/", dataset, "_", method, "_featImp.rds")
    #saveRDS(featImp, file = f)
    
    # In the list of results of the computed feature importance:
    ## Find max value from the importance column in the results list
    ## Return the associated feature -> The most important attribute
    mia = featImp$results[which.max(featImp$results$importance),'feature']
    print(paste0("The most important attribute: ", mia, " (for technique: ", method, ")"))
    
    # Record the results
    mia_df[nrow(mia_df) + 1,] = c(dataset, method, mia)
    aux_df[nrow(aux_df) + 1,] = c(dataset, method, mia)
    print("________________")
  }
  #########
  ## GLM ##
  #########
  if(FALSE) {
    method = "bayesglm"
    
    #Create a frequency table for the class in the 'Training' dataset
    #Calculate the frequency table as a proportion of all values
    print(paste("Dataset:", dataset))
    print("Probability table:")
    prop.table(table(df$class))
    
    #TRAINING#
    print(paste("Dataset:", dataset))
    print(paste("Technique:", method))
    print("BEGIN TRAINING")
    fit = caret::train(class ~ ., data = df, method="bayesglm", trControl = control)
    print("TRAINING SUCCESSFULL")
    
    # Create a predictor object from the trained data and
    # Compute feature importance for prediction models with 'classification' loss
    featImp = FeatureImp$new(Predictor$new(fit), loss = "ce")
    
    #f = paste0("results/mia_results/importance/", dataset, "_", method, "_featImp.rds")
    #saveRDS(featImp, file = f)
    
    # In the list of results of the computed feature importance:
    ## Find max value from the importance column in the results list
    ## Return the associated feature -> The most important attribute
    mia = featImp$results[which.max(featImp$results$importance),'feature']
    print(paste0("The most important attribute: ", mia, " (for technique: ", method, ")"))
    
    # Record the results
    mia_df[nrow(mia_df) + 1,] = c(dataset, method, mia)
    aux_df[nrow(aux_df) + 1,] = c(dataset, method, mia)
    print("________________")
  }
  
  print("Most important attribute obtained with all techniques")
  
  # Safeguard store by dataset
  filename = paste0("results/most_important_attr/by_dataset/", dataset, "_mia_KNN.rds")
  saveRDS(aux_df, file = filename)
  
  # Save list of most important attributes (unique)
  aux_df_unique <- aux_df
  aux_df_unique <- distinct(aux_df_unique, dataset_name, most_important_attribute)
  
  # Safeguard store by dataset
  filename = paste0("results/most_important_attr/by_dataset/", dataset, "_miaUnique_KNN.rds")
  saveRDS(aux_df_unique, file = filename)
  
  print("----------------")
}

# Save list of most important attributes (per method)
# mia_df <- mia_df %>% arrange(mia_df$dataset_name, mia_df$technique)
# saveRDS(mia_df, file = "results/most_important_attr/mia_df.rds")

# Save list of most important attributes (unique)
# mia_df_unique <- mia_df
# mia_df_unique <- distinct(mia_df_unique, dataset_name, most_important_attribute)
# saveRDS(mia_df_unique, file = "results/most_important_attr/miaUnique_df.rds")

print("****************")
print("RESULTS RECORDED")
print("****************")
