# /usr/bin/env/Rscript

# Part 1: Get MIA
# Packages that need to be loaded
pacman::p_load(caret, iml, citation, dplyr, earth, lime, data.table)

# Set the seed to make the experiment reproducible
set.seed(1)

# Load previous files
datasets <- c("iris")
fold_names <- readRDS("files/folds.rds")
methods <- readRDS("files/methods.rds")
method_names = readRDS("files/method_names.rds")
#control <- readRDS("files/control.rds")
noise_level <- readRDS("files/noise.rds")
noise_names <- readRDS("files/noise_names.rds")
instances <- append(readRDS("files/instances.rds"), c(0.25, 0.75))
instances_names <- append(readRDS("files/instances_names.rds"), c("25", "75"))
control = trainControl(method = "none", number = 1)

# Create a resulting dataframe - now capturing top 3 attributes
mia_df = data.frame(matrix(ncol = 4, nrow = 0))
colnames(mia_df) = c("dataset_name", "technique", "most_important_attribute", "rank")

# Calculate most important attributes for each dataset,
# depending on technique used
for (dataset in datasets) {
  
  # Load dataset
  ds_filename = paste0("datasets/", dataset, ".rds")
  df = readRDS(ds_filename)
  
  # Auxiliary dataframe to save per dataset
  aux_df = data.frame(matrix(ncol = 4, nrow = 0))
  colnames(aux_df) = c("dataset_name", "technique", "most_important_attribute", "rank")
  if(TRUE) {
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
      
      # Sort features by importance and get top 3
      importances <- as.data.frame(featImp$results)
      importances <- importances[order(-importances$importance), ]
      top_3_features <- head(importances$feature, 3)
      
      # Record the top 3 attributes with their ranks
      for(rank in 1:length(top_3_features)) {
        mia = top_3_features[rank]
        print(paste0("Rank ", rank, " - Most important attribute: ", mia, " (for technique: ", method, ")"))
        
        # Record the results
        mia_df[nrow(mia_df) + 1,] = c(dataset, method, mia, rank)
        aux_df[nrow(aux_df) + 1,] = c(dataset, method, mia, rank)
      }
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
    fit = caret::train(class ~ ., data = df, method="knn", tuneGrid = expand.grid(k = 5:5), preProcess = c("center", "scale"), trControl = control)
    print("TRAINING SUCCESSFULL")
    
    # Create a predictor object from the trained data and
    # Compute feature importance for prediction models with 'classification' loss
    featImp = FeatureImp$new(Predictor$new(fit), loss = "ce")
    
    #f = paste0("results/mia_results/importance/", dataset, "_", method, "_featImp.rds")
    #saveRDS(featImp, file = f)
    
    # Sort features by importance and get top 3
    importances <- as.data.frame(featImp$results)
    importances <- importances[order(-importances$importance), ]
    top_3_features <- head(importances$feature, 3)
    
    # Record the top 3 attributes with their ranks
    for(rank in 1:length(top_3_features)) {
      mia = top_3_features[rank]
      print(paste0("Rank ", rank, " - Most important attribute: ", mia, " (for technique: ", method, ")"))
      
      # Record the results
      mia_df[nrow(mia_df) + 1,] = c(dataset, method, mia, rank)
      aux_df[nrow(aux_df) + 1,] = c(dataset, method, mia, rank)
    }
    print("________________")
  }
  #########
  ## GLM ##
  #########
  if(TRUE) {
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
    
    # Sort features by importance and get top 3
    importances <- as.data.frame(featImp$results)
    importances <- importances[order(-importances$importance), ]
    top_3_features <- head(importances$feature, 3)
    
    # Record the top 3 attributes with their ranks
    for(rank in 1:length(top_3_features)) {
      mia = top_3_features[rank]
      print(paste0("Rank ", rank, " - Most important attribute: ", mia, " (for technique: ", method, ")"))
      
      # Record the results
      mia_df[nrow(mia_df) + 1,] = c(dataset, method, mia, rank)
      aux_df[nrow(aux_df) + 1,] = c(dataset, method, mia, rank)
    }
    print("________________")
  }
  
  print("Most important attributes obtained with all techniques")
  
  # Save list of most important attributes (keep all 3 per technique)
  aux_df_unique <- aux_df
  mia_df_unique <- aux_df_unique %>% 
    distinct(dataset_name, most_important_attribute, rank)
  
  print(paste("Total attributes to test for colinearity:", nrow(mia_df_unique)))
  print("----------------")
}

print("****************")
print("RESULTS RECORDED")
print("****************")

# PART 2: Inject noise

# Create a list to store the attribute lists per dataframe
dataset_list <- list()
noiseMIA_list <- list()
dataset_counter = 1

# Per dataset
for(dataset in datasets) {
  
  # Load dataset
  filename = paste0("datasets/", dataset, ".rds")
  df <- readRDS(filename)
  
  # Filter from the MIA list for only this dataset
  dataset_mia <- mia_df_unique %>% 
    filter(dataset_name == dataset) %>%
    arrange(rank)
  
  # Create a list of the most important attributes
  mia_list <- as.list(dataset_mia$most_important_attribute)
  
  # Create a list to store the noise lists per attribute
  attr_list <- list()
  attr_counter = 1
  
  # For each important attribute, process noise
  for(mia in mia_list) {
    
    # Create a list to store the dataframe with noise per level of noise
    noise_list <- list()
    noise_counter = 1
    
    # Add noise to most important attribute in the dataset and create a result file to pull data from later
    for(noise in noise_level){
      
      # Print information on current iteration
      print(paste("Dataset:", dataset))
      print(paste("Most important attribute:", mia))
      print(paste("Noise level:", noise))
      
      # Check if the class of the most important attribute is not a character
      print(paste("MIA's class:", class(df[[mia]])))
      if(class(df[[mia]]) == "factor") {
        df[[mia]] <- as.character(df[[mia]])
        transformed = TRUE
      }
      isNominal = class(df[[mia]]) == "character"
      print(paste("Is the MIA a nominal attribute?", isNominal))
      if(!isNominal){
        # Make the most important attribute the mean of the most important attribute
        mia_mean =  mean(df[[mia]])
        
        # Obtain the standard deviation
        std_dev = sd(df[[mia]])
        print(paste0("MIA's mean: ", mia_mean, ", and standard deviation: ", std_dev))
      }
      
      # Create an auxiliary dataframe to add 100% of noise to
      noise_df <- df
      
      # Injection of noise
      if(noise != 0){
        print("NOISE INJECTION")
        if(isNominal){
          
          # If the most important attribute is nominal, apply formula
          print("MIA is a nominal attribute")
          
          # Obtain frequency of each possible attribute as a probability in an array
          mia_new = df %>%
            group_by(df[[mia]]) %>%
            dplyr::summarise(n = n()) %>%
            mutate(Freq = n/sum(n))
          p = mia_new$Freq
          # Obtain the alpha
          alpha = 1 - exp(-noise)
          # Obtain vector t, used as one hot encoding to turn categorical data into integers
          ## Create a one-hot encoding
          t <- model.matrix(~ factor(df[[mia]]) - 1)
          ## Convert to a matrix and then to a data frame
          t <- as.data.frame(t)
          
          colnames(t) <- sort(unique(df[[mia]]))
          
          # Calculate the new probability in an array
          ## For each row, change the probability of the most important attributes for the new ones obtained through the formula
          ## Obtain a sampling to get a new  value for that column corresponding to the new probabilities
          test_df = data.frame(matrix(ncol = 3, nrow = 0))
          for(row in 1:nrow(t)){
            p1 = alpha*p+(1-alpha)*t[row, ]
            test_df[nrow(test_df) + 1,] <- p1
            sort(unique(df[[mia]]))
            sample(sort(unique(df[[mia]])),1,prob = p1, replace = FALSE)
            noise_df[[mia]][row] = sample(sort(unique(df[[mia]])),1,prob = p1, replace = FALSE)
          }
          
          # After injecting noise, coerce MIA's class to the original, in case it changed
          if(transformed) {
            noise_df[[mia]] <- as.factor(noise_df[[mia]])
          }
        }
        else {
          
          # If the most important attribute is not nominal, apply corresponding formula
          print("MIA is NOT a nominal attribute")
          
          ## Standard deviation * noise value
          dvar = std_dev * noise
          
          # For each value of the most important attribute, add normalized noise and round to two decimals
          noise_df[[mia]] <- sapply(noise_df[[mia]], function(x) rnorm(1, x, dvar))
          noise_df[[mia]] <- round(noise_df[[mia]], digits = 2)
        }
      }
      else {
        print("NO NOISE INJECTED")
        if(isNominal) {
          if(transformed) {
            noise_df[[mia]] <- as.factor(noise_df[[mia]])
          }
        }
      }
      
      # Store df with noise in list
      noise_list[[noise_counter]] <- noise_df
      
      # Advance noise counter
      noise_counter = noise_counter + 1
      
      print("________________")
    }
    
    # Alter names to make it easily readable
    names(noise_list) <- noise_names
    
    # Store df with noise in list
    attr_list[[attr_counter]] <- noise_list
    
    # Advance attribute counter
    attr_counter = attr_counter + 1
  }
  
  # Alter names to make it easily readable
  names(attr_list) <- mia_list
  
  # Store df with noise in list
  dataset_list[[dataset_counter]] <- attr_list
  
  # Advance attribute counter
  dataset_counter = dataset_counter + 1
  print(paste0("Finished with adding noise to dataset: ", dataset))
  
  print("----------------------")
} 

# Alter names to make it easily readable
names(dataset_list) <- datasets


print("****************")
print("RESULTS RECORDED")
print("****************")

# Save the noise list with the new naming convention
noiseMIA_list <- dataset_list

# Part 3: Alter instances

if(FALSE) {  # DISABLED - causes data structure issues
# Create a list to store the attribute lists per dataframe
instancesCM_list <- list()
dataset_counter = 1

# Create a list that contains the indices and order to be altered
index_list <- list()
index_counter = 1

# Per dataset
for(dataset in datasets) {
  
  # Load dataset
  filename = paste0("datasets/", dataset, ".rds")
  df <- readRDS(filename)
  
  # Get the list of attributes to test
  dataset_mia <- mia_df_unique %>% 
    filter(dataset_name == dataset) %>%
    arrange(rank)
  mia_list <- dataset_mia$most_important_attribute
  
  # Create fold indices, generating the training folds and the testing folds
  fold_train_indices <- createFolds(df$class, k = 5, list = TRUE, returnTrain = TRUE)
  fold_test_indices <- lapply(fold_train_indices, function(index) setdiff(1:nrow(df), index))
  
  # Create a list to store the attribute lists per dataframe
  folds_list <- list()
  folds_CM <- list()
  folds_counter = 1
  
  # Loop through the folds and create a dataframe for training
  for (i in 1:length(fold_train_indices)) {
    
    # Get the training indices for the current fold
    train_indices <- fold_train_indices[[i]]
    test_indices <- fold_test_indices[[i]]
    
    # Create the corresponding dataframes using the current fold
    train_df <- df[train_indices, ]
    test_df <- df[test_indices, ]
    
    # Create a frequency table for the class in the 'Training' dataset
    # Calculate the frequency table as a proportion of all values
    print(paste("Dataset:", dataset))
    print("Probability table:")
    print(prop.table(table(train_df$class)))
    
    # Create a list to store the attribute lists per MIA
    attr_list <- list()
    attr_counter = 1
    
    # Loop through each important attribute
    for(mia in mia_list) {
      
      # Create a list to store the noise lists per method
      method_list <- list()
      method_CM <- list()
      method_counter = 1
      if(FALSE) {
      # Train for each method selected (DISABLED - causes issues)
      for(method in methods) {
        # Train to obtain model
        print(paste("Dataset:", dataset))
        print(paste("Method:", method))
        print("BEGINNING TRAINING")
        if(method == "C5.0"){
          fit = caret::train(class ~ ., data = train_df, method = "C5.0")
        }else if(method == "ctree"){
          fit = caret::train(class~., data = train_df, method = "ctree")
        }else if(method == "fda"){
          fit = caret::train(class~., data = train_df, method = "fda")
        }else if(method == "gbm"){
          fit = caret::train(class~., data = train_df, method = "gbm")
        }else if(method == "gcvEarth"){
          fit = caret::train(class ~ ., data = train_df, method = "gcvEarth")
        }else if(method == "JRip"){
          fit = caret::train(class~., data = train_df, method = "JRip")
        }else if(method == "lvq"){
          fit = caret::train(class~., data = train_df, method = "lvq")
        }else if(method == "mlpML"){
          fit = caret::train(class ~ ., data = train_df, method = "mlpML")
        }else if(method == "multinom"){
          fit = caret::train(class ~ ., data = train_df, method = "multinom", trControl = control, tuneGrid = expand.grid(decay = c(0)), MaxNWts = 10000)
        }else if(method == "naive_bayes"){
          fit = caret::train(class ~ ., data = train_df, method = "naive_bayes")
        }else if(method == "PART"){
          fit = caret::train(class ~ ., data = train_df, method = "PART")
        }else if(method == "rbfDDA"){
          fit = caret::train(class ~ ., data = train_df, method = "rbfDDA", trControl = control)
        }else if(method == "rda"){
          fit = caret::train(class~., data = train_df, method = "rda")
        }else if(method == "rf"){
          fit = caret::train(class ~ ., data = train_df, method = "rf")
        }else if(method == "rpart"){
          fit = caret::train(class ~ ., data = train_df, method = "rpart")
        }else if(method == "simpls"){
          fit = caret::train(class ~ ., data = train_df, method = "simpls")
        }else if(method == "svmLinear"){
          fit = caret::train(class ~ ., data = train_df, method = "svmLinear")
        }else if(method == "svmRadial"){
          fit = caret::train(class ~ ., data = train_df, method = "svmRadial")
        }else if(method == "rfRules"){
          fit = caret::train(class ~ ., data = train_df, method = "rfRules")
        }
        #print(fit)
        print("TRAINING SUCCESSFULL")
        
        # Use the current MIA from the loop (already iterating through mia_list)
        # No need to look it up by method since we use the same MIA for all methods
        
        # Skip if no MIA found (safety check)
        if(is.na(mia) || is.null(mia) || length(mia) == 0) {
          print(paste("Skipping method:", method, "- no MIA found"))
          next
        }
        
        # Create a list to store the % instance lists per noise level
        noise_list <- list()
        noise_CM <- list()
        noise_counter = 1
        
        # Control variable to make sure sampling only happens on the first iteration
        semaphore = TRUE
        
        for(noise in noise_level){
          
          # Create a list to store the dataframe with instances per % of instances
          instances_list <- list()
          confMatrix_list <- list()
          instances_counter = 1
          
          for(percent in instances) {
            
            # Print relevant information for the iteration
            print(paste("Dataset:", dataset))
            print(paste("Fold:", i))
            print(paste("Method:", method))
            print(paste0("Noise level: ", noise * 100, "%"))
            print(paste0("Percentage of altered instances: ", percent * 100, "%"))
            
            # Create a new dataframe with the noise from noiseMIA_list
            noiselvl <- paste0("noise_", noise * 100)
            noise_df_orig <- noiseMIA_list[[dataset]][[mia]][[noiselvl]]
            
            # Create a new df with the altered number of instances desired
            instances_df <- test_df
            
            # Add rownames to manipulate dataframes (create indexed versions)
            test_df_indexed <- cbind(index = rownames(test_df), test_df)
            noise_df_indexed <- cbind(index = rownames(noise_df_orig), noise_df_orig)
            instances_df_indexed <- cbind(index = rownames(instances_df), instances_df)
            
            # Determine the indices list to alter the same instances consistently
            if(semaphore) {
              print("Obtain vector of altered instances")
              
              # Number of values to be altered (instances)
              sample_size = round(nrow(instances_df_indexed) * 1, 0)
              
              # Sample of ids that we want from the test df
              indices <- instances_df_indexed$index[sample(nrow(instances_df_indexed), sample_size)]
              
              # Store vector of indices in list
              index_list[[index_counter]] <- indices
              
              # Advance index counter
              index_counter = index_counter + 1
              
              # Set control variable to 0
              semaphore = FALSE
            }
            
            print(paste0("Alter ", percent * 100, "% of instances with noise"))
            
            # Number of values to be altered (instances)
            sample_size = round(nrow(instances_df_indexed) * percent, 0)
            
            # Sample of ids that we want from the test df
            indices <- tail(index_list, n = 1)
            indices <- indices[[1]][1:sample_size]
            
            # Create auxiliary dataframe to extract rows from and clean NAs
            aux_df_indexed <- noise_df_indexed[noise_df_indexed$index %in% indices,] 
            
            # Eliminate corresponding rows
            instances_df_indexed <- instances_df_indexed[!(instances_df_indexed$index %in% indices),]
            
            # Insert noised sample into clean dataframe
            instances_df_indexed <- rbind(instances_df_indexed, aux_df_indexed)
            
            # Reorder dataframe
            instances_df_indexed$index = as.numeric(as.character(instances_df_indexed$index))
            sapply(instances_df_indexed, class)
            instances_df_indexed = instances_df_indexed %>% arrange(index)
            rownames(instances_df_indexed) <- instances_df_indexed[,1]
            
            # Eliminate auxiliary index column
            instances_df <- instances_df_indexed %>% dplyr::select(-index)
            aux_df <- aux_df_indexed %>% dplyr::select(-index)
            test_df_used <- test_df_indexed %>% dplyr::select(-index)
            noise_df_used <- noise_df_indexed %>% dplyr::select(-index)
            
            # Prediction
            print("Calculate prediction")
            
            # Prediction without noise
            predict_unseen = predict(fit, test_df_used)
            predict_unseen
            
            # Prediction with noise 
            noise_predict = predict(fit, instances_df)
            noise_predict
            
            # Confusion matrix
            print("Confusion matrix")
            if(noise == 0) {
              conf_matrix = caret::confusionMatrix(predict_unseen, predict_unseen)
              #print(conf_matrix)
            }else {
              conf_matrix = caret::confusionMatrix(predict_unseen, noise_predict)
              #print(conf_matrix)
            }
            
            # Store df with noise in list
            instances_list[[instances_counter]] <- instances_df
            
            # Store Accuracy and Kappa results for % instances from the confusion matrix
            if(conf_matrix$overall["Kappa"] == "NaN"){ conf_matrix$overall["Kappa"] = 1 }
            #instances_AK[[instances_counter]] <- list(Accuracy = conf_matrix$overall["Accuracy"], Kappa = conf_matrix$overall["Kappa"])
            confMatrix_list[[instances_counter]] <- conf_matrix
            
            # Advance instances counter
            instances_counter = instances_counter + 1
            
            print("-")
          }
          
          # Alter names to make it easily readable
          names(instances_list) <- instances_names
          names(confMatrix_list) <- instances_names
          
          # Store % list with noise in list
          noise_list[[noise_counter]] <- instances_list
          noise_CM[[noise_counter]] <- confMatrix_list
          
          # Advance noise counter
          noise_counter = noise_counter + 1
        }
        
        # Alter names to make it easily readable
        names(noise_list) <- noise_names
        names(noise_CM) <- noise_names
        
        # Store noise list in method
        method_list[[method_counter]] <- noise_list
        method_CM[[method_counter]] <- noise_CM
        
        # Advance method counter
        method_counter = method_counter + 1
      }
    }
    
    # Standardize DF for KNN and GLM
    #train_df =  train_df %>% mutate_if(is.numeric, ~(scale(.) %>% as.vector))
    #test_df =  test_df %>% mutate_if(is.numeric, ~(scale(.) %>% as.vector))
    
    #########
    ## KNN ##
    #########
    if(TRUE){
      method = "knn"
      
      # Train to obtain model
      print(paste("Dataset:", dataset))
      print(paste("Method:", method))
      print("BEGIN TRAINING")
      fit = caret::train(class ~ ., data = df, method="knn", tuneGrid = expand.grid(k = 5:5), preProcess = c("center", "scale"), trControl = control)
      print("TRAINING SUCCESSFULL")
      
      # Use the current MIA from the outer loop (already iterating through mia_list)
      # No need to look it up by method since we use the same MIA for all methods
      
      # Skip if no MIA found (safety check)
      if(is.na(mia) || is.null(mia) || length(mia) == 0) {
        print(paste("Skipping method:", method, "- no MIA found"))
        next
      }
      
      # Create a list to store the % instance lists per noise level
      noise_list <- list()
      noise_CM <- list()
      noise_counter = 1
      
      # Control variable to make sure sampling only happens on the first iteration
      semaphore = TRUE
      
      for(noise in noise_level){
        
        # Create a list to store the dataframe with instances per % of instances
        instances_list <- list()
        confMatrix_list <- list()
        instances_counter = 1
        
        for(percent in instances) {
          
          # Print relevant information for the iteration
          print(paste("Dataset:", dataset))
          print(paste("Fold:", i))
          print(paste("Method:", method))
          print(paste0("Noise level: ", noise * 100, "%"))
          print(paste0("Percentage of altered instances: ", percent * 100, "%"))
          
          # Create a new dataframe with the noise from noiseMIA_list
          noiselvl <- paste0("noise_", noise * 100)
          noise_df_orig <- noiseMIA_list[[dataset]][[mia]][[noiselvl]]
          #noise_df <- noiseMIA_list[[mia]][[noiselvl]]
          
          # Create a new df with the altered number of instances desired
          instances_df <- test_df
          
          # Add rownames to manipulate dataframes
          test_df_indexed <- cbind(index = rownames(test_df), test_df)
          noise_df <- cbind(index = rownames(noise_df), noise_df)
          instances_df_indexed <- cbind(index = rownames(instances_df), instances_df)
          
          # Determine the indices list to alter the same instances consistently
          if(semaphore) {
            print("Obtain vector of altered instances")
            
            # Number of values to be altered (instances)
            sample_size = round(nrow(instances_df) * 1, 0)
            
            # Sample of ids that we want from the test df
            indices <- instances_df$index[sample(nrow(instances_df), sample_size)]
            
            # Store vector of indices in list
            index_list[[index_counter]] <- indices
            
            # Advance index counter
            index_counter = index_counter + 1
            
            # Set control variable to 0
            semaphore = FALSE
          }
          
          print(paste0("Alter ", percent * 100, "% of instances with noise"))
          
          # Number of values to be altered (instances)
          sample_size = round(nrow(instances_df) * percent, 0)
          
          # Sample of ids that we want from the test df
          indices <- tail(index_list, n = 1)
          indices <- indices[[1]][1:sample_size]
          
          # Create auxiliary dataframe to extract rows from and clean NAs
          aux_df <- noise_df[indices,] 
          
          # Eliminate corresponding rows
          instances_df <- instances_df[!(instances_df$index %in% indices),]
          
          # Insert noised sample into clean dataframe
          instances_df <- rbind(instances_df, aux_df)
          
          # Reorder dataframe
          instances_df$index = as.numeric(as.character(instances_df$index))
          sapply(instances_df, class)
          instances_df = instances_df %>% arrange(index)
          rownames(instances_df) <- instances_df[,1]
          
          # Eliminate auxiliary index column
          instances_df <- instances_df %>% dplyr::select(-index)
          aux_df <- aux_df_indexed %>% dplyr::select(-index)
          test_df_used <- test_df_indexed %>% dplyr::select(-index)
          noise_df_used <- noise_df_indexed %>% dplyr::select(-index)
          
          # Prediction
          print("Calculate prediction")
          
          # Prediction without noise
          predict_unseen = predict(fit, test_df_used)
          predict_unseen
          
          # Prediction with noise 
          noise_predict = predict(fit, instances_df)
          noise_predict
          
          # Confusion matrix
          print("Confusion matrix")
          if(noise == 0) {
            conf_matrix = caret::confusionMatrix(predict_unseen, predict_unseen)
            #print(conf_matrix)
          }else {
            conf_matrix = caret::confusionMatrix(predict_unseen, noise_predict)
            #print(conf_matrix)
          }
          
          # Store df with noise in list
          instances_list[[instances_counter]] <- instances_df
          
          # Store Accuracy and Kappa results for % instances from the confusion matrix
          if(conf_matrix$overall["Kappa"] == "NaN"){ conf_matrix$overall["Kappa"] = 1 }
          #instances_AK[[instances_counter]] <- list(Accuracy = conf_matrix$overall["Accuracy"], Kappa = conf_matrix$overall["Kappa"])
          confMatrix_list[[instances_counter]] <- conf_matrix
          
          # Advance instances counter
          instances_counter = instances_counter + 1
          
          print("-")
        }
        
        # Alter names to make it easily readable
        names(instances_list) <- instances_names
        names(confMatrix_list) <- instances_names
        
        # Store % list with noise in list
        noise_list[[noise_counter]] <- instances_list
        noise_CM[[noise_counter]] <- confMatrix_list
        
        # Advance noise counter
        noise_counter = noise_counter + 1
      }
      
      # Alter names to make it easily readable
      names(noise_list) <- noise_names
      names(noise_CM) <- noise_names
      
      # Store noise list in method
      method_list[[method_counter]] <- noise_list
      method_CM[[method_counter]] <- noise_CM
      
      # Advance method counter
      method_counter = method_counter + 1
      
    }
    #########
    ## GLM ##
    #########
    if(TRUE){
      method = "bayesglm"
      
      # Train to obtain model
      print(paste("Dataset:", dataset))
      print(paste("Method:", method))
      print("BEGINNING TRAINING")
      fit = caret::train(class ~ ., data = train_df, method="bayesglm", trControl = control)
      #print(fit)
      print("TRAINING SUCCESSFULL")
      
      # Select the MIA we will get the noise from 
      mia <- subset(mia_df, dataset_name == dataset & technique == method)$most_important_attribute
      #mia <- mia_df
      
      # Create a list to store the % instance lists per noise level
      noise_list <- list()
      noise_CM <- list()
      noise_counter = 1
      
      # Control variable to make sure sampling only happens on the first iteration
      semaphore = TRUE
      
      for(noise in noise_level){
        
        # Create a list to store the dataframe with instances per % of instances
        instances_list <- list()
        confMatrix_list <- list()
        instances_counter = 1
        
        for(percent in instances) {
          
          # Print relevant information for the iteration
          print(paste("Dataset:", dataset))
          print(paste("Fold:", i))
          print(paste("Method:", method))
          print(paste0("Noise level: ", noise * 100, "%"))
          print(paste0("Percentage of altered instances: ", percent * 100, "%"))
          
          # Create a new dataframe with the noise from noiseMIA_list
          noiselvl <- paste0("noise_", noise * 100)
          noise_df_orig <- noiseMIA_list[[dataset]][[mia]][[noiselvl]]
          #noise_df <- noiseMIA_list[[mia]][[noiselvl]]
          
          # Create a new df with the altered number of instances desired
          instances_df <- test_df
          
          # Add rownames to manipulate dataframes
          test_df_indexed <- cbind(index = rownames(test_df), test_df)
          noise_df <- cbind(index = rownames(noise_df), noise_df)
          instances_df_indexed <- cbind(index = rownames(instances_df), instances_df)
          
          # Determine the indices list to alter the same instances consistently
          if(semaphore) {
            print("Obtain vector of altered instances")
            
            # Number of values to be altered (instances)
            sample_size = round(nrow(instances_df) * 1, 0)
            
            # Sample of ids that we want from the test df
            indices <- instances_df$index[sample(nrow(instances_df), sample_size)]
            
            # Store vector of indices in list
            index_list[[index_counter]] <- indices
            
            # Advance index counter
            index_counter = index_counter + 1
            
            # Set control variable to 0
            semaphore = FALSE
          }
          
          print(paste0("Alter ", percent * 100, "% of instances with noise"))
          
          # Number of values to be altered (instances)
          sample_size = round(nrow(instances_df) * percent, 0)
          
          # Sample of ids that we want from the test df
          indices <- tail(index_list, n = 1)
          indices <- indices[[1]][1:sample_size]
          
          # Create auxiliary dataframe to extract rows from and clean NAs
          aux_df <- noise_df[indices,] 
          
          # Eliminate corresponding rows
          instances_df <- instances_df[!(instances_df$index %in% indices),]
          
          # Insert noised sample into clean dataframe
          instances_df <- rbind(instances_df, aux_df)
          
          # Reorder dataframe
          instances_df$index = as.numeric(as.character(instances_df$index))
          sapply(instances_df, class)
          instances_df = instances_df %>% arrange(index)
          rownames(instances_df) <- instances_df[,1]
          
          # Eliminate auxiliary index column
          instances_df <- instances_df %>% dplyr::select(-index)
          aux_df <- aux_df_indexed %>% dplyr::select(-index)
          test_df_used <- test_df_indexed %>% dplyr::select(-index)
          noise_df_used <- noise_df_indexed %>% dplyr::select(-index)
          
          # Prediction
          print("Calculate prediction")
          
          # Prediction without noise
          predict_unseen = predict(fit, test_df_used)
          predict_unseen
          
          # Prediction with noise 
          noise_predict = predict(fit, instances_df)
          noise_predict
          
          # Confusion matrix
          print("Confusion matrix")
          if(noise == 0) {
            conf_matrix = caret::confusionMatrix(predict_unseen, predict_unseen)
            #print(conf_matrix)
          }else {
            conf_matrix = caret::confusionMatrix(predict_unseen, noise_predict)
            #print(conf_matrix)
          }
          
          # Store df with noise in list
          instances_list[[instances_counter]] <- instances_df
          
          # Store Accuracy and Kappa results for % instances from the confusion matrix
          if(conf_matrix$overall["Kappa"] == "NaN"){ conf_matrix$overall["Kappa"] = 1 }
          #instances_AK[[instances_counter]] <- list(Accuracy = conf_matrix$overall["Accuracy"], Kappa = conf_matrix$overall["Kappa"])
          confMatrix_list[[instances_counter]] <- conf_matrix
          
          # Advance instances counter
          instances_counter = instances_counter + 1
          
          print("-")
        }
        
        # Alter names to make it easily readable
        names(instances_list) <- instances_names
        names(confMatrix_list) <- instances_names
        
        # Store % list with noise in list
        noise_list[[noise_counter]] <- instances_list
        noise_CM[[noise_counter]] <- confMatrix_list
        
        # Advance noise counter
        noise_counter = noise_counter + 1
      }
      
      # Alter names to make it easily readable
      names(noise_list) <- noise_names
      names(noise_CM) <- noise_names
      
      # Store noise list in method
      method_list[[method_counter]] <- noise_list
      method_CM[[method_counter]] <- noise_CM
      
    }
    
    # End of techniques
    # Alter names to make it easily readable
    #names(method_list) <- method_names
    #names(method_CM) <- method_names
    
    names(method_list) <- c("knn")
    names(method_CM) <- c("knn")
    # Store noise list in method
    attr_list[[attr_counter]] <- method_list
    
    # Store confusion matrices for this attribute
    # Create a nested list structure for this attribute
    # Initialize if needed
    if(!exists("attr_CM")) {
      attr_CM <- list()
      attr_CM_counter <- 1
    }
    attr_CM[[attr_counter]] <- method_CM
    
    # Advance attribute counter
    attr_counter = attr_counter + 1
    
    } # End of attribute loop
    
    # Alter names to make it easily readable  
    names(attr_list) <- mia_list
    
    # Store attribute list in fold
    folds_list[[folds_counter]] <- attr_list
    folds_CM[[folds_counter]] <- attr_CM
    
    # Advance fold counter
    folds_counter = folds_counter + 1
    
    # Reset the attribute_CM for the next fold
    rm(attr_CM, pos=.GlobalEnv)
  }
  
  # Alter names to make it easily readable
  names(folds_list) <- fold_names
  names(folds_CM) <- fold_names
  
  # Store fold list in dataset
  instancesCM_list[[dataset_counter]] <- folds_list
  
  # Advance dataset counter
  dataset_counter = dataset_counter + 1
  
  print("Altered instances with noise recorded")
  print("----------------")
}

# Alter names to make it easily readable
names(instancesCM_list) <- datasets

# Record full list
#saveRDS(dataset_list, file = "results/instances/instances_list.rds")
#saveRDS(dataset_CM, file = "results/instances/instancesCM_list.rds")

print("****************")
print("RESULTS RECORDED")
print("****************")
}  # End of Part 3 (disabled)

# Part 4: Confusion matrices (DISABLED)
if(FALSE) {
confusion_list <- list()
cm_counter = 1

for(dataset in datasets) {
  print(paste0("Dataset: ", dataset))
  # Load dataset
  filename = paste0("datasets/", dataset, ".rds")
  df <- readRDS(filename)
  
  # Get the list of attributes to test
  dataset_mia <- mia_df_unique %>% 
    filter(dataset_name == dataset) %>%
    arrange(rank)
  mia_list <- dataset_mia$most_important_attribute
  
  # Set counter to 1
  attrCM_list <- list()
  a_counter = 1
  
  # Loop through each attribute
  for(mia in mia_list) {
    print(paste0("Attribute: ", mia))
    
    # Set counter to 1
    methodCM_list <- list()
    m_counter = 1
    
    for(method in c("knn")) {
      print(paste0("Model: ", method))
      # Set counter to 1
      noiseCM_list <- list()
      n_counter = 1
      
      for(noise in noise_names) {
        print(paste0("Noise: ", noise))
        # Set counter to 1
        percentagesCM_list <- list()
        p_counter = 1
        
        for(instance in instances_names){
          print(paste0("Instance: ", instance))
          item_list <- list()
          
          # Obtain each confusion matrix from all five folds
          confusion_matrix_1 <- instancesCM_list[[dataset]][["Fold_1"]][[mia]][[method]][[noise]][[instance]]
          confusion_matrix_2 <- instancesCM_list[[dataset]][["Fold_2"]][[mia]][[method]][[noise]][[instance]]
          confusion_matrix_3 <- instancesCM_list[[dataset]][["Fold_3"]][[mia]][[method]][[noise]][[instance]]
          confusion_matrix_4 <- instancesCM_list[[dataset]][["Fold_4"]][[mia]][[method]][[noise]][[instance]]
          confusion_matrix_5 <- instancesCM_list[[dataset]][["Fold_5"]][[mia]][[method]][[noise]][[instance]]
          
          # Store confusion matrices in a list
          confusion_matrices <- list(confusion_matrix_1$table, confusion_matrix_2$table, confusion_matrix_3$table, confusion_matrix_4$table, confusion_matrix_5$table)
          
          # Initialize a sum matrix with zeros
          sum_matrix <- matrix(0, nrow = nrow(confusion_matrix_1$table), ncol = ncol(confusion_matrix_1$table))
          
          # Add confusion matrices to the sum matrix element-wise
          for (i in 1:length(confusion_matrices)) {
            sum_matrix <- sum_matrix + confusion_matrices[[i]]
          }
          print("Total Sum Confusion Matrix:")
          print(sum_matrix)
          
          print(paste("Checking matrices for:", dataset, mia, method, noise, instance))
          print(paste0("CM1: ", !is.null(confusion_matrix_1) , " has table ", !is.null(confusion_matrix_1) && !is.null(confusion_matrix_1$table)))
          
          # Calculate the average by dividing by the total number of confusion matrices
          average_matrix <- sum_matrix / length(confusion_matrices)
          
          if(noise == "noise_0") {
            
            kappa <- 1
            accuracy <- 1
            
          } else if(instance == "0") {
            
            kappa <- 1
            accuracy <- 1
            
          } else {
            
            # Obtain average Accuracy and Kappa
            # Compute the observed and expected agreements
            total_obs <- sum(average_matrix)
            P_o <- sum(diag(average_matrix)) / total_obs
            P_e <- sum(rowSums(average_matrix) * colSums(average_matrix)) / (total_obs^2)
            
            # Calculate Cohen's Kappa
            kappa <- (P_o - P_e) / (1 - P_e)
            
            # Calculate accuracy from a confusion matrix
            accuracy <- sum(diag(average_matrix)) / sum(average_matrix)
            
            if(kappa == "NaN") {kappa <- 1}
          }
          
          # Print relevant data
          print(paste0("Attribute: ", mia))
          print(paste0("Technique: ", method))
          print(paste0("Noise level: ", noise))
          print(paste0("Instance percentage: ", instance))
          
          # Print the average confusion matrix
          print("Average Confusion Matrix:")
          print(average_matrix)
          
          # Print the average Cohen's Kappa
          print("Average Cohen's Kappa:")
          print(kappa)
          
          # Print the accuracy
          print("Accuracy:")
          print(accuracy)
          
          # Store elements in list
          item_list[[1]] <- average_matrix
          item_list[[2]] <- kappa
          item_list[[3]] <- accuracy
          names(item_list) <-  c("confusion_matrix", "kappa", "accuracy")
          
          percentagesCM_list[[p_counter]] <- item_list
          p_counter = p_counter + 1
          
          print("---")
        }
        
        names(percentagesCM_list) <- instances_names
        print(instances_names)
        
        noiseCM_list[[n_counter]] <- percentagesCM_list
        n_counter = n_counter + 1
      }
      
      names(noiseCM_list) <- noise_names
      
      methodCM_list[[m_counter]] <- noiseCM_list
      m_counter = m_counter + 1
    }
    
    names(methodCM_list) <- c("knn")
    
    attrCM_list[[a_counter]] <- methodCM_list
    a_counter = a_counter + 1
    
    print(paste0("Recorded matrices, kappa and accuracy for attribute: ", mia))
    print("---")
  }
  
  # Store attribute list in dataset
  names(attrCM_list) <- mia_list
  confusion_list[[cm_counter]] <- attrCM_list
  cm_counter = cm_counter + 1
  
  print(paste0("Recorded matrices, kappa and accuracy for dataset: ", dataset))
  print("----------------")
}

names(confusion_list) <- datasets

print("****************")
print("RESULTS RECORDED")
print("****************")
}  # End of Part 4 (disabled)

# Save the colinearity results with the specified naming convention
saveRDS(mia_df, file = "results/collin_mia_df.rds")
saveRDS(noiseMIA_list, file = "results/collin_noise_list.rds")
#saveRDS(instancesCM_list, file = "results/collin_instancesCM_list.rds")
#saveRDS(confusion_list, file = "results/collin_confusion_matrices.rds")

print("All colinearity results saved successfully!")

