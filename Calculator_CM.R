# /usr/bin/env/Rscript

# Packages that need to be loaded
pacman::p_load(citation, dplyr, data.table)

# Set the seed to make the experiment reproducible
set.seed(1)

# Load important data
## Load files
args <- commandArgs(trailingOnly = TRUE)
datasets <- args
fold_names <- readRDS("files/folds.rds")
method_names <- readRDS("files/method_names.rds")
#method_names = c("C5.0", "ctree", "fda", "gbm", "gcvEarth", "JRip", "lvq", "mlpML", "multinom", "naive_bayes", "PART", "rbfDDA", "rda", "rf", "rpart", "simpls", "svmRadial", "rfRules", "knn", "bayesglm") # Without SVMLinear (for popular)
noise_names <- readRDS("files/noise_names.rds")
instances_names = readRDS("files/instances_names.rds")
#instances_names = append(readRDS("files/instances_names.rds"), c("25", "75")) # Without quartiles (for popular)

## Load results
mia_df <- readRDS("results/most_important_attr/mia_df.rds")
noiseMIA_list <- readRDS("results/noise/noise_list.rds")
#instancesCM_list = readRDS("results/instances/instancesCM_list.rds")
instancesCM_list = readRDS("results/instances/instancesCM_list_popular.rds")

confMatrices_list <- list()
cm_counter = 1

for(dataset in datasets) {
  print(paste0("Dataset: ", dataset))
  # Load dataset
  filename = paste0("datasets/", dataset, ".rds")
  df <- readRDS(filename)

  # Set counter to 1
  methodCM_list <- list()
  m_counter = 1
  
  for(method in method_names){
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
        confusion_matrix_1 <- instancesCM_list[[dataset]][["Fold_1"]][[method]][[noise]][[instance]]
        confusion_matrix_2 <- instancesCM_list[[dataset]][["Fold_2"]][[method]][[noise]][[instance]]
        confusion_matrix_3 <- instancesCM_list[[dataset]][["Fold_3"]][[method]][[noise]][[instance]]
        confusion_matrix_4 <- instancesCM_list[[dataset]][["Fold_4"]][[method]][[noise]][[instance]]
        confusion_matrix_5 <- instancesCM_list[[dataset]][["Fold_5"]][[method]][[noise]][[instance]]
        
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
     
        print(paste("Checking matrices for:", dataset, method, noise, instance))
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
  
  names(methodCM_list) <- method_names
  
  confMatrices_list[[cm_counter]] <- methodCM_list
  cm_counter = cm_counter + 1
  
  #filename1 = paste0("results/conf_matrices/by_dataset/", dataset, "_cm.rds")
  filename1 = paste0("results/conf_matrices/by_dataset_popular/", dataset, "_cm_popular.rds")
  saveRDS(confMatrices_list, file = filename1)
  
  print(paste0("Recorded matrices, kappa and accuracy for dataset: ", dataset))
  print("----------------")
}

names(confMatrices_list) <- datasets
#saveRDS(confMatrices_list, file = "results/conf_matrices/confusion_matrices.rds")
#saveRDS(confMatrices_list, file = "results/conf_matrices/confusion_matrices_popular.rds")

print("****************")
print("RESULTS RECORDED")
print("****************")

