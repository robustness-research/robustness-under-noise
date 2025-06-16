#instancesCM_list = readRDS("results/instances/instancesCM_list.rds")
instancesCM_list = readRDS("results/instances/instancesCM_list_popular.rds")
# Fix double nesting of method names like "knn"
for (dataset_name in names(instancesCM_list)) {
  dataset <- instancesCM_list[[dataset_name]]
  
  for (fold_name in names(dataset)) {
    fold <- dataset[[fold_name]]
    
    for (method_name in names(fold)) {
      method_entry <- fold[[method_name]]
      
      # Check if method contains itself as a redundant nested list
      if (is.list(method_entry) &&
          length(method_entry) == 1 &&
          method_name %in% names(method_entry)) {
        
        # Replace double-nested structure with inner one
        fold[[method_name]] <- method_entry[[method_name]]
      }
    }
    
    # Save back updated fold
    dataset[[fold_name]] <- fold
  }
  
  # Save back updated dataset
  instancesCM_list[[dataset_name]] <- dataset
}
#saveRDS(instancesCM_list, file = "results/instances/instancesCM_list.rds")
saveRDS(instancesCM_list, file = "results/instances/instancesCM_list_popular.rds")