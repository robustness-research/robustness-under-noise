# /usr/bin/env/Rscript

# Create data lists
datasets <- c("analcatdata_authorship", "badges2", "banknote", "blood-transfusion-service-center", "breast-w", "cardiotocography", "climate-model-simulation-crashes", "cmc", "credit-g", "diabetes", "eucalyptus", "iris", "kc1", "liver-disorders", "mfeat-karhunen", "mfeat-zernike", "ozone-level-8hr", "pc4", "phoneme", "qsar-biodeg", "tic-tac-toe", "vowel", "waveform-5000", "wdbc", "wilt") # Dataset names
folds <- c("Fold_1", "Fold_2", "Fold_3", "Fold_4", "Fold_5") # Set names to each fold
methods = c("C5.0", "ctree", "fda", "gbm", "gcvEarth", "JRip", "lvq", "mlpML", "multinom", "naive_bayes", "PART", "rbfDDA", "rda", "rf", "rpart", "simpls", "svmLinear", "svmRadial", "rfRules") #ML techniques in alphabetical order and excluding knn and glmBayes (memory issues if executed in the same for loop)
method_names = c("C5.0", "ctree", "fda", "gbm", "gcvEarth", "JRip", "lvq", "mlpML", "multinom", "naive_bayes", "PART", "rbfDDA", "rda", "rf", "rpart", "simpls", "svmLinear", "svmRadial", "rfRules", "knn", "bayesglm") # ML techniques with knn and glmBayes
control = trainControl(method = "none", number = 1) # Control the computational nuances of the 'train' function
noise_level = c(0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1)
noise_names = c("noise_0", "noise_5", "noise_10", "noise_20", "noise_30", "noise_40", "noise_50", "noise_60", "noise_70", "noise_80", "noise_90", "noise_100")
instances <- c(0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1) # Percentage of instances to be altered
instances_names <- c("0", "10", "20", "30", "40", "50", "60", "70", "80", "90", "100") # Set names for percentages of instances to be altered

# Save data lists
saveRDS(datasets, file = "files/datasets.rds")
saveRDS(folds, file = "files/folds.rds")
saveRDS(methods, file = "files/methods.rds")
saveRDS(method_names, file = "files/method_names.rds")
saveRDS(control, file = "files/control.rds")
saveRDS(noise_level, file = "files/noise.rds")
saveRDS(noise_names, file = "files/noise_names.rds")
saveRDS(instances, file = "files/instances.rds")
saveRDS(instances_names, file = "files/instances_names.rds")
