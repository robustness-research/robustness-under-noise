#!/bin/bash

arguments=("qsar-biodeg" "tic-tac-toe" "vowel" "waveform-5000" "wdbc" "wilt") #datasets

#  

# Done
# "analcatdata_authorship" "badges2" "banknote" "blood-transfusion-service-center" "breast-w"
# "cardiotocography" "climate-model-simulation-crashes" "cmc" "credit-g" "diabetes"
# "eucalyptus" "iris" "kc1" "liver-disorders" "mfeat-factors"
# "mfeat-karhunen" "mfeat-zernike" "ozone-level-8hr" "pc4" "phoneme"
# "qsar-biodeg" "tic-tac-toe" "vowel" "waveform-5000" "wdbc" "wilt"
arguments2=("C5.0" "ctree" "fda" "gbm" "gcvEarth" "JRip" "lvq" "mlpML" "multinom" "naive_bayes" "PART" "rbfDDA" "rda" "rf" "rpart" "simpls" "svmLinear" "svmRadial" "rfRules" "knn" "bayesglm") # ML techniques

for dataset in "${arguments[@]}"; do
    nohup Rscript markdown/Feature_Colinearity.R "$dataset" > output/rankings/"colinearity_output_${dataset}.log" 2>&1 &
    for method in "${arguments2[@]}"; do
        nohup Rscript markdown/Feature_Ranking.R "$dataset" "$method" > output/rankings/"ranking_output_${dataset}_${method}.log" 2>&1 &
    done
done