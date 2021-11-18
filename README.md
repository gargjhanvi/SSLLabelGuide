# R Package - SSLLabelGuide

#Intended use


Given n1 labelled samples and m1 unlabeled samples (m1 >> n1) and suppose user can label L more samples (from m1), the goal of this package is to suggest which L sample point user should label, in the aim to increase the overall prediction accuracy of the unlabeled data. we first compare the multiclass SVM and K nearest neighbor algorithm on the labeled data set by repeating  5 - fold cross-validation technique. By comparing average cross-validation error, we choose an algorithm from multiclass SVM and K nearest neighbor  The **"SSLconf"** function in this package arranges the unlabeled data points on the basis of the confidence in prediction. This function also outputs plot containing  points predicted with high, moderate and low confidence. The **SSLBudget** function suggest the L data points user should label. We cluster the low confidence points into L clusters and uniformly suggest one point from each cluster. The **SSLpredict** predicts the labels of the unlabeled data. The **Comparison_error** function compares the error (on known data sets) before including the labels of L points suggested by SSLBudget in the training data set and after including them. This package guides the user how to choose data points to perform experiment on in order to get better predictions. 

# Prospective Plan

1) Create some test examples 
2) Modify SSLBudget function based on tests 
3) Make Vignette
4) Create plots in ggplot instead of using base plot


