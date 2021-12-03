# R Package - SSLLabelGuide

### Introduction

This package guides the user how to choose data points to perform experiment on in order to get better predictions. 
Given n1 labelled samples and m1 unlabeled samples (m1 >> n1) and suppose user can label L more samples (from m1), the goal of this package is to suggest which L sample point user should label, in the aim to increase the overall prediction accuracy of the unlabeled data. we first compare the multiclass SVM, K nearest neighbor and Multiclass logistic regression algorithm on the labeled data set by repeating  5 - fold cross-validation technique. By comparing average cross-validation error, we choose an algorithm from Multiclass SVM, K nearest neighbor and Multiclass Logistic regression The **"SSLconf"** function in this package arranges the unlabeled data points on the basis of the confidence in prediction. This function also outputs plot containing  points predicted with high, moderate and low confidence. The **SSLBudget** function suggest the L data points user should label. We cluster the points whose confidence in prediction is less than a certain threshold into L clusters and uniformly suggest one point from each cluster. The **SSLpredict** predicts the labels of the unlabeled data. 

### Installation instructions
Run the following in R:

* devtools::install_github("gargjhanvi/SSLLabelGuide")
  
  
### Usuage

There are 8 functions in this package:

*  **standardizeX** - standardizeX function standardize a matrix X, i.e center and scale X and returns $tilde - standardized X, $means - the mean of columns of X and $weights - the weights after centering of X but before scaling 

```{r}
X <- matrix(c(3,2,4,20,3,42,12,61),4,2)
standardizeX(X)
#output
#$tilde
#           [,1]       [,2]
#[1,] -0.5747049 -1.1411712
#[2,] -0.7099296  0.5382883
#[3,] -0.4394802 -0.7536036
#[4,]  1.7241147  1.3564865

#$weights
#[1]  7.39510 23.22176

#$means
#[1]  7.25 29.50

```
 
* 2) **Algorithm** - If Xtilde is a n * p matrix of n data points with labels in vector Y and K is the number of classes (If K is NULL, it calculates K using Y), then Algorithm(Xtilde, Y, K) returns $error_svm - the 5 fold cross validation error when we use multiclass support vector machine, $error_knn - the 5  fold cross-validation error when K-nearest neighbor is used with K = round(sqrt(N)) where N is the number of training data points and $error_logistic - the 5  fold cross-validation error when Multiclass logistic regression is used with default numIter = 50, eta = 0.1, lambda = 1


```{r}

  Xtilde <- diamonds[1:2000, -c(1,2,3,4) ]
  Y <-  as.numeric(as.factor(as.matrix(diamonds[1:2000, 2 ])))
  Algorithm(Xtilde, Y, K=NULL)
#output
# $error_svm
# [1] 600
# 
# $error_knn
# [1] 1153
# 
# $error_logistic
# [1] 771

```

* 3) **ChooseAlgorithm** - Xtilde, Y, K  be as in the Algorithm function. ChooseAlgorithm(Xtilde, Y, K)  uses the output of the Algorithm function and choose the algorithm among multiclass SVM, multiclass logistic and KNN which has the least average cross validation error(The algorithm funtion is repeated many times) on the labelled data set. It outputs a string among one of the three "KNN","Logistic","SVM" depending on which algorithm performs better.
```{r}
ChooseAlgorithm(Xtilde, Y, K=NULL)
#output
# [1] "SVM"
```


* 4) **SSLconf** - The SSLconf function divides the unlabeled data into matrices "High", "Low", "Average" based on the confidence in predicting them. It inputs a n * p matrix of n data points **X**, a numeric n vector **Y** of labels of data points in X, a m * p matrix of m unlabeled  data points Z and plots which takes TRUE or FALSE depending on whether we want to display plots . The SSLconf(X, Y, Z, K, plots = TRUE) returns four matrices 
* $High - A matrix containing data points of Z that are predicted with high confidence

* $Low - A matrix containing data points of Z that are predicted with low confidence 

* $Average - A matrix containing data points of Z that are predicted with moderate confidence

* $Remat - A matrix containing data points of Z that should possibly be relabeled

and plots the points in matrices High, Low, Average.

```{r}
X <- diamonds[1:2000, -c(1,2,3,4) ]
Y <-  as.numeric(as.factor(as.matrix(diamonds[1:2000, 2 ])))
Z <-   diamonds[2001:4000, -c(1,2,3,4) ]

out <- SSLconf(X,Y,Z)

# 
# 
# 
head(out$High , 5)
#      depth table price    x    y    z
# 2003  61.5  54.0  3101 5.84 5.86 3.60
# 2004  62.1  55.0  3101 5.94 5.98 3.70
# 2008  61.3  56.0  3102 5.77 5.82 3.55
# 2020  61.9  56.0  3105 5.91 5.94 3.67
# 2021  60.7  56.0  3105 5.97 6.03 3.64
# 
 head(out$Low , 5)
# 
#      depth table price    x    y    z
# 2001  63.0  59.0  3099 6.16 6.10 3.86
# 2002  62.3  57.0  3101 6.20 6.22 3.87
# 2006  60.5  57.0  3102 5.85 5.89 3.55
# 2011  61.6  57.0  3103 5.57 5.64 3.45
# 2012  62.2  56.0  3103 5.70 5.75 3.56
# 
# 
 head(out$Average , 5)
# 
# 
#      depth table price    x    y    z
# 2007  62.6  55.0  3102 6.11 6.13 3.83
# 2009  62.6  54.0  3102 6.16 6.23 3.88
# 2010  61.9  53.0  3103 6.04 6.08 3.75
# 2016  55.9  62.0  3104 6.35 6.42 3.57
# 2017  64.7  58.0  3104 5.94 6.03 3.87
# 
 head(out$Remat, 5)
# 
#      depth table price    x    y    z
# 2001  63.0  59.0  3099 6.16 6.10 3.86
# 2002  62.3  57.0  3101 6.20 6.22 3.87
# 2006  60.5  57.0  3102 5.85 5.89 3.55
# 2007  62.6  55.0  3102 6.11 6.13 3.83
# 2010  61.9  53.0  3103 6.04 6.08 3.75


```

*  **SSLBudget** - X, Y, Z, K are as in function SSLconf. Let L be the number of points user can label for us from Z. The SSLBudget(X, Y, L, Z, K) function outputs  a matrix containing L data points that user should label in order to increase the overall prediction accuracy. It also plots the points that the funtion outputs.


```{r}


out <- SSLBudget(as.matrix(X),Y,5,as.matrix(Z),K = NULL)
#out

#      [,1] [,2] [,3] [,4] [,5] [,6]
# [1,] 62.7   56  407 4.27 4.28 2.68
# [2,] 63.1   55  567 4.42 4.46 2.80
# [3,] 62.5   56 3262 6.30 6.18 3.90
# [4,] 62.8   58 3399 6.28 6.33 3.96
# [5,] 61.9   57 3127 5.79 5.75 3.57

```


*  **SSLpredict** -Let X,Y,Z be as in the SSLconf funtion. SSLpredict function predicts the labels of the unlabeled data. SSLpredict(X, Y, Z, K) returns a vector containing the labels of data points in Z 
```{r}


out <- SSLpredict(as.matrix(X),Y,as.matrix(Z),K=NULL)
#out

#     depth table price    x    y    z SVM_fitting
# 1    61.5  54.0  3101 5.84 5.86 3.60           3
# 2    62.1  55.0  3101 5.94 5.98 3.70           3
# 3    61.3  56.0  3102 5.77 5.82 3.55           3
# 4    61.9  56.0  3105 5.91 5.94 3.67           3
# 5    60.7  56.0  3105 5.97 6.03 3.64           3
# 6    65.6  58.0  3105 6.05 5.95 3.95           1
# 7    61.2  56.0  3105 5.75 5.68 3.50           3
# 8    59.6  59.0  3105 6.38 6.24 3.76           4
# 9    62.1  54.0  3107 5.92 5.96 3.69           3
# 10   61.3  55.0   561 4.41 4.43 2.71           3
# 11   61.0  59.0   561 4.42 4.46 2.71           4

```


*  **LRMultiClass** -LRMultiClass function that implements multi-class logistic regression. It inputs training data points X, labels of training data set Y, test data points Xt, labels of test data set Yt and returns  $beta - p x K matrix of estimated beta values after numIter iterations    and  $error_test - (numIter + 1) length vector of testing error % at each iteration (+ starting value)

```{r}

X1 <- matrix(rnorm(50, -3, 1), 50, 1)
Y1 <- matrix(c(0), 50, 1)
X2 <- matrix(rnorm(50, 3, 1), 50, 1)
Y2 <- matrix(c(1), 50, 1)
X3 <- matrix(rnorm(50, 0, 1), 50, 1)
Y3 <- matrix(c(2), 50, 1)
X <- c(X1, X2, X3)
Y <- c(Y1, Y2, Y3)
X <- as.matrix(X)
Y <- as.matrix(Y)
random <- sample(nrow(X))
X <- as.matrix(X[random, ]) # training data from two normal distributions
Y <- as.matrix(Y[random, ]) 

# creating test data
X1t <- matrix(rnorm(10, -3, 1), 10, 1)
Y1t <- matrix(c(0), 10, 1)
X2t <- matrix(rnorm(10, 3, 1), 10, 1)
Y2t <- matrix(c(1), 10, 1)
X3t <- matrix(rnorm(10, 0, 1), 10, 1)
Y3t <- matrix(c(2), 10, 1)
Xt <- c(X1t, X2t, X3t)
Yt <- c(Y1t, Y2t, Y3t)
Xt <- as.matrix(Xt)
Yt <- as.matrix(Yt)
random <- sample(nrow(Xt))
Xt <- as.matrix(Xt[random, ]) # testing data from the same two normal distributions
Yt <- as.matrix(Yt[random, ])

# adding columns of 1's to training and testing data
colX1 <- rep(1, nrow(X))
X <- cbind(colX1, X)

colXt1 <- rep(1, nrow(Xt))
Xt <- cbind(colXt1, Xt)
output2 <- LRMultiClass(X, Y, Xt, Yt)

#output2

# $beta
#           [,1]      [,2]        [,3]
# [1,] -1.995264 -2.001957 0.585595954
# [2,] -1.746854  1.772867 0.003120814
# 
# $error_test
#  [1] 20 10  9  8  6  6  4  4  3  2  2  2  2  2  2  2  2  2  2  2  2  2  2  1  1  1  1  1  1  1
# [31]  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1
```



* **Mykmeans** - Mykmeans function that implements K-means algorithm. It inputs the n*p matrix X of n  data points that we want to  cluster and the number of clusters K and returns a vector of cluster labels. Mykmeans(X,K) will return a vector of cluster labels

```{r}


A <- rnorm(10, 0, 1) 
B <- rnorm(10, 1, 1)
C <- rnorm(10, -1, 1)

#Randomly shuffling the samples from normal distributions and storing in X 

X <- c(A, B, C)
X <- as.matrix(X)
set.seed(1) ## make reproducible here, but not if generating many random samples
random <- sample(nrow(X))
X <- as.matrix(X[random, ])



#Applying MyKmeans function with arbitrary M and 1000 iterations

MyKmeans(X, 3)

output
# [1] 2 3 3 3 2 2 2 1 3 2 3 3 3 3 2 3 2 2 1 3 1 1 3 1 3 1 2 3 2 1
```
