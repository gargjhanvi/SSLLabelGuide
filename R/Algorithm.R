# We will use cross validation technique to compare SVM and KNN on the labelled data.

# X - n1 * p matrix of labelled data points. Here the rows contains the n1 labelled data points. Each data point belongs to R^p
# Y - n1 vector of labels for data points in X
# C - number of folds for k-fold cross-validation, default is 5.
# K - number of classes

# Algorithm function compares SVM and KNN using cross-validation
Algorithm <- function(X, Y, C = 5, K = NULL) {

  # n1 is the number of rows in X
  n1 <- nrow(X)

  # K = max(Y) if K = NULL
  if (is.null(K) == T) {
    K <- length(unique(Y))
  }


  # fold_ids stores the fold_ids for cross validation
  fold_ids <- sample(rep(1:C, length.out = n1))

  # lstX is the list where i^th element is X^(i) for i = 1, 2, ..., C
  lstX <- vector(mode = "list", length = C)

  # lstY is the list where i^th element is Y^(i) for i = 1, 2, ..., C
  lstY <- vector(mode = "list", length = C)

  # Calculating lstX and lstY
  for (i in 1:C) {
    lstX[[i]] <- X[fold_ids == i, ]
    lstY[[i]] <- Y[fold_ids == i]
  }

  # lstdataX is a list where i^th element is X^(-i)
  lstdataX <- vector(mode = "list", length = C)

  # lstdataY is a list where i^th element is Y^(-i)
  lstdataY <- vector(mode = "list", length = C)

  # Calculating lstdataX and lstdataY
  for (i in 1:C) {
    lstdataX[[i]] <- X[fold_ids != i, ]
    lstdataY[[i]] <- Y[fold_ids != i]
  }

  # error_knn stores cross validation error when we use KNN algorithm
  error_knn <- 0

  # Calculating error_knn
  for (i in 1:C) {
    kNN_fitting <- class::knn(lstdataX[[i]], lstX[[i]], lstdataY[[i]], round(sqrt(nrow(X))))
    A <- table(lstY[[i]], kNN_fitting)
    error_knn <- error_knn + sum(A) - sum(diag(A))
  }

  # error_svm stores cross validation error when we use Multiclass SVM algorithm
  error_svm <- 0

  # Calculating error_svm
  for (i in 1:C) {
    labels <- as.factor(lstdataY[[i]])
    training_data <- cbind(lstdataX[[i]], labels)
    SVM_model <- e1071::svm(labels ~ ., data = training_data, type = "C")
    SVM_fitting <- predict(SVM_model, lstX[[i]])
    A <- table(lstY[[i]], SVM_fitting)
    error_svm <- error_svm + sum(A) - sum(diag(A))
  }
  return(list(error_svm = error_svm, error_knn = error_knn))
}
