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

# Returns which algorithm among SVM and KNN performs better on the labelled data set using cross validation technique
ChooseAlgorithm <- function(X, Y, C = 5, K = NULL) {
  Total_svm_error <- 0
  Total_KNN_error <- 0

  # Comparing cross validation error

  for (i in 1:25) {
    out <- Algorithm(X, Y, C, K)
    Total_svm_error <- Total_svm_error + out$error_svm
    Total_KNN_error <- Total_KNN_error + out$error_knn
  }

  if (Total_svm_error >= Total_KNN_error) {
    return("We will use KNN Algorithm")
  } else {
    return("We will use Multiclass SVM Algorithm")
  }
}



# Z - m1 * p matrix of unlabeled data

#' SSLconf function arranges the unlabeled data points based on the confidence of prediction
#'
#' @param X n1 * p matrix of labelled data points. Here the rows contains the n1 labelled data points. Each data point belongs to R^p
#' @param Y n1 vector of labels for data points in X
#' @param Z m1*p matrix of Unlabeled data points
#' @param K number of classes
#' @return A list with five components, conf - A matrix with data points of Z arranged from high to low confidence of prediction, High - A matrix containing data points of Z that are predicted with high confidence, Low - A matrix containing data points of Z that are predicted with low confidence, Average - A matrix containing data points of Z that are predicted with moderate confidence, Remat - A matrix containing data points of Z that should possibly be relabeled
#'
#' @export
#'
#' @examples
#' data = iris
#' data = data[sample(1:150), ]
#' X = data[1:25,1:4]
#' Y = data[1:25,5]
#' Z = data[26:150, 1:4]
#' K = 3
#' SSLconf(X, Y, Z, K)
#'
SSLconf <- function(X, Y, Z, K = NULL) {
  nZ <- nrow(Z)
  out <- ChooseAlgorithm(X, Y, C = 5, K)
  K_for_kNN <- round(sqrt(nrow(X)))
  if (out == "We will use KNN Algorithm") {
    Probability <- predict(caret::knn3(X, Y, k = K_for_kNN), Z)
    maximum <- apply(Probability, 1, max, na.rm = TRUE)
    second_maximum <- apply(Probability, 1, function(row) max(row[-which.max(row)]))
    Difference <- maximum - second_maximum
    # Coloring the unlabeled data points
    confidence_measure <- rep(0, nZ)
    Relabel <- rep(0, nZ)
    for (i in 1:nZ) {
      if (Difference[i] >= round(2 * K_for_kNN / 3) / K_for_kNN) {
        confidence_measure[i] <- "High"
      }

      if (Difference[i] >= round(K_for_kNN / 3) / K_for_kNN & Difference[i] < round(2 * K_for_kNN / 3) / K_for_kNN) {
        confidence_measure[i] <- "Average"
      }

      if (Difference[i] < round(K_for_kNN / 3) / K_for_kNN) {
        confidence_measure[i] <- "Low"
      }
      if (Difference[i] < 0.4) {
        Relabel[i] <- 1
      }
    }
  }

  if (out == "We will use Multiclass SVM Algorithm") {
    SVM_model <- e1071::svm(Y ~ ., data = cbind(X, Y), type = "C", probability = TRUE)
    SVM_fitting <- predict(SVM_model, Z, probability = T)
    Probability <- attr(SVM_fitting, "probabilities")
    # Reorder unlabelled data points
    maximum <- apply(Probability, 1, max, na.rm = TRUE)
    second_maximum <- apply(Probability, 1, function(row) max(row[-which.max(row)]))
    Difference <- maximum - second_maximum
    # Coloring the unlabeled data points
    confidence_measure <- rep(0, nZ)
    Relabel <- rep(0, nZ)
    for (i in 1:nZ) {
      if (Difference[i] >= 0.70) {
        confidence_measure[i] <- "High"
      }

      if (Difference[i] >= 0.30 & Difference[i] < 0.70) {
        confidence_measure[i] <- "Average"
      }

      if (Difference[i] < 0.30) {
        confidence_measure[i] <- "Low"
      }
      if (Difference[i] < 0.4) {
        Relabel[i] <- 1
      }
    }
  }

  # Reorder unlabeled data points

  attach <- cbind(Difference, maximum, 1:nZ)
  attach <- attach[order(attach[, 2], decreasing = T), ]
  attach <- attach[order(attach[, 1], decreasing = T), ]
  conf <- cbind(Z[attach[, 3], ], confidence_measure)
  High <- Z[confidence_measure == "High", ]
  Average <- Z[confidence_measure == "Average", ]
  Low <- Z[confidence_measure == "Low", ]
  Yes <- Z[Relabel == 1, ]
  No <- Z[Relabel == 0, ]
  nhigh <- nrow(High)
  nLow <- nrow(Low)
  nAverage <- nrow(Average)
  for (i in 1:ncol(Z)) {
    # dev.new()
    plot(rep(i, nhigh), High[, i],
         col = "green", main = "Confidence of Prediction",
         ylab = "Value at that feature",
         xlab = "Feature "
    )
    points(rep(i, nLow), Low[, i], col = "red")
    points(rep(i, nAverage), Average[, i], col = "yellow")
    legend("topleft",
           c("High", "Average", "Low"),
           fill = c("green", "yellow", "red")
    )
  }

  for (i in 1:ncol(Z)) {
    for (j in ncol(Z)) {
      if (j > i) {
        # dev.new()
        plot(High[, i], High[, j],
             col = "green", main = "Confidence of Prediction", ylab = j,
             xlab = i
        )
        points(Low[, i], Low[, j], col = "red")
        points(Average[, i], Average[, j], col = "yellow")
        legend("topleft",
               c("High", "Average", "Low"),
               fill = c("green", "yellow", "red")
        )
      }
    }
  }

  return(list(conf = conf, High = High, Low = Low, Average = Average, Remat = Yes))
}

# Function SSLBudget

#' SSLBudget outputs what unlabeled data points should be labelled based on the budget
#'
#' @param X n1 * p matrix of labelled data points. Here the rows contains the n1 labelled data points. Each data point belongs to R^p
#' @param Y n1 vector of labels for data points in X
#' @param L positive integer - number of data points that user can label
#' @param Z m1*p matrix of Unlabeled data points
#' @param K number of classes
#'
#' @return Tolabel - A L*p matrix of data points from Z that user should label
#' @export
#'
#' @examples
#' #' data = iris
#' data = data[sample(1:150), ]
#' X = data[1:25,1:4]
#' Y = data[1:25,5]
#' Z = data[26:150, 1:4]
#' K = 3
#' SSLBudget(X, Y, 5, Z, K)
#'
SSLBudget <- function(X, Y, L, Z, K = NULL) {
  out <- SSLconf(X, Y, Z, C = 5, K)
  Low <- out$Remat
  Cluster_labels <- MyKmeans(Low, L)
  Clusters <- vector(mode = "list", length = L)
  for (i in 1:L) {
    Clusters[[i]] <- as.matrix(Low[Cluster_labels == i, ])
  }
  ToLabel <- matrix(0, L, ncol(Z))
  for (i in 1:L) {
    ToLabel[i, ] <- Clusters[[i]][sample(1:nrow(Clusters[[i]]), 1), ]
  }

  return(ToLabel)
}


#' SSLpredict predicts the labels for unlabeled data
#'
#' @param X n1 * p matrix of labelled data points. Here the rows contains the n1 labelled data points. Each data point belongs to R^p
#' @param Y n1 vector of labels for data points in X
#' @param Z m1*p matrix of Unlabeled data points
#' @param K number of classes
#'
#' @return A m1 vector of predicted labels for data points in Z
#'
#' @export
#'
#' @examples
#' #' #' data = iris
#' data = data[sample(1:150), ]
#' X = data[1:25,1:4]
#' Y = data[1:25,5]
#' Z = data[26:150, 1:4]
#' K = 3
#' SSLpredict(X, Y, 5, Z, K)
#'
SSLpredict <- function(X, Y, Z, K = NULL) {
  out <- ChooseAlgorithm(X, Y, C = 5, K)
  if (out == "We will use KNN Algorithm") {
    kNN_fitting <- class::knn(X, Z, Y, round(sqrt(nrow(X))))
    return(kNN_fitting)
  }
  if (out == "We will use Multiclass SVM Algorithm") {
    labels <- as.factor(Y)
    training_data <- cbind(X, labels)
    SVM_model <- e1071::svm(labels ~ ., data = training_data, type = "C")
    SVM_fitting <- predict(SVM_model, Z)
    return(SVM_fitting)
  }
}
