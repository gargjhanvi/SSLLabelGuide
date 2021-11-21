# We will use cross validation technique to compare SVM and KNN on the labelled data.

# X - n1 * p matrix of labelled data points. Here the rows contains the n1 labelled data points. Each data point belongs to R^p
# Y - n1 vector of labels for data points in X
# C - number of folds for k-fold cross-validation, default is 5.
# K - number of classes


####################################################################
#Standardize X
standardizeX <- function(X) {
  # Calculating number of rows and columns in X
  n <- nrow(X) # n is the number of rows in X
  p <- ncol(X) # p is the number of columns in X


  #  Center and scale X

  # Calculating Xmeans, means of columns of X
  Xmeans <- colMeans(X)

  # Calculating Xcenter, centered X
  Xcenter <- X - matrix(Xmeans, byrow = T, n, p)

  # Calculating weights, weights of columns of X
  weights <- sqrt(colSums(Xcenter^2) / n)

  # Calculating Xtilde, Centered and Scaled X
  Xtilde <- Xcenter / matrix(weights, byrow = T, n, p)



  # Return:
  # Xtilde - centered and appropriately scaled X

    return(list(Xtilde = Xtilde, weights = weights, Xmeans = Xmeans))
}



####################################################################
# Algorithm function compares Multiclass SVM, KNN and Multiclass Logistic regression using cross-validation
Algorithm <- function(X, Y, K = NULL) {
  C <- 5


  # n1 is the number of rows in X
  n1 <- nrow(X)

  # Getting K if K is NULL
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
  for (i in 1:C) {
    kNN_fitting <- class::knn(lstdataX[[i]], lstX[[i]], lstdataY[[i]], round(sqrt(nrow(X))))
    A <- table(lstY[[i]], kNN_fitting)
    error_knn <- error_knn + sum(A) - sum(diag(A))
  }

  # error_svm stores cross validation error when we use Multiclass SVM algorithm
  error_svm <- 0
  for (i in 1:C) {
    labels <- as.factor(lstdataY[[i]])
    training_data <- cbind(lstdataX[[i]], labels)
    SVM_model <- e1071::svm(labels ~ ., data = training_data, type = "C")
    SVM_fitting <- stats::predict(SVM_model, lstX[[i]])
    A <- table(lstY[[i]], SVM_fitting)
    error_svm <- error_svm + sum(A) - sum(diag(A))
  }
  # error_logistic stores cross validation error when we use Multiclass SVM algorithm
  error_logistic = 0
  for (i in 1:C) {
    error_logistic = error_logistic + LRMultiClass (lstdataX[[i]], lstdataY[[i]],lstX[[i]],lstY[[i]])$error_test
  }

  return(list(error_svm = error_svm, error_knn = error_knn, error_logistic = error_logistic))
}


##############################################################################################
# Returns which algorithm among Multiclass SVM, KNN and Multiclass logistic  performs better on the labelled data set using cross validation technique
ChooseAlgorithm <- function(X, Y, K = NULL) {
  C <- 5
  Total_svm_error <- 0
  Total_KNN_error <- 0
  Total_Logistic_error <- 0

  # Comparing cross validation error

  for (i in 1:25) {
    out <- Algorithm(X, Y,K)
    Total_svm_error <- Total_svm_error + out$error_svm
    Total_KNN_error <- Total_KNN_error + out$error_knn
    Total_Logistic_error <- Total_Logistic_error + out$error_logistic
  }

  if (Total_svm_error > Total_KNN_error & Total_Logistic_error > Total_KNN_error) {
    return("KNN")
  }else if(Total_KNN_error > Total_svm_error  & Total_Logistic_error > Total_svm_error){
    return("SVM")
  }else if(Total_svm_error > Total_Logistic_error & Total_KNN_error > Total_Logistic_error){
    return("Logistic")
  }else if(Total_KNN_error > Total_svm_error & Total_KNN_error > Total_Logistic_error ){
    return("Logistic")
  }else{
    return("KNN")
  }

}

#####################################################################


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

#'
SSLconf <- function(X, Y, Z, K = NULL) {

  out <-standardizeX(rbind(X,Z))
  weights <- out$weights
  means <- out$Xmeans
  Xtilde <- X
  Ztilde <- Z
  X <- (Xtilde - matrix(means, byrow = T, nrow(X), ncol(X)))/matrix(weights, byrow = T, nrow(X), ncol(X))
  Z <- (Ztilde - matrix(means, byrow = T, nrow(Z), ncol(Z)))/matrix(weights, byrow = T, nrow(Z), ncol(Z))
  n_train <- nrow(X)
  p <- ncol(X)
  n_test <- nrow(Z)
  K_for_kNN <- round(sqrt(nrow(X)))
  confidence_measure <- rep(0, n_test)
  Relabel <- rep(0, n_test)
  High = matrix(c(0,0,0,0),2,2)

  out <- ChooseAlgorithm(X, Y, K)
  A <- Z
  while(is.null(High) == F){
 # If Algorithm is KNN
    if (out == "KNN"){
    Probability <- stats::predict(caret::knn3(X, Y, k = K_for_kNN), A)
    maximum <- apply(Probability, 1, max, na.rm = TRUE)
    second_maximum <- apply(Probability, 1, function(row) max(row[-which.max(row)]))
    Difference <- maximum - second_maximum

    for (i in 1:nrow(A)) {
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
 # If Algorithm is SVM
  if (out == "SVM") {
    SVM_model <- e1071::svm(Y ~ ., data = cbind(X, Y), type = "C", probability = TRUE)
    SVM_fitting <- stats::predict(SVM_model, A, probability = T)
    Probability <- attr(SVM_fitting, "probabilities")
    maximum <- apply(Probability, 1, max, na.rm = TRUE)
    second_maximum <- apply(Probability, 1, function(row) max(row[-which.max(row)]))
    Difference <- maximum - second_maximum
    for (i in 1:nrow(A)) {
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
# If Algorithm is Logistic
  if (out == "Logistic") {
    out1 <- LRMultiClass(X, Y, A, rep(1,nrow(A)))
    P <- exp(A %*% out1$beta)
    cols <- colSums(t(P))
    Probability <- P / cols
    maximum <- apply(Probability, 1, max, na.rm = TRUE)
    second_maximum <- apply(Probability, 1, function(row) max(row[-which.max(row)]))
    Difference <- maximum - second_maximum
    for (i in 1:nrow(A)) {
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
################# High low Average relabel calculated
  X <- rbind(X, A[confidence_measure == "High", ])
  Y <- c(Y, max.col(Probability)[confidence_measure == "High"])
  A <- A[confidence_measure !="High", ]
  }
################### Unstandardize X, A, Z
  A <- A %*% matrix(weights, byrow = T, nrow(A), p) + matrix(means, byrow = T, nrow(A), p)
  Z <- Z %*% matrix(weights, byrow = T, nrow(Z), p) + matrix(means, byrow = T, nrow(Z), p)

##################
  Remat <- A[Relabel == 1, ]
  Average <- A[confidence_measure == "Average", ]
  Low <- A[confidence_measure == "Low", ]
  remove <- rbind(Average, Low)
  index <- c()
  for (i in 1: nrow(remove)){
    index = append(index,  which(rowSums(abs(Z - matrix(remove[i] ,nrow(Z), ncol(Z),T)))==0))
    }
  High <- Z[-index, ]

  ############### We have High, Average, Low, Remat
  n_high <- nrow(High)
  n_Low <- nrow(Low)
  n_Average <- nrow(Average)

  for (i in 1:ncol(Z)) {
    # dev.new()
    plot(rep(i, n_high), High[, i],
         col = "green", main = "Confidence of Prediction",
         ylab = "Value at that feature",
         xlab = "Feature "
    )
    graphics::points(rep(i, n_Low), Low[, i], col = "red")
    graphics::points(rep(i, n_Average), Average[, i], col = "yellow")
    graphics::legend("topleft",
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
        graphics::points(Low[, i], Low[, j], col = "red")
        graphics::points(Average[, i], Average[, j], col = "yellow")
        graphics::legend("topleft",
               c("High", "Average", "Low"),
               fill = c("green", "yellow", "red")
        )
      }
    }
  }

  return(list( High = High, Low = Low, Average = Average, Remat = Remat))
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


SSLBudget <- function(X, Y, L, Z, K = NULL) {
  out <- SSLconf(X, Y, Z, K)
  Remat <- out$Remat
  Cluster_labels <- MyKmeans(Remat, L)
  Clusters <- vector(mode = "list", length = L)
  for (i in 1:L) {
    Clusters[[i]] <- as.matrix(Remat[Cluster_labels == i, ])
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
#' #'
#' data(iris)
#' data1 = iris[sample(1:150), ]
#' X = data1[1:25,1:4]
#' Y = data1[1:25,5]
#' Z = data1[26:150, 1:4]
#' SSLpredict(X, Y,  Z) # 1

SSLpredict <- function(X, Y, Z, K = NULL) {
  out <- ChooseAlgorithm(X, Y, K)
  if (out == "KNN") {
    kNN_fitting <- class::knn(X, Z, Y, round(sqrt(nrow(X))))
    return(kNN_fitting)
  }
  if (out == "SVM") {
    labels <- as.factor(Y)
    training_data <- cbind(X, labels)
    SVM_model <- e1071::svm(labels ~ ., data = training_data, type = "C")
    SVM_fitting <- stats::predict(SVM_model, Z)
    return(SVM_fitting)
  }
  if (out == "Logistic") {
    nz <- nrow(Z)
    out1 <- LRMultiClass(X, Y, Z, rep(1,nz))
    P <- exp(Z %*% out1$beta)
    rsum <- rowSums(P)
    rsum_mat <- matrix(rowsum, nrow(P), ncol(P))
    P <- P/rsum_mat
    maximum <- max.col(P)
    return(maximum)
  }

}



#' Comparison error compares the error of unlabeled data points before labeling L points and after labeling L points that were given by SSLBudget function
#'
#' @param X n1 * p matrix of labelled data points. Here the rows contains the n1 labelled data points. Each data point belongs to R^p
#' @param Y n1 vector of labels for data points in X
#' @param Z m1*p matrix of Unlabeled data points
#' @param W m1 vector of true labels of data points in Z
#' @param new_data L*(p+1) matrix of data points that were given by SSLBudget appended with their true labels
#' @param K number of classes
#'
#' @return A list with two components, error_before - hamming distance between true labels and predicted labels before user labels L data points suggested by SSLBudget, error_after - hamming distance between true labels and predicted labels after user labels L data points suggested by SSLBudget
#'
#' @export
#'
#'
Comparison_error <- function(X, Y, Z, W, new_data,  K = NULL) {
  p = ncol(new_data) - 1
  for (i in 1:nrow(new_data)) {
    A <- rowsum(abs(Z - matrix(new_data[i, 1:p], nrow = nrow(Z))))
    index <- which(A == 0)
    Z <- Z[-index, ]
  }
  out <- SSLpredict(X, Y, Z, K)

  X1 <- rbind(X, new_data[, 1:p])
  Y1 <- rbind(Y, new_data[, (p+1)])
  out1 <- SSLpredict(X1, Y1, Z, K)
  error1 <- table(W, out)
  error2 <- table(W, out1)
  return(list(error_before = error1, error_after = error2))
}
