
# We will use cross validation technique to compare Multiclass SVM, KNN and Multiclass logistic regression on the labelled data.

# X - n1 * p matrix of labelled data points. Here the rows contains the n1 labelled data points. Each data point belongs to R^p
# Xtilde -  n1 * p matrix of standarized X
# Y - n1 vector of labels for data points in X
# C - number of folds for k-fold cross-validation, default is 5.
# K - number of classes
# Z - m1 * p matrix of unlabeled data set
# L - budget/ number of data points user can label

####################################################################
# standardizeX function standardize X, i.e center and scale X
#' standardizeX function centers and scales data
#'
#' @param X A matrix to center and scale
#'
#' @return A list of 3 elements,
#' \item{tilde}{ contains the center and scaled matrix}
#' \item{weights}{ return weights after centering of X but before scaling}
#' \item{means}{ returns means of columns of X}
#' @export
#'
#'
#' @examples
#' X <- matrix(rnorm(20,2,1),5,4)
#' standardizeX(X)
#'#$ tilde
#'#[,1]       [,2]       [,3]       [,4]
#'#[1,]  1.6882137 -0.7593179 -0.6990462  1.8321703
#'#[2,]  0.1689194  0.2360252 -0.1677231 -0.4757201
#'#[3,] -0.8734264  0.1447277 -0.8047698 -1.1429930
#'#[4,]  0.1694665 -1.2727583 -0.2684332  0.1025888
#'#[5,] -1.1531731  1.6513233  1.9399723 -0.3160460

#'#$weights
#'#[1] 0.9099549 0.8239372 0.6899589 0.8253017

#'#$means
#'#[1] 2.446213 1.150378 2.978231 1.841284
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

  return(list(tilde = Xtilde, weights = weights, means = Xmeans))
}



####################################################################
# Algorithm function compares Multiclass SVM, KNN and Multiclass Logistic regression using cross-validation and returns error of each
#' Compares Multiclass SVM, K nearest neighbor and Multiclass Logistic regression on the labelled data points using five fold cross validation
#'
#' @param Xtilde n * p centered and scaled data points
#' @param Y A numeric n vector of labels of Xtilde
#' @param K Number of classes, ( If K is NULL, it calculates K as maximum entry in Y)
#'
#' @return A list of three elements a list of three elements:
#' \item{error_svm}{  returns five fold cross-validation error when Multiclass svm is used}
#' \item{error_knn}{ returns five fold cross-validation error when K-nearest neighbor is used with K = round(sqrt(N)) where N is the number of training data points}
#' \item{error_logistic}{  returns five fold cross-validation error when Multiclass logistic regression is used with default numIter = 50, eta = 0.1, lambda = 1}
#' @export
#'
#' @examples
#' X <- rbind(matrix(rnorm(10,0,1),5,2),matrix(rnorm(10,1,2),5,2), matrix(rnorm(10,4,3),5,2))
#' fold_ids <- sample(1:15)
#' X <- X[fold_ids, ]
#' Xtilde <- standardizeX(X)$tilde
#' Y <- c(1,1,1,1,1,2,2,2,2,2,3,3,3,3,3)
#' Y <- Y[fold_ids]
#' out <- Algorithm(Xtilde,Y,3)
#'
Algorithm <- function(Xtilde, Y, K = NULL) {
  C <- 5
  X <- Xtilde
  Y <- as.numeric(Y)
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

    error_logistic = error_logistic + LRMultiClass(cbind(1,as.matrix(lstdataX[[i]])), (lstdataY[[i]]-1), cbind(1,as.matrix(lstX[[i]])),(lstY[[i]]-1))$error_test[51]
  }

  return(list(error_svm = error_svm, error_knn = error_knn, error_logistic = error_logistic))
}


##############################################################################################
#' Returns which algorithm among Multiclass SVM, KNN and Multiclass logistic  performs better on the labelled data set using cross validation technique

#' @inheritParams Algorithm
#' @return A string among c("KNN", "Logistic", "SVM") which tells us which algorithm performs better on the labelled data set
#'
#' @export
#'
#' @examples
#'X <- rbind(matrix(rnorm(10,0,1),5,2),matrix(rnorm(10,1,2),5,2), matrix(rnorm(10,4,3),5,2))
#' fold_ids <- sample(1:15)
#' X <- X[fold_ids, ]
#' Xtilde <- standardizeX(X)$tilde
#' Y <- c(1,1,1,1,1,2,2,2,2,2,3,3,3,3,3)
#' Y <- Y[fold_ids]
#' out <- ChooseAlgorithm(Xtilde,Y,3)
#' #"KNN"
ChooseAlgorithm <- function(Xtilde, Y, K = NULL) {
  C <- 5
  Total_svm_error <- 0
  Total_KNN_error <- 0
  Total_Logistic_error <- 0

  # Comparing cross validation error

  for (i in 1:25) {
    out <- Algorithm(Xtilde, Y,K)
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


#' SSLconf divides the unlabeled data into matrices "High", "Low", "Average" based on the confidence in predicting them.
#'
#' @param X n * p matrix of labelled data points. Here the rows contains the n labelled data points. Each data point belongs to R^p
#' @param Z m * p matrix of Unlabeled data points
#' @param Plots Takes argument TRUE or FALSE, depending on whether  the function should output plots. Default is TRUE
#' @inheritParams Algorithm
#' @return A list with four elements,
#' \item{High}{A matrix containing data points of Z that are predicted with high confidence}
#' \item{Low}{A matrix containing data points of Z that are predicted with low confidence}
#' \item{Average}{A matrix containing data points of Z that are predicted with moderate confidence}
#' \item{Remat}{A matrix containing data points of Z that should possibly be relabeled}
#'
#' @export
#'
#' @examples
#' X <- rbind(matrix(rnorm(10,0,1),5,2),matrix(rnorm(10,1,2),5,2), matrix(rnorm(10,4,3),5,2))
#' fold_ids <- sample(1:15)
#' X <- X[fold_ids, ]
#' Y <- c(1,1,1,1,1,2,2,2,2,2,3,3,3,3,3)
#' Y <- Y[fold_ids]
#' Z <- rbind(matrix(rnorm(80,0,1),40,2),matrix(rnorm(80,1,2),40,2), matrix(rnorm(80,4,3),40,2))
#' Z <- Z[sample(1:120), ]
#' out <- SSLconf(X, Y, Z,3)
#'
#'
SSLconf <- function(X, Y, Z, K = NULL, Plots = T) {
  # Standardizing X and z
  n_train <- nrow(X)
  p <- ncol(X)
  n_test <- nrow(Z)
  out <- standardizeX(rbind(X,Z))
  weights <- out$weights
  means <- out$means
  tilde <- out$tilde
  Xtilde <- tilde[1:n_train, ]
  Ztilde <- tilde[(n_train + 1):(n_train + n_test), ]

  High <- matrix(c(0,0,0,0),2,2)
  High_increase <- NULL

  out1 <- ChooseAlgorithm(Xtilde, Y, K)

  Z_modified <- Ztilde
  X_modified <- Xtilde
  Y_modified <- Y


  while(nrow(High) !=0 ){
    K_for_kNN <- round(sqrt(nrow(X_modified)))
    confidence_measure <- rep(0, nrow(Z_modified))
    Relabel <- rep(0, nrow(Z_modified))

    # If Algorithm is KNN
    if (out1 == "KNN"){
      Probability <- stats::predict(caret::knn3(X_modified,as.factor( Y_modified), k = K_for_kNN), Z_modified)
    }

    # If Algorithm is SVM
    if (out1 == "SVM") {
      labels <- as.factor(Y_modified)
      SVM_model <- e1071::svm(labels ~ ., data = cbind(X_modified, labels), type = "C", probability = TRUE)
      SVM_fitting <- stats::predict(SVM_model, Z_modified, probability = T)
      Probability <- attr(SVM_fitting, "probabilities")
    }

    # If Algorithm is Logistic
    if (out1 == "Logistic") {
      out2 <- LRMultiClass(cbind(1,as.matrix(X_modified)), (Y_modified-1), cbind(1,as.matrix(Z_modified)), rep(1,nrow(Z_modified)))
      P <- exp(cbind(1,as.matrix(Z_modified)) %*% out2$beta)
      cols <- colSums(t(P))
      Probability <- P / cols
    }

    maximum <- apply(Probability, 1, max, na.rm = TRUE)
    second_maximum <- apply(Probability, 1, function(row) max(row[-which.max(row)]))
    Difference <- maximum - second_maximum

    for (i in 1:nrow(Z_modified)) {
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
    #################

    High <- as.matrix(Z_modified[confidence_measure =="High", ])
    if(ncol(High) == 1){
      High <- matrix(High, 1 , p,byrow=T)
    }

    High_increase <- rbind(High_increase, High)
    Low <- Z_modified[confidence_measure =="Low", ]
    Average <- Z_modified[confidence_measure =="Average", ]
    Remat <- Z_modified[Relabel == 1, ]
    X_modified <- as.matrix(rbind(X_modified, Z_modified[confidence_measure == "High", ]))
    Y_modified <- c(Y_modified, max.col(Probability)[confidence_measure == "High"])
    Z_modified <- as.matrix(Z_modified[confidence_measure !="High", ])

  }
  ###################

  ### Unscaling  Low, Average, High, Remat
  High <- High_increase * matrix(weights, byrow = T, nrow(High_increase), p) + matrix(means, byrow = T, nrow(High_increase), p)
  Low <- Low * matrix(weights, byrow = T, nrow(Low), p) + matrix(means, byrow = T, nrow(Low), p)
  Average <- Average * matrix(weights, byrow = T, nrow(Average), p) + matrix(means, byrow = T, nrow(Average), p)
  Remat<- Remat * matrix(weights, byrow = T, nrow(Remat), p) + matrix(means, byrow = T, nrow(Remat), p)

  ############### We have High, Average, Low, Remat
  n_high <- nrow(High)
  n_Low <- nrow(Low)
  n_Average <- nrow(Average)
  #grDevices::pdf("PLot1.pdf")
  if (Plots){
  for (i in 1:p) {

     plot( High[, i],rep(1.4, n_high), yaxt='n',
          col = "green", pch = 17, main = paste("Confidence of Prediction Vs feature", i),
          xlab = paste("Value at feature", i),
          ylab = " "
    )
    graphics::points (Low[, i],rep(1, n_Low), col = "red", pch = 16)
    graphics::points( Average[, i],rep(1.2, n_Average), col = "blue", pch = 15)
    graphics::legend("topleft",
                     c("High", "Average", "Low"),
                     fill = c("green", "blue", "red")
    )

  }

  #grDevices::pdf("PLot2.pdf")
  for (i in 1:p) {
    for (j in 1:p) {
      if (j > i) {

        plot(High[, i], High[, j],
             col = "green", main = "Confidence of Prediction", ylab = paste("Value at feature", j),
             xlab = paste("Value at feature", i), pch = 17
        )
        graphics::points(Low[, i], Low[, j], col = "red", pch = 16)
        graphics::points(Average[, i], Average[, j], col = "blue", pch = 15)
        graphics::legend("topleft",
                         c("High", "Average", "Low"),
                         fill = c("green", "blue", "red")
        )

      }
    }
  }
}
  return(list( High = High, Low = Low, Average = Average, Remat = Remat))
}
####################################################################
# Function SSLBudget

#' SSLBudget outputs what unlabeled data points should be labelled based on the budget
#'
#' @param L positive integer - number of data points that user can label
#' @inheritParams SSLconf
#' @return Tolabel - A L*p matrix of data points from Z that user should label
#' @export
#'
#' @examples
#' X <- rbind(matrix(rnorm(10,0,1),5,2),matrix(rnorm(10,1,2),5,2), matrix(rnorm(10,4,3),5,2))
#' fold_ids <- sample(1:15)
#' X <- X[fold_ids, ]
#' Y <- c(1,1,1,1,1,2,2,2,2,2,3,3,3,3,3)
#' Y <- Y[fold_ids]
#' Z <- rbind(matrix(rnorm(80,0,1),40,2),matrix(rnorm(80,1,2),40,2), matrix(rnorm(80,4,3),40,2))
#' Z <- Z[sample(1:120), ]
#' out <- SSLBudget(X, Y,5, Z,3)


SSLBudget <- function(X, Y, L, Z, K = NULL) {
  n <- nrow(Z)
  p <- ncol(Z)
  out <- SSLconf(X, Y, Z, K, Plots = F)
  Remat <- out$Remat
  Cluster_labels <- MyKmeans(as.matrix(Remat), L)
  Clusters <- vector(mode = "list", length = L)
  for (i in 1:L) {
    Clusters[[i]] <- as.matrix(Remat[Cluster_labels == i, ])
  }
  ToLabel <- matrix(0, L, ncol(Z))
  for (i in 1:L) {
    ToLabel[i, ] <- Clusters[[i]][sample(1:nrow(Clusters[[i]]), 1), ]
  }

for (i in 1:p) {
  plot( Z[, i],rep(1, n ), yaxt='n',
        col = "black", pch = 16, main = paste("Points that should be relabeled, only showing feature", i),
        xlab = paste("Value at feature", i),
        ylab = " "
  )
  graphics::points (ToLabel[, i],rep(1, L), col = "red", pch = 16)
  graphics::legend("topleft",
                   c("To Label"),
                   fill = c("red")
  )
}


  for (i in 1:p) {
    for (j in 1:p) {
      if (j > i) {
        # dev.new()
        plot(Z[, i], Z[, j],
             col = "black", main = paste(c("Points that should be relabeled, only showing feature",i,",",j)), ylab = paste("Value at feature", j),
             xlab = paste("Value at feature", i), pch = 16
        )
        graphics::points(ToLabel[, i], ToLabel[, j], col = "red", pch = 16)

        graphics::legend("topleft",
                         c("To Label"),
                         fill = c("red")
        )
      }
    }
  }

  return(ToLabel)
}
##################################################################

#' SSLpredict predicts the labels for unlabeled data
#' @inheritParams SSLconf
#' @return A m1 vector of predicted labels for data points in Z
#'
#' @export
#'
#' @examples
#'  X <- rbind(matrix(rnorm(10,0,1),5,2),matrix(rnorm(10,1,2),5,2), matrix(rnorm(10,4,3),5,2))
#' fold_ids <- sample(1:15)
#' X <- X[fold_ids, ]
#' Y <- c(1,1,1,1,1,2,2,2,2,2,3,3,3,3,3)
#' Y <- Y[fold_ids]
#' Z <- rbind(matrix(rnorm(80,0,1),40,2),matrix(rnorm(80,1,2),40,2), matrix(rnorm(80,4,3),40,2))
#' Z <- Z[sample(1:120), ]
#' out <- SSLpredict(X, Y, Z,3)


SSLpredict <- function(X, Y, Z, K = NULL) {
  p <- ncol(X)
  out <- standardizeX(rbind(X,Z))
  Xtilde <- out$tilde[1:nrow(X), ]
  Ztilde <- out$tilde[(nrow(X)+1):(nrow(X)+nrow(Z)), ]

  Ytilde <- as.numeric(Y)
  High = matrix(c(0,0,0,0),2,2)
  High_increase = NULL
  out1 <- ChooseAlgorithm(Xtilde, Y, K)

  if (out1 == "KNN") {
    while(nrow(High) !=0){
      K_for_kNN <- round(sqrt(nrow(Xtilde)))
      Probability <- stats::predict(caret::knn3(Xtilde, as.factor(Ytilde), k = K_for_kNN), Ztilde)
      maximum <- apply(Probability, 1, max, na.rm = TRUE)
      second_maximum <- apply(Probability, 1, function(row) max(row[-which.max(row)]))
      Difference <- maximum - second_maximum
      confidence_measure <- rep(0,nrow(Ztilde))
      for (i in 1:nrow(Ztilde)) {
        if (Difference[i] >= 0.70) {
          confidence_measure[i] <- "High"
        }

        if (Difference[i] >= 0.30 & Difference[i] < 0.70) {
          confidence_measure[i] <- "Average"
        }

        if (Difference[i] < 0.30) {
          confidence_measure[i] <- "Low"
        }
      }
      #################

      High <- Ztilde[confidence_measure =="High", ]
      print(High)
      print(Xtilde)
      print(Ytilde)
      if(is.vector(High) ){
        High = matrix(High,byrow=T, nrow = 1, ncol = length(High))
      }
      print(High)
      kNN_fitting <- class::knn(data.frame(Xtilde),High, as.factor(Ytilde), round(sqrt(nrow(Xtilde))))
      High_increase <- rbind(High_increase, cbind(High,kNN_fitting))
      Xtilde <- rbind(Xtilde, Ztilde[confidence_measure == "High", ])
      Ytilde <- c(Ytilde, max.col(Probability)[confidence_measure == "High"])
      Ztilde <- Ztilde[confidence_measure !="High", ]


      if(is.vector(Ztilde )){
        Ztilde= as.matrix(Ztilde,nrow = 1)
      }
    }

    kNN_fitting <- class::knn(Xtilde,Ztilde,as.factor(Ytilde), round(sqrt(nrow(Xtilde))))
    High_increase <- rbind(High_increase, cbind(Ztilde,kNN_fitting))
    High_increase[,1:p] <-High_increase[,1:p] * matrix(out$weights, byrow = T, nrow(High_increase), p) + matrix(out$means, byrow = T, nrow(High_increase), p)
    return(High_increase)
  }

  if (out1 == "SVM") {

    while(nrow(High) !=0){
      labels <- as.factor(Ytilde)
      SVM_model <- e1071::svm(labels ~ ., data = cbind(Xtilde, labels), type = "C", probability = TRUE)
      SVM_fitting <- stats::predict(SVM_model, Ztilde, probability = T)
      Probability <- attr(SVM_fitting, "probabilities")
      maximum <- apply(Probability, 1, max, na.rm = TRUE)
      second_maximum <- apply(Probability, 1, function(row) max(row[-which.max(row)]))
      Difference <- maximum - second_maximum
      confidence_measure <- rep(0,nrow(Ztilde))
      for (i in 1:nrow(Ztilde)) {
        if (Difference[i] >= 0.70) {
          confidence_measure[i] <- "High"
        }

        if (Difference[i] >= 0.30 & Difference[i] < 0.70) {
          confidence_measure[i] <- "Average"
        }

        if (Difference[i] < 0.30) {
          confidence_measure[i] <- "Low"
        }
      }
      #################

      High <- Ztilde[confidence_measure =="High", ]
      if(is.vector(High )){
        High = as.matrix(High,nrow = 1)
      }
      if(nrow(High) == 0){
        labels <- as.factor(Ytilde)
        SVM_model <- e1071::svm(labels ~ ., data = cbind(Xtilde, labels), type = "C")
        SVM_fitting <- stats::predict(SVM_model, Ztilde)
        High_increase <- rbind(High_increase, cbind(Ztilde,SVM_fitting))
        High_increase[,1:p] <-High_increase[,1:p] * matrix(out$weights, byrow = T, nrow(High_increase), p) + matrix(out$means, byrow = T, nrow(High_increase), p)
        return(High_increase)

      }

      labels <- as.factor(Ytilde)
      SVM_model <- e1071::svm(labels ~ ., data = cbind(Xtilde, labels), type = "C")
      SVM_fitting <- stats::predict(SVM_model, High)
      High_increase <- rbind(High_increase, cbind(High,SVM_fitting))
      Xtilde <- rbind(Xtilde, High)
      Ytilde <- c(Ytilde, max.col(Probability)[confidence_measure == "High"])
      Ztilde <- Ztilde[confidence_measure !="High", ]
      if(is.vector(Ztilde )){
        Ztilde= as.matrix(Ztilde,nrow = 1)
      }

    }
    labels <- as.factor(Ytilde)
    SVM_model <- e1071::svm(labels ~ ., data = cbind(Xtilde, labels), type = "C")
    SVM_fitting <- stats::predict(SVM_model, Ztilde)
    High_increase <- rbind(High_increase, cbind(Ztilde,SVM_fitting))
    High_increase[,1:p] <-High_increase[,1:p] * matrix(out$weights, byrow = T, nrow(High_increase), p) + matrix(out$means, byrow = T, nrow(High_increase), p)
    return(High_increase)
  }

  if (out1 == "Logistic") {

    while(nrow(High) !=0){
      out2 <- LRMultiClass(cbind(1,as.matrix(Xtilde)), (Ytilde-1), cbind(1,as.matrix(Ztilde)), rep(1,nrow(Ztilde)))
      P <- exp(cbind(1,as.matrix(Ztilde)) %*% out2$beta)
      cols <- colSums(t(P))
      Probability <- P / cols
      maximum <- apply(Probability, 1, max, na.rm = TRUE)
      second_maximum <- apply(Probability, 1, function(row) max(row[-which.max(row)]))
      Difference <- maximum - second_maximum
      confidence_measure <- rep(0,nrow(Ztilde))
      for (i in 1:nrow(Ztilde)) {
        if (Difference[i] >= 0.70) {
          confidence_measure[i] <- "High"
        }

        if (Difference[i] >= 0.30 & Difference[i] < 0.70) {
          confidence_measure[i] <- "Average"
        }

        if (Difference[i] < 0.30) {
          confidence_measure[i] <- "Low"
        }
      }
      #################
      max = max.col(maximum)
      High <- Ztilde[confidence_measure =="High", ]
      if(is.vector(High)){
        High <- as.matrix(High, nrow = 1)
      }
      log_fit <- max[confidence_measure == "High"]
      High_increase <- rbind(High_increase, cbind(High,log_fit))
      Xtilde <- rbind(Xtilde, Ztilde[confidence_measure == "High", ])
      Ytilde <- c(Ytilde, maximum[confidence_measure == "High"])
      Ztilde <- Ztilde[confidence_measure !="High", ]

    }

    P <- exp(cbind(1,as.matrix(Ztilde)) %*% out2$beta)
    rsum <- rowSums(P)
    rsum_mat <- matrix(rsum, nrow(P), ncol(P))
    Probability <- P/rsum_mat
    log_fit <- max.col(Probability)

    High_increase <- rbind(High_increase, cbind(Ztilde,log_fit))
    High_increase[,1:p] <-High_increase[,1:p] * matrix(out$weights, byrow = T, nrow(High_increase), p) + matrix(out$means, byrow = T, nrow(High_increase), p)
    return(High_increase)

  }

}


