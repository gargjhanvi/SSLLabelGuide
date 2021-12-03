# Function that implements multi-class logistic regression.
#############################################################
# Description of supplied parameters:
# X - n x p training data, 1st column should be 1s to account for intercept
# y - a vector of size n of class labels, from 0 to K-1
# Xt - ntest x p testing data, 1st column should be 1s to account for intercept
# yt - a vector of size ntest of test class labels, from 0 to K-1
# numIter - number of FIXED iterations of the algorithm, default value is 50
# eta - learning rate, default value is 0.1
# lambda - ridge parameter, default value is 0.1
# beta_init - (optional) initial starting values of beta for the algorithm, should be p x K matrix

## Return output
##########################################################################
# beta - p x K matrix of estimated beta values after numIter iterations
# error_train - (numIter + 1) length vector of training error % at each iteration (+ starting value)
# error_test - (numIter + 1) length vector of testing error % at each iteration (+ starting value)
# objective - (numIter + 1) length vector of objective values of the function that we are minimizing at each iteration (+ starting value)

#' Title
#'
#' @param X   n x p training data, 1st column should be 1s to account for intercept
#' @param y   a vector of size n of class labels, from 0 to K-1
#' @param Xt  ntest x p testing data, 1st column should be 1s to account for intercept
#' @param yt  a vector of size ntest of test class labels, from 0 to K-1

#' @return A list of two elements,
#' \item{beta}{p x K matrix of estimated beta values after numIter iterations}
#' \item{error_train}{(numIter + 1) length vector of training error % at each iteration (+ starting value)}
#'
#' @export
#'
#' @examples
#' X1 <- matrix(rnorm(50, -3, 1), 50, 1)
#'Y1 <- matrix(c(0), 50, 1)
#'X2 <- matrix(rnorm(50, 3, 1), 50, 1)
#'Y2 <- matrix(c(1), 50, 1)
#'X <- c(X1, X2)
#'Y <- c(Y1, Y2)
#'X <- as.matrix(X)
#'Y <- as.matrix(Y)
#'random <- sample(nrow(X))
#'X <- as.matrix(X[random, ]) # training data from two normal distributions
#'Y <- as.matrix(Y[random, ]) # class labels for training data
#'# creating test data
#'X1t <- matrix(rnorm(10, -3, 1), 10, 1)
#'Y1t <- matrix(c(0), 10, 1)
#'X2t <- matrix(rnorm(10, 3, 1), 10, 1)
#'Y2t <- matrix(c(1), 10, 1)
#'Xt <- c(X1t, X2t)
#'Yt <- c(Y1t, Y2t)
#'Xt <- as.matrix(Xt)
#'Yt <- as.matrix(Yt)
#'random <- sample(nrow(Xt))
#'Xt <- as.matrix(Xt[random, ]) # testing data from the same two normal distributions
#'Yt <- as.matrix(Yt[random, ]) # class labels for testing data
#'# adding columns of 1's to training and testing data
#'colX1 <- rep(1, nrow(X))
#'X <- cbind(colX1, X)

#'colXt1 <- rep(1, nrow(Xt))
#'Xt <- cbind(colXt1, Xt)
#'output <- LRMultiClass(X, Y, Xt, Yt, eta = 0.01, numIter = 100, lambda = 1, beta_init = NULL)









LRMultiClass <- function(X, y, Xt, yt) {
  ## Check the supplied parameters as described. You can assume that X, Xt are matrices; y, yt are vectors; and numIter, eta, lambda are scalars. You can assume that beta_init is either NULL (default) or a matrix.
  ###################################
  # Storing the number of rows and columns of X and Xt and the number of clusters in variables.

  n <- nrow(X) # n is the number of rows in X
  p <- ncol(X) # p is the number of columns in X
  nt <- nrow(Xt) # nt is number of rows in Xt
  K <- max(y, yt) + 1 # K is the number of clusters

  # Check that the first column of X and Xt are 1s, if not - display appropriate message and stop execution.
   numIter = 50
   eta = 0.1
   lambda = 1
   beta_init = NULL
  if (all(X[, 1] == matrix(c(1), 1, n)) == F) {
    stop("First column of X does not contain all 1's")
  }

  if (all(Xt[, 1] == matrix(c(1), 1, nt)) == F) {
    stop("First column of Xt does not contain all 1's")
  }

  # Check for compatibility of dimensions between X and y

  if (!(n == length(y)) == TRUE) {
    stop("X and y does not have compatible dimensions")
  }

  # Check for compatibility of dimensions between Xt and yt

  if (!(nt == length(yt)) == TRUE) {
    stop("Xt and yt does not have compatible dimensions")
  }

  # Check for compatibility of dimensions between X and Xt

  if (!(p == ncol(Xt)) == TRUE) {
    stop("X and Xt does not have compatible dimensions")
  }


  # Check eta is positive

  if (eta <= 0) {
    stop("eta is not positive")
  }


  # Check lambda is non-negative


  if (lambda < 0) {
    stop("lambda is negative")
  }


  # Check whether beta_init is NULL. If NULL, initialize beta with p x K matrix of zeroes. If not NULL, check for compatibility of dimensions with what has been already supplied.

  if (is.null(beta_init) == TRUE) {
    beta_init <- matrix(0, p, K)
  }

  if (!(nrow(beta_init) == p) == TRUE | !(ncol(beta_init) == K) == TRUE) {
    stop("beta_init does not have compatible dimensions")
  }

  ## Calculate corresponding pk, objective value f(beta_init), training error and testing error given the starting point beta_init
  ##########################################################################
  P <- matrix(0, n, K) # stores p_{k}(xi)  for training data
  iter <- 0 # keeps track of number of iteration
  error_train <- rep(NA, numIter + 1) # error_train stores training error % at each iteration (+ starting value)
  error_test <- rep(NA, numIter + 1) # error_test stores testing error % at each iteration (+ starting value)
  objective <- rep(NA, numIter + 1) # objective stores objective values of the function that we are minimizing at each iteration (+ starting value)
  Q <- matrix(NA, n, K) # for intermediate calculation
  for (i in 0:(K - 1)) {
    Q[, i + 1] <- ifelse(y == i, 1, 0)
  }

  P <- exp(X %*% beta_init)

  cols <- colSums(t(P))
  P <- P / cols

  lp <- log(P) # lp stores the log of the matrix P

  f1 <- lambda / 2 * (sum(beta_init^2)) # intermediate calculation of objective function
  f2 <- matrix(NA, 1, K)
  # intermediate calculation of  objective function
  for (i in 1:(K)) {
    f2[i] <- t(lp[, i]) %*% Q[, i]
  }

  objective[iter + 1] <- f1 - sum(f2) # appending/adding the value of objective function for this iteration to the vector objective.

  y1 <- rep(0, n) # stores the classification output for training data
  # calculating the classification output for training data
  y1 <- max.col(P, ties.method = "first") - 1

  Pt <- matrix(0, nt, K) # Pt stores P(yi=k|Xi)  for testing data
  # calculated  P(yi=k|Xi)  for testing data for each i in {1,2,...,ntest} and k in {0,...,K-1} and stored in Pt

  Pt <- exp(Xt %*% beta_init)

  colst <- colSums(t(Pt))
  Pt <- Pt / colst

  y1t <- rep(0, nt) # stores the classification output for testing data
  # calculating the classification output for testing data
  y1t <- max.col(Pt, ties.method = "first") - 1
  # calculating the training error and testing error in percentage and storing in error_train and error_test respectively for this iteration
  error_train[iter + 1] <- sum(y != y1)
  error_test[iter + 1] <- sum(yt != y1t)

  ## Newton's method cycle - implement the update EXACTLy numIter iterations
  ##########################################################################

  # Within one iteration: perform the update, calculate updated objective function and training/testing errors in %
  # Initializing the variables

  beta <- beta_init

  for (iter in 1:numIter) {
    # Updating the beta
    A <- P * (1 - P) # A stores the intermediate calculation

    for (k in 1:K) {
      XTWk <- matrix(NA, p, n)
      XTWk <- c(A[, k]) * X
      beta[, k] <- beta[, k] - eta * solve(lambda * diag(p) + t(XTWk) %*% X) %*% (t(X) %*% t((P[, k] - t(Q[, k]))) + lambda * beta[, k])
    }

    P <- exp(X %*% beta)

    cols <- colSums(t(P))
    P <- P / cols

    lp <- log(P) # lp stores the log of the matrix P

    f1 <- lambda / 2 * (sum(beta^2)) # intermediate calculation of objective function
    f2 <- matrix(NA, 1, K)
    # intermediate calculation of  objective function
    for (i in 1:(K)) {
      f2[i] <- t(lp[, i]) %*% Q[, i]
    }

    objective[iter + 1] <- f1 - sum(f2) # appending/adding the value of objective function for this iteration to the vector objective.

    y1 <- rep(0, n) # stores the classification output for training data
    # calculating the classification output for training data

    y1 <- max.col(P, ties.method = "first") - 1


    Pt <- matrix(0, nt, K) # Pt stores P(yi=k|Xi)  for testing data
    # calculated  P(yi=k|Xi)  for testing data for each i in {1,2,...,ntest} and k in {0,...,K-1} and stored in Pt

    Pt <- exp(Xt %*% beta)

    colst <- colSums(t(Pt))
    Pt <- Pt / colst

    y1t <- rep(0, nt) # stores the classification output for testing data
    # calculating the classification output for testing data

    y1t <- max.col(Pt, ties.method = "first") - 1

    # calculating the training error and testing error in percentage and storing in error_train and error_test respectively for this iteration
    error_train[iter + 1] <- sum(y != y1)
    error_test[iter + 1] <- sum(yt != y1t)



  }
  # converting  objective, error_train, error_test matrices to vectors

  ## Return output
  ##########################################################################
  # beta - p x K matrix of estimated beta values after numIter iterations
  # error_train - (numIter + 1) length vector of training error % at each iteration (+ starting value)
  # error_test - (numIter + 1) length vector of testing error % at each iteration (+ starting value)
  # objective - (numIter + 1) length vector of objective values of the function that we are minimizing at each iteration (+ starting value)
  return(list(beta = beta, error_test = error_test))
}
