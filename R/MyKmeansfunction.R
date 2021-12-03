#' Function that implements K-means algorithm.
#'
#' @param X n * p matrix of data points
#' @param K Number of classes

#'
#' @return
#' n vector of cluster labels
#' @export
#'
#' @examples
#' X <- matrix(c(12, 5, 2, 3, 1, 3, 4, 4, 2, 5, 5, 7), 4, 3)
#'MyKmeans(X, K = 2)
#'# output:
#'#      [,1] [,2] [,3] [,4]
#'# [1,]    1    2    2    2
MyKmeans <- function(X, K) {
  M = NULL
  numIter = 100

  # Check whether M is NULL or not. If NULL, initialize based on K randomly selected points from X. If not NULL, check for compatibility with X dimensions.

  p <- ncol(X) # p is the number of columns of X
  n <- nrow(X) # n is the number of rows of X

  # If M is NULL, then initializing M based on K randomly selected points from X

  if (is.null(M) == TRUE) {
    M <- X[sample(n, size = K, replace = FALSE), ]
  }


  # If M is a vector, then converting it into a 1*p matrix

  if (is.vector(M) == TRUE) {
    M <- matrix(M, nrow = K, byrow = TRUE)
  }

  # Checking if M has compatible dimensions

  if (is.null(M) == FALSE) {
    if (ncol(M) != p | nrow(M) != K) {
      stop("M does not have compatible dimensions")
    }
  }





  # Implement K-means algorithm. It should stop when either (i) the centroids don't change from one iteration to the next, or (ii) the maximal number of iterations was reached, or (iii) one of the clusters has disappeared after one of the iterations (in which case the error message is returned)

  # "iter"  track of number of iteration of the MyKmeans algorithm

  iter <- 1

  for (iter in 1:numIter) {

    #Initializing matrices that will change in every iteration

    M1 <- matrix(NA, K, p) # "M1" is a dummy matrix to check if M is changed in consecutive iteration

    Y <- matrix(NA, 1, n)  # "Y" is the variable that stores the output classification/clustering vector


    distance <- matrix(NA, K, n)# "distance"  stores distance of each data from cluster centers.

    #  Calculated the distance of each data point from centroid

    P <- t(-2 * X %*% t(M)) #intermediate calculation
    Q <- colSums(t(M^2))    #intermediate calculation
    R <- matrix(Q,K,n) #intermediate calculation
    distance <-P+R




    # Finding the output classification/clustering vector for this iteration

    Y <- max.col(t(-distance))




    # Checking if some cluster is disappeared in this iteration

    for (i in 1:K) {

      if (!(i %in% Y)==TRUE) {

        stop("Some clusters are dissappeared ! Change the supplied value
  of M.")

      }

    }



    # Finding M according to new cluster centers and storing in M1

    for (j in 1:K) {

      M1[j, ] <- colMeans(X[which(Y %in% c(j)), , drop = F])

    }

    # Checking If M has changed in this iteration, updating  M if it is changed and returning Y if M has not changed

    if (sum(abs(M1-M))==0) {
      return(Y)

    }

    else {
      M <- M1
    }
    iter <- iter+1
  }

  # Return the vector of assignments Y
  return(Y)
}
