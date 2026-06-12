partial_mcc_matrix <- function(A, BX) {
  A <- scale(A)
  BX <- scale(BX)
  
  n <- nrow(A)
  p <- ncol(A)
  C <- matrix(NA, nrow = p, ncol = p)
  
  for (i in 1:p) {
    idx_i <- setdiff(1:p, i)
    
    # Residuals of A[,i] ~ A[,-i]
    model_A <- lm(A[, i] ~ A[, idx_i])
    r_A <- scale(resid(model_A), center = TRUE, scale = TRUE)
    
    for (j in 1:p) {
      # Residuals of BX[,j] ~ A[,-i]
      model_BX <- lm(BX[, j] ~ A[, idx_i])
      r_BX <- scale(resid(model_BX), center = TRUE, scale = TRUE)
      
      # Cosine similarity = correlation
      C[i, j] <- sum(r_A * r_BX) / (sqrt(sum(r_A^2)) * sqrt(sum(r_BX^2)))
    }
  }
  
  rownames(C) <- colnames(A)
  colnames(C) <- colnames(BX)
  
  return(C)
}

optimal_partial_mcc_matrix <- function(A, BX) {
  C <- partial_mcc_matrix(A, BX)  # (p x p) matrix
  p <- ncol(C)
  
  # Solve the assignment problem (maximize trace)
  C_shifted <- C - min(C) # ensure all values are non-negative
  assignment <- solve_LSAP(C_shifted, maximum = TRUE)  # Hungarian algorithm
  
  # Permute columns of BX according to optimal assignment
  BX_perm <- BX[, assignment]
  
  # Recompute the final matrix after optimal alignment
  C_opt <- partial_mcc_matrix(A, BX_perm)
  
  return(list(
    matrix = C_opt,
    permutation = assignment,
    mean_partial_mcc = mean(diag(C_opt))
  ))
}
