#' @param data dataset
#' @param probs_name name of the column in data that contains the observed
#'  probabilities
#' @param target_fun target distribution function.
#' @param iter number of iterations.
#' @param draw if TRUE (default) the distribution of scores (gray bars) and the
#'  target distribution (in red) are plotted at each iteration.
#' @seed if not `NULL`, seed to use
#' @param verbose if `FALSE`, size of subsamplings at each iteration and KS test
#'  results are hiddent
subset_target <- function(data,
                          probs_name,
                          target_fun = function(x) dbeta(x,2,2),
                          iter = 1,
                          draw = TRUE,
                          seed = NULL,
                          verbose = TRUE){
  select <- rep(nrow(data),iter + 1)
  if (!is.null(seed)) set.seed(seed)

  # Get the scores from the dataset
  probs_01 <- data |> pull(!!probs_name)
  if (verbose == TRUE) cat("1) Size ...... ", nrow(data), "\n", sep = "")

  # Kolmogorov-Smirnov Test
  fun <- Vectorize(function(x) integrate(target_fun, 0, x)$value)
  K <- ks.test(probs_01, fun)

  if (verbose) {
    cat("1)  ks ............ ", K$statistic, "\n", sep = "")
    cat("1)  (pvalue) ...... ", K$p.value, "\n", sep = "")
  }

  if (draw) {
    # Histogram of scores (gray) and target distribution (red)
    hist(probs_01,probability = TRUE, xlab = "", ylab = "", main = "Initial")
    val_x <- seq(0,1,length = 601)
    lines(val_x,target_fun(val_x), col = "red")
  }

  data_subset <- data

  for (k in 1:iter) {
    n <- nrow(data_subset)
    initial_density <- kde(x = probs_01, eval.points = probs_01)
    # Probability to include each observation in the current subset
    prob_acceptation <- target_fun(probs_01) / initial_density$estimate
    prob_acceptation <- pmin(prob_acceptation / max(prob_acceptation), 1)
    # For each scores from the current data subset, decide whether or not to
    # include it based on a random draw from a Ber(prob_acceptation)
    index_acceptation <- rbinom(n, size = 1, prob = prob_acceptation)
    # Use this index to keep only the selected data
    data_subset <- data_subset[which(index_acceptation ==1 ), ]
    select[k + 1] <- nrow(data_subset)
    probs_01 <- data_subset |> pull(!!probs_name)
    if (verbose == TRUE)
      cat(k + 1, ") Size ...... ", nrow(data_subset), "\n", sep = "")
    # Kolmogorov-Smirnov Test
    K <- ks.test(probs_01, fun)
    if (verbose) {
      cat(k + 1, ")   ks ............ ", K$statistic, "\n", sep = "")
      cat(k + 1, ")   (pvalue) ...... ", K$p.value, "\n", sep = "")
    }
    if (draw) {
      hist(
        probs_01, probability = TRUE, xlab = "", ylab = "",
        main = paste("Iteration ", k)
      )
      val_x <- seq(0, 1, length = 601)
      lines(val_x, target_fun(val_x), col = "red")
    }
  }
  data_subset
}
