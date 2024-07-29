# Targeted Distribution

library(ks)
library(tidyverse)
library(splines)

source("functions/subsample_target_distrib.r")

# 1. Algorithm----


#' @param data dataset
#' @param score_name name of the column in data that contains the scores
#' @param target_fun target distribution function.
#' @param iter number of iterations.
#' @param draw if TRUE (default) the distribution of scores (gray bars) and the
#'  target distribution (in red) are plotted at each iteration.
#' @seed if not `NULL`, seed to use
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

# 2. First Example----

n <- 1e5 # Number of obs.
# Covariates
x1 <- rnorm(n)
x2 <- rnorm(n)
# True probabilities
p <- function(x1, x2) .4 * x1 - .2*x2
# Observed event
y <- rnorm(n,p(x1, x2), .4)
tb <- tibble(y = y, x1 = x1, x2 = x2)

# Linear model to predict the observed event:
reg <- lm(y ~ x1 + x2, data = tb)
scores <- predict(reg)
tb$scores <- scores

# Scores in [0,1]
tb_01 <- tb[(scores > 0) & (scores < 1), ]
data <- tb_01

# Sampling
B <- subset_target(data = data, probs_name = "scores", iter = 4)

# 3. Second Example----

n <- 1e6
x <- rbeta(n, 1, 2)
y <- rbinom(n, size = 1, prob = x)
base <- tibble(
  x = x,
  y = y,
  id = 1:n
)

# Let us assume that the scores are estimated using a logistic model.
reg <- glm(y ~ bs(x), data = base, family = binomial)
base$scores <- predict(reg, type = "response")

# Subsampling with probabilities following a Beta(2,1)
B <- subset_target(
  data = base,
  probs_name = "scores",
  iter = 1,
  target_fun = function(x) dbeta(x,2,1)
)

reg2 <- glm(y ~ bs(x), data = B, family = binomial)

val_x <- seq(0, 1, length = 601)
plot(
  val_x,
  predict(reg, type = "response", newdata = data.frame(x = val_x)),
  type = "l", lwd = 2
)
lines(
  val_x,
  predict(reg2, type = "response", newdata = data.frame(x = val_x)),
  type = "l", lwd = 2, col = "red"
)
