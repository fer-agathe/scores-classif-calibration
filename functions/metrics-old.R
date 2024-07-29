# 1. Standard Metrics For Classification / Regression----

#' Computes goodness of fit metrics
#'
#' @param true_prob true probabilities. If `NULL` (default), True MSE is not
#'   computed and the `NA` value is returned for this metric
#' @param obs observed values (binary outcome)
#' @param pred predicted scores
#' @param threshold classification threshold (default to `.5`)
#'
#' @returns tibble with MSE, accuracy, missclassification rate, sensititity
#'   (TPR), specificity (TNR), FPR, and used threshold
#'
#' @importFrom tibble tibble
#' @importFrom dplyr filter pull count
#' @importFrom pROC auc
#' @export
compute_gof <- function(true_prob = NULL,
                        obs,
                        pred,
                        threshold = .5) {

  # MSE
  if (!is.null(true_prob)) {
    mse <- mean((true_prob - pred)^2)
  } else {
    mse = NA
  }

  # pred_class <- as.numeric(pred > threshold)
  # confusion_tb <- tibble(
  #   obs = obs,
  #   pred = pred_class
  # ) |>
  #   count(obs, pred)
  #
  # TN <- confusion_tb |> filter(obs == 0, pred == 0) |> pull(n)
  # TP <- confusion_tb |> filter(obs == 1, pred == 1) |> pull(n)
  # FP <- confusion_tb |> filter(obs == 0, pred == 1) |> pull(n)
  # FN <- confusion_tb |> filter(obs == 1, pred == 0) |> pull(n)
  #
  # if (length(TN) == 0) TN <- 0
  # if (length(TP) == 0) TP <- 0
  # if (length(FP) == 0) FP <- 0
  # if (length(FN) == 0) FN <- 0
  #
  # n_pos <- sum(obs == 1)
  # n_neg <- sum(obs == 0)
  #
  # # Accuracy
  # acc <- (TP + TN) / (n_pos + n_neg)
  # # Missclassification rate
  # missclass_rate <- 1 - acc
  # # Sensitivity (True positive rate)
  # # proportion of actual positives that are correctly identified as such
  # TPR <- TP / n_pos
  # # Specificity (True negative rate)
  # # proportion of actual negatives that are correctly identified as such
  # TNR <- TN / n_neg
  # # False positive Rate
  # FPR <- FP / n_neg
  # AUC
  AUC <- as.numeric(pROC::auc(obs, pred, levels = c("0", "1")))

  tibble(
    mse = mse,
    AUC = AUC
  )
}


# 2. Metrics for Calibration----

## Brier Score----

#' Brier Score
#'
#' The Brier Score \citep{brier_1950}, is expressed as: \deqn{\text{BS} =
#' \frac{1}{n}\sum_{i=1}^{n} \big(\hat{s}(\mathbf{x}_i) - d_i\big)^{2}} where
#' \eqn{d_i \in \{0,1\}} is the observed event for observation \eqn{i}.
#'
#' @param scores vector of scores
#' @param obs vector of observed binary events
#'
#' @references Brier, G. W. (1950). Verification of forecasts expressed in terms
#' of probability. Monthly Weather Review 78: 1–3.
#'
#' @export
brier_score <- function(obs, scores) mean((scores - obs)^2)

## Expected Calibration Error (ECE)----

#' Computes summary statistics for binomial observed data and predicted scores
#' returned by a model
#'
#' @param obs vector of observed events
#' @param scores vector of predicted probabilities
#' @param k number of classes to create (quantiles, default to `10`)
#' @param threshold classification threshold (default to `.5`)
#'
#' @importFrom tibble tibble
#' @importFrom stats quantile
#' @importFrom dplyr group_by slice_tail ungroup mutate summarise arrange n
#'
#' @returns A tibble where each row corresponds to a bin, and each column
#'   represents:
#' \itemize{
#'   \item \code{score_class}: Level of the decile that the bin represents.
#'   \item \code{nb}: Number of observations.
#'   \item \code{mean_obs}: Average of observations (proportion of positive events).
#'   \item \code{mean_score}: Average predicted score (confidence).
#'   \item \code{sum_obs}: Number of positive events.
#'   \item \code{accuracy}: Accuracy (share of correctly predicted, using the threshold).
#' }
#' @export
get_summary_bins <- function(obs,
                             scores,
                             k = 10,
                             threshold = .5) {
  breaks <- quantile(scores, probs = (0:k) / k)
  tb_breaks <- tibble(breaks = breaks, labels = 0:k) |>
    group_by(breaks) |>
    slice_tail(n = 1) |>
    ungroup()

  x_with_class <- tibble(
    obs = obs,
    score = scores,
  ) |>
    mutate(
      score_class = cut(
        score,
        breaks = tb_breaks$breaks,
        labels = tb_breaks$labels[-1],
        include.lowest = TRUE
      ),
      pred_class = ifelse(score > threshold, 1, 0),
      correct_pred = obs == pred_class
    )

  x_with_class |>
    group_by(score_class) |>
    summarise(
      nb = n(),
      mean_obs = mean(obs),
      mean_score = mean(score), # confidence
      sum_obs = sum(obs),
      accuracy = mean(correct_pred)
    ) |>
    ungroup() |>
    mutate(
      score_class = as.character(score_class) |> as.numeric()
    ) |>
    arrange(score_class)
}

#' Expected Calibration Error
#'
#' @description Given a sample size \eqn{n}, the Expected Calibration Error
#'   (ECE) is determined using two metrics within each bin \eqn{b\in\{1, \ldots,
#'   B\}} of quantile-binned predicted scores \eqn{\hat{s}(\mathbf{x})}: accuracy
#'   \eqn{\text{acc}(b)}, which measures the average of empirical probabilities
#'   or fractions of correctly predicted classes, and confidence
#'   \eqn{\text{conf}(b)}, indicating the model's average confidence within bin
#'   \eqn{b} by averaging predicted scores. The ECE is then computed as the
#'   average over the bins using: \deqn{\text{ECE} = \sum_{b=1}^{B}
#'   \frac{n_b}{n} \mid \text{acc}(b) - \text{conf}(b) \mid}
#'   where \eqn{n_b} is the number of observations in bin \eqn{b}. Given that
#'   each bin \eqn{b} is associated with set \eqn{\mathcal{I}_b} containing the
#'   indices of instances within that bin, \deqn{\text{acc}(b) = \frac{1}{n_b}
#'   \sum_{i \in \mathcal{I}_b} \mathds{1}_{\hat{d}_i = d_i} \quad
#'   \text{and}\quad \text{conf}(b) = \frac{1}{n_b} \sum_{i \in \mathcal{I}_b}
#'   \hat{s}(\mathbf{x}_i)} are, respectively, the accuracy and the confidence
#'   of the model in bin \eqn{b}. The predicted class \eqn{\hat{d}_i} for
#'   observation \eqn{i} is determined based on a classification threshold
#'   \eqn{\tau\in [0,1]}, where \eqn{\hat{d}_i = 1} if
#'   \eqn{\hat{s}(\mathbf{x}_i) \geq \tau} and \eqn{0} otherwise.
#'
#'
#' @param obs vector of observed events
#' @param scores vector of predicted probabilities
#' @param k number of classes to create (quantiles, default to `10`)
#' @param threshold classification threshold (default to `.5`)
#'
#' @references Pakdaman Naeini, M., Cooper, G. and Hauskrecht, M. (2015).
#'   Obtaining well calibrated probabilities using bayesian binning.
#'   Proceedings of the AAAI Conference on Artificial Intelligence 29:
#'   2901–2907, doi:10.1609/aaai.v29i1.9602.
#'
#' @importFrom dplyr mutate summarise pull
#' @seealso [get_summary_bins()] for computing statistics on each bin
#' @export
e_calib_error <- function(obs,
                          scores,
                          k = 10,
                          threshold = .5) {
  summary_bins <- get_summary_bins(
    obs = obs, scores = scores, k = k, threshold = .5
  )
  summary_bins |>
    mutate(ece_bin = nb * abs(accuracy - mean_score)) |>
    summarise(ece = 1 / sum(nb) * sum(ece_bin)) |>
    pull(ece)
}



## Quantile-based Mean Squared Error----

#' Quantile-Based MSE
#'
#' @description Given a sample size \eqn{n}, the Quantile-based Mean Squared
#'   Error (QMSE) is determined using two metrics within each bin
#'   \eqn{b\in\{1,\ldots, B\}} of quantile-binned predicted scores
#'   \eqn{\hat{s}(\mathbf{x})}: the average of the observed events in the bin,
#'   and the confidence in the bin. The average of observed events is calculated
#'   as \deqn{\bar{d}_b = \frac{1}{n_b} \sum_{i \in \mathcal{I}_b} d_i,}, and
#'   the confidence is calculated as \deqn{\text{conf}(b) = \frac{1}{n_b} \sum_{i \in \mathcal{I}_b}
#'   \hat{s}(\mathbf{x}_i)}.
#'   The QMSE writes: \deqn{\text{QMSE} = \frac{1}{n}\sum_{b=1}^{B} n_b
#'   \left[\bar{d}_b - \text{conf}(b)\right]^2.}
#'
#' @param obs vector of observed events
#' @param scores vector of predicted probabilities
#' @param k number of classes to create (quantiles, default to `10`)
#' @param threshold classification threshold (default to `.5`)
#'
#' @importFrom dplyr mutate summarise pull
#' @seealso [get_summary_bins()] for computing statistics on each bin
#' @export
qmse_error <- function(obs,
                       scores,
                       k = 10,
                       threshold = .5) {
  summary_bins <- get_summary_bins(
    obs = obs, scores = scores, k = k, threshold = .5
  )
  summary_bins |>
    mutate(qmse_bin = nb * (mean_obs - mean_score)^2) |>
    summarise(qmse = 1/sum(nb) * sum(qmse_bin)) |>
    pull(qmse)
}

## Weighted Mean Squared Error----

#' Weighted Mean Squared Error
#'
#' @param obs vector of observed events
#' @param scores vector of predicted probabilities
#' @param tau value at which to compute the confidence interval
#' @param nn fraction of nearest neighbors
#' @param prob level of the confidence interval (default to `.95`)
#' @param method Which method to use to construct the interval. Any combination
#'  of c("exact", "ac", "asymptotic", "wilson", "prop.test", "bayes", "logit",
#'  "cloglog", "probit") is allowed. Default is "all".
#' @returns A tibble with a single row that corresponds to estimations made in
#' the neighborhood of a probability $p=\tau$, using the fraction `nn` of
#' neighbors, where the columns are:
#' \itemize{
#'   \item \code{score}: Score tau in the neighborhood of which statistics are computed.
#'   \item \code{mean}: Estimation of $E(d | s(x) = \tau)$.
#'   \item \code{lower}: Lower bound of the confidence interval.
#'   \item \code{upper}: Upper bound of the confidence interval.
#' }
#' @importFrom binom binom.confint
#' @importFrom tibble tibble
#' @importFrom dplyr mutate relocate
#' @export
local_ci_scores <- function(obs,
                            scores,
                            tau,
                            nn,
                            prob = .95,
                            method = "probit") {

  # Identify the k nearest neighbors based on hat{p}
  k <- round(length(scores) * nn)
  rgs <- rank(abs(scores - tau), ties.method = "first")
  idx <- which(rgs <= k)

  binom.confint(
    x = sum(obs[idx]),
    n = length(idx),
    conf.level = prob,
    methods = method
  )[, c("mean", "lower", "upper")] |>
    tibble() |>
    mutate(xlim = tau) |>
    relocate(xlim, .before = mean)
}

#' Compute the Weighted Mean Squared Error to assess the calibration of a model
#'
#' @param local_scores tibble with expected scores obtained with the
#'   `local_ci_scores()` function
#' @param scores vector of raw predicted probabilities
#'
#' @importFrom stats density
#' @importFrom dplyr mutate summarise pull
#' @export
weighted_mse <- function(local_scores, scores) {
  # To account for border bias (support is [0,1])
  scores_reflected <- c(-scores, scores, 2 - scores)
  dens <- density(
    x = scores_reflected, from = 0, to = 1,
    n = length(local_scores$xlim)
  )
  # The weights
  weights <- dens$y
  local_scores |>
    mutate(
      wmse_p = (xlim - mean)^2,
      weight = !!weights
    ) |>
    summarise(wmse = sum(weight * wmse_p) / sum(weight)) |>
    pull(wmse)
}

#' Calibration score using Local Regression
#'
#' @description The Local Calibration Score (LCS) is based on the calibration
#'   curve constructed using `locfit()`. The calculation of the LCS relies on
#'   the disparities between this curve and the bisector, in the range from 0 to
#'   1, weighted by the density of the predicted scores
#'   \eqn{\hat{s}(\mathbf{x})}. To execute this, a local regression of degree 0,
#'   denoted as \eqn{\hat{g}}, is fitted to the predicted scores
#'   \eqn{\hat{s}(\mathbf{x})}. This fit is then applied to a vector of linearly
#'   spaced values in the interval \eqn{[0,1]}, where each point is denoted by
#'   \eqn{l_i} related to observation \eqn{i \in \{1, \ldots, n\}}. The LCS is
#'   calculated by averaging the squared differences between each predicted
#'   score \eqn{\hat{g}(l_i)} and its corresponding linearly spaced value
#'   \eqn{l_i}, weighted by the density of the observed scores, \eqn{w_i}:
#'   \deqn{\text{LCS} = \sum_{i=1}^{n}w_i \big(\hat{g}(l_i) - l_i\big)^2.}
#'
#' @param obs vector of observed events
#' @param scores vector of predicted probabilities
#'
#' @importFrom stats density rnorm weighted.mean
#' @importFrom tibble tibble
#' @importFrom locfit locfit lp
#' @export
local_calib_score <- function(obs,
                              scores) {

  # Add a little noise to the scores, to avoir crashing R
  scores <- scores + rnorm(length(scores), 0, .001)
  locfit_0 <- locfit(
    formula = d ~ lp(scores, nn = 0.15, deg = 0),
    kern = "rect", maxk = 200,
    data = tibble(
      d = obs,
      scores = scores
    )
  )
  # Predictions on [0,1]
  linspace_raw <- seq(0, 1, length.out = 100)
  # Restricting this space to the range of observed scores
  keep_linspace <- which(linspace_raw >= min(scores) & linspace_raw <= max(scores))
  linspace <- linspace_raw[keep_linspace]

  locfit_0_linspace <- predict(locfit_0, newdata = linspace)
  locfit_0_linspace[locfit_0_linspace > 1] <- 1
  locfit_0_linspace[locfit_0_linspace < 0] <- 0

  # Squared difference between predicted value and the bisector, weighted by
  # the density of values
  scores_reflected <- c(-scores, scores, 2 - scores)
  dens <- density(
    x = scores_reflected, from = 0, to = 1,
    n = length(linspace_raw)
  )
  # The weights
  weights <- dens$y[keep_linspace]

  weighted.mean((linspace - locfit_0_linspace)^2, weights)
}

## Wrapper----

#' Computes the calibration metrics for a set of observed and predicted
#' probabilities
#'
#' @returns
#' \itemize{
#'   \item \code{mse}: Mean Squared Error based on true probability.
#'   \item \code{lcs}: Local Calibration Score.
#'   \item \code{brier}: Brier score.
#'   \item \code{ece}: Expected Calibration Error.
#'   \item \code{qmse}: MSE on bins defined by the quantiles of the predicted scores.
#'   \item \code{wmse}: MSE weighted by the density of the predicted scores.
#' }
#'
#' @param obs observed events
#' @param scores predicted scores
#' @param true_probas true probabilities from the PGD (to compute MSE)
#' @param linspace vector of values at which to compute the WMSE
#' @param k number of classes (bins) to create (default to `10`)
#'
#' @importFrom purrr map
#' @importFrom tibble tibble
#' @importFrom dplyr bind_rows
#'
#' @export
compute_calib_metrics <- function(obs,
                            scores,
                            true_probas = NULL,
                            linspace = NULL,
                            k = 10) {

  ## True MSE
  if (!is.null(true_probas)) {
    mse <- mean((true_probas - scores)^2)
  } else {
    mse <- NA
  }

  brier <- brier_score(obs = obs, scores = scores)
  if (length(unique(scores)) > 1) {
    ece <- e_calib_error(obs = obs, scores = scores, k = k, threshold = .5)
    qmse <- qmse_error(obs = obs, scores = scores, k = k, threshold = .5)
  } else {
    ece <- NA
    qmse <- NA
  }


  lcs <- local_calib_score(obs = obs, scores = scores)

  if (is.null(linspace)) linspace <- seq(0, 1, length.out = 101)
  expected_events <- map(
    .x = linspace,
    .f = ~local_ci_scores(
      obs = obs,
      scores = scores,
      tau = .x, nn = .15, prob = .95, method = "probit")
  ) |>
    bind_rows()
  wmse <- weighted_mse(local_scores = expected_events, scores = scores)
  tibble(
    mse = mse,
    lcs = lcs,
    brier = brier,
    ece = ece,
    qmse = qmse,
    wmse = wmse
  )
}


# 3. Both----
#' Computes the calibration metrics for a set of observed and predicted
#' probabilities
#'
#' @returns
#' \itemize{
#'   \item \code{mse}: True Mean Squared Error based on true probability.
#'   \item \code{acc}: accuracy with a .5 probability threshold
#'   \item \code{AUC}: Area Under the ROC Curve
#'   \item \code{lcs}: Local Calibration Score.
#'   \item \code{brier}: Brier score.
#' }
#'
#' @param obs observed events
#' @param scores predicted scores
#' @param true_probas true probabilities from the PGD (to compute MSE)
#'
#' @importFrom purrr map
#' @importFrom tibble tibble
#' @importFrom dplyr bind_rows
#'
#' @export
compute_metrics <- function(obs,
                            scores,
                            true_probas = NULL) {

  # True MSE
  if (!is.null(true_probas)) {
    mse <- mean((true_probas - scores)^2)
  } else {
    mse <- NA
  }

  # True MAE
  if (!is.null(true_probas)) {
    mae <- mean(abs(true_probas - scores))
  } else {
    mae <- NA
  }

  # AUC
  AUC <- pROC::auc(obs, scores, levels = c("0", "1"), quiet = TRUE) |>
    as.numeric()

  # Brier Score
  brier <- brier_score(obs = obs, scores = scores)

  # LCS
  lcs <- local_calib_score(obs = obs, scores = scores)

  # Accuracy
  pred_class <- ifelse(scores > .5, yes = 1, no = 0)
  acc <- sum(diag(table(obs = obs, pred = pred_class))) / length(scores)

  tibble(
    mse = mse,
    mae = mae,
    acc = acc,
    AUC = AUC,
    brier = brier,
    lcs = lcs
  )

}

# 4. Dispersion metrics----
#' Computes the dispersion metrics for a set of observed and predicted
#' probabilities
#'
#' @returns
#' \itemize{
#'   \item \code{inter_quantile_25_75}: Difference of inter-quantile between 25% and 75%
#'   \item \code{inter_quantile_10_90}: Difference of inter-quantile between 10% and 90%
#'   \item \code{KL_10_true_probas}: KL of of predicted probabilities w.r. to true probabilities with 10 bins
#'   \item \code{KL_10_scores}: KL of of true probabilities w.r. to predicted probabilities with 10 bins
#'   \item \code{KL_20_true_probas}: KL of of predicted probabilities w.r. to true probabilities with 20 bins
#'   \item \code{KL_20_scores}: KL of of true probabilities w.r. to predicted probabilities with 20 bins
#'   \item \code{ind_cov}: Difference between the variance of true probabilities and the covariance between true probabilities and predicted scores
#' }
#'
#' @param true_probas true probabilities from simulations
#' @param scores predicted scores
#'
dispersion_metrics <- function(true_probas, scores){

  # Inter-quantiles
  inter_q_80 <- diff(quantile(scores, c(.9, .1))) /
    diff(quantile(true_probas, c(.9, .1)))
  inter_q_50 <- diff(quantile(scores, c(.75,.25))) /
    diff(quantile(true_probas, c(.75, .25)))

  # KL divergences
  m <- 10 # Number of bins
  h_p <- hist(true_probas, breaks = seq(0, 1, length = m + 1), plot = FALSE)
  h_phat <- hist(scores, breaks=seq(0, 1, length = m + 1), plot = FALSE)
  # Densities
  h1 <- rbind(h_phat$density / m, h_p$density / m) # Reference : true probabilities
  h2 <- rbind(h_p$density / m, h_phat$density / m) # Reference : predicted scores
  KL_10_true_probas <- distance(
    h1, method = "kullback-leibler", unit = "log2", mute.message = TRUE)
  KL_10_scores <- distance(
    h2, method = "kullback-leibler", unit = "log2", mute.message = TRUE)


  m <- 20 # Number of bins
  h_p <- hist(true_probas,breaks = seq(0, 1, length = m + 1), plot = FALSE)
  h_phat <- hist(scores, breaks = seq(0, 1, length = m + 1), plot = FALSE)
  # Densities
  h1 <- rbind(h_phat$density / m,h_p$density / m) # Reference : true probabilities
  h2 <- rbind(h_p$density / m, h_phat$density / m) # Reference : predicted scores
  KL_20_true_probas <- distance(
    h1, method = "kullback-leibler", unit = "log2", mute.message = TRUE)
  KL_20_scores <- distance(
    h2, method = "kullback-leibler", unit = "log2", mute.message = TRUE)

  # Indicator of the difference between variance and covariance
  var_p <- var(true_probas)
  cov_p_phat <- cov(true_probas, scores)
  ind_cov <- abs(cov_p_phat - var_p)

  # Collection
  dispersion_metrics <- tibble(
    "inter_quantile_25_75" = as.numeric(inter_q_50),
    "inter_quantile_10_90" = as.numeric(inter_q_80),
    "KL_10_true_probas" = as.numeric(KL_10_true_probas),
    "KL_10_scores" = as.numeric(KL_10_scores),
    "KL_20_true_probas" = as.numeric(KL_20_true_probas),
    "KL_20_scores" = as.numeric(KL_20_scores),
    "ind_cov" = ind_cov
    )

  dispersion_metrics
}


#' Computes \hat{P}(q_1 < s < q_2)
#'
#' @param s scores
#' @param q1 lower quantile
#' @param q2 upper quantile (default to 1-q2)
prop_btw_quantiles <- function(s, q1, q2 = 1 - q1) {
  tibble(q1 = q1, q2 = q2, freq = mean(s < q2 & s > q1))
}
