# Dependencies:
# tidyverse, gmish, pROC, purrr

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
  brier <- brier_score(obs = as.numeric(as.character(obs)), scores = scores)
  # gmish::brier(pred = scores, obs = obs) #same results

  # ICI
  ici_quiet <- purrr::quietly(gmish::ici)
  ici <- ici_quiet(pred = scores, obs = as.numeric(as.character(obs)))
  ici <- ici$result

  # Accuracy
  pred_class <- ifelse(scores > .5, yes = 1, no = 0)
  acc <- sum(diag(table(obs = obs, pred = pred_class))) / length(scores)

  tibble(
    mse = mse,
    mae = mae,
    acc = acc,
    AUC = AUC,
    brier = brier,
    ici = ici
  )

}

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
  h_phat <- hist(scores, breaks = seq(0, 1, length = m + 1), plot = FALSE)
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
