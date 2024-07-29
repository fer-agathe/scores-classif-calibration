#' Recalibrates scores using a calibration
#'
#' @param obs_train vector of observed events in the train set
#' @param scores_train vector of predicted probabilities in the train set
#' @param obs_calib vector of observed events in the calibration set
#' @param scores_calib vector of predicted probabilities in the calibration set
#' @param obs_test vector of observed events in the test set
#' @param scores_test vector of predicted probabilities in the test set
#' @param method recalibration method (`"platt"` for Platt-Scaling, `"isotonic"`
#'   for isotonic regression, `"beta"` for beta calibration, `"locfit"` for
#'   local regression)
#' @param params list of named parameters to use in the local regression (`nn`
#'   for fraction of nearest neighbors to use, `deg` for degree)
#' @param linspace vector of alues at which to compute the recalibrated scores
#' @returns list of three elements: recalibrated scores on the calibration set,
#'   recalibrated scores on the test set, and recalibrated scores on a segment
#'   of values
#'
#' @references
#' Platt, J. (1999). Probabilistic outputs for support vector
#' machines and comparisons to regularized likelihood methods. Advances in large
#' margin classifiers 10:61–74.
#'
#' Zadrozny, B. and Elkan, C. (2002).
#' Transforming classifier scores into accurate multiclass probability
#' estimates. In Proceedings of the eighth ACM SIGKDD international conference
#' on Knowledge discovery and data mining, 694–699.
#'
#' Kull, M., Filho, T. M. S. and Flach, P. (2017). Beyond sigmoids: How to
#' obtain well-calibrated probabilities from binary classifiers with beta
#' calibration. Electronic Journal of Statistics 11: 5052 – 5080,
#' doi:10.1214/17-EJS1338SI.
#'
#' @importFrom stats glm binomial isoreg as.stepfun rnorm
#' @importFrom utils capture.output
#' @importFrom betacal beta_calibration beta_predict
#' @importFrom locfit locfit lp
#' @importFrom tibble tibble
#' @importFrom dplyr mutate
#' @importFrom stringr str_c
#' @export
recalibrate <- function(obs_train,
                        pred_train,
                        obs_calib,
                        pred_calib,
                        obs_test,
                        pred_test,
                        method = c("platt", "isotonic", "beta", "locfit"),
                        params = NULL,
                        linspace = NULL) {

  if (is.null(linspace)) linspace <- seq(0, 1, length.out = 101)
  data_train <- tibble(d = obs_train, scores = pred_train)
  data_calib <- tibble(d = obs_calib, scores = pred_calib)
  data_test <- tibble(d = obs_test, scores = pred_test)

  # Recalibrator trained on calibration data
  if (method == "platt") {
    # Recalibrator
    lr <- glm(
      d ~ scores, family = binomial(link = 'logit'), data = data_calib
    )
    # Recalibrated scores on calib/test sets
    score_c_train <- predict(lr, newdata = data_train, type = "response")
    score_c_calib <- predict(lr, newdata = data_calib, type = "response")
    score_c_test <- predict(lr, newdata = data_test, type = "response")
    # Recalibrated scores on [0,1]
    score_c_linspace <- predict(
      lr, newdata = tibble(scores = linspace), type = "response"
    )
  } else if (method == "isotonic") {
    iso <- isoreg(x = data_calib$scores, y = data_calib$d)
    fit_iso <- as.stepfun(iso)
    score_c_train <- fit_iso(data_train$scores)
    score_c_calib <- fit_iso(data_calib$scores)
    score_c_test <- fit_iso(data_test$scores)
    score_c_linspace <- fit_iso(linspace)
  } else if (method == "beta") {
    capture.output({
      bc <- try(beta_calibration(
        p = data_calib$scores,
        y = data_calib$d,
        parameters = "abm" # 3 parameters a, b & m
      ))
    })
    if (!inherits(bc, "try-error")) {
      score_c_train <- beta_predict(p = data_train$scores, bc)
      score_c_calib <- beta_predict(p = data_calib$scores, bc)
      score_c_test <- beta_predict(p = data_test$scores, bc)
      score_c_linspace <- beta_predict(p = linspace, bc)
    } else {
      score_c_train <- score_c_calib <- score_c_test <- score_c_linspace <- NA
    }

  } else if (method == "locfit") {
    noise_scores <- data_calib$scores + rnorm(nrow(data_calib), 0, 0.01)
    noise_data_calib <- data_calib %>% mutate(scores = noise_scores)
    locfit_reg <- locfit(
      formula = d ~ lp(scores, nn = params$nn, deg = params$deg),
      kern = "rect", maxk = 200, data = noise_data_calib
    )
    score_c_train <- predict(locfit_reg, newdata = data_train)
    score_c_train[score_c_train < 0] <- 0
    score_c_train[score_c_train > 1] <- 1
    score_c_calib <- predict(locfit_reg, newdata = data_calib)
    score_c_calib[score_c_calib < 0] <- 0
    score_c_calib[score_c_calib > 1] <- 1
    score_c_test <- predict(locfit_reg, newdata = data_test)
    score_c_test[score_c_test < 0] <- 0
    score_c_test[score_c_test > 1] <- 1
    score_c_linspace <- predict(locfit_reg, newdata = linspace)
    score_c_linspace[score_c_linspace < 0] <- 0
    score_c_linspace[score_c_linspace > 1] <- 1
  } else {
    stop(str_c(
      'Wrong method. Use one of the following:',
      '"platt", "isotonic", "beta", "locfit"'
    ))
  }

  # Format results in tibbles:
  # For train set
  tb_score_c_train <- tibble(
    d = obs_train,
    p_u = pred_train,
    p_c = score_c_train
  )
  # For calibration set
  tb_score_c_calib <- tibble(
    d = obs_calib,
    p_u = pred_calib,
    p_c = score_c_calib
  )
  # For test set
  tb_score_c_test <- tibble(
    d = obs_test,
    p_u = pred_test,
    p_c = score_c_test
  )
  # For linear space
  tb_score_c_linspace <- tibble(
    linspace = linspace,
    p_c = score_c_linspace
  )

  list(
    tb_score_c_train = tb_score_c_train,
    tb_score_c_calib = tb_score_c_calib,
    tb_score_c_test = tb_score_c_test,
    tb_score_c_linspace = tb_score_c_linspace
  )

}
