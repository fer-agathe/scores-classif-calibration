# Estimations with real-world data

library(tidyverse)
library(philentropy)
library(ranger)
library(xgboost)
library(pbapply)
library(parallel)
library(gam)
library(gamsel)

# Colours for train/test
colour_samples <- c(
  "Train" = "#0072B2",
  "Test" = "#D55E00"
)

# Functions----

## Metrics----

# metrics
source("functions/metrics.R")
# Pre-processing functions and estimation functions
source("functions/real-data.R")

### Performance and Calibration Metrics----

#' Get the performance and calibration metrics for estimated scores
#'
#' @param scores_train vector of scores on the train test
#' @param scores_valid vector of scores on the validation test
#' @param scores_test vector of scores on the test test
#' @param tb_train train set
#' @param tb_valid valdation set
#' @param tb_test test set
#' @param target_name name of target variable
get_perf_metrics <- function(scores_train,
                             scores_valid,
                             scores_test,
                             tb_train,
                             tb_valid,
                             tb_test,
                             target_name) {
  # We add very small noise to predicted scores
  # otherwise the local regression may crash
  scores_train_noise <- scores_train +
    runif(n = length(scores_train), min = 0, max = 0.01)
  scores_train_noise[scores_train_noise > 1] <- 1
  metrics_train <- compute_metrics(
    obs = tb_train |> pull(!!target_name),
    scores = scores_train_noise, true_probas = NULL
  ) |> mutate(sample = "train")

  scores_valid_noise <- scores_valid +
    runif(n = length(scores_valid), min = 0, max = 0.01)
  scores_valid_noise[scores_valid_noise > 1] <- 1
  metrics_valid <- compute_metrics(
    obs = tb_valid |> pull(!!target_name),
    scores = scores_valid_noise, true_probas = NULL
  ) |> mutate(sample = "validation")

  scores_test_noise <- scores_test +
    runif(n = length(scores_test), min = 0, max = 0.01)
  scores_test_noise[scores_test_noise > 1] <- 1
  metrics_test <- compute_metrics(
    obs = tb_test |> pull(!!target_name),
    scores = scores_test_noise, true_probas = NULL
  ) |> mutate(sample = "test")

  tb_metrics <- metrics_train |>
    bind_rows(metrics_valid) |>
    bind_rows(metrics_test)
  tb_metrics
}

#' Computes the dispersion and divergence metrics for a vector of scores and
#' a Beta distribution
#'
#' @param shape_1 first parameter of the beta distribution
#' @param shape_2 second parameter of the beta distribution
#' @param scores predicted scores
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
dispersion_metrics_beta <- function(shape_1 = 1, shape_2 = 1, scores){

  # Inter-quantiles
  inter_q_80 <- diff(quantile(scores, c(.9, .1))) /
    diff(qbeta(c(.9, .1), shape_1, shape_2))
  inter_q_50 <- diff(quantile(scores, c(.75,.25))) /
    diff(qbeta(c(.75,.25), shape_1, shape_1))

  # KL divergences
  m <- 10 # Number of bins
  h_phat <- hist(scores, breaks = seq(0, 1, length = m + 1), plot = FALSE)
  h_p <- list(breaks = h_phat$breaks, mids = h_phat$mids)
  h_p$density = diff(pbeta(h_p$breaks, shape_1, shape_2))
  h_p$counts =  h_p$density*length(scores)

  # Densities
  h1 <- rbind(h_phat$density / m, h_p$density / m) # Reference : true probabilities
  h2 <- rbind(h_p$density / m, h_phat$density / m) # Reference : predicted scores
  KL_10_true_probas <- distance(
    h1, method = "kullback-leibler", unit = "log2", mute.message = TRUE)
  KL_10_scores <- distance(
    h2, method = "kullback-leibler", unit = "log2", mute.message = TRUE)


  m <- 20 # Number of bins
  h_phat <- hist(scores, breaks = seq(0, 1, length = m + 1), plot = FALSE)
  h_p <- list(breaks = h_phat$breaks, mids = h_phat$mids)
  h_p$density = diff(pbeta(h_p$breaks, shape_1, shape_2))
  h_p$counts =  h_p$density * length(scores)
  # Densities
  h1 <- rbind(h_phat$density / m, h_p$density) # Reference : true probabilities
  h2 <- rbind(h_p$density, h_phat$density / m) # Reference : predicted scores
  KL_20_true_probas <- distance(
    h1, method = "kullback-leibler", unit = "log2", mute.message = TRUE)
  KL_20_scores <- distance(
    h2, method = "kullback-leibler", unit = "log2", mute.message = TRUE)

  # Indicator of the difference between variance and covariance
  var_p <- shape_1 * shape_2 / ((shape_1 + shape_2)^2 * (shape_1 + shape_2 + 1))
  cov_p_phat <- cov(
    qbeta(
      rank(scores, ties.method = "average") / (1 + length(scores)),
      shape_1,
      shape_2),
    scores
  )
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

#' Computes the dispersion and divergence metrics between estimated scores and
#' the Beta distributions whose parameters were estimated using scores estimated
#' with a GLM-loistic, a GAM and a GAM with model selection.
#' (helper function)
#'
#' @param priors priors obtained with `get_beta_fit()`
#' @param scores estimated scores from a model
disp_metrics_dataset <- function(priors, scores) {
  # GLM priors
  shape_1_glm <- priors$mle_glm$estimate["shape1"]
  shape_2_glm <- priors$mle_glm$estimate["shape2"]
  # GAM priors
  shape_1_gam <- priors$mle_gam$estimate["shape1"]
  shape_2_gam <- priors$mle_gam$estimate["shape2"]
  # GAMSEL priors
  shape_1_gamsel <- priors$mle_gamsel$estimate["shape1"]
  shape_2_gamsel <- priors$mle_gamsel$estimate["shape2"]

  # Divergence metrics
  dist_prior_glm <- dispersion_metrics_beta(
    shape_1 = shape_1_glm, shape_2 = shape_2_glm, scores = scores
  )
  dist_prior_gam <- dispersion_metrics_beta(
    shape_1 = shape_1_gam, shape_2 = shape_2_gam, scores = scores
  )
  dist_prior_gamsel <- dispersion_metrics_beta(
    shape_1 = shape_1_gamsel, shape_2 = shape_2_gamsel, scores = scores
  )

  dist_prior_glm |>
    mutate(prior = "glm", shape_1 = shape_1_glm, shape_2 = shape_2_glm) |>
    bind_rows(
      dist_prior_gam |>
        mutate(prior = "gam", shape_1 = shape_1_gam, shape_2 = shape_2_gam)
    ) |>
    bind_rows(
      dist_prior_gamsel |>
        mutate(
          prior = "gamsel", shape_1 = shape_1_gamsel, shape_2 = shape_2_gamsel
        )
    )
}

#' Counts the number of scores in each of the 20 equal-sized bins over [0,1]
#'
#' @param scores_train vector of scores on the train test
#' @param scores_valid vector of scores on the validation test
#' @param scores_test vector of scores on the test test
get_histogram <- function(scores_train,
                          scores_valid,
                          scores_test) {
  breaks <- seq(0, 1, by = .05)
  scores_train_hist <- hist(scores_train, breaks = breaks, plot = FALSE)
  scores_valid_hist <- hist(scores_valid, breaks = breaks, plot = FALSE)
  scores_test_hist <- hist(scores_test, breaks = breaks, plot = FALSE)
  scores_hist <- list(
    train = scores_train_hist,
    valid = scores_valid_hist,
    test = scores_test_hist
  )
  scores_hist
}

#' Estimation of P(q1 < score < q2)
#'
#' @param scores_train vector of scores on the train test
#' @param scores_valid vector of scores on the validation test
#' @param scores_test vector of scores on the test test
#' @param q1 vector of desired values for q1 (q2 = 1-q1)
estim_prop <- function(scores_train,
                       scores_valid,
                       scores_test,
                       q1 = c(.1, .2, .3, .4)) {
  proq_scores_train <- map(
    q1,
    ~prop_btw_quantiles(s = scores_train, q1 = .x)
  ) |>
    list_rbind() |>
    mutate(sample = "train")
  proq_scores_valid <- map(
    q1,
    ~prop_btw_quantiles(s = scores_valid, q1 = .x)
  ) |>
    list_rbind() |>
    mutate(sample = "validation")
  proq_scores_test <- map(
    q1,
    ~prop_btw_quantiles(s = scores_test, q1 = .x)
  ) |>
    list_rbind() |>
    mutate(sample = "test")

  proq_scores_train |>
    bind_rows(proq_scores_valid) |>
    bind_rows(proq_scores_test)
}

### Wrapper Metrics Function----

#' Get the performance/calibration/dispersion/divergence metrics
#' given estimated scores on the train, validation, and test sets
#'
#' @param scores_train scores estimated on train set
#' @param scores_valid scores estimated on validation set
#' @param scores_test scores estimated on test set
#' @param tb_train train set
#' @param tb_valid validation set
#' @param tb_test test set
#' @param target_name name of the target variable
#' @param priors priors
#'
#' @returns A list with 4 elements:
#'  - `tb_metrics`: performance / calibration metrics
#'  - `tb_disp_metrics`: disp and div metrics
#'  - `tb_prop_scores`: table with P(q1 < score < q2)
#'  - `scores_hist`: histogram of scores
get_metrics_simul <- function(scores_train,
                              scores_valid,
                              scores_test,
                              tb_train,
                              tb_valid,
                              tb_test,
                              priors,
                              target_name) {
  ## Histogram of scores----
  scores_hist <- get_histogram(scores_train, scores_valid, scores_test)

  # Performance and Calibration Metrics----
  tb_metrics <- get_perf_metrics(
    scores_train = scores_train,
    scores_valid = scores_valid,
    scores_test = scores_test,
    tb_train = tb_train,
    tb_valid = tb_valid,
    tb_test = tb_test,
    target_name = target_name)

  # Dispersion Metrics----
  tb_disp_metrics <-
    disp_metrics_dataset(priors = priors, scores = scores_train) |>
    mutate(sample = "train") |>
    bind_rows(
      disp_metrics_dataset(priors = priors, scores = scores_valid) |>
        mutate(sample = "validation")
    ) |>
    bind_rows(
      disp_metrics_dataset(priors = priors, scores = scores_test) |>
        mutate(sample = "test")
    )

  # Estimation of P(q1 < score < q2)----
  tb_prop_scores <- estim_prop(
    scores_train = scores_train,
    scores_valid = scores_valid,
    scores_test = scores_test
  )

  list(
    tb_metrics = tb_metrics,           # performance / calibration metrics
    tb_disp_metrics = tb_disp_metrics, # disp and div metrics
    tb_prop_scores = tb_prop_scores,   # table with P(q1 < score < q2)
    scores_hist = scores_hist          # histogram of scores
  )
}

## Estimation Functions----

### Random Forests----
#' Train a random forest on a dataset for a binary task for various
#' hyperparameters and computes metrics based on scores and on a set of prior
#' distributions of the underlying probability
#'
#' @param data dataset
#' @param target_name name of the target variable
#' @param priors priors obtained with `get_beta_fit()`
#' @param seed desired seed (default to `NULL`)
#'
#' @returns A list with two elements:
#'  - `res`: results for each estimated model of the grid. Each element is a
#'   list with the following 4 arguments:
#'      - `tb_metrics`: performance / calibration metrics
#'      - `tb_disp_metrics`: disp and div metrics
#'      - `tb_prop_scores`: table with P(q1 < score < q2)
#'      - `scores_hist`: histogram of scores.
#'  - `grid`: the grid search.
simul_forest_real <- function(data,
                              target_name,
                              priors,
                              seed = NULL) {

  if (!is.null(seed)) set.seed(seed)

  min_bucket_values <- unique(round(2^seq(1, 14, by = .4)))
  min_bucket_values <- min_bucket_values[min_bucket_values <=  nrow(data)]

  mtry <- c(2, 4, 10)
  mtry <- mtry[mtry <= ncol(data)]

  # Grid for hyperparameters
  grid <- expand_grid(
    mtry = mtry,
    num_trees = 500,
    min_node_size = min_bucket_values
  ) |>
    mutate(ind = row_number())

  # Split data into train and test set
  data_splitted <- split_train_test(data = data, prop_train = .8, seed = seed)

  data_encoded <- encode_dataset(
    data_train = data_splitted$train,
    data_test = data_splitted$test,
    target_name = target_name,
    intercept = FALSE
  )

  # Further split train intro two samples (train/valid)
  data_splitted_train <-
    split_train_test(data = data_encoded$train, prop_train = .8, seed = seed)

  progressr::with_progress({
    p <- progressr::progressor(steps = nrow(grid))
    res_grid <- furrr::future_map(
      .x = seq_len(nrow(grid)),
      .f = ~{
        p()
        simul_forest_helper(
          data_train = data_splitted_train$train,
          data_valid = data_splitted_train$test,
          data_test = data_encoded$test,
          target_name = target_name,
          params = grid |> dplyr::slice(.x),
          priors = priors
        )
      },
      .options = furrr::furrr_options(seed = NULL)
    )
  })

  list(
    res = res_grid,
    grid = grid
  )
}

#' Fit a random forest and returns metrics based on scores. The divergence
#' metrics are obtained using the prior distributions.
#'
#' @param data_train train set
#' @param data_valid validation set
#' @param data_test test set
#' @param target_name name of the target variable
#' @param parms tibble with hyperparameters for the current estimation
#' @param priors priors obtained with `get_beta_fit()`
#'
#' @returns A list with 4 elements:
#'  - `tb_metrics`: performance / calibration metrics
#'  - `tb_disp_metrics`: disp and div metrics
#'  - `tb_prop_scores`: table with P(q1 < score < q2)
#'  - `scores_hist`: histogram of scores
simul_forest_helper <- function(data_train,
                                data_valid,
                                data_test,
                                target_name,
                                params,
                                priors) {
  ## Estimation----
  fit_rf <- ranger(
    str_c(target_name, " ~ ."),
    data = data_train,
    min.bucket = params$min_node_size,
    mtry = params$mtry,
    num.trees = params$num_trees
  )
  ## Predicted scores----
  scores_train <- predict(fit_rf, data = data_train, type = "response")$predictions
  scores_valid <- predict(fit_rf, data = data_valid, type = "response")$predictions
  scores_test <- predict(fit_rf, data = data_test, type = "response")$predictions
  ## Metrics----
  metrics <- get_metrics_simul(
    scores_train = scores_train,
    scores_valid = scores_valid,
    scores_test = scores_test,
    tb_train = data_train,
    tb_valid = data_valid,
    tb_test = data_test,
    priors = priors,
    target_name = target_name
  )
  # Add index of the grid search
  metrics$tb_metrics <- metrics$tb_metrics |> mutate(ind = params$ind)
  metrics$tb_disp_metrics <- metrics$tb_disp_metrics |> mutate(ind = params$ind)
  metrics$tb_prop_scores <- metrics$tb_prop_scores |> mutate(ind = params$ind)
  metrics$scores_hist$ind <- params$ind

  metrics
}

### Extreme Gradient Boosting----
#' Train an XGB on a dataset for a binary task for various
#' hyperparameters and computes metrics based on scores and on a set of prior
#' distributions of the underlying probability
#'
#' @param data dataset
#' @param target_name name of the target variable
#' @param priors priors obtained with `get_beta_fit()`
#' @param seed desired seed (default to `NULL`)
#'
#' @returns A list with two elements:
#'  - `res`: results for each estimated model of the grid. Each element is a
#'  list with the following elements:
#'      - `tb_metrics`: performance / calibration metrics
#'      - `tb_disp_metrics`: disp and div metrics
#'      - `tb_prop_scores`: table with P(q1 < score < q2)
#'      - `scores_hist`: histogram of scores.
#'  - `grid`: the grid search.
simul_xgb_real <- function(data,
                           target_name,
                           priors,
                           seed = NULL) {

  if (!is.null(seed)) set.seed(seed)

  # Grid for hyperparameters
  grid <- expand_grid(
    max_depth = c(2, 4, 6),
    nb_iter_total = 500,
    eta = 0.3
  ) |>
    mutate(ind = row_number())

  # Split data into train and test set
  data_splitted <- split_train_test(data = data, prop_train = .8, seed = seed)
  data_encoded <- encode_dataset(
    data_train = data_splitted$train,
    data_test = data_splitted$test,
    target_name = target_name,
    intercept = FALSE
  )

  # Further split train intro two samples (train/valid)
  data_splitted_train <-
    split_train_test(data = data_encoded$train, prop_train = .8, seed = seed)

  res_grid <- vector(mode = "list", length = nrow(grid))
  for (i_grid in 1:nrow(grid)) {
    res_grid[[i_grid]] <- simul_xgb_helper(
      data_train = data_splitted_train$train,
      data_valid = data_splitted_train$test,
      data_test = data_encoded$test,
      target_name = target_name,
      params = grid |> dplyr::slice(i_grid),
      priors = priors
    )
  }

  list(
    res = res_grid,
    grid = grid
  )
}

#' Get the metrics based on scores estimated at a given boosting iteration
#'
#' @param scores scores estimated a boosting iteration `nb_iter` (list with
#'   train and test scores, returned by `predict_score_iter()`)
#' @param data_train train set
#' @param data_valid validation set
#' @param data_test test set
#' @param target_name name of the target variable
#' @param ind index of the grid search
#' @param nb_iter boosting iteration to consider
#'
#' @returns A list with 4 elements:
#'  - `tb_metrics`: performance / calibration metrics
#'  - `tb_disp_metrics`: disp and div metrics
#'  - `tb_prop_scores`: table with P(q1 < score < q2)
#'  - `scores_hist`: histogram of scores
get_metrics_xgb_iter <- function(scores,
                                 data_train,
                                 data_valid,
                                 data_test,
                                 target_name,
                                 ind,
                                 nb_iter) {

  scores_train <- scores$scores_train
  scores_valid <- scores$scores_valid
  scores_test <- scores$scores_test

  ## Metrics----
  metrics <- get_metrics_simul(
    scores_train = scores_train,
    scores_valid = scores_valid,
    scores_test = scores_test,
    tb_train = data_train,
    tb_valid = data_valid,
    tb_test = data_test,
    priors = priors,
    target_name = target_name
  )
  # Add index of the grid search
  metrics$tb_metrics <- metrics$tb_metrics |>
    mutate(ind = ind, nb_iter = nb_iter)
  metrics$tb_disp_metrics <- metrics$tb_disp_metrics |>
    mutate(ind = ind, nb_iter = nb_iter)
  metrics$tb_prop_scores <- metrics$tb_prop_scores |>
    mutate(ind = ind, nb_iter = nb_iter)
  metrics$scores_hist$ind <- ind
  metrics$scores_hist$nb_iter <- nb_iter

  metrics
}

#' Predicts the scores at a given iteration of the XGB model
#'
#' @param fit_xgb estimated XGB model
#' @param tb_train_xgb train set
#' @param tb_valid_xgb validation set
#' @param tb_test_xgb test set
#' @param ind index of the grid search
#' @param nb_iter boosting iteration to consider
#'
#' @returns A list with three elements: `scores_train`, `scores_valid`, and
#'  `scores_train` which contain the estimated scores on the train and on the
#'  test score, resp.
predict_score_iter <- function(fit_xgb,
                               tb_train_xgb,
                               tb_valid_xgb,
                               tb_test_xgb,
                               nb_iter) {

  ## Predicted scores----
  scores_train <- predict(fit_xgb, tb_train_xgb, iterationrange = c(1, nb_iter))
  scores_valid <- predict(fit_xgb, tb_valid_xgb, iterationrange = c(1, nb_iter))
  scores_test <- predict(fit_xgb, tb_test_xgb, iterationrange = c(1, nb_iter))

  list(
    scores_train = scores_train,
    scores_valid = scores_valid,
    scores_test = scores_test
  )
}


#' Fit an XGB and returns metrics based on scores. The divergence metrics are
#' obtained using the prior distributions.
#'
#' @param data_train train set
#' @param data_valid validation set
#' @param data_test test set
#' @param target_name name of the target variable
#' @param parms tibble with hyperparameters for the current estimation
#' @param priors priors obtained with `get_beta_fit()`
#'
#' @returns A list with 4 elements:
#'  - `tb_metrics`: performance / calibration metrics
#'  - `tb_disp_metrics`: disp and div metrics
#'  - `tb_prop_scores`: table with P(q1 < score < q2)
#'  - `scores_hist`: histogram of scores
simul_xgb_helper <- function(data_train,
                             data_valid,
                             data_test,
                             target_name,
                             params,
                             priors) {

  ## Format data for xgboost----
  tb_train_xgb <- xgb.DMatrix(
    data = data_train |> dplyr::select(-!!target_name) |> as.matrix(),
    label = data_train |> dplyr::pull(!!target_name) |> as.matrix()
  )
  tb_valid_xgb <- xgb.DMatrix(
    data = data_valid |> dplyr::select(-!!target_name) |> as.matrix(),
    label = data_valid |> dplyr::pull(!!target_name) |> as.matrix()
  )
  tb_test_xgb <- xgb.DMatrix(
    data = data_test |> dplyr::select(-!!target_name) |> as.matrix(),
    label = data_test |> dplyr::pull(!!target_name) |> as.matrix()
  )
  # Parameters for the algorithm
  param <- list(
    max_depth = params$max_depth, #Note: root node is indexed 0
    eta = params$eta,
    nthread = 1,
    objective = "binary:logistic",
    eval_metric = "auc"
  )
  watchlist <- list(train = tb_train_xgb, eval = tb_valid_xgb)
  ## Estimation----
  fit_xgb <- xgb.train(
    param, tb_train_xgb,
    nrounds = params$nb_iter_total,
    watchlist,
    verbose = 0
  )

  # First, we estimate the scores at each boosting iteration
  # As the xgb.Dmatrix objects cannot be easily serialised, we first estimate
  # these scores in a classical way, without parallelism...
  scores_iter <- vector(mode = "list", length = params$nb_iter_total)
  for (i_iter in 1:params$nb_iter_total) {
    scores_iter[[i_iter]] <- predict_score_iter(
      fit_xgb = fit_xgb,
      tb_train_xgb = tb_train_xgb,
      tb_valid_xgb = tb_valid_xgb,
      tb_test_xgb = tb_test_xgb,
      nb_iter = i_iter)
  }

  # Then, to compute the metrics, as it is a bit slower, we can use parallelism

  ncl <- detectCores() - 1
  (cl <- makeCluster(ncl))
  clusterEvalQ(cl, {
    library(tidyverse)
    library(locfit)
    library(philentropy)
  }) |>
    invisible()

  clusterExport(cl, c(
    "scores_iter", "data_train", "data_valid", "data_test", "priors", "params",
    "target_name"
  ), envir = environment())
  clusterExport(cl, c(
    "get_metrics_xgb_iter", "get_metrics_simul", "get_histogram",
    "brier_score",
    "get_perf_metrics", "compute_metrics",
    "disp_metrics_dataset", "dispersion_metrics_beta", "prop_btw_quantiles",
    "estim_prop"
  ))

  metrics_iter <-
    pbapply::pblapply(
      X = seq_len(params$nb_iter_total),
      FUN = function(i_iter) {
        get_metrics_xgb_iter(
          scores = scores_iter[[i_iter]],
          data_train = data_train,
          data_valid = data_valid,
          data_test = data_test,
          target_name = target_name,
          ind = params$ind,
          nb_iter = i_iter
        )
      },
      cl = cl
    )
  stopCluster(cl)

  # Merge tibbles from each iteration into a single one
  tb_metrics <-
    map(metrics_iter, "tb_metrics") |>
    list_rbind()
  tb_disp_metrics <-
    map(metrics_iter, "tb_disp_metrics") |>
    list_rbind()
  tb_prop_scores <-
    map(metrics_iter, "tb_prop_scores") |>
    list_rbind()
  scores_hist <- map(metrics_iter, "scores_hist")

  list(
    tb_metrics = tb_metrics,
    tb_disp_metrics = tb_disp_metrics,
    tb_prop_scores = tb_prop_scores,
    scores_hist = scores_hist
  )
}

### GLM----
#' Train GLM (with logistic link) on a dataset for a binary task
#' and computes metrics based on scores and on a set of prior
#' distributions of the underlying probability (assumed to be "true" probs)
#'
#' @param data dataset
#' @param target_name name of the target variable
#' @param priors priors obtained with `get_beta_fit()`
#' @param seed desired seed (default to `NULL`)
#'
#' @returns A list with one elements:
#'  - `res`: results for each estimated model of the grid. Each element is a
#'   list with the following 4 arguments:
#'      - `tb_metrics`: performance / calibration metrics
#'      - `tb_disp_metrics`: disp and div metrics
#'      - `tb_prop_scores`: table with P(q1 < score < q2)
#'      - `scores_hist`: histogram of scores.
simul_glm <- function(data,
                      target_name,
                      priors,
                      seed = NULL) {

  # Split data into train and test set
  data_splitted <- split_train_test(data = data, prop_train = .8, seed = seed)

  # Further split train intro two samples (train/valid)
  # Should not be done, but we need to do it here to have comparable results
  # with ml models
  data_splitted_train <- split_train_test(
    data = data_splitted$train, prop_train = .8, seed = seed
  )

  ## Estimation----
  fit <- train_glm(
    data_train = data_splitted_train$train,
    data_test = data_splitted$test,
    target_name = target_name,
    return_model = FALSE
  )

  ## Predicted scores----
  scores_train <- fit$scores_train
  scores_test <- fit$scores_test
  ## Metrics----
  metrics <- get_metrics_simul(
    scores_train = scores_train,
    scores_valid = scores_test,# will not be used, sorry it is dirty
    scores_test = scores_test,
    tb_train = data_splitted_train$train,
    tb_valid = data_splitted$test,# same here, will not be used
    tb_test = data_splitted$test,
    priors = priors,
    target_name = target_name
  )

  # Remove wrong info on validation sample, since there is no validation sample
  metrics$tb_metrics <-
    metrics$tb_metrics |> filter(sample != "validation")
  metrics$tb_disp_metrics <-
    metrics$tb_disp_metrics |> filter(sample != "validation")
  metrics$tb_prop_scores <-
    metrics$tb_prop_scores |> filter(sample != "validation")
  metrics$scores_hist$valid <- NULL
  metrics
}

### GAM----

#' Train GAM on a dataset for a binary task
#' and computes metrics based on scores and on a set of prior
#' distributions of the underlying probability (assumed to be "true" probs)
#'
#' @param data dataset
#' @param target_name name of the target variable
#' @param spline_df degree of freedom for the splines
#' @param priors priors obtained with `get_beta_fit()`
#' @param seed desired seed (default to `NULL`)
#'
#' @returns A list with one elements:
#'  - `res`: results for each estimated model of the grid. Each element is a
#'   list with the following 4 arguments:
#'      - `tb_metrics`: performance / calibration metrics
#'      - `tb_disp_metrics`: disp and div metrics
#'      - `tb_prop_scores`: table with P(q1 < score < q2)
#'      - `scores_hist`: histogram of scores.
simul_gam <- function(data,
                      target_name,
                      spline_df = 6,
                      priors,
                      seed = NULL) {

  # Split data into train and test set
  data_splitted <- split_train_test(data = data, prop_train = .8, seed = seed)

  # Further split train intro two samples (train/valid)
  # Should not be done, but we need to do it here to have comparable results
  # with ml models
  data_splitted_train <- split_train_test(
    data = data_splitted$train, prop_train = .8, seed = seed
  )

  ## Estimation----
  fit <- train_gam(
    data_train = data_splitted_train$train,
    data_test = data_splitted$test,
    target_name = target_name,
    spline_df = spline_df,
    return_model = FALSE
  )

  ## Predicted scores----
  scores_train <- fit$scores_train
  scores_test <- fit$scores_test
  ## Metrics----
  metrics <- get_metrics_simul(
    scores_train = scores_train,
    scores_valid = scores_test,# will not be used, sorry it is dirty
    scores_test = scores_test,
    tb_train = data_splitted_train$train,
    tb_valid = data_splitted$test,# same here, will not be used
    tb_test = data_splitted$test,
    priors = priors,
    target_name = target_name
  )

  # Remove wrong info on validation sample, since there is no validation sample
  metrics$tb_metrics <-
    metrics$tb_metrics |> filter(sample != "validation")
  metrics$tb_disp_metrics <-
    metrics$tb_disp_metrics |> filter(sample != "validation")
  metrics$tb_prop_scores <-
    metrics$tb_prop_scores |> filter(sample != "validation")
  metrics$scores_hist$valid <- NULL
  metrics
}

### GAMSEL----
#' Train GAMSEL on a dataset for a binary task
#' and computes metrics based on scores and on a set of prior
#' distributions of the underlying probability (assumed to be "true" probs)
#'
#' @param data dataset
#' @param target_name name of the target variable
#' @param degrees degree for the splines
#' @param priors priors obtained with `get_beta_fit()`
#' @param seed desired seed (default to `NULL`)
#'
#' @returns A list with one elements:
#'  - `res`: results for each estimated model of the grid. Each element is a
#'   list with the following 4 arguments:
#'      - `tb_metrics`: performance / calibration metrics
#'      - `tb_disp_metrics`: disp and div metrics
#'      - `tb_prop_scores`: table with P(q1 < score < q2)
#'      - `scores_hist`: histogram of scores.
simul_gamsel <- function(data,
                         target_name,
                         degrees = 6,
                         priors,
                         seed = NULL) {

  # Split data into train and test set
  data_splitted <- split_train_test(data = data, prop_train = .8, seed = seed)

  # Further split train intro two samples (train/valid)
  # Should not be done, but we need to do it here to have comparable results
  # with ml models
  data_splitted_train <- split_train_test(
    data = data_splitted$train, prop_train = .8, seed = seed
  )

  ## Estimation----
  fit <- train_gamsel(
    data_train = data_splitted_train$train,
    data_test = data_splitted$test,
    target_name = target_name,
    degrees = degrees,
    return_model = FALSE
  )

  ## Predicted scores----
  scores_train <- fit$scores_train
  scores_test <- fit$scores_test

  ind_na_test <- which(is.na(scores_test))
  if (length(ind_na_test) > 0) {
    scores_test <- scores_test[-ind_na_test]
    data_splitted$test <- data_splitted$test[-ind_na_test,]
  }

  ## Metrics----
  metrics <- get_metrics_simul(
    scores_train = scores_train,
    scores_valid = scores_test,# will not be used, sorry it is dirty
    scores_test = scores_test,
    tb_train = data_splitted_train$train,
    tb_valid = data_splitted$test,# same here, will not be used
    tb_test = data_splitted$test,
    priors = priors,
    target_name = target_name
  )

  # Remove wrong info on validation sample, since there is no validation sample
  metrics$tb_metrics <-
    metrics$tb_metrics |> filter(sample != "validation")
  metrics$tb_disp_metrics <-
    metrics$tb_disp_metrics |> filter(sample != "validation")
  metrics$tb_prop_scores <-
    metrics$tb_prop_scores |> filter(sample != "validation")
  metrics$scores_hist$valid <- NULL
  metrics
}


# Data----
datasets <- tribble(
  ~name, ~target_name,
  "abalone", "Sex",
  "adult", "high_income",
  "bank", "y",
  "default", "default",
  "drybean", "is_dermason",
  "coupon", "y",
  "mushroom", "edible",
  "occupancy", "Occupancy",
  "winequality", "high_quality",
  "spambase", "is_spam"
)

# Load data
for (name in datasets$name) {
  # The data
  load(str_c("output/real-data/tb_", name, ".rda"))
  # The Prior on the distribution of the scores
  load(str_c("output/real-data/priors_", name, ".rda"))
}

# Estimations----
library(future)
nb_cores <- future::availableCores() - 1

seed <- 1234
for (name in datasets$name) {
  current_data <- get(str_c("tb_", name))
  current_priors <- get(str_c("priors_", name))
  current_target_name <- datasets |>
    filter(name == !!name) |> pull(target_name)
  ## Random Forests----
  plan(multisession, workers = nb_cores)
  rf_resul <- simul_forest_real(
    data = current_data,
    target_name = current_target_name,
    priors = current_priors,
    seed = seed
  )
  save(rf_resul, file = str_c("output/real-data/rf_resul_", name, ".rda"))

  ## Extreme Gradient Boosting----
  xgb_resul <- simul_xgb_real(
    data = current_data,
    target_name = current_target_name,
    priors = current_priors,
    seed = seed
  )
  save(xgb_resul, file = str_c("output/real-data/xgb_resul_", name, ".rda"))

  ## GLM----
  glm_resul <- simul_glm(
    data = current_data,
    target_name = current_target_name,
    priors = current_priors,
    seed = seed
  )
  save(glm_resul, file = str_c("output/real-data/glm_resul_", name, ".rda"))

  ## GAM----
  gam_resul <- simul_gam(
    data = current_data,
    target_name = current_target_name,
    spline_df = 6,
    priors = current_priors,
    seed = seed
  )
  save(gam_resul, file = str_c("output/real-data/gam_resul_", name, ".rda"))

  ## GAMSEL----
  gamsel_resul <- simul_gamsel(
    data = current_data,
    target_name = current_target_name,
    degrees = 6,
    priors = current_priors,
    seed = seed
  )
  save(gamsel_resul, file = str_c("output/real-data/gamsel_resul_", name, ".rda"))
}
