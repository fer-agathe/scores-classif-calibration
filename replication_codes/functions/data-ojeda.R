#' Simulates train/test
#'
#' @details
#' This function is a modified version of the function 'simulateData' in the
#' R script 'functions-for-calibrating-random-forests.R' provided in the
#' supplementary material of  Dankowski, T., & Ziegler, A. (2016). Calibrating
#' random forests for probability estimation. Statistics in medicine, 35(22),
#' 3949-3960.
#'
#' @param n_num number of numerical covariates
#' @param add_categ if `TRUE`, add 5 categorical variables
#' @param coeff vector of coefficients (of length n_num + 5)
#' @param n_noise number of noise variables (drawn from N(0,1))
#' @param mean_num vector of mean for the numerical variables
#' @param sd_num vector of standard deviations for the numerical variables
#' @param size_train size for the train set
#' @param size_valid size for the validation set
#' @param size_test size for the test set
#' @param transform_probs if `TRUE`, the true probability is taken to the power of 3
#' @param linear_predictor if `TRUE`, the predictor of the true probability is a
#'  linear combination of the covariates. Otherwise, the squared term for x1 is
#'  added, as well as an interaction term between x2 and x3 (`n_num` thus need
#'  to be at least 3).
#' @param seed desired seed (default to `NULL`)
#' @param linear_predictor_factor if `transform_probs = TRUE`, scalar used to
#'  draw more observation before subsampling. Default to 3 (a sample 3 times
#'  larger than `size_train` and `size_test` will first be generated before
#'  subsampling so that the true probability follows a Beta(2,2).
#'
#' @returns A list with the following components:
#'  - train: train set
#'  - valid: validation set
#'  - test: test set
#'  - probs_train: true probabilities for binary event in train set
#'  - probs_valid: true probabilities for binary event in validation set
#'  - probs_test: true probabilities for binary event in test set
simulate_data <- function(n_num = 2,
                          add_categ = FALSE,
                          coeff,
                          n_noise = 0,
                          mean_num,
                          sd_num,
                          size_train,
                          size_valid,
                          size_test,
                          transform_probs = FALSE,
                          linear_predictor = TRUE,
                          linear_predictor_factor = 3,
                          seed = NULL) {

  if (linear_predictor == TRUE) {
    n_obs <- size_train + size_test + size_valid
  } else {
    n_obs <- (size_train + size_test + size_valid) * linear_predictor_factor
  }

  if (!is.null(seed)) {
    set.seed(seed)
  }

  # Numerical covariates
  covariates <- map2(
    .x = mean_num,
    .y = sd_num,
    .f = ~rnorm(n = n_obs, mean = .x, sd = .y)
  )
  names(covariates) <- str_c("x", 1:n_num)
  covariates <- as_tibble(covariates)

  # Categorical covariates
  if (add_categ == TRUE) {
    x_c1 <- base::sample(c(0, 1), n_obs, replace = TRUE)
    x_c2 <- base::sample(c(0, 1), n_obs, replace = TRUE)
    x_c3 <- base::sample(c(1, 2, 3), n_obs, replace = TRUE)
    x_c4 <- base::sample(c(1, 2, 3, 4), n_obs, replace = TRUE)
    x_c5 <- base::sample(c(1, 2, 3, 4, 5), n_obs, replace = TRUE)

    categ_covariates <- tibble(x_c1, x_c2, x_c3, x_c4, x_c5)
    colnames(categ_covariates) <- str_c("x", (n_num + 1):(n_num + 5))
    covariates <- bind_cols(covariates, categ_covariates)
  }

  if (linear_predictor == TRUE) {
    # Linear predictor
    eta <- as.matrix(covariates) %*% coeff
  } else {
    if (n_num < 3) stop("If linear_predictor=TRUE, n_num must be greater than 2")
    eta <- as.matrix(covariates) %*% coeff +
      covariates$x1^2 + covariates$x2^2 * covariates$x3
  }

  # True probability
  true_prob <- as.numeric(1 / (1 + exp(-eta)))
  if (transform_probs) true_prob <- true_prob^3

  # Observed event
  y <- rbinom(n_obs, size = 1, prob = true_prob)

  # Create dataset with observed event and covariates
  tb <- tibble(y, covariates)

  if (linear_predictor == FALSE) {
    # We would like the probabilities to be distributed as a Beta(2,2)
    tb <- tb |> mutate(p = true_prob)
    tb <- subset_target(
      data = tb,
      probs_name = "p",
      target_fun = function(x) dbeta(x,2,2),
      iter = 1, draw = FALSE,
      seed = seed,
      verbose = FALSE
    )
    n_obs <- size_train + size_test + size_valid
    if (nrow(tb) < n_obs) {
      stop(
        str_c("The number of observation generated is lower than the ",
              "desired number. Increase `linear_predictor_factor`.")
      )
    }
    true_prob <- tb$p[1:n_obs]
    tb <- tb |> select(-p) |> dplyr::slice_head(n = n_obs)
  }


  # Noise variables
  if (n_noise > 0) {
    noise <- matrix(
      rnorm(n_noise * n_obs, mean = 0, sd = 1),
      ncol = n_noise,
      nrow = n_obs,
      byrow = FALSE
    ) |>
      as_tibble()
    colnames(noise) <- str_c("noise_", 1:n_noise)
    tb <- bind_cols(tb, noise)
  }

  # Split data into train/valid/test
  tb_train <- tb |> dplyr::slice(1:size_train)
  true_prob_train = true_prob[1:size_train]
  if (size_valid > 0) {
    ind_valid <- (size_train + 1):(size_train+size_valid)
    tb_valid <- tb |> dplyr::slice(ind_valid)
    true_prob_valid = true_prob[ind_valid]
  } else {
    tb_valid <- NULL
    true_prob_valid <- NULL
  }
  if (size_test > 0) {
    ind_test <- (size_train + ifelse(is.null(size_valid), 0, size_valid) + 1):n_obs
    tb_test <- tb |> dplyr::slice(ind_test)
    true_prob_test = true_prob[ind_test]
  } else {
    tb_test <- NULL
    true_prob_test <- NULL
  }

  list(
    train = tb_train,
    valid = tb_valid,
    test = tb_test,
    probs_train = true_prob_train,
    probs_valid = true_prob_valid,
    probs_test = true_prob_test
  )
}

#' Generates data for a given simulation scenario.
#'
#' @details
#' Wrapper of 'simulate_data' function that generates the data for a given
#' simulation scenario.
#'
#' @param scenario simulation scenario number.
#' @param params_df data frame containing the parameters to be passed to the
#'  `simulate_data` for each simulation scenario.
#' @param repn Number of current replication to be generated for the given
#'  simulation scenario.
#'
#' @returns A list with the following components:
#'  - scenario: the scenario ID
#'  - params_df: the parameters used for the data generation for the given
#'               scenario.
#'  - repn: Number of current replication that was generated for the given
#'          simulation scenario.
#'  - data: list with the simulated data (train, valid, test, probs_train,
#'          probs_valid and probs_test)
#'          see result of `simulate_data()`.
simulate_data_wrapper <- function(scenario, params_df, repn) {
  params <- params_df[params_df[["scenario"]] == scenario, ]
  if(nrow(params) != 1) stop("More than one row from params_df chosen")

  seed_for_repn <- pull(params, "seed") + repn

  args <- list(
    coeff = params |> pull("coefficients") |> pluck(1),
    n_num = params |> pull("n_num"),
    add_categ = params |> pull("add_categ"),
    n_noise = params |> pull("n_noise"),
    mean_num = params |> pull("mean_num") |> pluck(1),
    sd_num = params |> pull("sd_num") |> pluck(1),
    size_train = params |> pull("size_train"),
    size_valid = params |> pull("size_valid"),
    size_test = params |> pull("size_test"),
    transform_probs = params |> pull("transform_probs"),
    linear_predictor = params |> pull("linear_predictor"),
    seed = seed_for_repn
  )
  sim_data <- do.call("simulate_data", args)

  list(
    scenario = scenario,
    params_df = params,
    repn = repn,
    data = sim_data
  )

}
