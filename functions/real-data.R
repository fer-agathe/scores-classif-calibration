# Dependencies:
# tidyverse, gam, gamsel, recipes, MASS, latex2exp
# functions/metrics.R

#' Split dataset into train and test set
#'
#' @param data dataset
#' @param prop_train proportion in the train test (default to .8)
#' @param seed desired seed (default to `NULL`)
#'
#' @returns a list with two elements: the train set, the test set
split_train_test <- function(data,
                             prop_train = .8,
                             seed = NULL) {
  if (!is.null(seed)) set.seed(seed)
  size_train <- round(prop_train * nrow(data))
  ind_sample <- sample(1:nrow(data), replace = FALSE, size = size_train)

  list(
    train = data |> dplyr::slice(ind_sample),
    test = data |> dplyr::slice(-ind_sample)
  )
}

#' One-hot encoding, and renaming variables to avoid naming that do not respect
#' r old naming conventions
#'
#' @param data_train train set
#' @param data_test test set
#' @param target_name name of the target (response) variable
#' @param intercept should a column for an intercept be added? Default to
#'  `FALSE`
#'
#' @returns list with five elements:
#'  - `train`: train set
#'  - `test`: test set
#'  - `initial_covariate_names`: vector of names of all explanatory variables
#'  - `categ_names`: vector of new names of categorical variables (if any)
#'  - `covariate_names`: vector of new names of all explanatory variables (including
#'     categorical ones).
encode_dataset <- function(data_train,
                           data_test,
                           target_name,
                           intercept = FALSE) {

  col_names <- colnames(data_train)
  col_names_covariates <- col_names[-which(col_names == target_name)]
  new_names_covariates <- str_c("X_", 1:length(col_names_covariates))
  data_train <- data_train |>
    rename_with(.cols = all_of(col_names_covariates), .fn = ~new_names_covariates)
  data_test <- data_test |>
    rename_with(.cols = all_of(col_names_covariates), .fn = ~new_names_covariates)

  data_rec <- recipes::recipe(
    formula(str_c(target_name, " ~ .")),
    data = data_train
  )

  ref_cell <- data_rec |> recipes::step_dummy(
    recipes::all_nominal(), -recipes::all_outcomes(),
    one_hot = TRUE
  ) |>
    recipes::prep(training = data_train)

  X_train_dmy <- recipes::bake(ref_cell, new_data = data_train)
  X_test_dmy  <- recipes::bake(ref_cell, new_data = data_test)

  # Identify categorical variables
  # Bake the recipe to apply the transformation
  df_transformed <- recipes::bake(ref_cell, new_data = NULL)
  # Get the names of the transformed data
  new_names <- names(X_train_dmy)
  original_vars <- names(data_train)
  categ_names <- setdiff(new_names, original_vars)
  covariate_names <- colnames(X_train_dmy)
  covariate_names <- covariate_names[!covariate_names == target_name]
  categ_names <- categ_names[!categ_names == target_name]
  list(
    train = X_train_dmy,
    test = X_test_dmy,
    initial_covariate_names = col_names_covariates,
    categ_names = categ_names,
    covariate_names = covariate_names
  )
}

## Estimation Functions----

#' Train a GLM-logistic model
#'
#' @param data_train train set
#' @param data_test test set
#' @param target_name name of the target (response) variable
#' @param return_model if TRUE, the estimated model is returned
#'
#' @returns list with estimated scores on train set (`scores_train`) and on
#'  test set (`scores_test`)
train_glm <- function(data_train,
                      data_test,
                      target_name,
                      return_model = FALSE) {
  # Encode dataset so that categorical variables become dummy variables
  data_dmy <- encode_dataset(
    data_train = data_train,
    data_test = data_test,
    target_name = target_name,
    intercept = FALSE
  )
  # Formula for the model
  form <- str_c(target_name, "~.") |> as.formula()
  # Estimation
  fit <- glm(form, data = data_dmy$train, family = "binomial")
  # Scores on train and test set
  scores_train <- predict(fit, newdata = data_dmy$train, type = "response")
  scores_test <- predict(fit, newdata = data_dmy$test, type = "response")

  if (return_model == TRUE) {
    res <- list(
      scores_train = scores_train,
      scores_test = scores_test,
      fit = fit)
  } else {
    list(scores_train = scores_train, scores_test = scores_test, fit = NULL)
  }
}

#' Train a GAM model
#'
#' @param data_train train set
#' @param data_test test set
#' @param target_name name of the target (response) variable
#' @param spline_df degree of freedom for the splines
#' @param return_model if TRUE, the estimated model is returned
#'
#' @returns list with estimated scores on train set (`scores_train`) and on
#'  test set (`scores_test`)
train_gam <- function(data_train,
                      data_test,
                      target_name,
                      spline_df = 5,
                      return_model = FALSE) {
  # Encode dataset so that categorical variables become dummy variables
  data_dmy <- encode_dataset(
    data_train = data_train,
    data_test = data_test,
    target_name = target_name,
    intercept = FALSE
  )

  # Formula for the model
  ## Names of numerical variables
  num_names <- data_dmy$covariate_names[!data_dmy$covariate_names %in% data_dmy$categ_names]
  num_names <- num_names[!num_names %in% target_name]
  if (length(num_names) > 0) {
    ## Number of unique values
    num_nb_val <- map_dbl(num_names, ~data_dmy$train |> pull(.x) |> unique() |> length())
    ## Degree for numerical variables
    deg_num <- ifelse(num_nb_val < spline_df, num_nb_val, spline_df)
    num_term <- str_c("s(", num_names, ", df = ", deg_num, ")", collapse = " + ")
  } else {
    num_term <- NULL
  }
  if (length(data_dmy$categ_names > 0)) {
    categ_terms <- str_c(data_dmy$categ_names, collapse = " + ")
  } else {
    categ_terms <- NULL
  }

  form_terms <- str_c(num_term, categ_terms, sep = " + ")

  form_gam <- str_c(target_name, " ~ ", form_terms) |> as.formula()
  # Estimation
  fit <- gam::gam(formula = form_gam, family = binomial, data = data_dmy$train)
  # Scores on train and test set
  scores_train <- predict(fit, newdata = data_dmy$train, type = "response")
  scores_test <- predict(fit, newdata = data_dmy$test, type = "response")

  if (return_model == TRUE) {
    res <- list(
      scores_train = scores_train,
      scores_test = scores_test,
      fit = fit)
  } else {
    list(scores_train = scores_train, scores_test = scores_test, fit = NULL)
  }
}

#' Train a GAMSEL model
#'
#' @param data_train train set
#' @param data_test test set
#' @param target_name name of the target (response) variable
#' @param degrees degree for the splines
#' @param return_model if TRUE, the estimated model is returned
#'
#' @returns list with estimated scores on train set (`scores_train`) and on
#'  test set (`scores_test`)
train_gamsel <- function(data_train,
                         data_test,
                         target_name,
                         degrees = 6,
                         return_model = FALSE) {
  # Encode dataset so that categorical variables become dummy variables
  data_dmy <- encode_dataset(
    data_train = data_train,
    data_test = data_test,
    target_name = target_name,
    intercept = FALSE
  )
  # Estimation
  X_dmy_train <- data_dmy$train |> dplyr::select(-!!target_name)
  X_dmy_train <- X_dmy_train |> mutate(across(everything(), as.numeric))
  X_dmy_test <- data_dmy$test |> dplyr::select(-!!target_name)
  X_dmy_test <- X_dmy_test |> mutate(across(everything(), as.numeric))
  y_train <- data_dmy$train |> dplyr::pull(!!target_name)
  y_test <- data_dmy$test |> dplyr::pull(!!target_name)

  deg <- rep(NA, ncol(X_dmy_train))
  col_names_X <- colnames(X_dmy_train)
  nb_val <- map_dbl(
    col_names_X, ~X_dmy_train |> pull(.x) |> unique() |> length()
  )
  for (i_var_name in 1:ncol(X_dmy_train)) {
    var_name <- col_names_X[i_var_name]
    if (var_name %in% data_dmy$categ_names) {
      deg[i_var_name] <- 1
    } else {
      deg[i_var_name] <- min(nb_val[i_var_name]-1, degrees)
    }
  }
  gamsel_cv <- gamsel::cv.gamsel(
    x = as.data.frame(X_dmy_train), y = y_train, family = "binomial",
    degrees = deg
  )
  gamsel_out <- gamsel::gamsel(
    x = as.data.frame(X_dmy_train), y = y_train, family = "binomial",
    degrees = deg,
    lambda = gamsel_cv$lambda.min
  )
  # Scores on train and test set
  scores_train <- predict(
    gamsel_out, newdata = as.data.frame(X_dmy_train), type = "response")[, 1]
  scores_test <- predict(
    gamsel_out, newdata = as.data.frame(X_dmy_test), type = "response")[, 1]

  if (return_model == TRUE) {
    res <- list(
      scores_train = scores_train,
      scores_test = scores_test,
      fit = fit)
  } else {
    list(scores_train = scores_train, scores_test = scores_test, fit = NULL)
  }
}

## Distribution of scores----

#' Maximum-likelihood fitting of Beta distribution on scores
#'
#' @param scores vector of estimated scores
#' @param shape1 non-negative first parameter of the Beta distribution
#' @param shape1 non-negative second parameter of the Beta distribution
#'
#' @returns An object of class `fitdistr`, a list with four components
#'  (see: MASS::fitdistr())
#'  - `estimate`: the parameter estimates
#'  - `sd`: the estimated standard errors
#'  - `vcov`: the estimated variance-covariance matrix
#'  - `loglik`: the log-likelihood
fit_beta_scores <- function(scores, shape1 = 1, shape2 = 1) {
  # Fit a beta distribution
  mle_fit <- MASS::fitdistr(
    scores, "beta", start = list(shape1 = 1, shape2 = 1)
  )
  mle_fit
}

#' Estimation of a GLM-logistic, a GAM and a GAMSEL model on a classification
#' task. Then, on estimated scores from the test set, fits a Beta distribution.
#'
#' @param dataset dataset with response variable and predictors
#' @param target_name name of the target (response) variable
#' @param seed desired seed (default to `NULL`)
#'
#' @returns A list with the following elements:
#'  - `scores_glm`: scores on train and test set (in a list) from the GLM
#'  - `scores_gam`: scores on train and test set (in a list) from the GAM
#'  - `scores_gamsel`: scores on train and test set (in a list) from the GAMSEL
#'  - `mle_glm`: An object of class "fitdistr" for the GLM model
#'    (see fit_beta_scores())
#'  - `mle_gamsel`: An object of class "fitdistr" for the GAM model
#'    (see fit_beta_scores())
#'  - `mle_gamsel`: An object of class "fitdistr" for the GAMSEL model
#'    (see fit_beta_scores())
get_beta_fit <- function(dataset,
                         target_name,
                         seed = NULL) {
  # Split data into train/test
  data <- split_train_test(data = dataset, prop_train = .8, seed = seed)

  # Train a GLM-logistic model
  scores_glm <- train_glm(
    data_train = data$train, data_test = data$test, target_name = target_name
  )
  # Train a GAM model
  scores_gam <- train_gam(
    data_train = data$train, data_test = data$test, target_name = target_name,
    spline_df = 6
  )
  # Train a GAMSEL model
  scores_gamsel <- train_gamsel(
    data_train = data$train, data_test = data$test, target_name = target_name,
    degrees = 6
  )
  # Add a little noise to the estimated scores to avoid being in [0,1] and be
  # in (0,1) instead.
  x_glm <- (scores_glm$scores_test * (1 - 1e-6)) + 1e-6 / 2
  x_gam <- (scores_gam$scores_test * (1 - 1e-6)) + 1e-6 / 2
  x_gamsel <- (scores_gamsel$scores_test * (1 - 1e-6)) + 1e-6 / 2
  # Fit a Beta distribution on these scores
  mle_gam <- fit_beta_scores(scores = x_gam)
  mle_glm <- fit_beta_scores(scores = x_glm)
  mle_gamsel <- fit_beta_scores(scores = x_gamsel[!is.nan(x_gamsel)])

  list(
    scores_glm = scores_glm,
    scores_gam = scores_gam,
    scores_gamsel = scores_gamsel,
    mle_glm = mle_glm,
    mle_gam = mle_gam,
    mle_gamsel = mle_gamsel
  )
}

## Plots----

#' Plots the histogram of scores estimated with GAMSEL
#' add densities of Beta distribution for whith the parameters have been
#' estimated using scores from the GLM, the GAM, or the GAMSEL model
#'
#' @param fit_resul results obtained from get_beta_fit()
#' @param title title of the graph (e.g.: dataset name)
plot_hist_scores_beta <- function(fit_resul, title = NULL) {
  val_u <- seq(0, 1, length = 651)
  layout(mat = matrix(1:2), heights = c(3,1))

  dbeta_val <- vector(mode = "list", length = 3)
  for (i_type in 1:3) {
    type <- c("mle_glm", "mle_gam", "mle_gamsel")[i_type]
    dbeta_val[[i_type]] <- dbeta(
      val_u,
      fit_resul[[type]]$estimate[1],
      fit_resul[[type]]$estimate[2]
    )
  }
  y_lim <- c(
    0,
    map(dbeta_val,
        ~range(.x[!is.infinite(.x)], na.rm = TRUE)) |>
      unlist() |> max(na.rm = TRUE)
  )
  # Histogram of scores obtained with the GAMSEL, on test set
  par(mar = c(4.1, 4.1, 1, 2.1))
  hist(
    fit_resul$scores_gamsel$scores_test,
    breaks = seq(0, 1, by = .05), probability = TRUE,
    main = title, xlab = latex2exp::TeX("$\\hat{s}(x)$"),
    ylim = y_lim
  )

  legend_name <- c("GLM-logistic", "GAM", "GAMSEL")
  colours <- c(
    "mle_glm" = "#D55E00",
    "mle_gam" = "#0072B2",
    "mle_gamsel" = "#E69F00"
  )
  for (i_type in 1:3) {
    type <- c("mle_glm", "mle_gam", "mle_gamsel")[i_type]
    lines(
      val_u,
      dbeta_val[[i_type]],
      col = colours[i_type],lwd = 1.5
    )
  }
  par(mar = c(0, 4.1, 0, 2.1))
  plot.new()
  legend(
    xpd = TRUE, ncol = 3,
    "center",
    title = "Model",
    lwd = 1.5,
    col = colours,
    legend = legend_name
  )
}
