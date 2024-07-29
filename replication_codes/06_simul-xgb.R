# XGBoost

library(tidyverse)
library(ggh4x)
library(ggrepel)
library(rpart)
library(locfit)
library(philentropy)
library(xgboost)

# Colours for train/test
colour_samples <- c(
  "Train" = "#0072B2",
  "Validation" = "#009E73",
  "Test" = "#D55E00"
)

# 1. Setup----

# ggplot2 theme
source("functions/utils.R")
# data simulations
source("functions/data-ojeda.R")
# subsampling to targeted distribution
source("functions/subsample_target_distribution.R")
# prepare scenarios
source("functions/data-setup-dgp-scenarios.R")
# metrics
source("functions/metrics.R")

# Number of replications for each scenario
repns_vector <- 1:100


# 2. Estimation Function----

#' Computes the performance and calibration metrics for an xgb model,
#' depending on the number of iterations kept.
#'
#' @param nb_iter number of boosting iterations to keep
#' @param params hyperparameters of the current model
#' @param fitted_xgb xgb estimated model
#' @param tb_train_xgb train data (in xgb.DMatrix format)
#' @param tb_valid_xgb validation data (in xgb.DMatrix format)
#' @param tb_test_xgb test data (in xgb.DMatrix format)
#' @param simu_data simulated dataset
#' @param true_prob list with true probabilities on train, validation and
#'  test sets
get_metrics_nb_iter <- function(nb_iter,
                                params,
                                fitted_xgb,
                                tb_train_xgb,
                                tb_valid_xgb,
                                tb_test_xgb,
                                simu_data,
                                true_prob) {

  ind <- params$ind
  max_depth <- params$max_depth
  tb_train <- simu_data$data$train |> rename(d = y)
  tb_valid <- simu_data$data$valid |> rename(d = y)
  tb_test <- simu_data$data$test |> rename(d = y)

  # Predicted scores
  scores_train <- predict(fitted_xgb, tb_train_xgb, iterationrange = c(1, nb_iter))
  scores_valid <- predict(fitted_xgb, tb_valid_xgb, iterationrange = c(1, nb_iter))
  scores_test <- predict(fitted_xgb, tb_test_xgb, iterationrange = c(1, nb_iter))

  ## Histogram of scores----
  breaks <- seq(0, 1, by = .05)
  scores_train_hist <- hist(scores_train, breaks = breaks, plot = FALSE)
  scores_valid_hist <- hist(scores_valid, breaks = breaks, plot = FALSE)
  scores_test_hist <- hist(scores_test, breaks = breaks, plot = FALSE)
  scores_hist <- list(
    train = scores_train_hist,
    valid = scores_valid_hist,
    test = scores_test_hist,
    scenario = simu_data$scenario,
    ind = ind,
    repn = simu_data$repn,
    max_depth = params$max_depth,
    nb_iter = nb_iter
  )

  ## Estimation of P(q1 < score < q2)----
  proq_scores_train <- map(
    c(.1, .2, .3, .4),
    ~prop_btw_quantiles(s = scores_train, q1 = .x)
  ) |>
    list_rbind() |>
    mutate(sample = "train")
  proq_scores_valid <- map(
    c(.1, .2, .3, .4),
    ~prop_btw_quantiles(s = scores_valid, q1 = .x)
  ) |>
    list_rbind() |>
    mutate(sample = "valid")
  proq_scores_test <- map(
    c(.1, .2, .3, .4),
    ~prop_btw_quantiles(s = scores_test, q1 = .x)
  ) |>
    list_rbind() |>
    mutate(sample = "test")

  ## Dispersion Metrics----
  disp_train <- dispersion_metrics(
    true_probas = true_prob$train, scores = scores_train
  ) |> mutate(sample = "train")

  disp_valid <- dispersion_metrics(
    true_probas = true_prob$valid, scores = scores_valid
  ) |>mutate(sample = "valid")

  disp_test <- dispersion_metrics(
    true_probas = true_prob$test, scores = scores_test
  ) |> mutate(sample = "test")

  # Performance and Calibration Metrics
  # We add very small noise to predicted scores
  # otherwise the local regression may crash
  scores_train_noise <- scores_train +
    runif(n = length(scores_train), min = 0, max = 0.01)
  scores_train_noise[scores_train_noise > 1] <- 1
  metrics_train <- compute_metrics(
    obs = tb_train$d, scores = scores_train_noise, true_probas = true_prob$train
  ) |> mutate(sample = "train")

  scores_valid_noise <- scores_valid +
    runif(n = length(scores_valid), min = 0, max = 0.01)
  scores_valid_noise[scores_valid_noise > 1] <- 1
  metrics_valid <- compute_metrics(
    obs = tb_valid$d, scores = scores_valid_noise, true_probas = true_prob$valid
  ) |> mutate(sample = "valid")

  scores_test_noise <- scores_test +
    runif(n = length(scores_test), min = 0, max = 0.01)
  scores_test_noise[scores_test_noise > 1] <- 1
  metrics_test <- compute_metrics(
    obs = tb_test$d, scores = scores_test_noise, true_probas = true_prob$test
  ) |> mutate(sample = "test")

  tb_metrics <- metrics_train |>
    bind_rows(metrics_valid) |>
    bind_rows(metrics_test) |>
    left_join(
      disp_train |>
        bind_rows(disp_valid) |>
        bind_rows(disp_test),
      by = "sample"
    ) |>
    mutate(
      scenario = simu_data$scenario,
      ind = ind,
      repn = simu_data$repn,
      max_depth = params$max_depth,
      # type = !!type,
      nb_iter = nb_iter
    )

  tb_prop_scores <- proq_scores_train |>
    bind_rows(proq_scores_valid) |>
    bind_rows(proq_scores_test) |>
    mutate(
      scenario = simu_data$scenario,
      ind = ind,
      repn = simu_data$repn,
      max_depth = params$max_depth,
      nb_iter = nb_iter
    )

  list(
    scenario = simu_data$scenario,     # data scenario
    ind = ind,                         # index for grid
    repn = simu_data$repn,             # data replication ID
    nb_iter = nb_iter,                 # number of boosting iterations
    tb_metrics = tb_metrics,           # table with performance/calib/divergence
                                       #  metrics
    tb_prop_scores = tb_prop_scores,   # table with P(q1 < score < q2)
    scores_hist = scores_hist          # histogram of scores
  )
}

#' Train an xgboost model and compute performance, calibration, and dispersion
#' metrics
#'
#' @param params tibble with hyperparameters for the simulation
#' @param ind index of the grid (numerical ID)
#' @param simu_data simulated data obtained with `simulate_data_wrapper()`
simul_xgb <- function(params,
                      ind,
                      simu_data) {
  tb_train <- simu_data$data$train |> rename(d = y)
  tb_valid <- simu_data$data$valid |> rename(d = y)
  tb_test <- simu_data$data$test |> rename(d = y)
  true_prob <-
    list(
      train = simu_data$data$probs_train,
      valid = simu_data$data$probs_valid,
      test = simu_data$data$probs_test
    )

  ## Format data for xgboost----
  tb_train_xgb <- xgb.DMatrix(
    data = model.matrix(d ~ -1 + ., tb_train), label = tb_train$d
  )
  tb_valid_xgb <- xgb.DMatrix(
    data = model.matrix(d ~ -1 + ., tb_valid), label = tb_valid$d
  )
  tb_test_xgb <- xgb.DMatrix(
    data = model.matrix(d ~ -1 + ., tb_test), label = tb_test$d
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
  xgb_fit <- xgb.train(
    param, tb_train_xgb,
    nrounds = params$nb_iter_total,
    watchlist,
    verbose = 0
  )

  # Number of leaves
  # dt_tree <- xgb.model.dt.tree(model = xgb_fit)
  # path_depths <- xgboost:::get.leaf.depth(dt_tree)
  # path_depths |> count(Tree) |> select(n) |> table()

  # Then, for each boosting iteration number up to params$nb_iter_total
  # compute the predicted scores and evaluate the metrics
  resul <- map(
    seq(2, params$nb_iter_total),
    ~get_metrics_nb_iter(
      nb_iter = .x,
      params = params,
      fitted_xgb = xgb_fit,
      tb_train_xgb = tb_train_xgb,
      tb_valid_xgb = tb_valid_xgb,
      tb_test_xgb = tb_test_xgb,
      simu_data = simu_data,
      true_prob = true_prob
    ),
  )
  resul
}

simulate_xgb_scenario <- function(scenario, params_df, repn) {
  # Generate Data
  simu_data <- simulate_data_wrapper(
    scenario = scenario,
    params_df = params_df,
    repn = repn
  )

  # Looping over the grid hyperparameters for the scenario
  res_simul <- vector(mode = "list", length = nrow(grid))
  cli::cli_progress_bar("Iteration grid", total = nrow(grid), type = "tasks")
  for (j in 1:nrow(grid)) {
    curent_params <- grid |> dplyr::slice(!!j)
    res_simul[[j]] <- simul_xgb(
      params = curent_params,
      ind = curent_params$ind,
      simu_data = simu_data
    )
    cli::cli_progress_update()
  }


  # The metrics computed for all set of hyperparameters (identified with `ind`)
  # and for each number of boosting iterations (`nb_iter`), for the current
  # scenario (`scenario`) and current replication number (`repn`)
  metrics_simul <- map(
    res_simul,
    function(simul_grid_j) map(simul_grid_j, "tb_metrics") |> list_rbind()
  ) |>
    list_rbind()

  # Sanity check
  # metrics_simul |> count(scenario, repn, ind, sample, nb_iter) |>
  #   filter(n > 1)

  # P(q_1<s(x)<q_2)
  prop_scores_simul <- map(
    res_simul,
    function(simul_grid_j) map(simul_grid_j, "tb_prop_scores") |> list_rbind()
  ) |>
    list_rbind()

  # Sanity check
  # prop_scores_simul |> count(scenario, repn, ind, sample, nb_iter)

  # Histogram of estimated scores
  scores_hist <- map(
    res_simul,
    function(simul_grid_j) map(simul_grid_j, "scores_hist")
  )

  list(
    metrics_simul = metrics_simul,
    scores_hist = scores_hist,
    prop_scores_simul = prop_scores_simul
  )
}

# 3. Estimations----

grid <- expand_grid(
  max_depth = c(2, 4, 6),
  nb_iter_total = 400,
  eta = 0.3
) |>
  mutate(ind = row_number())


library(pbapply)
library(parallel)
ncl <- detectCores()-1
(cl <- makeCluster(ncl))

clusterEvalQ(cl, {
  library(tidyverse)
  library(locfit)
  library(philentropy)
  library(xgboost)
  library(ks)
}) |>
  invisible()

clusterExport(
  cl, c(
    # Functions
    "brier_score",
    "compute_metrics",
    "dispersion_metrics",
    "prop_btw_quantiles",
    "subset_target",
    "simulate_data",
    "simulate_data_wrapper",
    "simul_xgb",
    "simulate_xgb_scenario",
    "get_metrics_nb_iter",
    # Objects
    "grid",
    "params_df",
    "repns_vector"
  )
)

for (i_scenario in 1:16) {
  scenario <- i_scenario
  print(str_c("Scenario ", scenario, "/", nrow(params_df)))
  clusterExport(cl, c("scenario"))
  resul_xgb_scenario <-
    pblapply(
      1:length(repns_vector), function(i) simulate_xgb_scenario(
        scenario = scenario, params_df = params_df, repn = repns_vector[i]
      ),
      cl = cl
    )
  save(
    resul_xgb_scenario,
    file = str_c("../output/simul/dgp-ojeda/resul_xgb_scenario_", scenario, ".rda")
  )
}
stopCluster(cl)

# 4. Results----

scenarios <- 1:16
files <- str_c(
  "../output/simul/dgp-ojeda/resul_xgb_scenario_", scenarios, ".rda"
)
resul_xgb <- map(files[file.exists(files)], ~{load(.x) ; resul_xgb_scenario})


metrics_xgb_all <- map(
  resul_xgb,
  function(resul_xgb_sc) map(resul_xgb_sc, "metrics_simul") |> list_rbind()
) |>
  list_rbind() |>
  mutate(
    sample = factor(
      sample,
      levels = c("train", "valid", "test"),
      labels = c("Train","Validation" ,"Test")
    )
  )

# Sanity check
metrics_xgb_all |> count(scenario, ind, sample, nb_iter) |>
  filter(n != max(repns_vector))

# Models of interest
# - `smallest`: model with the lowest number of boosting iteration
# - `largest`: model with the highest number of boosting iteration
# - `largest_auc`: model with the highest AUC on validation set
# - `lowest_mse`: model with the lowest MSE on validation set
# - `lowest_ici`: model with the lowest ICI on validation set
# - `lowest_kl`: model with the lowest KL Divergence on validation set

# Identify the model with the smallest number of boosting iterations
smallest_xgb <-
  metrics_xgb_all |>
  filter(sample == "Validation") |>
  group_by(scenario, repn) |>
  arrange(nb_iter) |>
  slice_head(n = 1) |>
  select(scenario, repn, ind, nb_iter) |>
  mutate(result_type = "smallest") |>
  ungroup()

# Identify the largest tree
largest_xgb <-
  metrics_xgb_all |>
  filter(sample == "Validation") |>
  group_by(scenario, repn) |>
  arrange(desc(nb_iter)) |>
  slice_head(n = 1) |>
  select(scenario, repn, ind, nb_iter) |>
  mutate(result_type = "largest") |>
  ungroup()

# Identify tree with highest AUC on test set
highest_auc_xgb <-
  metrics_xgb_all |>
  filter(sample == "Validation") |>
  group_by(scenario, repn) |>
  arrange(desc(AUC)) |>
  slice_head(n = 1) |>
  select(scenario, repn, ind, nb_iter) |>
  mutate(result_type = "largest_auc") |>
  ungroup()

# Identify tree with lowest MSE
lowest_mse_xgb <-
  metrics_xgb_all |>
  filter(sample == "Validation") |>
  group_by(scenario, repn) |>
  arrange(mse) |>
  slice_head(n = 1) |>
  select(scenario, repn, ind, nb_iter) |>
  mutate(result_type = "lowest_mse") |>
  ungroup()

# Identify tree with lowest brier
lowest_brier_xgb <-
  metrics_xgb_all |>
  filter(sample == "Validation") |>
  group_by(scenario, repn) |>
  arrange(brier) |>
  slice_head(n = 1) |>
  select(scenario, repn, ind, nb_iter) |>
  mutate(result_type = "lowest_brier") |>
  ungroup()

# Identify tree with lowest ICI
lowest_ici_xgb <-
  metrics_xgb_all |>
  filter(sample == "Validation") |>
  group_by(scenario, repn) |>
  arrange(ici) |>
  slice_head(n = 1) |>
  select(scenario, repn, ind, nb_iter) |>
  mutate(result_type = "lowest_ici") |>
  ungroup()

# Identify tree with lowest KL
lowest_kl_xgb <-
  metrics_xgb_all |>
  filter(sample == "Validation") |>
  group_by(scenario, repn) |>
  arrange(KL_20_true_probas) |>
  slice_head(n = 1) |>
  select(scenario, repn, ind, nb_iter) |>
  mutate(result_type = "lowest_kl") |>
  ungroup()

# Merge these
models_of_interest_xgb <-
  smallest_xgb |>
  bind_rows(largest_xgb) |>
  bind_rows(highest_auc_xgb) |>
  bind_rows(lowest_mse_xgb) |>
  bind_rows(lowest_brier_xgb) |>
  bind_rows(lowest_ici_xgb) |>
  bind_rows(lowest_kl_xgb)

# Add metrics now
models_of_interest_metrics <-
  models_of_interest_xgb |>
  left_join(
    metrics_xgb_all,
    by = c("scenario", "repn", "ind", "nb_iter"),
    relationship = "many-to-many" # (train, valid, test)
  )

# Sanity check
models_of_interest_metrics |> count(scenario, sample, result_type)

## Figures----

### Metrics vs. No. boosting iterations----
plot_metrics <- function(dgp) {
  df_plot <-
    metrics_xgb_all |>
    mutate(
      dgp = case_when(
        scenario %in% 1:4 ~ 1,
        scenario %in% 5:8 ~ 2,
        scenario %in% 9:12 ~ 3,
        scenario %in% 13:16 ~ 4
      ),
      no_noise = c(0, 10, 50, 100)[(scenario-1)%%4 + 1],
      no_noise = factor(
        no_noise,
        levels = c(no_noise),
        labels = str_c(no_noise, " noise variables")
      )
    ) |>
    filter(dgp == !!dgp) |>
    select(
      dgp, no_noise, scenario, ind, sample, nb_iter, max_depth,
      AUC, brier, ici, KL_20_true_probas
    ) |>
    pivot_longer(
      cols = -c(dgp, no_noise, scenario, ind, sample, nb_iter, max_depth),
      names_to = "metric", values_to = "value"
    ) |>
    group_by(
      dgp, no_noise, scenario, ind, sample, nb_iter, max_depth, metric
    ) |>
    summarise(
      value_lower = quantile(value, probs = 2.5/100),
      value_upper = quantile(value, probs = 97.5/100),
      value = mean(value)
    ) |>
    mutate(
      max_depth = factor(max_depth),
      metric = factor(
        metric,
        levels = c("AUC", "brier", "ici", "KL_20_true_probas"),
        labels = c("AUC", "brier", "ICI", "KL Divergence")
      )
    ) |>
    filter(max_depth %in% c(2,4,6))

  ggplot(
    data = df_plot,
    mapping = aes(x = nb_iter, y = value)
  ) +
    geom_ribbon(
      mapping = aes(
        ymin = value_lower, ymax = value_upper,
        fill = sample
      ),
      alpha = .1
    ) +
    geom_line(mapping = aes(colour = sample, linetype = max_depth)) +
    geom_text_repel(
      data = df_plot |> filter(nb_iter == 380),
      mapping = aes(
        x = nb_iter, y = value, label = max_depth, colour = sample
      ),
      size = 4, # font size in the text labels
      point.padding = 0, # additional padding around each point
      min.segment.length = 0, # draw all line segments
      max.time = 1, max.iter = 1e5, # stop after 1 second, or after 100,000 iterations
      box.padding = .3, # additional padding around each text label
      segment.size = .25 # line segment thickness
    ) +
    ggh4x::facet_grid2(metric~no_noise, scales = "free_y", independent = "y") +
    labs(
      x = "Boosting Iterations", y = NULL
    ) +
    scale_colour_manual(
      "Sample", values = colour_samples,
      guide = guide_legend(
        override.aes = list(label = "")
      )
    ) +
    scale_linetype_discrete("Max Depth") +
    scale_fill_manual("Sample", values = colour_samples) +
    theme_paper()
}

for (dgp in 1:4) {
  plot_metrics(dgp = dgp)
}

### Distribution of Scores----
scores_hist_all <-
  map(
    resul_xgb,
    function(resul_xgb_sc) map(resul_xgb_sc, "scores_hist")
  )

plot_bp_interest <- function(metrics_interest, scores_hist_interest, label) {
  subtitle <- str_c(
    "Depth = ", metrics_interest$max_depth, ", ",
    "MSE = ", round(metrics_interest$mse, 2), ", ",
    "AUC = ", round(metrics_interest$AUC, 2), ", \n",
    "Brier = ", round(metrics_interest$brier, 2), ",",
    "ICI = ", round(metrics_interest$ici, 2), ", ",
    "KL = ", round(metrics_interest$KL_20_true_probas, 2)
  )

  plot(
    main = str_c(label, " (iter = ", metrics_interest$nb_iter,")"),
    scores_hist_interest$test,
    xlab = latex2exp::TeX("$\\hat{s}(x)$"),
    ylab = ""
  )
  mtext(side = 3, line = -0.25, adj = .5, subtitle, cex = .5)
}

plot_bp_xgb <- function(scenario, repn, max_depth) {
  # Focus on current scenario
  scores_hist_scenario <- scores_hist_all[[scenario]]
  # Focus on a particular replication
  scores_hist_repn <- scores_hist_scenario[[repn]]
  # Focus on a value for max_depth
  max_depth_val <- map_dbl(scores_hist_repn, ~.x[[1]]$max_depth)
  i_max_depth <- which(max_depth_val == max_depth)
  scores_hist <- scores_hist_repn[[i_max_depth]]

  # The metrics for the corresponding simulations, on the validation set
  metrics_xgb_current_valid <-
    metrics_xgb_all |>
    filter(
      scenario == !!scenario,
      repn == !!repn,
      max_depth == !!max_depth,
      sample == "Validation"
    )
  # and on the test set
  metrics_xgb_current_test <-
    metrics_xgb_all |>
    filter(
      scenario == !!scenario,
      repn == !!repn,
      max_depth == !!max_depth,
      sample == "Test"
    )

  # True Probabilities
  simu_data <- simulate_data_wrapper(
    scenario = scenario,
    params_df = params_df,
    repn = repn # only one replication here
  )
  true_prob <- simu_data$data$probs_train
  hist(
    true_prob,
    breaks = seq(0, 1, by = .05),
    xlab = "p", ylab = "",
    main = "True Probabilities",
    xlim = c(0, 1)
  )
  mtext(
    side = 2, str_c("Max Depth = ", max_depth), line = 2.5, cex = 1,
    font.lab = 2
  )

  # Iterations of interest----
  ## Start of iterations
  scores_hist_start <- scores_hist[[1]]
  metrics_start <- metrics_xgb_current_test |>
    filter(nb_iter == scores_hist_start$nb_iter)
  plot_bp_interest(
    metrics_interest = metrics_start,
    scores_hist_interest = scores_hist_start,
    label = "Start"
  )

  ## End of iterations
  scores_hist_end <- scores_hist[[length(scores_hist)]]
  metrics_end <- metrics_xgb_current_test |>
    filter(nb_iter == scores_hist_end$nb_iter)
  plot_bp_interest(
    metrics_interest = metrics_end,
    scores_hist_interest = scores_hist_end,
    label = "End"
  )


  ## Iteration with min MSE on validation set
  nb_iter_mse <-
    metrics_xgb_current_valid |> arrange(mse) |>
    dplyr::slice(1) |>
    pull("nb_iter")
  # Metrics at the same iteration on the test set
  metrics_min_mse <-
    metrics_xgb_current_test |> filter(nb_iter == !!nb_iter_mse)
  # Note: indexing at 0 here...
  ind_mse <- which(map_dbl(scores_hist, "nb_iter") == nb_iter_mse)
  scores_hist_min_mse <- scores_hist[[ind_mse]]
  plot_bp_interest(
    metrics_interest = metrics_min_mse,
    scores_hist_interest = scores_hist_min_mse,
    label = "MSE*"
  )

  ## Iteration with max AUC on validation set
  nb_iter_auc <-
    metrics_xgb_current_valid |> arrange(desc(AUC)) |>
    dplyr::slice(1) |>
    pull("nb_iter")
  metrics_max_auc <-
    metrics_xgb_current_test |> filter(nb_iter == !!nb_iter_auc)
  # Note: indexing at 0 here...
  ind_auc <- which(map_dbl(scores_hist, "nb_iter") == nb_iter_auc)
  scores_hist_max_auc <- scores_hist[[ind_auc]]
  plot_bp_interest(
    metrics_interest = metrics_max_auc,
    scores_hist_interest = scores_hist_max_auc,
    label = "AUC*"
  )
  mtext(
    side = 2, str_c("Max Depth = ", max_depth), line = 2.5, cex = 1,
    font.lab = 2
  )

  ## Min Brier on validation set
  nb_iter_brier <-
    metrics_xgb_current_valid |> arrange(brier) |>
    dplyr::slice(1) |>
    pull("nb_iter")
  metrics_min_brier <-
    metrics_xgb_current_test |> filter(nb_iter == !!nb_iter_brier)
  ind_brier <- which(map_dbl(scores_hist, "nb_iter") == nb_iter_brier)
  scores_hist_min_brier <- scores_hist[[ind_brier]]
  plot_bp_interest(
    metrics_interest = metrics_min_brier,
    scores_hist_interest = scores_hist_min_brier,
    label = "Brier*"
  )

  ## Min ICI on validation set
  nb_iter_ici <-
    metrics_xgb_current_valid |> arrange(ici) |>
    dplyr::slice(1) |>
    pull("nb_iter")
  metrics_min_ici <-
    metrics_xgb_current_test |> filter(nb_iter == !!nb_iter_ici)
  ind_ici <- which(map_dbl(scores_hist, "nb_iter") == nb_iter_ici)
  scores_hist_min_ici <- scores_hist[[ind_ici]]
  plot_bp_interest(
    metrics_interest = metrics_min_ici,
    scores_hist_interest = scores_hist_min_ici,
    label = "ICI*"
  )

  ## Min KL on validation set
  nb_iter_kl <-
    metrics_xgb_current_valid |> arrange(KL_20_true_probas) |>
    dplyr::slice(1) |>
    pull("nb_iter")
  metrics_min_kl <-
    metrics_xgb_current_test |> filter(nb_iter == !!nb_iter_kl)
  ind_kl <- which(map_dbl(scores_hist, "nb_iter") == nb_iter_kl)
  scores_hist_min_kl <- scores_hist[[ind_kl]]
  plot_bp_interest(
    metrics_interest = metrics_min_kl,
    scores_hist_interest = scores_hist_min_kl,
    label = "KL*"
  )
}

# Showing results for replication 1 only
repn <- 1

for (scenario in 1:16) {
  par(mfrow = c(2*3,4))
  plot_bp_xgb(scenario = scenario, repn = repn, max_depth = 2)
  plot_bp_xgb(scenario = scenario, repn = repn, max_depth = 4)
  plot_bp_xgb(scenario = scenario, repn = repn, max_depth = 6)
}


plot_bp_xgb_2 <- function(scenario, repn) {
  # Focus on current scenario
  scores_hist_scenario <- scores_hist_all[[scenario]]
  # Focus on a particular replication
  scores_hist_repn <- scores_hist_scenario[[repn]]

  ref_max_depth <- grid$max_depth

  # # Focus on a value for max_depth
  # max_depth_val <- map_dbl(scores_hist_repn, ~.x[[1]]$max_depth)
  # i_max_depth <- which(max_depth_val == max_depth)
  # scores_hist <- scores_hist_repn[[i_max_depth]]

  # The metrics for the corresponding simulations
  metrics_xgb_current_valid <-
    metrics_xgb_all |>
    filter(
      scenario == !!scenario,
      repn == !!repn,
      max_depth == !!ref_max_depth,
      sample == "Validation"
    )
  # and on the test set
  metrics_xgb_current_test <-
    metrics_xgb_all |>
    filter(
      scenario == !!scenario,
      repn == !!repn,
      max_depth == !!ref_max_depth,
      sample == "Test"
    )

  # True Probabilities
  simu_data <- simulate_data_wrapper(
    scenario = scenario,
    params_df = params_df,
    repn = repn # only one replication here
  )
  true_prob <- simu_data$data$probs_train
  hist(
    true_prob,
    breaks = seq(0, 1, by = .05),
    xlab = "p", ylab = "",
    main = "True Probabilities",
    xlim = c(0, 1)
  )
  mtext(
    side = 2,
    str_c(c(0, 10, 50, 100)[(scenario-1) %% 4 + 1], " noise variables"),
    line = 2.5, cex = 1,
    font.lab = 2
  )

  ## Iteration with min MSE on validation set
  nb_iter_mse <-
    metrics_xgb_current_valid |> arrange(mse) |>
    dplyr::slice(1) |>
    pull("nb_iter")
  # Metrics at the same iteration on the test set
  metrics_min_mse <-
    metrics_xgb_current_test |> filter(nb_iter == !!nb_iter_mse)
  # Note: indexing at 0 here...
  ind_mse <- which(map_dbl(scores_hist, "nb_iter") == nb_iter_mse)
  scores_hist_min_mse <- scores_hist[[ind_mse]]
  plot_bp_interest(
    metrics_interest = metrics_min_mse,
    scores_hist_interest = scores_hist_min_mse,
    label = "MSE*"
  )

  ## Iteration with max AUC on validation set
  nb_iter_auc <-
    metrics_xgb_current_valid |> arrange(desc(AUC)) |>
    dplyr::slice(1) |>
    pull("nb_iter")
  metrics_max_auc <-
    metrics_xgb_current_test |> filter(nb_iter == !!nb_iter_auc)
  # Note: indexing at 0 here...
  ind_auc <- which(map_dbl(scores_hist, "nb_iter") == nb_iter_auc)
  scores_hist_max_auc <- scores_hist[[ind_auc]]
  plot_bp_interest(
    metrics_interest = metrics_max_auc,
    scores_hist_interest = scores_hist_max_auc,
    label = "AUC*"
  )
  mtext(
    side = 2, str_c("Max Depth = ", max_depth), line = 2.5, cex = 1,
    font.lab = 2
  )

  ## Min Brier on validation set
  nb_iter_brier <-
    metrics_xgb_current_valid |> arrange(brier) |>
    dplyr::slice(1) |>
    pull("nb_iter")
  metrics_min_brier <-
    metrics_xgb_current_test |> filter(nb_iter == !!nb_iter_brier)
  ind_brier <- which(map_dbl(scores_hist, "nb_iter") == nb_iter_brier)
  scores_hist_min_brier <- scores_hist[[ind_brier]]
  plot_bp_interest(
    metrics_interest = metrics_min_brier,
    scores_hist_interest = scores_hist_min_brier,
    label = "Brier*"
  )

  ## Min ICI on validation set
  nb_iter_ici <-
    metrics_xgb_current_valid |> arrange(ici) |>
    dplyr::slice(1) |>
    pull("nb_iter")
  metrics_min_ici <-
    metrics_xgb_current_test |> filter(nb_iter == !!nb_iter_ici)
  ind_ici <- which(map_dbl(scores_hist, "nb_iter") == nb_iter_ici)
  scores_hist_min_ici <- scores_hist[[ind_ici]]
  plot_bp_interest(
    metrics_interest = metrics_min_ici,
    scores_hist_interest = scores_hist_min_ici,
    label = "ICI*"
  )

  ## Min KL on validation set
  nb_iter_kl <-
    metrics_xgb_current_valid |> arrange(KL_20_true_probas) |>
    dplyr::slice(1) |>
    pull("nb_iter")
  metrics_min_kl <-
    metrics_xgb_current_test |> filter(nb_iter == !!nb_iter_kl)
  ind_kl <- which(map_dbl(scores_hist, "nb_iter") == nb_iter_kl)
  scores_hist_min_kl <- scores_hist[[ind_kl]]
  plot_bp_interest(
    metrics_interest = metrics_min_kl,
    scores_hist_interest = scores_hist_min_kl,
    label = "KL*"
  )
}

for (dgp in 1:4) {
  scenarios <- (1:4) + 4 * (dgp - 1)
  par(mfrow = c(4,6))
  for (scenario in scenarios) {
    plot_bp_xgb_2(scenario = scenario, repn = repn)
  }
}


### KL Divergence and Calibration along Boosting Iterations----
df_plot <-
  metrics_xgb_all |>
  mutate(
    dgp = case_when(
      scenario %in% 1:4 ~ 1,
      scenario %in% 5:8 ~ 2,
      scenario %in% 9:12 ~ 3,
      scenario %in% 13:16 ~ 4
    ),
    dgp = factor(dgp, levels = 1:4, labels = str_c("DGP ", 1:4)),
    no_noise = c(0, 10, 50, 100)[(scenario-1)%%4 + 1],
    no_noise = factor(
      no_noise, levels = c(no_noise),
      labels = str_c(no_noise, " noise variables")
    )
  ) |>
  select(
    dgp, no_noise, scenario, ind, sample, nb_iter, max_depth,
    brier, ici, KL_20_true_probas
  ) |>
  group_by(dgp, no_noise, scenario, ind, sample, nb_iter, max_depth) |>
  summarise(
    brier = mean(brier),
    ici = mean(ici),
    KL_20_true_probas = mean(KL_20_true_probas),
    .groups = "drop"
  ) |>
  mutate(
    max_depth = factor(
      max_depth,
      levels = c(2, 4, 6)
    )
  )

formatter1000 <- function(x) x*1000

# With Brier score
p_brier <- ggplot(
  data = df_plot |> arrange(nb_iter),
  mapping = aes(x = brier, y = KL_20_true_probas)
) +
  geom_path(
    mapping = aes(colour = sample, linetype = max_depth),
    arrow = arrow(type = "closed", ends = "last",
                  length = unit(0.08, "inches"))
  ) +
  # facet_wrap(~scenario) +
  ggh4x::facet_grid2(dgp~no_noise, scales = "free_y", independent = "y") +
  labs(
    x = latex2exp::TeX("Calibration (Brier), $\\times 10^{3}$, log scale"),
    y = "KL Divergence"
  ) +
  scale_x_log10(labels = formatter1000) + scale_y_log10() +
  scale_colour_manual("Sample", values = colour_samples) +
  scale_linetype_discrete("Max Depth") +
  theme_paper() +
  theme(legend.key.width = unit(1.5, "cm"))


p_brier


# With ICI
p_ici <- ggplot(
  data = df_plot |> arrange(nb_iter),
  mapping = aes(x = ici, y = KL_20_true_probas)
) +
  geom_path(
    mapping = aes(colour = sample, linetype = max_depth),
    arrow = arrow(type = "closed", ends = "last",
                  length = unit(0.08, "inches"))
  ) +
  # facet_wrap(~scenario) +
  ggh4x::facet_grid2(dgp~no_noise, scales = "free_y", independent = "y") +
  labs(
    x = latex2exp::TeX("Calibration (ICI), $\\times 10^{3}$, log scale"),
    y = "KL Divergence"
  ) +
  scale_x_log10(labels = formatter1000) + scale_y_log10() +
  scale_colour_manual("Sample", values = colour_samples) +
  scale_linetype_discrete("Max Depth") +
  theme_paper() +
  theme(legend.key.width = unit(1.5, "cm"))

p_ici
