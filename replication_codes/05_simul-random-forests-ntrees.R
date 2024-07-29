# Random Forests: number of trees

library(tidyverse)
library(ggh4x)
library(rpart)
library(locfit)
library(philentropy)
library(ks)
library(ranger)

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

# 2. Functions----

#' Train a random forest and compute performance, calibration, and dispersion
#' metrics
#'
#' @param params tibble with hyperparameters for the simulation
#' @param ind index of the grid (numerical ID)
#' @param simu_data simulated data obtained with `simulate_data_wrapper()`
#'  for probability trees
simul_forest <- function(params,
                         ind,
                         simu_data
) {
  tb_train <- simu_data$data$train |> rename(d = y)
  tb_valid <- simu_data$data$valid |> rename(d = y)
  tb_test <- simu_data$data$test |> rename(d = y)

  true_prob <-
    list(
      train = simu_data$data$probs_train,
      valid = simu_data$data$probs_valid,
      test = simu_data$data$probs_test
    )

  ## Estimation----
  fit_rf <- ranger(
    d ~ .,
    data = tb_train,
    min.bucket = params$min_node_size,
    mtry = params$mtry,
    num.trees = params$num_trees
  )

  # Average number of leaves per trees in the forest
  nb_leaves <- map_dbl(fit_rf$forest$child.nodeIDs, ~sum(pluck(.x, 1) == 0)) |>
    mean()


  ## Raw Scores----
  # Predicted scores
  scores_train <- predict(fit_rf, data = tb_train, type = "response")$predictions
  scores_valid <- predict(fit_rf, data = tb_valid, type = "response")$predictions
  scores_test <- predict(fit_rf, data = tb_test, type = "response")$predictions

  ## Histogram of scores----
  breaks <- seq(0, 1, by = .05)
  scores_train_hist <- hist(scores_train, breaks = breaks, plot = FALSE)
  scores_valid_hist <- hist(scores_valid, breaks = breaks, plot = FALSE)
  scores_test_hist <- hist(scores_test, breaks = breaks, plot = FALSE)
  scores_hist <- list(
    train = scores_train_hist,
    valid = scores_valid_hist,
    test = scores_test_hist
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
  ) |> mutate(sample = "valid")

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
    mutate(
      scenario = simu_data$scenario,
      ind = ind,
      repn = simu_data$repn,
      min_bucket = params$min_node_size,
      nb_leaves = nb_leaves,
      prop_leaves = nb_leaves / nrow(tb_train)
    )

  tb_prop_scores <- proq_scores_train |>
    bind_rows(proq_scores_valid) |>
    bind_rows(proq_scores_test) |>
    mutate(
      scenario = simu_data$scenario,
      ind = ind,
      repn = simu_data$repn,
      min_bucket = params$min_node_size,
      nb_leaves = nb_leaves,
      prop_leaves = nb_leaves / nrow(tb_train)
    )

  tb_disp_metrics <- disp_train |>
    bind_rows(disp_valid) |>
    bind_rows(disp_test) |>
    mutate(
      scenario = simu_data$scenario,
      ind = ind,
      repn = simu_data$repn,
      min_bucket = params$min_node_size,
      nb_leaves = nb_leaves,
      prop_leaves = nb_leaves / nrow(tb_train)
    )

  list(
    scenario = simu_data$scenario,     # data scenario
    ind = ind,                         # index for grid
    repn = simu_data$repn,             # data replication ID
    min_bucket = params$min_node_size, # min number of obs in terminal leaf node
    metrics = tb_metrics,              # table with performance/calib metrics
    disp_metrics = tb_disp_metrics,    # table with divergence metrics
    tb_prop_scores = tb_prop_scores,   # table with P(q1 < score < q2)
    scores_hist = scores_hist,         # histogram of scores
    nb_leaves = nb_leaves              # number of terminal leaves
  )
}

#' Simulations for a scenario (single replication)
#'
#' @returns list with the following elements:
#'  - `metrics_all`: computed metrics for each set of hyperparameters.
#'    Each row gives the values for unique keys
#'    (sample, min_bucket)
#'  - `scores_hist`: histograms of scores computed on train, validation and test
#'    sets
#'  - `prop_scores_simul` P(q1 < s(x) < q2) for various values of q1 and q2
#'    Each row gives the values for unique keys
#'    (sample, min_bucket, q1, q2)
#'    q1, q2, min_bucket, type, sample)
simulate_rf_scenario <- function(scenario, params_df, repn) {
  # Generate Data
  simu_data <- simulate_data_wrapper(
    scenario = scenario,
    params_df = params_df,
    repn = repn
  )

  res_simul <- vector(mode = "list", length = nrow(grid))
  # cli::cli_progress_bar("Iteration grid", total = nrow(grid), type = "tasks")
  for (j in 1:nrow(grid)) {
    curent_params <- grid |> slice(!!j)
    n_var <- simu_data$params_df$n_num + simu_data$params_df$add_categ * 5 +
      simu_data$params_df$n_noise
    if (curent_params$mtry > n_var) {
      # cli::cli_progress_update()
      next()
    }

    res_simul[[j]] <- simul_forest(
      params = curent_params,
      ind = curent_params$ind,
      simu_data = simu_data
    )
    # cli::cli_progress_update()
  }

  metrics_simul <- map(res_simul, "metrics") |> list_rbind()
  disp_metrics_simul <- map(res_simul, "disp_metrics") |> list_rbind()
  metrics <- suppressMessages(
    left_join(metrics_simul, disp_metrics_simul)
  )
  scores_hist <- map(res_simul, "scores_hist")
  prop_scores_simul <- map(res_simul, "tb_prop_scores") |> list_rbind()

  list(
    metrics_all = metrics,
    scores_hist = scores_hist,
    prop_scores_simul = prop_scores_simul
  )
}

# 3. Estimations----

min_bucket_values <- unique(round(2^seq(1, 14, by = .4)))
min_bucket_values <- min_bucket_values[min_bucket_values <=  max(params_df$size_train)]

grid <- expand_grid(
  mtry = c(2, 4, 10),
  num_trees = c(1,2,5),
  min_node_size = min_bucket_values
) |>
  mutate(ind = row_number())

library(pbapply)
library(parallel)
ncl <- detectCores()-1
(cl <- makeCluster(ncl))

clusterEvalQ(cl, {
  library(tidyverse)
  library(philentropy)
  library(ranger)
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
    "simul_forest",
    "simulate_rf_scenario",
    # Objects
    "grid",
    "params_df",
    "repns_vector"
  )
)

for (i_scenario in 10:12) {
  scenario <- i_scenario
  print(str_c("Scenario ", scenario, "/", nrow(params_df)))
  clusterExport(cl, c("scenario"))
  resul_rf_scenario <-
    pblapply(
      1:length(repns_vector), function(i) simulate_rf_scenario(
        scenario = scenario, params_df = params_df, repn = repns_vector[i]
      ),
      cl = cl
    )
  save(
    resul_rf_scenario,
    file = str_c("../output/simul/dgp-ojeda/resul_rf_ntrees_scenario_", scenario, ".rda")
  )
}
stopCluster(cl)

# 4. Results----
files <- str_c(
  "../output/simul/dgp-ojeda/resul_rf_ntrees_scenario_", 1:16, ".rda"
)
resul_rf <- map(files[file.exists(files)], ~{load(.x) ; resul_rf_scenario})

metrics_rf_all <- map(
  resul_rf,
  function(resul_rf_sc) map(resul_rf_sc, "metrics_all") |> list_rbind()
) |>
  list_rbind() |>
  mutate(
    sample = factor(
      sample,
      levels = c("train", "valid", "test"),
      labels = c("Train", "Validation", "Test")
    )
  ) |>
  left_join(
    grid |> select(ind, mtry, min_node_size, num_trees),
    by = "ind"
  )

rf_prop_scores_simul <- map(
  resul_rf,
  function(resul_rf_sc) map(resul_rf_sc, "prop_scores_simul") |> list_rbind()
) |>
  list_rbind()

smallest <- metrics_rf_all |>
  filter(sample == "Validation") |>
  group_by(scenario, type, num_trees, repn) |>
  arrange(nb_leaves) |>
  slice_head(n = 1) |>
  select(scenario, type, num_trees, repn, ind) |>
  mutate(result_type = "smallest") |>
  ungroup()

# Forest with the largest average number of leaves per tree
largest <- metrics_rf_all |>
  filter(sample == "Validation") |>
  group_by(scenario, type, num_trees, repn) |>
  arrange(desc(nb_leaves)) |>
  slice_head(n = 1) |>
  select(scenario, type, num_trees, repn, ind) |>
  mutate(result_type = "largest") |>
  ungroup()

# Identify tree with highest AUC on test set
highest_auc <-  metrics_rf_all |>
  filter(sample == "Validation") |>
  group_by(scenario, type, num_trees, repn) |>
  arrange(desc(AUC)) |>
  slice_head(n = 1) |>
  select(scenario, type, num_trees, repn, ind) |>
  mutate(result_type = "largest_auc")

# Identify tree with lowest MSE
lowest_mse <- metrics_rf_all |>
  filter(sample == "Validation") |>
  group_by(scenario, type, num_trees, repn) |>
  arrange(mse) |>
  slice_head(n = 1) |>
  select(scenario, type, num_trees, repn, ind) |>
  mutate(result_type = "lowest_mse")

# Identify tree with lowest KL
lowest_kl <- metrics_rf_all |>
  filter(sample == "Validation") |>
  group_by(scenario, type, num_trees, repn) |>
  arrange(KL_20_true_probas) |>
  slice_head(n = 1) |>
  select(scenario, type, num_trees, repn, ind) |>
  mutate(result_type = "lowest_kl")

rf_of_interest <-
  smallest |>
  bind_rows(largest) |>
  bind_rows(highest_auc) |>
  bind_rows(lowest_mse) |>
  bind_rows(lowest_kl) |>
  # Add metrics
  left_join(
    metrics_rf_all,
    by = c("scenario", "type", "num_trees", "repn", "ind"),
    relationship = "many-to-many" # (train, valid, test)
  ) |>
  left_join(
    rf_prop_scores_simul
  ) |>
  mutate(
    result_type = factor(
      result_type,
      levels = c("largest_auc", "smallest", "largest", "lowest_mse", "lowest_kl"),
      labels = c("Max AUC", "Smallest", "Largest", "Min MSE", "Min KL")
    )
  )

## Figures----

### Distribution of Scores----

get_bp <- function(interest,
                   scenario,
                   repn,
                   num_tree) {
  # Identify the row number of the grid
  rf_of_interest_plot <-
    rf_of_interest |>
    filter(
      result_type == !!interest,
      scenario == !!scenario,
      repn == !!repn,
      num_trees == !!num_tree,
      sample == "Test"
    )

  ind_grid <- rf_of_interest_plot |>
    pull("ind")

  # The corresponding boxplot data
  data_bp <-
    map(resul_rf[[scenario]], "scores_hist")[[repn]] |>
    pluck(ind_grid)

  subtitle <- str_c(
    "No. Leaves = ", rf_of_interest_plot$nb_leaves, ", ",
    "AUC = ", round(rf_of_interest_plot$AUC, 2), ", \n",
    "ICI = ", round(rf_of_interest_plot$ici, 2), ", ",
    "KL = ", round(rf_of_interest_plot$KL_20_true_probas, 2)
  )

  plot(
    main = interest,
    data_bp$test,
    xlab = latex2exp::TeX("$\\hat{s}(x)$"),
    ylab = ""
  )
  mtext(side = 3, line = -0.25, adj = .5, subtitle, cex = .6)
}

plot_hist_scenario <- function(scenario,
                               repn,
                               num_tree) {
  par(mfrow = c(1,5))
  # True probabilities
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

  for (interest in c("Smallest", "Largest", "Max AUC", "Min KL")) {
    get_bp(
      interest = interest,
      scenario = scenario,
      repn = repn,
      num_tree = num_tree
    )
  }
}

# Showing plots for the first replication only
repn <- 1

for (num_trees in c(1, 2, 5)) {
  for (scenario in 1:16) {
    plot_hist_scenario(scenario = scenario, repn = repn, num_tree = num_tree)
  }
}


### Metrics vs. Number of Leaves----
plot_metric_leaves <- function(scenario_number) {
  data_plot <-
    metrics_rf_all |>
    filter(scenario %in% scenario_number) |>
    group_by(
      sample, scenario, ind, mtry, num_trees
    ) |>
    summarise(
      KL_20_true_probas = mean(KL_20_true_probas),
      auc = mean(AUC),
      ici = mean(ici),
      nb_leaves = mean(nb_leaves),
      .groups = "drop"
    ) |>
    mutate(
      mtry_lab = factor(str_c("mtry = ", mtry)),
      mtry_lab = fct_reorder(mtry_lab, mtry),
      num_trees_lab = factor(num_trees),
      num_trees_lab = fct_reorder(num_trees_lab, num_trees),
      scenario = str_c("Scenario ", scenario)
    ) |>
    pivot_longer(cols = c(auc, ici, KL_20_true_probas), names_to = "metric") |>
    mutate(metric = factor(
      metric,
      levels = c("auc", "ici", "KL_20_true_probas"),
      labels = c("AUC", "ICI", "KL Divergence")
    ))


  ggplot(
    data = data_plot |> group_by(num_trees_lab) |> arrange(nb_leaves),
    mapping = aes(x = nb_leaves, y = value)
  ) +
    geom_line(
      mapping = aes(colour = sample, linetype = num_trees_lab, group = )) +
    ggh4x::facet_grid2(mtry_lab~metric, scales = "free_y", independent = "y") +
    scale_colour_manual("Sample", values = colour_samples) +
    scale_linetype_discrete("No. Trees") +
    theme_paper() +
    labs(x = "Average Number of leaves", y = NULL)
}

for (scenario in 1:16) {
  plot_metric_leaves(scenario_number = scenario_number)
}

