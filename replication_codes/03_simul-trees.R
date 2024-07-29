# Decision Trees

library(tidyverse)
library(ggh4x)
library(rpart)
library(locfit)
library(philentropy)
library(ks)
# remotes::install_github("gweissman/gmish")

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
#| code-fold: true
#| code-summary: Function `simul_tree()`{.R}
#' Train a tree and compute performance, calibration, and dispersion metrics.
#'
#' @param prune should the tree be pruned?
#' @param min_bucket minimal number of observations in terminal nodes
#' @param type either `"regression"` for regression tree or `"classification"`
#'  for classification tree
#' @param simu_data simulated data obtained with `simulate_data_wrapper()`
#' @param ind numerical ID of the simulation in the grid: different from the
#'  seed ID)
simul_tree <- function(prune = c(TRUE, FALSE),
                       min_bucket,
                       type = c("regression", "classification"),
                       simu_data,
                       ind) {

  tb_train <- simu_data$data$train |> rename(d = y)
  tb_valid <- simu_data$data$valid |> rename(d = y)
  tb_test <- simu_data$data$test |> rename(d = y)

  true_prob <-
    list(
      train = simu_data$data$probs_train,
      valid = simu_data$data$probs_valid,
      test = simu_data$data$probs_test
    )

  # Estimation----
  if (type == "regression") {
    estim_tree <- rpart(
      d ~ x1 + x2, data = tb_train,
      method = "anova",
      minsplit = min_bucket * 3,
      minbucket = min_bucket,
      cp = 0
    )
  } else {
    estim_tree <- rpart(
      d ~ x1 + x2, data = tb_train,
      method = "class",
      minsplit = min_bucket * 3,
      minbucket = min_bucket,
      cp = 0
    )
  }
  if (prune == TRUE) {
    ind_min_cp <- which.min(estim_tree$cptable[,"xerror"])
    min_cp <- estim_tree$cptable[ind_min_cp, "CP"]
    estim_tree <- prune(estim_tree, cp = min_cp)
  }

  nb_leaves <- sum(estim_tree$frame$var=="<leaf>")
  depth <- max(rpart:::tree.depth(as.numeric(rownames(estim_tree$frame))))

  # Raw Scores----
  # Predicted scores
  if (type == "regression") {
    scores_train <- predict(estim_tree, newdata = tb_train)
    scores_valid <- predict(estim_tree, newdata = tb_valid)
    scores_test <- predict(estim_tree, newdata = tb_test)
  } else {
    scores_train <- predict(estim_tree, newdata = tb_train)[,"1"]
    scores_valid <- predict(estim_tree, newdata = tb_valid)[,"1"]
    scores_test <- predict(estim_tree, newdata = tb_test)[,"1"]
  }

  # Histogram of scores
  breaks <- seq(0, 1, by = .05)
  scores_train_hist <- hist(scores_train, breaks = breaks, plot = FALSE)
  scores_valid_hist <- hist(scores_valid, breaks = breaks, plot = FALSE)
  scores_test_hist <- hist(scores_test, breaks = breaks, plot = FALSE)
  scores_hist <- list(
    train = scores_train_hist,
    valid = scores_valid_hist,
    test = scores_test_hist
  )

  # Estimation of P(q1 < score < q2)
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

  # Dispersion Metrics
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
      prune = prune,
      min_bucket = min_bucket,
      type = !!type,
      nb_leaves = nb_leaves,
      depth = depth,
      prop_leaves = nb_leaves / nrow(tb_train)
    )

  tb_prop_scores <- proq_scores_train |>
    bind_rows(proq_scores_valid) |>
    bind_rows(proq_scores_test) |>
    mutate(
      scenario = simu_data$scenario,
      ind = ind,
      repn = simu_data$repn,
      prune = prune,
      min_bucket = min_bucket,
      type = !!type,
      nb_leaves = nb_leaves,
      depth = depth,
      prop_leaves = nb_leaves / nrow(tb_train)
    )

  tb_disp_metrics <- disp_train |>
    bind_rows(disp_valid) |>
    bind_rows(disp_test) |>
    mutate(
      scenario = simu_data$scenario,
      ind = ind,
      repn = simu_data$repn,
      prune = prune,
      min_bucket = min_bucket,
      type = !!type,
      nb_leaves = nb_leaves,
      depth = depth,
      prop_leaves = nb_leaves / nrow(tb_train)
    )

  list(
    scenario = simu_data$scenario,   # data scenario
    ind = ind,                       # index for grid
    repn = simu_data$repn,           # data replication ID
    prune = prune,                   # pruned tree?
    min_bucket = min_bucket,         # min number of obs in terminal leaf node
    type = type,                     # tree type: regression or classification
    metrics = tb_metrics,            # table with performance/calib metrics
    disp_metrics = tb_disp_metrics,  # table with divergence metrics
    tb_prop_scores = tb_prop_scores, # table with P(q1 < score < q2)
    scores_hist = scores_hist,       # histogram of scores
    nb_leaves = nb_leaves,           # number of terminal leaves
    depth = depth                    # tree depth
  )
}


#' Simulations for a scenario (single replication)
#'
#' @returns list with the following elements:
#'  - `metrics_all`: computed metrics for each set of hyperparameters.
#'    Each row gives the values for unique keys
#'    (type, prune, sample, min_bucket)
#'  - `scores_hist`: histograms computed on the train/valid/test samples
#'  - `prop_scores_simul` P(q1 < s(x) < q2) for various values of q1 and q2
#'    Each row gives the values for unique keys
#'    (type, prune, sample, min_bucket, q1, q2)
simulate_tree_scenario <- function(scenario, params_df, repn) {
  # Generate Data
  simu_data <- simulate_data_wrapper(
    scenario = scenario,
    params_df = params_df,
    repn = repn
  )
  # Simulate Trees
  progressr::with_progress({
    p <- progressr::progressor(steps = nrow(grid))
    res_simul <- furrr::future_map(
      .x = seq_len(nrow(grid)),
      .f = ~{
        p()
        simul_tree(
          prune = grid$prune[.x],
          min_bucket = grid$min_bucket[.x],
          type = grid$type[.x],
          simu_data = simu_data,
          ind = grid$ind[.x]
        )
      },
      .options = furrr::furrr_options(seed = NULL)
    )
  })
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

## Grid
grid <- expand_grid(
  prune = FALSE,
  min_bucket = unique(round(2^seq(1, 10, by = .1))),
  type = "regression"
) |>
  mutate(ind = row_number())

if (run_estim == TRUE) {

  library(pbapply)
  library(parallel)
  ncl <- detectCores()-1
  (cl <- makeCluster(ncl))

  clusterEvalQ(cl, {
    library(tidyverse)
    library(philentropy)
    library(rpart)
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
      "simul_tree",
      "simulate_tree_scenario",
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
    resul_trees_scenario <-
      pblapply(
        1:length(repns_vector), function(i) simulate_tree_scenario(
          scenario = scenario, params_df = params_df, repn = repns_vector[i]
        ),
        cl = cl
      )
    save(
      resul_trees_scenario,
      file = str_c("../output/simul/dgp-ojeda/resul_trees_scenario_", scenario, ".rda")
    )
  }
  stopCluster(cl)
}

# Load results
files <- str_c(
  "../output/simul/dgp-ojeda/resul_trees_scenario_", 1:16, ".rda"
)
resul_trees <- map(files[file.exists(files)], ~{load(.x) ; resul_trees_scenario})

# 4. Results----
metrics_trees_all <- map(
  resul_trees,
  function(resul_trees_sc) map(resul_trees_sc, "metrics_all") |> list_rbind()
) |>
  list_rbind() |>
  mutate(
    sample = factor(
      sample,
      levels = c("train", "valid", "test"),
      labels = c("Train", "Validation", "Test")
    )
  )

## Trees of interest----
# On the validation set
# - `smallest`: tree with the smallest average number of leaves
# - `largest`: tree with the highest average number of leaves
# - `largest_auc`: tree with the highest AUC on validation set
# - `lowest_mse`: tree with the lowest MSE on validation set
# - `lowest_ici`: tree with the lowest ICI on validation set
# - `lowest_kl`: tree with the lowest KL Divergence on validation set


# Identify the smallest tree
smallest_tree <-
  metrics_trees_all |>
  filter(sample == "Validation") |>
  group_by(scenario, repn) |>
  arrange(nb_leaves) |>
  slice_head(n = 1) |>
  select(scenario, repn, ind, nb_leaves) |>
  mutate(result_type = "smallest") |>
  ungroup()

# Identify the largest tree
largest_tree <-
  metrics_trees_all |>
  filter(sample == "Validation") |>
  group_by(scenario, repn) |>
  arrange(desc(nb_leaves)) |>
  slice_head(n = 1) |>
  select(scenario, repn, ind, nb_leaves) |>
  mutate(result_type = "largest") |>
  ungroup()

# Identify tree with highest AUC on test set
highest_auc_tree <-
  metrics_trees_all |>
  filter(sample == "Validation") |>
  group_by(scenario, repn) |>
  arrange(desc(AUC)) |>
  slice_head(n = 1) |>
  select(scenario, repn, ind, nb_leaves) |>
  mutate(result_type = "largest_auc") |>
  ungroup()

# Identify tree with lowest MSE
lowest_mse_tree <-
  metrics_trees_all |>
  filter(sample == "Validation") |>
  group_by(scenario, repn) |>
  arrange(mse) |>
  slice_head(n = 1) |>
  select(scenario, repn, ind, nb_leaves) |>
  mutate(result_type = "lowest_mse") |>
  ungroup()

# Identify tree with lowest ICI
lowest_ici_tree <-
  metrics_trees_all |>
  filter(sample == "Validation") |>
  group_by(scenario, repn) |>
  arrange(ici) |>
  slice_head(n = 1) |>
  select(scenario, repn, ind, nb_leaves) |>
  mutate(result_type = "lowest_ici") |>
  ungroup()

# Identify tree with lowest Brier's score
lowest_brier_tree <-
  metrics_trees_all |>
  filter(sample == "Validation") |>
  group_by(scenario, repn) |>
  arrange(brier) |>
  slice_head(n = 1) |>
  select(scenario, repn, ind, nb_leaves) |>
  mutate(result_type = "lowest_brier") |>
  ungroup()

# Identify tree with lowest KL
lowest_kl_tree <-
  metrics_trees_all |>
  filter(sample == "Validation") |>
  group_by(scenario, repn) |>
  arrange(KL_20_true_probas) |>
  slice_head(n = 1) |>
  select(scenario, repn, ind, nb_leaves) |>
  mutate(result_type = "lowest_kl") |>
  ungroup()

# Merge these
trees_of_interest_tree <-
  smallest_tree |>
  bind_rows(largest_tree) |>
  bind_rows(highest_auc_tree) |>
  bind_rows(lowest_mse_tree) |>
  bind_rows(lowest_ici_tree) |>
  bind_rows(lowest_brier_tree) |>
  bind_rows(lowest_kl_tree)

# Add metrics now
trees_of_interest_metrics_tree <-
  trees_of_interest_tree |>
  left_join(
    metrics_trees_all,
    by = c("scenario", "repn", "ind", "nb_leaves"),
    relationship = "many-to-many" # (train, valid, test)
  ) |>
  mutate(
    result_type = factor(
      result_type,
      levels = c(
        "smallest", "largest", "lowest_mse", "largest_auc",
        "lowest_brier", "lowest_ici", "lowest_kl"),
      labels = c(
        "Smallest", "Largest", "MSE*", "AUC*",
        "Brier*", "ICI*", "KL*"
      )
    )
  )

# Sanity check
trees_of_interest_metrics_tree |> count(scenario, sample, result_type)


trees_prop_scores_simul <- map(
  resul_trees,
  function(resul_trees_sc) map(resul_trees_sc, "prop_scores_simul") |> list_rbind()
) |>
  list_rbind()

trees_of_interest_prop_scores_simul <-
  trees_of_interest_tree |>
  left_join(
    trees_prop_scores_simul,
    by = c("scenario", "repn", "ind", "nb_leaves"),
    relationship = "many-to-many" # (train, validation, test, (q1,q2))
  )


## Tables----
n_digits <- 3
table_to_print <-
  trees_of_interest_metrics_tree |>
  group_by(scenario, sample, result_type) |>
  summarise(
    across(
      all_of(c(
        "AUC", "ici", "KL_20_true_probas", "inter_quantile_10_90")
      ),
      ~str_c(round(mean(.x), n_digits), " (", round(sd(.x), n_digits), ")")
    ), .groups = "drop"
  ) |>
  arrange(scenario, result_type) |>
  mutate(scenario = str_c("Scenario ", scenario))

table_to_print |>
  DT::datatable(rownames = FALSE,
                colnames = c(
                  "Scenario" = "scenario",
                  "Selected Tree" = "result_type",
                  "Sample" = "sample",
                  "ICI" = "ici",
                  "KL Div." = "KL_20_true_probas",
                  "Quant. Ratio" = "inter_quantile_10_90"
                ),
                filter = "top",
                extensions = 'RowGroup',
                options = list(
                  rowGroup = list(dataSrc = c(0)),
                  iDisplayLength = 21
                )
  ) |>
  DT::formatStyle(
    1:ncol(table_to_print),
    target = 'row',
    backgroundColor = DT::styleEqual(
      c("Smallest", "Largest", "AUC*", "MSE*", "Brier*", "ICI*", "KL*"),
      c("#332288", "#117733", "#AA4499", "#882255","#DDCC77", "#44AA99", "#949698")
    ),
    color = "white"
  )

## Figures----

### Distribution of Scores----
# On the test set

get_bp <- function(interest,
                   scenario,
                   repn) {
  # Identify the row number of the grid
  tree_of_interest <-
    trees_of_interest_metrics_tree |>
    filter(
      result_type == !!interest,
      scenario == !!scenario,
      repn == !!repn,
      sample == "Test"
    )

  ind_grid <- tree_of_interest |>
    pull("ind")

  # The corresponding barplot data
  data_bp <-
    map(resul_trees[[scenario]], "scores_hist")[[repn]] |>
    pluck(ind_grid)

  subtitle <- str_c(
    "No. Leaves = ", tree_of_interest$nb_leaves, ", ",
    "AUC = ", round(tree_of_interest$AUC, 2), ", \n",
    "Brier = ", round(tree_of_interest$brier, 2), ", ",
    "ICI = ", round(tree_of_interest$ici, 2), ", ",
    "KL = ", round(tree_of_interest$KL_20_true_probas, 2)
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
                               repn) {
  par(mfrow = c(2,4), mar = c(4.1, 4.1, 4.1, 2.1))
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

  for (interest in c("Smallest", "Largest", "MSE*" ,"AUC*", "Brier*", "ICI*", "KL*")) {
    get_bp(
      interest = interest,
      scenario = scenario,
      repn = repn
    )
  }
}

# Showing only plots for 1st replication
repn <- 1
for (scenario in 1:16) {
  plot_hist_scenario(scenario = scenario, repn = repn)
}



### Metrics vs Number of Leaves----

plot_metric_leaves <- function(dgp) {
  data_plot <-
    metrics_trees_all |>
    mutate(
      dgp = case_when(
        scenario %in% 1:4 ~ 1,
        scenario %in% 5:8 ~ 2,
        scenario %in% 9:12 ~ 3,
        scenario %in% 13:16 ~ 4
      ),
      no_noise = c(0, 10, 50, 100)[(scenario-1)%%4 + 1],
      no_noise = factor(
        no_noise, levels = c(no_noise),
        labels = str_c(no_noise, " noise variables")
      )
    ) |>
    filter(dgp == !!dgp) |>
    group_by(
      sample, dgp, no_noise, scenario, ind, min_bucket
    ) |>
    summarise(
      KL_20_true_probas = mean(KL_20_true_probas),
      auc = mean(AUC),
      brier = mean(brier),
      ici = mean(ici),
      nb_leaves = mean(nb_leaves),
      .groups = "drop"
    ) |>
    pivot_longer(cols = c(auc, brier, ici, KL_20_true_probas), names_to = "metric") |>
    mutate(metric = factor(
      metric,
      levels = c("auc", "brier" ,"ici", "KL_20_true_probas"),
      labels = c("AUC", "Brier", "ICI", "KL Divergence")
    ))

  ggplot(
    data = data_plot |> arrange(nb_leaves),
    mapping = aes(x = nb_leaves, y = value)
  ) +
    geom_line(
      mapping = aes(colour = sample, group = )
    ) +
    ggh4x::facet_grid2(metric~no_noise, scales = "free_y", independent = "y") +
    scale_colour_manual(
      "Sample", values = colour_samples,
      labels = c("Train", "Validation", "Test")
    ) +
    theme_paper() +
    labs(x = "Average Number of leaves", y = NULL)
}

for (dgp in 1:4) {
  plot_metric_leaves(dgp = 1)
}


### Relationship between calibration, KL divergence and tree complexity----

plot_kl_vs_calib <- function(scenario_number,
                             calib_metric,
                             log_scale = FALSE) {
  data_plot <- metrics_trees_all |>
    filter(scenario == !!scenario_number) |>
    group_by(sample, scenario, ind) |>
    summarise(
      KL_20_true_probas = mean(KL_20_true_probas),
      ici = mean(ici),
      brier = mean(brier),
      nb_leaves = mean(nb_leaves),
      auc = mean(AUC),
      .groups = "drop"
    )

  data_plot_max_auc <-
    data_plot |>
    filter(sample == "Test") |>
    group_by(sample, scenario) |>
    mutate(is_max_auc = auc == max(auc)) |>
    ungroup() |>
    filter(is_max_auc == TRUE) |>
    select(sample, scenario, !!calib_metric, KL_20_true_probas)

  x_lab <- latex2exp::TeX(
    str_c(
      "Calibration (", ifelse(calib_metric == "ici", "ICI", "Brier Score"),
      ifelse(log_scale == TRUE, ", log scale)", ")")
    )
  )

  p <- ggplot(
    data = data_plot |> arrange(nb_leaves),
    mapping = aes(
      x = !!sym(calib_metric), y = KL_20_true_probas
    )
  ) +
    geom_path(
      mapping = aes(colour = sample),
      arrow = arrow(type = "closed", ends = "last",
                    length = unit(0.08, "inches"))
    ) +
    geom_vline(
      data = data_plot_max_auc,
      mapping = aes(xintercept = !!sym(calib_metric)),
      linetype = "dashed"
    ) +
    geom_hline(
      data = data_plot_max_auc,
      mapping = aes(yintercept = KL_20_true_probas),
      linetype = "dashed"
    ) +
    scale_colour_manual("Sample", values = colour_samples,
                        labels = c("Train", "Validation", "Test")) +
    labs(x = x_lab, y = "KL Divergence") +
    scale_size_binned_area("Number of Leaves") +
    theme_paper() +
    theme(legend.key.width = unit(1.5, "cm"))

  if (log_scale) p <- p + scale_x_log10() + scale_y_log10()
  p
}

# Using Brier score as the calibration metric
calib_metric <- "brier"
for (scenario_number in 1:16) {
  plot_kl_vs_calib(
    scenario_number = scenario_number,
    calib_metric = calib_metric,
    log_scale = TRUE
  )
}
# Using ICI as the calibration metric
calib_metric <- "ici"
for (scenario_number in 1:16) {
  plot_kl_vs_calib(
    scenario_number = scenario_number,
    calib_metric = calib_metric,
    log_scale = TRUE
  )
}


# With a single graph
data_plot <- metrics_trees_all |>
  group_by(
    sample, scenario, ind
  ) |>
  summarise(
    KL_20_true_probas = mean(KL_20_true_probas),
    ici = mean(ici),
    brier = mean(brier),
    nb_leaves = mean(nb_leaves),
    auc = mean(AUC),
    .groups = "drop"
  ) |>
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
      no_noise, levels = c(0, 10, 50, 100),
      labels = str_c(c(0, 10, 50, 100), " noise variables")
    )
  )


data_plot_max_auc <-
  data_plot |>
  filter(sample == "Test") |>
  group_by(sample, scenario) |>
  mutate(is_max_auc = auc == max(auc)) |>
  ungroup() |>
  filter(is_max_auc == TRUE) |>
  select(sample, scenario, dgp, no_noise, ici, brier, KL_20_true_probas)


formatter1000 <- function(x) x*1000

p_ici <- ggplot(
  data = data_plot |> arrange(nb_leaves),
  mapping = aes(
    x = ici, y = KL_20_true_probas
  )
) +
  geom_path(
    mapping = aes(colour = sample),
    arrow = arrow(type = "closed", ends = "last",
                  length = unit(0.08, "inches"))
  ) +
  geom_vline(
    data = data_plot_max_auc,
    mapping = aes(xintercept = ici),
    linetype = "dashed"
  ) +
  geom_hline(
    data = data_plot_max_auc,
    mapping = aes(yintercept = KL_20_true_probas),
    linetype = "dashed"
  ) +
  facet_grid(dgp~no_noise) +
  # ggh4x::facet_grid2(dgp~no_noise, scales = "free_y") +
  scale_colour_manual(
    "Sample", values = colour_samples,
    labels = c("Train", "Validation", "Test")) +
  labs(
    x = latex2exp::TeX("Calibration (ICI), $\\times 10^{3}$, log scale"),
    y = "KL Divergence"
  ) +
  theme_paper() +
  theme(legend.key.width = unit(1.5, "cm")) +
  scale_x_log10(labels = formatter1000) + scale_y_log10()

p_ici


p_brier <- ggplot(
  data = data_plot |> arrange(nb_leaves),
  mapping = aes(
    x = brier, y = KL_20_true_probas
  )
) +
  geom_path(
    mapping = aes(colour = sample),
    arrow = arrow(type = "closed", ends = "last",
                  length = unit(0.08, "inches"))
  ) +
  geom_vline(
    data = data_plot_max_auc,
    mapping = aes(xintercept = ici),
    linetype = "dashed"
  ) +
  geom_hline(
    data = data_plot_max_auc,
    mapping = aes(yintercept = KL_20_true_probas),
    linetype = "dashed"
  ) +
  facet_grid(dgp~no_noise) +
  # ggh4x::facet_grid2(dgp~no_noise, scales = "free_y") +
  scale_colour_manual(
    "Sample", values = colour_samples,
    labels = c("Train", "Validation", "Test")) +
  labs(
    x = latex2exp::TeX("Calibration (Brier), $\\times 10^{3}$, log scale"),
    y = "KL Divergence"
  ) +
  theme_paper() +
  theme(legend.key.width = unit(1.5, "cm")) +
  scale_x_log10(labels = formatter1000) + scale_y_log10()

p_brier
