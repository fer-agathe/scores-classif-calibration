# Comparison of Models

# Make sure to start with a fresh session to avoid conflicts with function
# names (e.g., slice from {dplyr} and {xgboost} causes issues...)

library(tidyverse)
library(locfit)
library(philentropy)

# Colours for train/validation/test
colour_samples <- c(
  "Train" = "#0072B2",
  "Validation" = "#009E73",
  "Test" = "#D55E00"
)

# Colour for the models of interest
colour_result_type <- c(
  "AUC*" = "#D55E00",
  "Smallest" = "#56B4E9",
  "Largest" = "#009E73",
  "Brier*" = "gray",
  "MSE*" = "#0072B2",
  "ICI*" = "#CC79A7",
  "KL*" = "#E69F00",
  "None" = "black"
)

# 1. Setup----

# ggplot2 theme
source("functions/utils.R")

# 2. Load Previous Results----

## Trees----
files <- str_c(
  "../output/simul/dgp-ojeda/resul_trees_scenario_", 1:16, ".rda"
)
resul_trees <- map(files[file.exists(files)], ~{load(.x) ; resul_trees_scenario})

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

# Trees of interest (based on validation set):
# - `smallest`: tree with the smallest number of leaves
# - `largest`: tree with the highest number of leaves
# - `largest_auc`: tree with the highest AUC on validation set
# - `lowest_mse`: tree with the lowest MSE on validation set
# - `lowest_brier`: tree with the lowest Brier on validation set
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
  filter(sample == "Test") |>
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

# Average metrics over the 100 replications
models_interest_trees <-
  trees_of_interest_metrics_tree |>
  group_by(scenario, sample, result_type) |>
  summarise(
    AUC_lower = quantile(AUC, probs = 2.5/100),
    AUC_upper = quantile(AUC, probs = 97.5/100),
    AUC_sd = sd(AUC),
    AUC = mean(AUC),
    brier_lower = quantile(brier, probs = 2.5/100),
    brier_upper = quantile(brier, probs = 97.5/100),
    brier_sd = sd(brier),
    brier = mean(brier),
    ici_lower = quantile(ici, probs = 2.5/100),
    ici_upper = quantile(ici, probs = 97.5/100),
    ici_sd = sd(ici),
    ici = mean(ici),
    KL_20_true_probas_lower = quantile(KL_20_true_probas, probs = 2.5/100),
    KL_20_true_probas_upper = quantile(KL_20_true_probas, probs = 97.5/100),
    KL_20_true_probas_sd = sd(KL_20_true_probas),
    KL_20_true_probas = mean(KL_20_true_probas),
    quant_ratio_sd = sd(inter_quantile_10_90),
    quant_ratio = mean(inter_quantile_10_90),
    .groups = "drop"
  ) |>
  mutate(model = "tree")

## Random Forests----

files <- str_c(
  "../output/simul/dgp-ojeda/resul_rf_scenario_", 1:16, ".rda"
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
      levels = c("train", "test", "valid"),
      labels = c("Train", "Test", "Validation")
    )
  )

# Forests of interest:
# - `smallest`: forest with the smallest average number of leaves in the trees
# - `largest`: forest with the highest average number of leaves in the trees
# - `largest_auc`: forest with the highest AUC on validation set
# - `lowest_mse`: forest with the lowest MSE on validation set
# - `lowest_ici`: forest with the lowest ICI on validation set
# - `lowest_brier`: forest with the lowest Brier score on validation set
# - `lowest_kl`: forest with the lowest KL Divergence on validation set

# Identify the model with the smallest number of leaves on average on
# validation set
smallest_rf <-
  metrics_rf_all |>
  filter(sample == "Validation") |>
  group_by(scenario, repn) |>
  arrange(nb_leaves) |>
  slice_head(n = 1) |>
  select(scenario, repn, ind, nb_leaves) |>
  mutate(result_type = "smallest") |>
  ungroup()

# Identify the largest tree
largest_rf <-
  metrics_rf_all |>
  filter(sample == "Validation") |>
  group_by(scenario, repn) |>
  arrange(desc(nb_leaves)) |>
  slice_head(n = 1) |>
  select(scenario, repn, ind, nb_leaves) |>
  mutate(result_type = "largest") |>
  ungroup()

# Identify tree with highest AUC on test set
highest_auc_rf <-
  metrics_rf_all |>
  filter(sample == "Validation") |>
  group_by(scenario, repn) |>
  arrange(desc(AUC)) |>
  slice_head(n = 1) |>
  select(scenario, repn, ind, nb_leaves) |>
  mutate(result_type = "largest_auc") |>
  ungroup()

# Identify tree with lowest MSE
lowest_mse_rf <-
  metrics_rf_all |>
  filter(sample == "Validation") |>
  group_by(scenario, repn) |>
  arrange(mse) |>
  slice_head(n = 1) |>
  select(scenario, repn, ind, nb_leaves) |>
  mutate(result_type = "lowest_mse") |>
  ungroup()

# Identify tree with lowest Brier
lowest_brier_rf <-
  metrics_rf_all |>
  filter(sample == "Validation") |>
  group_by(scenario, repn) |>
  arrange(brier) |>
  slice_head(n = 1) |>
  select(scenario, repn, ind, nb_leaves) |>
  mutate(result_type = "lowest_brier") |>
  ungroup()

# Identify tree with lowest ICI
lowest_ici_rf <-
  metrics_rf_all |>
  filter(sample == "Validation") |>
  group_by(scenario, repn) |>
  arrange(ici) |>
  slice_head(n = 1) |>
  select(scenario, repn, ind, nb_leaves) |>
  mutate(result_type = "lowest_ici") |>
  ungroup()

# Identify tree with lowest KL
lowest_kl_rf <-
  metrics_rf_all |>
  filter(sample == "Validation") |>
  group_by(scenario, repn) |>
  arrange(KL_20_true_probas) |>
  slice_head(n = 1) |>
  select(scenario, repn, ind, nb_leaves) |>
  mutate(result_type = "lowest_kl") |>
  ungroup()

# Merge these
rf_of_interest <-
  smallest_rf |>
  bind_rows(largest_rf) |>
  bind_rows(highest_auc_rf) |>
  bind_rows(lowest_mse_rf) |>
  bind_rows(lowest_brier_rf) |>
  bind_rows(lowest_ici_rf) |>
  bind_rows(lowest_kl_rf)

# Add metrics now
rf_of_interest <-
  rf_of_interest |>
  left_join(
    metrics_rf_all,
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
trees_of_interest_metrics_rf |> count(scenario, sample, result_type)

# Average of metrics over the 100 replications
models_interest_rf <- rf_of_interest |>
  group_by(scenario, sample, result_type) |>
  summarise(
    AUC_lower = quantile(AUC, probs = 2.5/100),
    AUC_upper = quantile(AUC, probs = 97.5/100),
    AUC_sd = sd(AUC),
    AUC = mean(AUC),
    brier_lower = quantile(brier, probs = 2.5/100),
    brier_upper = quantile(brier, probs = 97.5/100),
    brier_sd = sd(brier),
    brier = mean(brier),
    ici_lower = quantile(ici, probs = 2.5/100),
    ici_upper = quantile(ici, probs = 97.5/100),
    ici_sd = sd(ici),
    ici = mean(ici),
    KL_20_true_probas_lower = quantile(KL_20_true_probas, probs = 2.5/100),
    KL_20_true_probas_upper = quantile(KL_20_true_probas, probs = 97.5/100),
    KL_20_true_probas_sd = sd(KL_20_true_probas),
    KL_20_true_probas = mean(KL_20_true_probas),
    quant_ratio_sd = sd(inter_quantile_10_90),
    quant_ratio = mean(inter_quantile_10_90),
    .groups = "drop"
  ) |>
  mutate(model = "rf")


## Extreme Gradient Boosting----

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

# Models of interest:
# - `smallest`: model with the lowest number of boosting iteration
# - `largest`: model with the highest number of boosting iteration
# - `largest_auc`: model with the highest AUC on validation set
# - `lowest_mse`: model with the lowest MSE on validation set
# - `lowest_brier`: model with the lowest Brier score on validation set
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
models_of_interest_metrics |> count(scenario, sample, result_type)

# Average of metrics over the 100 replications
models_interest_xgb <- models_of_interest_metrics |>
  group_by(scenario, sample, result_type) |>
  summarise(
    AUC_lower = quantile(AUC, probs = 2.5/100),
    AUC_upper = quantile(AUC, probs = 97.5/100),
    AUC_sd = sd(AUC),
    AUC = mean(AUC),
    brier_lower = quantile(brier, probs = 2.5/100),
    brier_upper = quantile(brier, probs = 97.5/100),
    brier_sd = sd(brier),
    brier = mean(brier),
    ici_lower = quantile(ici, probs = 2.5/100),
    ici_upper = quantile(ici, probs = 97.5/100),
    ici_sd = sd(ici),
    ici = mean(ici),
    KL_20_true_probas_lower = quantile(KL_20_true_probas, probs = 2.5/100),
    KL_20_true_probas_upper = quantile(KL_20_true_probas, probs = 97.5/100),
    KL_20_true_probas_sd = sd(KL_20_true_probas),
    KL_20_true_probas = mean(KL_20_true_probas),
    quant_ratio_sd = sd(inter_quantile_10_90),
    quant_ratio = mean(inter_quantile_10_90),
    .groups = "drop"
  ) |>
  mutate(
    model = "xgb",
    sample = str_to_lower(as.character(sample))
  )

## Generalized Linear Models----

files <- str_c(
  "../output/simul/dgp-ojeda/resul_glm_scenario_", 1:16, ".rda"
)
resul_glm <- map(files[file.exists(files)], ~{load(.x) ; resul_glm_scenario})

metrics_glm_all <- map(
  resul_glm,
  function(resul_glm_sc) map(resul_glm_sc, "metrics") |> list_rbind()
) |>
  list_rbind() |>
  mutate(
    sample = factor(
      sample,
      levels = c("train", "test"), labels = c("Train", "Test")
    )
  )

# Metrics over the 100 replications
models_interest_glm <-
  metrics_glm_all |>
  group_by(scenario, sample) |>
  summarise(
    AUC_lower = quantile(AUC, probs = 2.5/100),
    AUC_upper = quantile(AUC, probs = 97.5/100),
    AUC_sd = sd(AUC),
    AUC = mean(AUC),
    brier_lower = quantile(brier, probs = 2.5/100),
    brier_upper = quantile(brier, probs = 97.5/100),
    brier_sd = sd(brier),
    brier = mean(brier),
    ici_lower = quantile(ici, probs = 2.5/100),
    ici_upper = quantile(ici, probs = 97.5/100),
    ici_sd = sd(ici),
    ici = mean(ici),
    KL_20_true_probas_lower = quantile(KL_20_true_probas, probs = 2.5/100),
    KL_20_true_probas_upper = quantile(KL_20_true_probas, probs = 97.5/100),
    KL_20_true_probas_sd = sd(KL_20_true_probas),
    KL_20_true_probas = mean(KL_20_true_probas),
    quant_ratio_sd = sd(inter_quantile_10_90),
    quant_ratio = mean(inter_quantile_10_90),
    .groups = "drop"
  ) |>
  mutate(
    model = "glm",
    sample = str_to_lower(as.character(sample))
  ) |>
  mutate(result_type = "None")

## Generalized Additive Models----

files <- str_c(
  "../output/simul/dgp-ojeda/resul_gam_scenario_", 1:16, ".rda"
)
resul_gam <- map(files[file.exists(files)], ~{load(.x) ; resul_gam_scenario})

metrics_gam_all <- map(
  resul_gam,
  function(resul_gam_sc) map(resul_gam_sc, "metrics") |> list_rbind()
) |>
  list_rbind() |>
  mutate(
    sample = factor(
      sample,
      levels = c("train", "test"), labels = c("Train", "Test")
    )
  )

# Metrics over the 100 replications
models_interest_gam <-
  metrics_gam_all |>
  group_by(scenario, sample) |>
  summarise(
    AUC_lower = quantile(AUC, probs = 2.5/100),
    AUC_upper = quantile(AUC, probs = 97.5/100),
    AUC_sd = sd(AUC),
    AUC = mean(AUC),
    brier_lower = quantile(brier, probs = 2.5/100),
    brier_upper = quantile(brier, probs = 97.5/100),
    brier_sd = sd(brier),
    brier = mean(brier),
    ici_lower = quantile(ici, probs = 2.5/100),
    ici_upper = quantile(ici, probs = 97.5/100),
    ici_sd = sd(ici),
    ici = mean(ici),
    KL_20_true_probas_lower = quantile(KL_20_true_probas, probs = 2.5/100),
    KL_20_true_probas_upper = quantile(KL_20_true_probas, probs = 97.5/100),
    KL_20_true_probas_sd = sd(KL_20_true_probas),
    KL_20_true_probas = mean(KL_20_true_probas),
    quant_ratio_sd = sd(inter_quantile_10_90),
    quant_ratio = mean(inter_quantile_10_90),
    .groups = "drop"
  ) |>
  mutate(
    model = "gam",
    sample = str_to_lower(as.character(sample))
  ) |>
  mutate(result_type = "None")

## Generalized Additive Models Selection----

files <- str_c(
  "../output/simul/dgp-ojeda/resul_gamsel_scenario_", 1:16, ".rda"
)
resul_gamsel <- map(files[file.exists(files)], ~{load(.x) ; resul_gamsel_scenario})

metrics_gamsel_all <- map(
  resul_gamsel,
  function(resul_gamsel_sc) map(resul_gamsel_sc, "metrics") |> list_rbind()
) |>
  list_rbind() |>
  mutate(
    sample = factor(
      sample,
      levels = c("train", "test"), labels = c("Train", "Test")
    )
  )

# Metrics over the 100 replications
models_interest_gamsel <-
  metrics_gamsel_all |>
  group_by(scenario, sample) |>
  summarise(
    AUC_lower = quantile(AUC, probs = 2.5/100),
    AUC_upper = quantile(AUC, probs = 97.5/100),
    AUC_sd = sd(AUC),
    AUC = mean(AUC),
    brier_lower = quantile(brier, probs = 2.5/100),
    brier_upper = quantile(brier, probs = 97.5/100),
    brier_sd = sd(brier),
    brier = mean(brier),
    ici_lower = quantile(ici, probs = 2.5/100),
    ici_upper = quantile(ici, probs = 97.5/100),
    ici_sd = sd(ici),
    ici = mean(ici),
    KL_20_true_probas_lower = quantile(KL_20_true_probas, probs = 2.5/100),
    KL_20_true_probas_upper = quantile(KL_20_true_probas, probs = 97.5/100),
    KL_20_true_probas_sd = sd(KL_20_true_probas),
    KL_20_true_probas = mean(KL_20_true_probas),
    quant_ratio_sd = sd(inter_quantile_10_90),
    quant_ratio = mean(inter_quantile_10_90),
    .groups = "drop"
  ) |>
  mutate(
    model = "gamsel",
    sample = str_to_lower(as.character(sample))
  ) |>
  mutate(result_type = "None")

# 3. Comparison of Models----

models_interest <- models_interest_trees |>
  filter(sample %in% c("Train", "Test")) |>
  mutate(
    sample = fct_recode(sample, "train" = "Train", "test" = "Test")
  ) |>
  bind_rows(
    models_interest_rf |>
      filter(sample %in% c("Train", "Test")) |>
      mutate(
        sample = fct_recode(sample, "train" = "Train", "test" = "Test")
      )
  ) |>
  bind_rows(models_interest_xgb |> filter(sample %in% c("train", "test"))) |>
  bind_rows(models_interest_glm) |>
  bind_rows(models_interest_gam) |>
  bind_rows(models_interest_gamsel) |>
  mutate(
    sample = factor(
      sample,
      levels = c("train", "test"),
      labels = c("Train", "Test")
    ),
    model = factor(
      model,
      levels = c("tree", "rf", "xgb", "glm", "gam", "gamsel"),
      labels = c("Trees", "Random Forests", "XGB", "GLM", "GAM", "GAMSEL")
    )
  ) |>
  # filter(result_type != "lowest_mse") |>
  mutate(
    result_type = factor(
      result_type,
      levels = c(
        "Smallest", "Largest", "MSE*",
        "AUC*", "Brier*",
        "ICI*", "KL*", "None"),
      labels = c(
        "Smallest", "Largest", "MSE*",
        "AUC*", "Brier*",
        "ICI*", "KL*", "None")
    )
  )

## Figures----

plot_comparison <- function(scenario, calib_metric) {
  df_plot <- models_interest |> filter(scenario == !!scenario)
  model_shapes <- c(
    "Trees" = 1,
    "Random Forests" = 2,
    "XGB" = 4,
    "GLM" = 5,
    "GAM" = 6,
    "GAMSEL" = 7
  )
  model_shapes <- model_shapes[names(model_shapes) %in% df_plot$model]

  ggplot(
    data = df_plot,
    mapping = aes(
      colour = result_type,
      shape = model
    )
  ) +
    geom_segment(
      mapping = aes(
        x = !!sym(str_c(calib_metric, "_lower")),
        y = KL_20_true_probas,
        xend = !!sym(str_c(calib_metric, "_upper")),
        yend = KL_20_true_probas
      ),
      linetype = "solid",
      linewidth = .5
    ) +
    geom_segment(
      mapping = aes(
        x = !!sym(calib_metric),
        y = KL_20_true_probas_lower,
        xend = !!sym(calib_metric),
        yend = KL_20_true_probas_upper
      ),
      linetype = "solid",
      linewidth = .5
    ) +
    geom_point(
      mapping = aes(x = !!sym(calib_metric), y = KL_20_true_probas),
      size = 4
    ) +
    labs(
      x = str_c("Calibration (", ifelse(calib_metric == "ici", "ICI", "Brier"), ")"),
      y = "KL Divergence"
    ) +
    scale_colour_manual("Type", values = colour_result_type) +
    scale_shape_manual(
      "Model", values = c(model_shapes)) +
    facet_wrap(~sample) +
    theme_paper() +
    scale_x_log10() +
    scale_y_log10() +
    guides(colour=guide_legend(ncol = 3))
}

for (scenario in 1:16) {
  plot_comparison(scenario = scenario, calib_metric = "brier")
}
for (scenario in 1:16) {
  plot_comparison(scenario = scenario, calib_metric = "ici")
}

## Tables----

table_models_interest_mean <-
  models_interest |>
  filter(sample == "Test") |>
  select(
    scenario, sample, model, result_type,
    AUC, brier, ici, kl = KL_20_true_probas, quant_ratio
  ) |>
  filter(
    result_type %in% c("AUC*", "Brier*", "ICI*", "KL*", "None")
  ) |>
  mutate(result_type = fct_recode(result_type, "KL*" = "None")) |>
  mutate(value_type = "mean")

table_models_interest_sd <-
  models_interest |>
  filter(sample == "Test") |>
  select(
    scenario, sample, model, result_type,
    AUC = AUC_sd, brier = brier_sd, ici = ici_sd, kl = KL_20_true_probas_sd, quant_ratio = quant_ratio_sd
  ) |>
  filter(
    result_type %in% c("AUC*", "Brier*", "ICI*", "KL*", "None")
  ) |>
  mutate(result_type = fct_recode(result_type, "KL*" = "None")) |>
  mutate(value_type = "sd")


red_colours <- c(
  "#FFD6D6", "#FFCCCC", "#FFC2C2", "#FFB8B8", "#FFADAD",
  "#FFA3A3", "#FF9999", "#FF8F8F", "#FF8585", "#FF7A7A"
)
red_colours_txt <- c(
  "#333333", "#333333", "#2B2B2B", "#2B2B2B", "#232323",
  "#1F1F1F", "#1A1A1A", "#141414", "#101010", "#0A0A0A"
)
green_colours <- c(
  "#E9F6E9", "#D4F2D4", "#BFEFBF", "#AADCA9", "#96C996",
  "#81B781", "#6CA56C", "#578252", "#426F42", "#2F5D2F"
)
green_colours_txt <- c(
  "#1A1A1A", "#1A1A1A", "#1A1A1A", "#1A1A1A", "#1A1A1A",
  "#E6E6E6", "#E6E6E6", "#E6E6E6", "#E6E6E6", "#E6E6E6"
)

accuracy_digits <- 0.01

table_kb <-
  table_models_interest_mean |>
  bind_rows(table_models_interest_sd) |>
  pivot_wider(
    names_from = "result_type",
    values_from = c("AUC", "brier", "ici", "kl", "quant_ratio")
  ) |>
  mutate(
    value_type = factor(value_type, levels = c("mean", "sd")),
    scenario = factor(scenario)
  ) |>
  select(
    scenario, model, value_type,
    # # columns for GLM/GAM/GAMSEL
    # AUC_None, ici_None, kl_None,
    # columns for ML models selected based on AUC
    `AUC_AUC*`, `brier_AUC*`, `ici_AUC*`, `kl_AUC*`, `quant_ratio_AUC*`,
    # columns for ML models selected based on Brier score
    `AUC_Brier*`,  `brier_Brier*`, `ici_Brier*`, `kl_Brier*`, `quant_ratio_Brier*`,
    # columns for ML models selected based on ICI
    `AUC_ICI*`, `brier_ICI*`, `ici_ICI*`, `kl_ICI*`, `quant_ratio_ICI*`,
    # columns for ML models selected based on KL dist
    `AUC_KL*`, `brier_KL*`, `ici_KL*`, `kl_KL*`, `quant_ratio_KL*`
  ) |>
  arrange(scenario, model, value_type) |>
  mutate(
    # Difference in metrics computed when minnimizing Brier wrt when maximizing AUC
    diff_AUC_Brier = `AUC_Brier*` - `AUC_AUC*`,
    diff_brier_Brier = `brier_Brier*` - `brier_AUC*`,
    diff_ICI_Brier = `ici_Brier*` - `ici_AUC*`,
    diff_KL_Brier = `kl_Brier*` - `kl_AUC*`,
    # Difference in metrics computed when minnimizing ICI wrt when maximizing AUC
    diff_AUC_ICI = `AUC_ICI*` - `AUC_AUC*`,
    diff_brier_ICI = `brier_ICI*` - `brier_AUC*`,
    diff_ICI_ICI = `ici_ICI*` - `ici_AUC*`,
    diff_KL_ICI = `kl_ICI*` - `kl_AUC*`,
    # Difference in metrics computed when minnimizing KL wrt when maximizing AUC
    diff_AUC_KL = `AUC_KL*` - `AUC_AUC*`,
    diff_brier_KL = `brier_KL*` - `brier_AUC*`,
    diff_ICI_KL = `ici_KL*` - `ici_AUC*`,
    diff_KL_KL = `kl_KL*` - `kl_AUC*`
  ) |>
  ungroup()

get_range_for_colours <- function(variable_name) {
  value <- table_kb |>
    filter(value_type == "mean") |>
    pull(!!variable_name) |>
    range(na.rm = TRUE) |> abs() |> max()
  value * c(-1, 1)
}

get_colour <- function(variable, value_type, min_or_max, colour_type) {
  if (value_type == "mean") {
    variable_string <- deparse(substitute(variable))
    if (colour_type == "bg") {
      # background colour
      if (min_or_max == "min") {
        colours <- rev(c(rev(red_colours), green_colours))
      } else {
        colours <- c(rev(red_colours), rev(green_colours))
      }
    } else {
      # text colour
      if (min_or_max == "min") {
        colours <- rev(c(rev(red_colours_txt), green_colours_txt))
      } else {
        colours <- c(rev(red_colours_txt), rev(green_colours_txt))
      }
    }
    res <- kableExtra::spec_color(
      variable,
      palette = colours,
      scale_from = get_range_for_colours(variable_string),
      na_color = "white"
    )
  } else {
    res <- "white"
  }
  res
}

table_kb <-
  table_kb |>
  rowwise() |>
  mutate(
    # Difference in metrics computed when minnimizing ICI wrt when maximizing AUC
    diff_AUC_Brier_bgcol = get_colour(diff_AUC_Brier, value_type, "max", "bg"),
    diff_AUC_Brier_txtcol = get_colour(diff_AUC_Brier, value_type, "max", "txt"),
    diff_brier_Brier_bgcol = get_colour(diff_brier_Brier, value_type, "min", "bg"),
    diff_brier_Brier_txtcol = get_colour(diff_brier_Brier, value_type, "min", "txt"),
    diff_ICI_Brier_bgcol = get_colour(diff_ICI_Brier, value_type, "min", "bg"),
    diff_ICI_Brier_txtcol = get_colour(diff_ICI_Brier, value_type, "min", "txt"),
    diff_KL_Brier_bgcol = get_colour(diff_KL_Brier, value_type, "min", "bg"),
    diff_KL_Brier_txtcol = get_colour(diff_KL_Brier, value_type, "min", "txt"),
    # Difference in metrics computed when minnimizing ICI wrt when maximizing AUC
    diff_AUC_ICI_bgcol = get_colour(diff_AUC_ICI, value_type, "max", "bg"),
    diff_AUC_ICI_txtcol = get_colour(diff_AUC_ICI, value_type, "max", "txt"),
    diff_brier_ICI_bgcol = get_colour(diff_brier_ICI, value_type, "min", "bg"),
    diff_brier_ICI_txtcol = get_colour(diff_brier_ICI, value_type, "min", "txt"),
    diff_ICI_ICI_bgcol = get_colour(diff_ICI_ICI, value_type, "min", "bg"),
    diff_ICI_ICI_txtcol = get_colour(diff_ICI_ICI, value_type, "min", "txt"),
    diff_KL_ICI_bgcol = get_colour(diff_KL_ICI, value_type, "min", "bg"),
    diff_KL_ICI_txtcol = get_colour(diff_KL_ICI, value_type, "min", "txt"),
    # Difference in metrics computed when minnimizing KL wrt when maximizing AUC
    diff_AUC_KL_bgcol = get_colour(diff_AUC_KL, value_type, "max", "bg"),
    diff_AUC_KL_txtcol = get_colour(diff_AUC_KL, value_type, "max", "txt"),
    diff_brier_KL_bgcol = get_colour(diff_brier_KL, value_type, "min", "bg"),
    diff_brier_KL_txtcol = get_colour(diff_brier_KL, value_type, "min", "txt"),
    diff_ICI_KL_bgcol = get_colour(diff_ICI_KL, value_type, "min", "bg"),
    diff_ICI_KL_txtcol = get_colour(diff_ICI_KL, value_type, "min", "txt"),
    diff_KL_KL_bgcol = get_colour(diff_KL_KL, value_type, "min", "bg"),
    diff_KL_KL_txtcol = get_colour(diff_KL_KL, value_type, "min", "txt")
  ) |>
  mutate(
    across(
      where(is.numeric),
      ~ifelse(value_type == "mean",
              scales::number(.x, accuracy = accuracy_digits),
              str_c("(", scales::number(.x, accuracy = accuracy_digits), ")")
      )
    )
  )


opts <- options(knitr.kable.NA = "")


print_table <- function(scenario) {

  table_kb <- table_kb |> filter(scenario == !!scenario) |>
    select(
      scenario, model,
      # Max AUC
      `AUC_AUC*`, `brier_AUC*`, `ici_AUC*`, `kl_AUC*`, `quant_ratio_AUC*`,
      # Min Brier
      `AUC_Brier*`, `brier_Brier*`, `ici_Brier*`, `kl_Brier*`, `quant_ratio_Brier*`,
      diff_AUC_Brier, diff_brier_Brier, diff_ICI_Brier, diff_KL_Brier,
      # Min ICI
      `AUC_ICI*`, `brier_ICI*`, `ici_ICI*`, `kl_ICI*`, `quant_ratio_ICI*`,
      diff_AUC_ICI, diff_brier_ICI, diff_ICI_ICI, diff_KL_ICI,
      # Min KL
      `AUC_KL*`, `brier_KL*`, `ici_KL*`, `kl_KL*`, `quant_ratio_KL*`,
      diff_AUC_KL, diff_brier_KL, diff_ICI_KL, diff_KL_KL,
      # colouring variables
      diff_AUC_Brier_bgcol, diff_brier_Brier_bgcol, diff_ICI_Brier_bgcol, diff_KL_Brier_bgcol,
      diff_AUC_Brier_txtcol, diff_brier_Brier_txtcol, diff_ICI_Brier_txtcol, diff_KL_Brier_txtcol,
      diff_AUC_ICI_bgcol, diff_brier_ICI_bgcol, diff_ICI_ICI_bgcol, diff_KL_ICI_bgcol,
      diff_AUC_ICI_txtcol, diff_brier_ICI_txtcol, diff_ICI_ICI_txtcol, diff_KL_ICI_txtcol,
      diff_AUC_KL_bgcol, diff_brier_KL_bgcol, diff_ICI_KL_bgcol, diff_KL_KL_bgcol,
      diff_AUC_KL_txtcol, diff_brier_KL_txtcol, diff_ICI_KL_txtcol, diff_KL_KL_txtcol
    )

  knitr::kable(
    table_kb |>
      select(
        scenario, model,
        # Max AUC
        `AUC_AUC*`, `brier_AUC*`, `ici_AUC*`, `kl_AUC*`, `quant_ratio_AUC*`,
        # Min Brier
        `AUC_Brier*`, `brier_Brier*`, `ici_Brier*`, `kl_Brier*`, `quant_ratio_Brier*`,
        diff_AUC_Brier, diff_brier_Brier, diff_ICI_Brier, diff_KL_Brier,
        # Min ICI
        `AUC_ICI*`, `brier_ICI*`, `ici_ICI*`, `kl_ICI*`, `quant_ratio_ICI*`,
        diff_AUC_ICI, diff_brier_ICI, diff_ICI_ICI, diff_KL_ICI,
        # Min KL
        `AUC_KL*`, `brier_KL*`, `ici_KL*`, `kl_KL*`, `quant_ratio_KL*`,
        diff_AUC_KL, diff_brier_KL, diff_ICI_KL, diff_KL_KL
      ),
    col.names = c(
      "Scenario", "Model",
      # # columns for GLM/GAM/GAMSEL
      # "AUC", "ICI", "KL",
      # columns for ML models selected based on AUC
      "AUC", "Brier", "ICI", "KL", "Quant. Ratio",
      # columns for ML models selected based on Brier
      "AUC", "Brier", "ICI", "KL", "Quant. Ratio", "ΔAUC", "ΔBrier", "ΔICI", "ΔKL",
      # columns for ML models selected based on ICI
      "AUC", "Brier", "ICI", "KL", "Quant. Ratio", "ΔAUC", "ΔBrier", "ΔICI", "ΔKL",
      # columns for ML models selected based on KL dist
      "AUC", "Brier", "ICI", "KL", "Quant. Ratio", "ΔAUC", "ΔBrier","ΔICI", "ΔKL"
    ),
    align = str_c("cl", str_c(rep("c", 5+9*3), collapse = ""), collapse = ""),
    escape = FALSE, booktabs = T, digits = 3, format = "markdown") |>
    # Difference in metrics computed when minnimizing Brier wrt when maximizing AUC
    kableExtra::column_spec(
      which(colnames(table_kb) == "diff_AUC_Brier"),
      background = table_kb$diff_AUC_Brier_bgcol,
      color = table_kb$diff_AUC_Brier_txtcol
    ) |>
    kableExtra::column_spec(
      which(colnames(table_kb) == "diff_brier_Brier"),
      background = table_kb$diff_brier_Brier_bgcol,
      color = table_kb$diff_brier_Brier_txtcol
    ) |>
    kableExtra::column_spec(
      which(colnames(table_kb) == "diff_ICI_Brier"),
      background = table_kb$diff_ICI_Brier_bgcol,
      color = table_kb$diff_ICI_Brier_txtcol
    ) |>
    kableExtra::column_spec(
      which(colnames(table_kb) == "diff_KL_Brier"),
      background = table_kb$diff_KL_Brier_bgcol,
      color = table_kb$diff_KL_Brier_txtcol
    ) |>
    # Difference in metrics computed when minnimizing ICI wrt when maximizing AUC
    kableExtra::column_spec(
      which(colnames(table_kb) == "diff_AUC_ICI"),
      background = table_kb$diff_AUC_ICI_bgcol,
      color = table_kb$diff_AUC_ICI_txtcol
    ) |>
    kableExtra::column_spec(
      which(colnames(table_kb) == "diff_brier_ICI"),
      background = table_kb$diff_brier_ICI_bgcol,
      color = table_kb$diff_brier_ICI_txtcol
    ) |>
    kableExtra::column_spec(
      which(colnames(table_kb) == "diff_ICI_ICI"),
      background = table_kb$diff_ICI_ICI_bgcol,
      color = table_kb$diff_ICI_ICI_txtcol
    ) |>
    kableExtra::column_spec(
      which(colnames(table_kb) == "diff_KL_ICI"),
      background = table_kb$diff_KL_ICI_bgcol,
      color = table_kb$diff_KL_ICI_txtcol
    ) |>
    # Difference in metrics computed when minnimizing KL wrt when maximizing AUC
    kableExtra::column_spec(
      which(colnames(table_kb) == "diff_AUC_KL"),
      background = table_kb$diff_AUC_KL_bgcol,
      color = table_kb$diff_AUC_KL_txtcol
    ) |>
    kableExtra::column_spec(
      which(colnames(table_kb) == "diff_brier_KL"),
      background = table_kb$diff_brier_KL_bgcol,
      color = table_kb$diff_brier_KL_txtcol
    ) |>
    kableExtra::column_spec(
      which(colnames(table_kb) == "diff_ICI_KL"),
      background = table_kb$diff_ICI_KL_bgcol,
      color = table_kb$diff_ICI_KL_txtcol
    ) |>
    kableExtra::column_spec(
      which(colnames(table_kb) == "diff_KL_KL"),
      background = table_kb$diff_KL_KL_bgcol,
      color = table_kb$diff_KL_KL_txtcol
    ) |>
    kableExtra::collapse_rows(columns = 1:2, valign = "top") |>
    kableExtra::add_header_above(
      c(" " = 2,
        # "Generalized Lin. Models" = 3,
        "AUC*" = 5,
        "Brier*" = 9,
        "ICI*" = 9,
        "KL*" = 9
      )
    )
}

for (scenario in 1:16) {
  print_table(scenario = scenario)
}
