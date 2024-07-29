# Generalized Linear Models

library(tidyverse)
library(philentropy)
library(ks)

# Colours for train/test
colour_samples <- c(
  "Train" = "#0072B2",
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
#' Counts the number of scores in each of the 20 equal-sized bins over [0,1]
#'
#' @param scores_train vector of scores on the train test
#' @param scores_test vector of scores on the test test
get_histogram <- function(scores_train, scores_test) {
  breaks <- seq(0, 1, by = .05)
  scores_train_hist <- hist(scores_train, breaks = breaks, plot = FALSE)
  scores_test_hist <- hist(scores_test, breaks = breaks, plot = FALSE)
  scores_hist <- list(
    train = scores_train_hist,
    test = scores_test_hist
  )
  scores_hist
}

#' Get KL divergence metrics for estimated scores and true probabilities
#'
#' @param scores_train vector of scores on the train test
#' @param scores_test vector of scores on the test test
#' @param true_prob list of true probabilities on train and test set
get_disp_metrics <- function(scores_train, scores_test, true_prob) {
  disp_train <- dispersion_metrics(
    true_probas = true_prob$train, scores = scores_train
  ) |> mutate(sample = "train")
  disp_test <- dispersion_metrics(
    true_probas = true_prob$test, scores = scores_test
  ) |> mutate(sample = "test")

  tb_disp_metrics <- disp_train |>
    bind_rows(disp_test)
  tb_disp_metrics
}

#' Get the performance and calibration metrics for estimated scores
#'
#' @param scores_train vector of scores on the train test
#' @param scores_test vector of scores on the test test
#' @param tb_train train set
#' @param tb_test test set
#' @param true_prob list of true probabilities on train and test set
get_perf_metrics <- function(scores_train,
                             scores_test,
                             tb_train,
                             tb_test,
                             true_prob) {
  # We add very small noise to predicted scores
  # otherwise the local regression may crash
  scores_train_noise <- scores_train +
    runif(n = length(scores_train), min = 0, max = 0.01)
  scores_train_noise[scores_train_noise > 1] <- 1
  metrics_train <- compute_metrics(
    obs = tb_train$d, scores = scores_train_noise, true_probas = true_prob$train
  ) |> mutate(sample = "train")

  scores_test_noise <- scores_test +
    runif(n = length(scores_test), min = 0, max = 0.01)
  scores_test_noise[scores_test_noise > 1] <- 1
  metrics_test <- compute_metrics(
    obs = tb_test$d, scores = scores_test_noise, true_probas = true_prob$test
  ) |> mutate(sample = "test")

  tb_metrics <- metrics_train |>
    bind_rows(metrics_test)
  tb_metrics
}

#' Estimation of P(q1 < score < q2)
#'
#' @param scores_train vector of scores on the train test
#' @param scores_test vector of scores on the test test
#' @param q1 vector of desired values for q1 (q2 = 1-q1)
estim_prop <- function(scores_train,
                       scores_test,
                       q1 = c(.1, .2, .3, .4)) {
  proq_scores_train <- map(
    q1,
    ~prop_btw_quantiles(s = scores_train, q1 = .x)
  ) |>
    list_rbind() |>
    mutate(sample = "train")
  proq_scores_test <- map(
    q1,
    ~prop_btw_quantiles(s = scores_test, q1 = .x)
  ) |>
    list_rbind() |>
    mutate(sample = "test")

  proq_scores_train |>
    bind_rows(proq_scores_test)
}

simul_glm <- function(scenario, params_df, repn) {
  # Generate Data----
  simu_data <- simulate_data_wrapper(
    scenario = scenario,
    params_df = params_df,
    repn = repn
  )

  tb_train <- simu_data$data$train |> rename(d = y)
  tb_test <- simu_data$data$test |> rename(d = y)

  true_prob <-
    list(
      train = simu_data$data$probs_train,
      test = simu_data$data$probs_test
    )

  # Estimation----
  estim_logit <- glm(d ~ ., data = tb_train, family = "binomial")

  # Predicted scores----
  scores_train <- predict(estim_logit, newdata = tb_train, type = "response")
  scores_test <- predict(estim_logit, newdata = tb_test, type = "response")

  # Histogram of scores----
  scores_hist <- get_histogram(scores_train, scores_test)

  # Performance and Calibration Metrics----
  tb_metrics <- get_perf_metrics(
    scores_train = scores_train,
    scores_test = scores_test,
    tb_train = tb_train,
    tb_test = tb_test,
    true_prob = true_prob) |>
    mutate(
      scenario = simu_data$scenario,
      repn = simu_data$repn
    )

  # Dispersion Metrics----
  tb_disp_metrics <- get_disp_metrics(
    scores_train = scores_train,
    scores_test = scores_test,
    true_prob = true_prob
  ) |>
    mutate(
      scenario = simu_data$scenario,
      repn = simu_data$repn
    )

  metrics <- suppressMessages(
    left_join(tb_metrics, tb_disp_metrics)
  )

  # Estimation of P(q1 < score < q2)----
  tb_prop_scores <- estim_prop(scores_train, scores_test) |>
    mutate(
      scenario = simu_data$scenario,
      repn = simu_data$repn
    )

  list(
    scenario = simu_data$scenario,   # data scenario
    repn = simu_data$repn,           # data replication ID
    metrics = metrics,               # table with performance/calib/divergence
    tb_prop_scores = tb_prop_scores, # table with P(q1 < score < q2)
    scores_hist = scores_hist        # histogram of scores
  )
}


# 3. Estimations----
library(pbapply)
library(parallel)
ncl <- detectCores()-1
(cl <- makeCluster(ncl))

clusterEvalQ(cl, {
  library(tidyverse)
}) |>
  invisible()

clusterExport(cl, c("simulate_data"))

for (i_scenario in 1:16) {
  scenario <- i_scenario
  print(str_c("Scenario ", scenario, "/", nrow(params_df)))

  resul_glm_scenario <-
    pblapply(
      1:length(repns_vector), function(i) simul_glm(
        scenario = scenario, params_df = params_df, repn = repns_vector[i]
      )
    )
  save(
    resul_glm_scenario,
    file = str_c("output/simul/dgp-ojeda/resul_glm_scenario_", scenario, ".rda")
  )
}
stopCluster(cl)

# 4. Results----
files <- str_c(
  "output/simul/dgp-ojeda/resul_glm_scenario_", 1:16, ".rda"
)
resul_glm <- map(files[file.exists(files)], ~{load(.x) ; resul_glm_scenario})

metrics_glm_all <- map(
  resul_glm,
  function(resul_glm_sc) map(resul_glm_sc, "metrics") |> list_rbind()
) |>
  list_rbind()

## Boxplots of metrics for each scenario----
df_plot <- metrics_glm_all |>
  select(scenario, sample, AUC, mse, ici, KL_20_true_probas) |>
  pivot_longer(cols = -c(scenario, sample), names_to = "metric") |>
  mutate(
    scenario = factor(scenario),
    sample = factor(
      sample,
      levels = c("train", "test"),
      labels = c("Train", "Test")
    ),
    metric = factor(
      metric,
      levels = c("AUC", "mse", "ici", "KL_20_true_probas"),
      labels = c("AUC", "MSE", "ICI", "KL Divergence")
    )
  )

ggplot(
  data = df_plot,
  mapping = aes(x = scenario, y = value, fill = sample)
) +
  geom_boxplot() +
  facet_wrap(~metric, scales = "free") +
  scale_fill_manual("Sample", values = colour_samples) +
  labs(x = "Scenario", y = NULL) +
  theme_paper()

## Distribution of Scores----
scores_hist_all <-
  map(
    resul_glm,
    function(resul_glm_sc) map(resul_glm_sc, "scores_hist")
  )

# Showing histograms for the first replication only
repn <- 1

plot_hist <- function(metrics_interest, scores_hist_interest) {
  subtitle <- str_c(
    "AUC = ", round(metrics_interest$AUC, 2), ", ",
    "Brier = ", round(metrics_interest$ici, 2), ", \n",
    "ICI = ", round(metrics_interest$ici, 2), ", ",
    "KL = ", round(metrics_interest$KL_20_true_probas, 2)
  )
  plot(
    # main = "Test Set",
    main = "",
    scores_hist_interest$test,
    xlab = latex2exp::TeX("$\\hat{s}(x)$"),
    ylab = ""
  )
  mtext(side = 3, line = -0.25, adj = .5, subtitle, cex = .6)
}


plot_hist_dgp <- function(repn) {
  layout(
    matrix(c(1:5, (1:20)+5), ncol = 5, byrow = TRUE),
    heights = c(.3, rep(3, 4))
  )
  par(mar = c(0, 4.1, 0, 2.1))
  col_titles <- c("True Probabilities", str_c(c(0, 10, 50, 100), " noise variables"))
  for (i in 1:5) {
    plot(c(0, 1), c(0, 1), ann = F, bty = 'n', type = 'n', xaxt = 'n', yaxt = 'n')
    text(x = 0.5, y = 0.5, col_titles[i], cex = 1.6, col = "black")
  }

  par(mar = c(4.1, 4.1, 1.6, 2.1))
  for (dgp in 1:4) {
    scenarios <- (1:4) + 4*(dgp-1)
    # True Probabilities
    simu_data <- simulate_data_wrapper(
      scenario = scenarios[1],
      params_df = params_df,
      repn = repn # only one replication here
    )
    true_prob <- simu_data$data$probs_train
    hist(
      true_prob,
      breaks = seq(0, 1, by = .05),
      xlab = "p", ylab = "",
      main = "",
      xlim = c(0, 1)
    )
    mtext(
      side = 2, str_c("DGP ", dgp), line = 2.5, cex = 1,
      font.lab = 2
    )

    for (i_scenario in scenarios) {
      metrics_interest <- metrics_glm_all |>
        filter(scenario == !!i_scenario, repn == !!repn, sample == "test")
      scores_hist_interest <- scores_hist_all[[i_scenario]][[repn]]
      plot_hist(
        metrics_interest = metrics_interest,
        scores_hist_interest = scores_hist_interest
      )
    }
  }
}

plot_hist_dgp(repn = 1)

