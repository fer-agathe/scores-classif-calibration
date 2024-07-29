# Simulated Data
library(tidyverse)
library(ks)

colour_samples <- c(
  "Train" = "#0072B2",
  "Validation" = "#009E73",
  "Test" = "#D55E00"
)
# Theme for ggplot2
source("functions/utils.R")


# 1. Data Simulation----
repns_vector <- 1:100

# Functions to generate datasets
source("functions/data-ojeda.R")
# Subsampling function
source("functions/subsample_target_distribution.R")

# Creation of the 16 scenarios: 4 per GDP, varying the number of noise
# variables within each GDP type.

# Coefficients beta
coefficients <- list(
  # First category (baseline, 2 covariates)
  c(0.5, 1),  # scenario 1, 0 noise variable
  c(0.5, 1),  # scenario 2, 10 noise variables
  c(0.5, 1),  # scenario 3, 50 noise variables
  c(0.5, 1),  # scenario 4, 100 noise variables
  # Second category (same as baseline, with lower number of 1s)
  c(0.5, 1),  # scenario 5, 0 noise variable
  c(0.5, 1),  # scenario 6, 10 noise variables
  c(0.5, 1),  # scenario 7, 50 noise variables
  c(0.5, 1),  # scenario 8, 100 noise variables
  # Third category (same as baseline but with 5 num. and 5 categ. covariates)
  c(0.1, 0.2, 0.3, 0.4, 0.5, 0.01, 0.02, 0.03, 0.04, 0.05),
  c(0.1, 0.2, 0.3, 0.4, 0.5, 0.01, 0.02, 0.03, 0.04, 0.05),
  c(0.1, 0.2, 0.3, 0.4, 0.5, 0.01, 0.02, 0.03, 0.04, 0.05),
  c(0.1, 0.2, 0.3, 0.4, 0.5, 0.01, 0.02, 0.03, 0.04, 0.05),
  # Fourth category (nonlinear predictor, 3 covariates)
  c(0.5, 1, .3),  # scenario 5, 0 noise variable
  c(0.5, 1, .3),  # scenario 6, 10 noise variables
  c(0.5, 1, .3),  # scenario 7, 50 noise variables
  c(0.5, 1, .3)  # scenario 8, 100 noise variables
)

# Mean parameter for the normal distribution to draw from to draw num covariates
mean_num <- list(
  # First category (baseline, 2 covariates)
  rep(0, 2),  # scenario 1, 0 noise variable
  rep(0, 2),  # scenario 2, 10 noise variables
  rep(0, 2),  # scenario 3, 50 noise variables
  rep(0, 2),  # scenario 4, 100 noise variables
  # Second category (same as baseline, with lower number of 1s)
  rep(0, 2),  # scenario 5, 0 noise variable
  rep(0, 2),  # scenario 6, 10 noise variables
  rep(0, 2),  # scenario 7, 50 noise variables
  rep(0, 2),  # scenario 8, 100 noise variables
  # Third category (same as baseline but with 5 num. and 5 categ. covariates)
  rep(0, 5),
  rep(0, 5),
  rep(0, 5),
  rep(0, 5),
  # Fourth category (nonlinear predictor, 3 covariates)
  rep(0, 3),
  rep(0, 3),
  rep(0, 3),
  rep(0, 3)
)
# Sd parameter for the normal distribution to draw from to draw num covariates
sd_num <- list(
  # First category (baseline, 2 covariates)
  rep(1, 2),  # scenario 1, 0 noise variable
  rep(1, 2),  # scenario 2, 10 noise variables
  rep(1, 2),  # scenario 3, 50 noise variables
  rep(1, 2),  # scenario 4, 100 noise variables
  # Second category (same as baseline, with lower number of 1s)
  rep(1, 2),  # scenario 5, 0 noise variable
  rep(1, 2),  # scenario 6, 10 noise variables
  rep(1, 2),  # scenario 7, 50 noise variables
  rep(1, 2),  # scenario 8, 100 noise variables
  # Third category (same as baseline but with 5 num. and 5 categ. covariates)
  rep(1, 5),
  rep(1, 5),
  rep(1, 5),
  rep(1, 5),
  # Fourth category (nonlinear predictor, 3 covariates)
  rep(1, 3),
  rep(1, 3),
  rep(1, 3),
  rep(1, 3)
)

params_df <- tibble(
  scenario = 1:16,
  coefficients = coefficients,
  n_num = c(rep(2, 8), rep(5, 4), rep(3, 4)),
  add_categ = c(rep(FALSE, 8), rep(TRUE, 4), rep(FALSE, 4)),
  n_noise = rep(c(0, 10, 50, 100), 4),
  mean_num = mean_num,
  sd_num = sd_num,
  size_train = rep(10000, 16),
  size_valid = rep(10000, 16),
  size_test = rep(10000, 16),
  transform_probs = c(rep(FALSE, 4), rep(TRUE, 4), rep(FALSE, 4), rep(FALSE, 4)),
  linear_predictor = c(rep(TRUE, 12), rep(FALSE, 4)),
  seed = 202105
)
rm(coefficients, mean_num, sd_num)


# Creates 2 plots here
par(mfrow = c(4, 6), mar = c(2, 2, 2, 2))
for (scenario in 1:16) {
  simu_data <- simulate_data_wrapper(
    scenario = scenario,
    params_df = params_df,
    repn = repns_vector[1] # only one replication here
  )
  colour_samples <- c(
    "Train" = "#0072B2",
    "Valid" = "#009E73",
    "Test" = "#D55E00"
  )
  for (sample in c("train", "valid", "test")) {
    if (sample == "train") {
      true_prob <- simu_data$data$probs_train
      i_scores <- 1
    } else if (sample == "valid") {
      true_prob <- simu_data$data$probs_valid
      i_scores <- 2
    }else {
      true_prob <- simu_data$data$probs_test
      i_scores <- 3
    }
    hist(
      true_prob,
      breaks = seq(0, 1, by = .05),
      col = colour_samples[i_scores],
      border = "white",
      xlab = "p", ylab = "",
      main = str_c("Scen. ", scenario, ", ", c("Train", "Valid", "Test")[i_scores]),
      xlim = c(0, 1)
    )
  }
}

# Only 1 scenario per DGP
par(mfrow = c(1, 4), mar = c(4.1, 3.1, 2.1, 1.1))
for (i_dgp in 1:4) {
  scenario <- c(1, 5, 9, 13)[i_dgp]
  simu_data <- simulate_data_wrapper(
    scenario = scenario,
    params_df = params_df,
    repn = repns_vector[1] # only one replication here
  )

  true_prob <- simu_data$data$probs_test
  title <- str_c("DGP ", i_dgp)

  hist(
    true_prob,
    breaks = seq(0, 1, by = .05),
    # col = ,
    # border = "white",
    xlab = "p", ylab = "",
    main = title,
    xlim = c(0, 1)
  )
}
