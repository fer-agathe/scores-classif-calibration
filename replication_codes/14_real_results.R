# Real-data estimation results

library(tidyverse)
library(philentropy)
library(ranger)
library(xgboost)
library(pbapply)
library(parallel)

# Colours for train/test
colour_samples <- c(
  "Train" = "#0072B2",
  "Test" = "#D55E00"
)

# Helper Functions----

#' From the training results obtained across the grid search of the random
#' forest, extract models of interest (based on scores from test set):
#' max AUC, min ICI, min KL divergence with each of the prior distributions
#'
#' @param resul training results from `simul_xgb_real()`
get_model_interest_rf <- function(resul, tb_metrics, tb_disp_metrics) {

  # Identify best model according to AUC from test set
  ind_max_auc <- tb_metrics |>
    filter(sample == "test") |>
    arrange(desc(AUC)) |>
    pull("ind") |> pluck(1)

  # Identify best model according to Brier score from test set
  ind_min_brier <- tb_metrics |>
    filter(sample == "test") |>
    arrange(brier) |>
    pull("ind") |> pluck(1)

  # Identify best model according to ICI from test set
  ind_min_ici <- tb_metrics |>
    filter(sample == "test") |>
    arrange(ici) |>
    pull("ind") |> pluck(1)

  # Identify best model according to KL divergence
  ind_min_kl <- tb_disp_metrics |>
    filter(sample == "test") |>
    group_by(prior) |>
    arrange(KL_20_true_probas) |>
    slice_head(n = 1) |>
    select(model_interest = prior, ind)

  model_of_interest <-
    tibble(model_interest = "max_auc", ind = ind_max_auc) |>
    bind_rows(
      tibble(model_interest = "min_brier", ind = ind_min_brier)
    ) |>
    bind_rows(
      tibble(model_interest = "min_ici", ind = ind_min_ici)
    ) |>
    bind_rows(ind_min_kl)

  model_of_interest |>
    left_join(
      tb_metrics |> select(AUC, brier, ici, sample, ind),
      by = "ind",
      relationship = "many-to-many" # train and test
    ) |>
    left_join(
      tb_disp_metrics |>
        select(
          KL_20_true_probas, inter_quantile_10_90,
          sample, ind, prior, shape_1, shape_2
        ),
      by = c("ind", "sample", "model_interest" = "prior")
    ) |>
    # Add KL divergence
    left_join(
      tb_disp_metrics |>
        select(
          KL = KL_20_true_probas, quant_ratio = inter_quantile_10_90,
          sample, ind, prior
        ) |>
        pivot_wider(
          names_from = prior,
          values_from = c(KL, quant_ratio)
          # names_glue = "KL_{prior}"
        ),
      by = c("ind", "sample")
    )
}

#' From the training results obtained across the grid search of the random
#' forest, extract models of interest (based on scores from test set):
#' max AUC, min ICI, min KL divergence with each of the prior distributions
#'
#' @param resul training results from `simul_xgb_real()`
get_model_interest_xgb <- function(resul, tb_metrics, tb_disp_metrics) {

  # Identify best model according to AUC from test set
  ind_max_auc <- tb_metrics |>
    filter(sample == "validation") |>
    arrange(desc(AUC)) |>
    slice_head(n = 1) |>
    mutate(model_interest = "max_auc") |>
    select(model_interest, ind, nb_iter)

  # Identify best model according to Brier from test set
  ind_min_brier <- tb_metrics |>
    filter(sample == "validation") |>
    arrange(brier) |>
    slice_head(n = 1) |>
    mutate(model_interest = "min_brier") |>
    select(model_interest, ind, nb_iter)

  # Identify best model according to ICI from test set
  ind_min_ici <- tb_metrics |>
    filter(sample == "validation") |>
    arrange(ici) |>
    slice_head(n = 1) |>
    mutate(model_interest = "min_ici") |>
    select(model_interest, ind, nb_iter)

  # Identify best model according to KL divergence
  ind_min_kl <- tb_disp_metrics |>
    filter(sample == "validation") |>
    group_by(prior) |>
    arrange(KL_20_true_probas) |>
    slice_head(n = 1) |>
    select(model_interest = prior, ind, nb_iter)

  model_of_interest <-
    ind_max_auc |>
    bind_rows(ind_min_brier) |>
    bind_rows(ind_min_ici) |>
    bind_rows(ind_min_kl)

  model_of_interest |>
    left_join(
      tb_metrics |> select(AUC, brier, ici, sample, ind, nb_iter),
      by = c("ind", "nb_iter"),
      relationship = "many-to-many" # train and test
    ) |>
    left_join(
      tb_disp_metrics |>
        select(
          KL_20_true_probas, inter_quantile_10_90,
          sample, ind, nb_iter, prior, shape_1, shape_2
        ),
      by = c("ind", "sample", "nb_iter", "model_interest" = "prior")
    ) |>
    # Add KL divergence
    left_join(
      tb_disp_metrics |>
        select(
          KL = KL_20_true_probas, quant_ratio = inter_quantile_10_90,
          sample, ind, nb_iter, prior
        ) |>
        pivot_wider(
          names_from = prior,
          values_from = c(KL, quant_ratio)
          # names_glue = "KL_{prior}"
        ),
      by = c("ind", "nb_iter", "sample")
    )
}

#' From the training results obtained for GLM (or GAM, or GAMSEL), extracts
#' AUC, ICI, KL divergence with each of the prior distributions
#'
#' @param resul training results from `simul_xgb_real()`
get_model_interest_glm <- function(resul) {
  resul$tb_metrics |>
    mutate(model_interest = "none") |>
    left_join(
      resul$tb_disp_metrics |>
        select(
          KL = KL_20_true_probas, quant_ratio = inter_quantile_10_90,
          sample, prior
        ),
      by = "sample"
    ) |>
    pivot_wider(
      names_from = "prior",
      values_from = c(KL, quant_ratio)
      # names_glue = "KL_{prior}"
    )
}

#' From the training results obtained across the grid search, extract
#' models of interest (based on scores from test set): max AUC, min ICI,
#' min KL divergence with each of the prior distributions
#'
#' @param resul training results from `simul_xgb_real()`
get_model_interest <- function(resul,
                               model_type = c("rf", "xgb", "glm", "gam", "gamsel")) {
  tb_metrics <- map(resul$res, "tb_metrics") |> list_rbind()
  tb_disp_metrics <- map(resul$res, "tb_disp_metrics") |> list_rbind()
  if (model_type == "rf") {
    res <- get_model_interest_rf(resul, tb_metrics, tb_disp_metrics)
  } else if (model_type == "xgb") {
    res <- get_model_interest_xgb(resul, tb_metrics, tb_disp_metrics)
  } else {
    res <- get_model_interest_glm(resul)
  }
  res |> mutate(model_type = !!model_type)
}

#' @param model_interest computed metrics for the model of interest
#' @param name name of the dataset
get_row_table <- function(model_interest, name) {
  model_interest |>
    filter(sample == "test") |>
    select(
      model_type, model_interest, AUC, brier, ici,
      quant_ratio_glm, quant_ratio_gam, quant_ratio_gamsel,
      KL = KL_20_true_probas,
      KL_glm, KL_gam, KL_gamsel
    ) |>
    mutate(dataset = !!name) |>
    pivot_wider(
      names_from = "model_interest",
      values_from = c(
        "AUC", "brier", "ici", "KL",
        "quant_ratio_glm", "quant_ratio_gam", "quant_ratio_gamsel",
        "KL_glm", "KL_gam", "KL_gamsel"
      )
    ) |>
    mutate(
      # variation in AUC
      diff_auc_glm = AUC_glm - AUC_max_auc,
      diff_auc_gam = AUC_gam - AUC_max_auc,
      diff_auc_gamsel = AUC_gamsel - AUC_max_auc,
      # variation in Brier score
      diff_brier_glm = brier_glm - brier_max_auc,
      diff_brier_gam = brier_gam - brier_max_auc,
      diff_brier_gamsel = brier_gamsel - brier_max_auc,
      # variation in ICI
      diff_ici_glm = ici_glm - ici_max_auc,
      diff_ici_gam = ici_gam - ici_max_auc,
      diff_ici_gamsel = ici_gamsel - ici_max_auc,
      # variation in KL divergence
      diff_kl_glm = KL_glm_glm - KL_glm_max_auc,
      diff_kl_gam = KL_gam_gam - KL_gam_max_auc,
      diff_kl_gamsel = KL_gamsel_gamsel - KL_gamsel_max_auc
    )
}


get_row_table_glm <- function(model_glms, name) {
  model_glms |>
    filter(sample == "test") |>
    select(
      model_type, model_interest, AUC, brier, ici,
      quant_ratio_glm, quant_ratio_gam, quant_ratio_gamsel,
      KL_glm, KL_gam, KL_gamsel
    ) |>
    mutate(
      dataset = !!name,
      tmp = model_type
    ) |>
    rename(
      KL_glm_glm = KL_glm,
      KL_gam_gam = KL_gam,
      KL_gamsel_gamsel = KL_gamsel,
      quant_ratio_glm_glm = quant_ratio_glm,
      quant_ratio_gam_gam = quant_ratio_gam,
      quant_ratio_gamsel_gamsel = quant_ratio_gamsel
    ) |>
    mutate(
      AUC_glm = AUC,
      AUC_gam = AUC,
      AUC_gamsel = AUC,
      brier_glm = brier,
      brier_gam = brier,
      brier_gamsel = brier,
      ici_glm = ici,
      ici_gam = ici,
      ici_gamsel = ici
    ) |>
    select(-AUC, -brier, -ici)
}

# Estimated Metrics----

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
result_table <- vector(mode = "list", length = nrow(datasets))
priors <- vector(mode = "list", length = nrow(datasets))
names(priors) <- datasets$name
scores_hist <- list()
for (i_model in 1:nrow(datasets)) {
  name <- datasets$name[i_model]
  # Load priors
  load(str_c("output/real-data/priors_", name, ".rda"))
  priors[[i_model]] <- get(str_c("priors_", name))
  # Load results
  load(str_c("output/real-data/rf_resul_", name, ".rda"))
  load(str_c("output/real-data/xgb_resul_", name, ".rda"))
  model_interest <-
    get_model_interest(resul = rf_resul, model_type = "rf") |>
    bind_rows(get_model_interest(resul = xgb_resul, model_type = "xgb"))
  result_table_ml <- get_row_table(model_interest = model_interest, name = name)

  load(str_c("output/real-data/glm_resul_", name, ".rda"))
  load(str_c("output/real-data/gam_resul_", name, ".rda"))
  load(str_c("output/real-data/gamsel_resul_", name, ".rda"))
  model_glms <- get_model_interest(resul = glm_resul, model_type = "glm") |>
    bind_rows(get_model_interest(resul = gam_resul, model_type = "gam")) |>
    bind_rows(get_model_interest(resul = gamsel_resul, model_type = "gamsel"))
  result_table_gl <- get_row_table_glm(model_glms = model_glms, name = name)

  result_table[[i_model]] <- result_table_ml |> bind_rows(result_table_gl)

  tb_ind_model_interest <-
    model_interest |> filter(sample == "test") |>
    select(model_interest, model_type, ind)

  # Extract histograms for model of interest
  scores_hist_current <- list()
  for (model in c("rf", "xgb")) {
    scores_hist_current_model <-
      tb_ind_model_interest |>
      filter(model_type == model) |>
      pull("ind") |>
      map(~rf_resul$res[[.x]]$scores_hist)
    for (j in 1:length(scores_hist_current_model)) {
      scores_hist_current_model[[j]]$model_interest <-
        tb_ind_model_interest |>
        filter(model_type == !!model) |>
        pull(model_interest) |> pluck(j)
      scores_hist_current_model[[j]]$model_type <- model
      scores_hist_current_model[[j]]$name <- name
    }
    scores_hist_current <- c(scores_hist_current, scores_hist_current_model)
  }
  scores_hist <- c(scores_hist, scores_hist_current)
}

## Tables----
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

get_range_for_colours <- function(variable_name, table_kb) {
  value <- table_kb |>
    # filter(value_type == "mean") |>
    pull(!!variable_name) |>
    range(na.rm = TRUE) |> abs() |> max()
  value * c(-1, 1)
}

get_colour <- function(variable, value_type, min_or_max, colour_type, table_kb) {
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
  kableExtra::spec_color(
    variable,
    palette = colours,
    scale_from = get_range_for_colours(variable_string, table_kb = table_kb),
    na_color = "white"
  )
}

print_table <- function(format,
                        table_kb,
                        prior_model = c("glm", "gam", "gamsel")) {
  tb_with_colours <- table_kb |>
    rowwise() |>
    mutate(
      # When min KL
      ## Delta AUC
      diff_auc_bgcol = get_colour(
        !!sym(str_c("diff_auc_", prior_model)), value_type, "max", "bg", table_kb
      ),
      diff_auc_txtcol = get_colour(
        !!sym(str_c("diff_auc_", prior_model)), value_type, "max", "txt", table_kb
      ),
      ## Delta Brier
      diff_brier_bgcol = get_colour(
        !!sym(str_c("diff_brier_", prior_model)), value_type, "min", "bg", table_kb
      ),
      diff_brier_txtcol = get_colour(
        !!sym(str_c("diff_brier_", prior_model)), value_type, "min", "txt", table_kb
      ),
      ## Delta ICI
      diff_ici_bgcol = get_colour(
        !!sym(str_c("diff_ici_", prior_model)), value_type, "min", "bg", table_kb
      ),
      diff_ici_txtcol = get_colour(
        !!sym(str_c("diff_ici_", prior_model)), value_type, "min", "txt", table_kb
      ),
      ## Delta KL
      diff_kl_bgcol = get_colour(
        !!sym(str_c("diff_kl_", prior_model)), value_type, "min", "bg", table_kb
      ),
      diff_kl_txtcol = get_colour(
        !!sym(str_c("diff_kl_", prior_model)), value_type, "min", "txt", table_kb
      ),
    )

  table_kb |>
    mutate(
      across(
        where(is.numeric),
        ~scales::number(.x, accuracy = accuracy_digits)
      )
    ) |>
    knitr::kable(
      col.names = c(
        "Dataset", "Model",
        "AUC", "brier", "ICI", "KL", "Quant. Ratio", # model with max AUC
        "AUC", "brier", "ICI", "KL", "Quant. Ratio", # model with min Brier
        "AUC", "brier", "ICI", "KL", "Quant. Ratio", # model with min ICI
        "AUC", "brier", "ICI", "KL", "Quant. Ratio", "ΔAUC", "ΔBrier", "ΔICI", "ΔKL"
      ),
      escape = FALSE, booktabs = T, digits = 3, format = format) |>
    ## Delta AUC
    kableExtra::column_spec(
      which(colnames(table_kb) == str_c("diff_auc_", prior_model)),
      background = tb_with_colours$diff_auc_bgcol,
      color = tb_with_colours$diff_auc_txtcol
    ) |>
    ## Delta Brier
    kableExtra::column_spec(
      which(colnames(table_kb) == str_c("diff_brier_", prior_model)),
      background = tb_with_colours$diff_brier_bgcol,
      color = tb_with_colours$diff_brier_txtcol
    ) |>
    ## Delta ICI
    kableExtra::column_spec(
      which(colnames(table_kb) == str_c("diff_ici_", prior_model)),
      background = tb_with_colours$diff_ici_bgcol,
      color = tb_with_colours$diff_ici_txtcol
    ) |>
    ## Delta KL
    kableExtra::column_spec(
      which(colnames(table_kb) == str_c("diff_kl_", prior_model)),
      background = tb_with_colours$diff_kl_bgcol,
      color = tb_with_colours$diff_kl_txtcol
    ) |>
    kableExtra::collapse_rows(columns = 1:2, valign = "top") |>
    kableExtra::add_header_above(
      c(" " = 2,
        "AUC*" = 5,
        "Brier*" = 5,
        "ICI*" = 5,
        "KL*" = 9
      )
    )
}

opts <- options(knitr.kable.NA = "")

tbl_kb <- result_table |>
  list_rbind() |>
  filter(model_type %in% c("glm", "gam", "gamsel")) |>
  select(
    dataset, model_type,
    AUC_glm, brier_glm, ici_glm,
    quant_ratio_glm_glm, quant_ratio_gam_gam, quant_ratio_gamsel_gamsel,
    KL_glm_glm, KL_gam_gam, KL_gamsel_gamsel
  ) |>
  mutate(
    model_type = factor(
      model_type,
      levels = c("glm", "gam", "gamsel"),
      labels = c("GLM", "GAM", "GAMSEL")
    )
  ) |>
  mutate(
    across(
      where(is.numeric),
      ~scales::number(.x, accuracy = accuracy_digits)
    )
  )

cbind(
  tbl_kb |> filter(dataset %in% datasets$name[1:5]),
  tbl_kb |> filter(dataset %in% datasets$name[6:10])
) |>
  knitr::kable(
    col.names = c(
      rep(c("Dataset","Model",
            "AUC", "brier", "ICI",
            "QR-GLM", "QR-GAM", "QR-GAMSEL",
            "KL-GLM", "KL-GAM", "KL-GAMSEL"
      ), 2)
    ),
    escape = FALSE, booktabs = T, digits = 3, format = "markdown") |>
  kableExtra::collapse_rows(columns = c(1, 12), valign = "top")

### Priors from GLM----

result_table_glm <-
  result_table |>
  list_rbind() |>
  filter(model_type %in% c("rf", "xgb")) |>
  select(
    dataset, model_type,
    # model with max AUC
    AUC_max_auc, brier_max_auc, ici_max_auc, KL_glm_max_auc, quant_ratio_glm_max_auc,
    # model with min Brier
    AUC_min_brier, brier_min_brier, ici_min_brier, KL_glm_min_brier, quant_ratio_glm_min_brier,
    # model with min ICI
    AUC_min_ici, brier_min_ici, ici_min_ici, KL_glm_min_ici, quant_ratio_glm_min_ici,
    # model with min KL distance with prior from GLM
    AUC_glm, brier_glm, ici_glm, KL_glm_glm, quant_ratio_glm_glm,
    diff_auc_glm, diff_brier_glm, diff_ici_glm, diff_kl_glm
  ) |>
  mutate(
    model_type = factor(
      model_type,
      levels = c("rf", "xgb", "glm", "gam", "gamsel"),
      labels = c("RF", "XGB", "GLM", "GAM", "GAMSEL")
    )
  )

print_table(
  format = "markdown", table_kb = result_table_glm, prior_model = "glm"
)

### Priors from GAM----
result_table_gam <-
  result_table |>
  list_rbind() |>
  filter(model_type %in% c("rf", "xgb")) |>
  select(
    dataset, model_type,
    # model with max AUC
    AUC_max_auc, brier_max_auc, ici_max_auc, KL_gam_max_auc, quant_ratio_gam_max_auc,
    # model with min Brier
    AUC_min_brier, brier_min_brier, ici_min_brier, quant_ratio_gam_min_brier, KL_gam_min_brier,
    # model with min ICI
    AUC_min_ici, brier_min_ici, ici_min_ici, KL_gam_min_ici, quant_ratio_gam_min_ici,
    # model with min KL distance with prior from GAM
    AUC_gam, brier_gam, ici_gam, KL_gam_gam, quant_ratio_gam_gam,
    diff_auc_gam, diff_brier_gam, diff_ici_gam, diff_kl_gam
  ) |>
  mutate(
    model_type = factor(
      model_type,
      levels = c("rf", "xgb", "glm", "gam", "gamsel"),
      labels = c("RF", "XGB", "GLM", "GAM", "GAMSEL")
    )
  )


print_table(
  format = "markdown", table_kb = result_table_gam, prior_model = "gam"
)

### Priors from GAMSEL----

result_table_gamsel <-
  result_table |>
  list_rbind() |>
  filter(model_type %in% c("rf", "xgb")) |>
  select(
    dataset, model_type,
    # model with max AUC
    AUC_max_auc, brier_max_auc, ici_max_auc, KL_gamsel_max_auc, quant_ratio_gamsel_max_auc,
    # model with min Brier
    AUC_min_brier, brier_min_brier, ici_min_brier, KL_gamsel_min_brier, quant_ratio_gamsel_min_brier,
    # model with min ICI
    AUC_min_ici, brier_min_ici, ici_min_ici, KL_gamsel_min_ici, quant_ratio_gamsel_min_ici,
    # model with min KL distance with prior from GAMSEL
    AUC_gamsel, brier_gamsel, ici_gamsel, KL_gamsel_gamsel, quant_ratio_gamsel_gamsel,
    diff_auc_gamsel, diff_brier_gamsel, diff_ici_gamsel, diff_kl_gamsel
  ) |>
  mutate(
    model_type = factor(
      model_type,
      levels = c("rf", "xgb", "glm", "gam", "gamsel"),
      labels = c("RF", "XGB", "GLM", "GAM", "GAMSEL")
    )
  )


print_table(
  format = "markdown", table_kb = result_table_gamsel, prior_model = "gamsel"
)

# Distribution of Scores----

scores_ref_tibble <-
  map(scores_hist, ~tibble(
    ind = .x$ind,
    model_interest = .x$model_interest,
    model_type = .x$model_type,
    name = .x$name
  )) |>
  list_rbind() |>
  mutate(ind_list = row_number())

prior_model_names <- tribble(
  ~name, ~label, ~colour,
  "glm", "GLM", "#D55E00",
  "gam", "GAM", "#0072B2",
  "gamsel", "GAMSEL", "#E69F00"
)

print_plot <- function(prior_model, names) {
  prior_name <- prior_model_names |> filter(name == !!prior_model) |>
    pull("label")
  col_titles <- c(prior_name, "AUC*", "KL*")
  layout(
    matrix(
      data = c(
        1:3,
        4:(length(names)*3+3),
        rep(length(names)*3+4, 3)
      ),
      ncol = 3, byrow = TRUE
    ),
    heights = c(.5, rep(3, length(names)), .75)
  )

  # layout(matrix(c(1:6), ncol=3, byrow=T),heights = c(.5,3))
  par(mar = c(0, 4.1, 0, 2.1))
  for (i in 1:3) {
    plot(c(0, 1), c(0, 1), ann = F, bty = 'n', type = 'n', xaxt = 'n', yaxt = 'n')
    text(x = 0.5, y = 0.5, col_titles[i], cex = 1.6, col = "black")
  }

  colour_rf <- "#009E73"
  colour_xgb <- "#CC79A7"

  for (name in names) {
    # Get the histogram of scores estimated with the generalized linear model
    scores_prior <- priors[[name]][[str_c("scores_", prior_model)]]$scores_test
    priors_shapes <- priors[[name]][[str_c("mle_", prior_model)]]
    colour_prior <- prior_model_names |> filter(name == !!prior_model) |>
      pull("colour")


    par(mar = c(4.1, 4.1, 1.1, 1.1))
    breaks <- seq(0, 1, by = .05)
    p_scores_prior <- hist(
      scores_prior,
      breaks = breaks,
      plot = FALSE
    )
    val_u <- seq(0, 1, length = 651)
    dens_prior <-
      dbeta(val_u, priors_shapes$estimate[1], priors_shapes$estimate[2])
    # Scores estimates with RF and XBG, maximizing AUC
    ind_score_hist_rf_auc <-
      scores_ref_tibble |>
      filter(model_interest == "max_auc", model_type == "rf", name == !!name) |>
      pull("ind_list")
    ind_score_hist_xgb_auc <-
      scores_ref_tibble |>
      filter(model_interest == "max_auc", model_type == "xgb", name == !!name) |>
      pull("ind_list")
    # Scores estimates with RF and XBG, minimizing KL
    ind_score_hist_rf_kl <-
      scores_ref_tibble |>
      filter(model_interest == !!prior_model, model_type == "rf", name == !!name) |>
      pull("ind_list")
    ind_score_hist_xgb_kl <-
      scores_ref_tibble |>
      filter(model_interest == !!prior_model, model_type == "xgb", name == !!name) |>
      pull("ind_list")

    p_max_auc_rf <- scores_hist[[ind_score_hist_rf_auc]]$test
    p_max_auc_xgb <- scores_hist[[ind_score_hist_xgb_auc]]$test
    p_min_kl_rf <- scores_hist[[ind_score_hist_rf_kl]]$test
    p_min_kl_xgb <- scores_hist[[ind_score_hist_xgb_kl]]$test

    y_lim <- c(
      range(dens_prior[!is.infinite(dens_prior)]),
      range(p_scores_prior$density),
      range(p_max_auc_rf$density),
      range(p_max_auc_xgb$density),
      range(p_min_kl_rf$density),
      range(p_min_kl_xgb$density)
    ) |> range()

    x_lab <- latex2exp::TeX("$\\hat{s}(x)$")
    plot(
      p_scores_prior,
      main = "",
      xlab = x_lab,
      ylab = "",
      freq = FALSE,
      ylim = y_lim,
      col = adjustcolor(colour_prior, alpha.f = .5)
    )
    lines(val_u, dens_prior, col = colour_prior, lwd = 1.5)
    # mtext(text = substitute(paste(bold(name))), side = 2,
    #       line = 3, cex = 1, las = 0)
    mtext(text = name, side = 2, line = 3, cex = 1.1, las = 0)

    # Plot for max AUC
    plot(
      p_max_auc_rf,
      # main = "AUC*",
      main = "",
      xlab = x_lab,
      ylab = "",
      freq = FALSE,
      col = adjustcolor(colour_rf, alpha.f = .5),
      ylim = y_lim
    )
    plot(
      p_max_auc_xgb,
      add = TRUE,
      freq = FALSE,
      col = adjustcolor(colour_xgb, alpha.f = .5),
      y_lim = y_lim
    )
    lines(val_u, dens_prior, col = colour_prior, lwd = 1.5)

    # Plot for min KL
    plot(
      p_min_kl_rf,
      # main = "KL*",
      main = "",
      xlab = x_lab,
      ylab = "",
      freq = FALSE,
      col = adjustcolor(colour_rf, alpha.f = .5),
      ylim = y_lim
    )
    plot(
      p_min_kl_xgb,
      add = TRUE,
      freq = FALSE,
      col = adjustcolor(colour_xgb, alpha.f = .5),
      ylim = y_lim
    )
    lines(val_u, dens_prior, col = colour_prior, lwd = 1.5)
  }


  par(mar = c(0, 4.1, 0, 1.1))
  plot.new()
  legend(
    xpd = TRUE, ncol = 4,
    "center",
    lwd = c(1.5, rep(NA, 3)),
    col = c(colour_prior, rep(NA, 3)),
    fill = c(0, colour_prior, colour_rf, colour_xgb),
    legend = c(str_c("Prior distribution (", prior_name,")"), prior_name, "Random forest", "Extreme Gradient Boosting"),
    border=c(NA, "black","black","black")
  )
}

## Prior from GLM----

# First 5 datasets
print_plot(prior_model = "glm", names = datasets$name[1:5])
# Remaining 5
print_plot(prior_model = "glm", names = datasets$name[6:10])

## Prior from GAM----

# First 5 datasets
print_plot(prior_model = "gam", names = datasets$name[1:5])
# Remaining 5
print_plot(prior_model = "gam", names = datasets$name[6:10])

## Prior from GAMSEL----

# First 5 datasets
print_plot(prior_model = "gamsel", names = datasets$name[1:5])
# Remaining 5
print_plot(prior_model = "gamsel", names = datasets$name[6:10])
