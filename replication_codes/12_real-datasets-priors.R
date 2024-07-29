# Datasets and Priors for real data
library(tidyverse)
library(gam)
library(gamsel)

download_data <- TRUE # if FALSE, loads the data previously downloaded

# Colours for train/test
colour_samples <- c(
  "Train" = "#0072B2",
  "Test" = "#D55E00"
)

# Pre-processing functions and estimation functions
source("functions/real-data.R")

if (!dir.exists("output/real-data/")) dir.create("output/real-data/")

# Datasets----

## Abalone----
# URL to the data: <https://archive.ics.uci.edu/dataset/1/abalone>
# Description: Predict the age of abalone from physical measurements.
# Number of instances: 4,177
# Features: 8

name <- "abalone"
if (download_data) {
  if (!dir.exists("data")) dir.create("data")
  download.file(
    url = str_c("https://archive.ics.uci.edu/static/public/1/", name, ".zip"),
    destfile = str_c("data/", name, ".zip")
  )
}

tb_abalone <- read_csv(
  file = unz(str_c("data/", name, ".zip"), str_c(name, ".data")),
  col_names = c(
    "Sex", "Length", "Diameter", "Height", "Whole_weight",
    "Shucked_weight", "Viscera_weight", "Shell_weight", "Rings"),
  show_col_types = FALSE
)

tb_abalone <- tb_abalone |>
  mutate(Sex = ifelse(Sex == "M", 1, 0))
target_name <- "Sex"

# Fit Beta distribution
priors_abalone <- get_beta_fit(
  dataset = tb_abalone, target_name = target_name, seed = 1234
)

save(priors_abalone, file = "output/real-data/priors_abalone.rda")
save(tb_abalone, file = "output/real-data/tb_abalone.rda")

plot_hist_scores_beta(priors_abalone, "abalone")

## Adult----
# URL to the data: <https://archive.ics.uci.edu/dataset/2/adult>
# Description: Predict whether income exceeds $50K/yr based on census data. Also known as "Census Income" dataset.
# Number of instances: 48,842
# Features: 14

name <- "adult"

if (download_data) {
  if (!dir.exists("data")) dir.create("data")
  download.file(
    url = str_c("https://archive.ics.uci.edu/static/public/2/", name, ".zip"),
    destfile = str_c("data/", name, ".zip")
  )
}

tb_adult <- read_csv(
  file = unz(str_c("data/", name, ".zip"), str_c(name, ".data")),
  col_names = c(
    "age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week", "native-country",
    "income"
  ),
  show_col_types = FALSE
)

tb_adult <- tb_adult |>
  mutate(high_income = ifelse(income == ">50K", 1, 0)) |>
  dplyr::select(-income)
target_name <- "high_income"

# Fit Beta distribution
priors_adult <- get_beta_fit(
  dataset = tb_adult, target_name = target_name, seed = 1234
)

save(priors_adult, file = "output/real-data/priors_adult.rda")
save(tb_adult, file = "output/real-data/tb_adult.rda")

plot_hist_scores_beta(priors_adult, "adult")

## Bank Marketing----
# URL to the data: <https://archive.ics.uci.edu/dataset/222/bank+marketing>
# Description: The data is related with direct marketing campaigns (phone
#  calls) of a Portuguese banking institution. The classification goal is to
#  predict if the client will subscribe a term deposit (variable y).
# Number of instances: 45,211
# Features: 16

name <- "bank"

if (download_data) {
  if (!dir.exists("data")) dir.create("data")
  download.file(
    url = "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip",
    destfile = str_c("data/", name, ".zip")
  )
}

dir.create("data/bank/")
system("unzip data/bank.zip -d data/bank/")
system("unzip data/bank/bank.zip -d data/bank/")
tb_bank <- read_csv2(
  file = unz(str_c("data/bank/", name, ".zip"), str_c("bank-full.csv")),
  skip = 1,
  col_names = c(
    "age", "job", "marital", "education", "default", "balance", "housing",
    "loan", "contact", "day", "month", "duration", "campaign", "pdays",
    "previous", "poutcome", "y"
  ),
  show_col_types = FALSE
)
system("rm -rf data/bank/")

tb_bank <- tb_bank |>
  mutate(y = ifelse(y == "yes", 1, 0))
target_name <- "y"

# Fit Beta distribution
priors_bank <- get_beta_fit(
  dataset = tb_bank, target_name = target_name, seed = 1234
)

save(priors_bank, file = "output/real-data/priors_bank.rda")
save(tb_bank, file = "output/real-data/tb_bank.rda")

plot_hist_scores_beta(priors_bank, "bank")


## Default of Credit Card Clients----
# URL to the data: <https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients>
# Description: This research aimed at the case of customers' default payments in
#   Taiwan and compares the predictive accuracy of probability of default among
#   six data mining methods.
# Number of instances: 30,000
# Features: 23

name <- "default"

if (download_data) {
  if (!dir.exists("data")) dir.create("data")
  download.file(
    url = str_c("https://archive.ics.uci.edu/static/public/350/",
                "default+of+credit+card+clients.zip"
    ),
    destfile = str_c("data/", name, ".zip")
  )
}

dir.create("data/default/")
system("unzip data/default.zip -d data/default/")
tb_default <- readxl::read_excel(
  path = "data/default/default of credit card clients.xls",
  skip = 1
) |>
  select(-ID)
system("rm -rf data/default")


tb_default <-
  tb_default |>
  mutate(
    across(all_of(c(
      "SEX", "EDUCATION", "MARRIAGE",
      "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6")), as.factor)
  ) |>
  mutate(
    across(all_of(c(
      "EDUCATION", "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"
    )), ~fct_lump(.x, prop = .05)
    )
  ) |>
  rename(default = `default payment next month`)
target_name <- "default"

# Fit Beta distribution
priors_default <- get_beta_fit(
  dataset = tb_default, target_name = target_name, seed = 1234
)

save(priors_default, file = "output/real-data/priors_default.rda")
save(tb_default, file = "output/real-data/tb_default.rda")

plot_hist_scores_beta(priors_default, "default")

## Dry Bean----
# URL to the data: <https://archive.ics.uci.edu/dataset/602/dry+bean+dataset>
# Description: Images of 13,611 grains of 7 different registered dry beans
#   were taken with a high-resolution camera. A total of 16 features;
#   12 dimensions and 4 shape forms, were obtained from the grains.
# Number of instances: 13,611
# Features: 16

name <- "drybean"


if (download_data) {
  if (!dir.exists("data")) dir.create("data")
  download.file(
    url = "https://archive.ics.uci.edu/static/public/602/dry+bean+dataset.zip",
    destfile = str_c("data/", name, ".zip")
  )
}

dir.create("data/drybean/")
system("unzip data/drybean.zip -d data/drybean/")
tb_drybean <- readxl::read_excel(
  path = "data/drybean/DryBeanDataset/Dry_Bean_Dataset.xlsx"
)
system("rm -rf data/drybean")

tb_drybean <- tb_drybean |>
  mutate(is_dermason = ifelse(Class == "DERMASON", 1, 0)) |>
  select(-Class)
target_name <- "is_dermason"

# Fit Beta distribution
priors_drybean <- get_beta_fit(
  dataset = tb_drybean, target_name = target_name, seed = 1234
)

save(priors_drybean, file = "output/real-data/priors_drybean.rda")
save(tb_drybean, file = "output/real-data/tb_drybean.rda")

plot_hist_scores_beta(priors_drybean, "drybean")

## In-Vehicle Coupon Recommendation----
# URL to the data: <https://archive.ics.uci.edu/dataset/603/in+vehicle+coupon+recommendation>
# Description: This data studies whether a person will accept the coupon
#   recommended to him in different driving scenarios.
# Number of instances: 12,684
# Features: 25


if (download_data) {
  if (!dir.exists("data")) dir.create("data")
  download.file(
    url = str_c("https://archive.ics.uci.edu/static/public/603/",
                "in+vehicle+coupon+recommendation.zip"),
    destfile = str_c("data/", name, ".zip")
  )
}

tb_coupon <- read_csv(
  file = unz(str_c("data/", name, ".zip"), "in-vehicle-coupon-recommendation.csv"),
  show_col_types = FALSE
)

tb_coupon <-
  tb_coupon |>
  mutate(
    temperature = as.factor(temperature),
    has_children = as.factor(has_children),
    toCoupon_GEQ15min = as.factor(toCoupon_GEQ15min),
    toCoupon_GEQ25min = as.factor(toCoupon_GEQ25min),
    direction_same = as.factor(direction_same)
  ) |>
  select(-toCoupon_GEQ5min, -direction_opp, -car) |>
  rename(y = Y)

tb_coupon <- na.omit(tb_coupon)

target_name <- "y"

# Fit Beta distribution
priors_coupon <- get_beta_fit(
  dataset = tb_coupon, target_name = target_name, seed = 1234
)

save(priors_coupon, file = "output/real-data/priors_coupon.rda")
save(tb_coupon, file = "output/real-data/tb_coupon.rda")

plot_hist_scores_beta(priors_coupon, "coupon")


## Mushroom----
# URL to the data: <https://archive.ics.uci.edu/dataset/73/mushroom>
# Description: From Audobon Society Field Guide; mushrooms described in terms
#   of physical characteristics; classification: poisonous or edible.
# Number of instances: 8,124
# Features: 22

name <- "mushroom"


if (download_data) {
  if (!dir.exists("data")) dir.create("data")
  download.file(
    url = str_c("https://archive.ics.uci.edu/static/public/73/mushroom.zip"),
    destfile = str_c("data/", name, ".zip")
  )
}

tb_mushroom <- read_csv(
  file = unz(str_c("data/", name, ".zip"), "agaricus-lepiota.data"),
  col_names = c(
    "edible",
    "cap_shape", "cap_surface", "cap_color", "bruises", "odor",
    "gill_attachment", "gill_spacing", "gill_size", "gill_color",
    "stalk_shape", "stalk_root", "stalk_surface_above_ring",
    "stalk_surface_below_ring", "stalk_color_above_ring",
    "stalk_color_below_ring", "veil_type", "veil_color", "ring_number",
    "ring_type", "spore_print_color", "population", "habitat"
  ),
  show_col_types = FALSE
)

tb_mushroom <- tb_mushroom |>
  mutate(bruises = ifelse(bruises == TRUE, "yes", "no")) |>
  mutate(edible = ifelse(edible == "e", 1, 0)) |>
  select(-veil_type)
target_name <- "edible"

# Fit Beta distribution
priors_mushroom <- get_beta_fit(
  dataset = tb_mushroom, target_name = target_name, seed = 1234
)

save(priors_mushroom, file = "output/real-data/priors_mushroom.rda")
save(tb_mushroom, file = "output/real-data/tb_mushroom.rda")

plot_hist_scores_beta(priors_mushroom, "mushroom")




## Occupancy Detection----
# URL to the data: <https://archive.ics.uci.edu/dataset/357/occupancy+detection>
# Description: Predict the age of occupancy from physical measurements.
# Number of instances: 20,560
# Features: 6

name <- "occupancy"

if (download_data) {
  if (!dir.exists("data")) dir.create("data")
  download.file(
    url = str_c("https://archive.ics.uci.edu/static/public/357/",
                "occupancy+detection.zip"),
    destfile = str_c("data/", name, ".zip")
  )
}

tb_occupancy <- read_csv(
  file = unz(str_c("data/", name, ".zip"), "datatraining.txt"),
  col_names = c(
    "id", "date","Temperature","Humidity","Light","CO2",
    "HumidityRatio","Occupancy"
  ),
  show_col_types = FALSE, skip = 1
) |>
  bind_rows(
    read_csv(
      file = unz(str_c("data/", name, ".zip"), "datatest.txt"),
      col_names = c(
        "id", "date","Temperature","Humidity","Light","CO2",
        "HumidityRatio","Occupancy"
      ),
      show_col_types = FALSE, skip = 1,
    )
  ) |>
  bind_rows(
    read_csv(
      file = unz(str_c("data/", name, ".zip"), "datatest2.txt"),
      show_col_types = FALSE, skip = 1,
      col_names = c(
        "id", "date","Temperature","Humidity","Light","CO2",
        "HumidityRatio","Occupancy"
      ),
    )
  ) |>
  select(-id)

tb_occupancy <- tb_occupancy |> select(-date)
target_name <- "Occupancy"

# Fit Beta distribution
priors_occupancy <- get_beta_fit(
  dataset = tb_occupancy, target_name = target_name, seed = 1234
)

save(priors_occupancy, file = "output/real-data/priors_occupancy.rda")
save(tb_occupancy, file = "output/real-data/tb_occupancy.rda")

plot_hist_scores_beta(priors_occupancy, "occupancy")


## Wine Quality

# URL to the data: <https://archive.ics.uci.edu/dataset/186/wine+quality>
# Description: Two datasets are included, related to red and white vinho verde
#  wine samples, from the north of Portugal.
#  The goal is to model wine quality based on physicochemical tests
#   (see [Cortez et al., 2009], http://www3.dsi.uminho.pt/pcortez/wine/).
# Number of instances: 4,898
# Features: 11

name <- "winequality"

if (download_data) {
  if (!dir.exists("data")) dir.create("data")
  download.file(
    url = str_c("https://archive.ics.uci.edu/static/public/186/",
                "wine+quality.zip"),
    destfile = str_c("data/", name, ".zip")
  )
}

red_wine <- read_csv2(
  file = unz(str_c("data/", name, ".zip"), "winequality-red.csv"),
  show_col_types = FALSE) |>
  mutate(wine_type = "red")
white_wine <- read_csv2(
  file = unz(str_c("data/", name, ".zip"), "winequality-white.csv"),
  show_col_types = FALSE) |>
  mutate(wine_type = "white") |>
  mutate(`residual sugar` = as.numeric(`residual sugar`))

tb_winequality <- red_wine |> bind_rows(white_wine) |>
  mutate(high_quality = ifelse(quality >= 6, 1, 0)) |>
  mutate(across(all_of(c(
    "density", "chlorides", "volatile acidity", "sulphates", "citric acid"
  )), ~as.numeric(.x))) |>
  select(-quality)
tb_winequality <- na.omit(tb_winequality)
target_name <- "high_quality"

# Fit Beta distribution
priors_winequality <- get_beta_fit(
  dataset = tb_winequality, target_name = target_name, seed = 1234
)

save(priors_winequality, file = "output/real-data/priors_winequality.rda")
save(tb_winequality, file = "output/real-data/tb_winequality.rda")

plot_hist_scores_beta(priors_winequality, "winequality")


## Spambase

# URL to the data: <https://archive.ics.uci.edu/dataset/94/spambase>
# Description: Classifying Email as Spam or Non-Spam
# Number of instances: 4,601
# Features: 57


if (download_data) {
  if (!dir.exists("data")) dir.create("data")
  download.file(
    url = str_c("https://archive.ics.uci.edu/static/public/2/", name, ".zip"),
    destfile = str_c("data/", name, ".zip")
  )
}

tb_spambase <- read_csv(
  file = unz(str_c("data/", name, ".zip"), str_c(name, ".data")),
  col_names = c(
    "word_freq_make", "word_freq_address", "word_freq_all", "word_freq_3d",
    "word_freq_our", "word_freq_over", "word_freq_remove", "word_freq_internet",
    "word_freq_order", "word_freq_mail", "word_freq_receive", "word_freq_will",
    "word_freq_people", "word_freq_report", "word_freq_addresses",
    "word_freq_free", "word_freq_business", "word_freq_email", "word_freq_you",
    "word_freq_credit", "word_freq_your", "word_freq_font", "word_freq_000",
    "word_freq_money", "word_freq_hp", "word_freq_hpl", "word_freq_george",
    "word_freq_650", "word_freq_lab", "word_freq_labs", "word_freq_telnet",
    "word_freq_857", "word_freq_data", "word_freq_415", "word_freq_85",
    "word_freq_technology", "word_freq_1999", "word_freq_parts", "word_freq_pm",
    "word_freq_direct", "word_freq_cs", "word_freq_meeting",
    "word_freq_original", "word_freq_project", "word_freq_re", "word_freq_edu",
    "word_freq_table", "word_freq_conference", "char_freq_;", "char_freq_(",
    "char_freq_[", "char_freq_!", "char_freq_$", "char_freq_#",
    "capital_run_length_average", "capital_run_length_longest",
    "capital_run_length_total", "is_spam"
  ),
  show_col_types = FALSE
)

target_name <- "is_spam"

# Fit Beta distribution
priors_spambase <- get_beta_fit(
  dataset = tb_spambase, target_name = target_name, seed = 1234
)

save(priors_spambase, file = str_c("output/real-data/priors_spambase.rda"))
save(tb_spambase, file = "output/real-data/tb_spambase.rda")

plot_hist_scores_beta(priors_spambase, "spambase")

# Summary----

datasets <- tribble(
  ~name, ~target_name, ~reference,
  "abalone", "Sex", "@misc_abalone_1",
  "adult", "high_income", "@misc_adult_2",
  "bank", "y", "@misc_bank_marketing_222",
  "default", "default", "@misc_default_of_credit_card_clients_350",
  "drybean", "is_dermason", "@misc_dry_bean_602",
  "coupon", "y", "@misc_vehicle_coupon_recommendation_603",
  "mushroom", "edible", "@misc_mushroom_73",
  "occupancy", "Occupancy", "@misc_occupancy_detection__357",
  "winequality", "high_quality", "@misc_wine_quality_186",
  "spambase", "is_spam", "@misc_spambase_94"
)

dataset_info <- vector(mode = "list", length = nrow(datasets))
for (i in 1:nrow(datasets)) {
  name <- datasets$name[i]
  target_name <- datasets$target_name[i]
  current_data <- get(str_c('tb_', name))
  current_target <- current_data |> pull(!!target_name)
  current_ref <- datasets$reference[i]
  n <- nrow(current_data)
  n_col <- ncol(current_data)
  n_numeric <- current_data |> select(-!!target_name) |>
    select(where(is.numeric)) |>
    ncol()
  dataset_info[[i]] <- tibble(
    Dataset = name,
    n = n,
    `# features` = n_col-1,
    `# numeric features` = n_numeric,
    `Prop. target = 1` = round(sum(current_target == 1) / n, 2),
    Reference = current_ref
  )
}

dataset_info <- list_rbind(dataset_info)
knitr::kable(dataset_info, booktabs = TRUE, format.args = list(big.mark = ","))
