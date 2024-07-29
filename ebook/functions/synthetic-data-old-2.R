#' Simulates binary data, depending on a seed
#'
#' @description Consider a binary variable that is assumed to follow a Bernoulli
#'   distribution: \eqn{D_i\sim B(p_i)}, where \eqn{p_i} is the probability of
#'   observing \eqn{D_i = 1}. We define this probability \eqn{p_i} according to
#'   the following function \deqn{p_i^{u} =
#'   \left(\frac{1}{1+\exp(-\eta_i^u)}\right)^{\alpha} \tag{1.3},} where
#'   \eqn{\eta_i^u} is defined as follows: \deqn{\eta_i^u =\gamma \times \left( a_1 x_1 + a_2 x_2 \right),} where \eqn{(a_1,
#'   a_2) = (-0.1, 0.05)} by default.
#'
#' @param n_obs number of desired observations
#' @param seed seed to use to generate the data
#' @param alpha scale parameter for the latent probability (if different from 1,
#'   the probabilities are transformed and it may induce decalibration)
#' @param gamma scale parameter for the latent score (if different from 1, the
#'   probabilities are transformed and it may induce decalibration)
#' @param a vector of length 4 with values for \eqn{a}. Default to \eqn{(a_1,
#'   a_2) = (-0.1, 0.055)}
#'
#' @importFrom stats rnorm runif rbinom
#' @importFrom tibble tibble
#'
#' @returns tibble with simulated data:
#'
#' \describe{
#'   \item{\code{p}}{True probability}
#'   \item{\code{d}}{Observed binary event $\{0,1\}$}
#'   \item{\code{x1, x2, x3, x4}}{Features}
#' }
#' @examples
#' toy_data <- sim_data(n_obs = 2000, seed = 1)
#' @export
sim_data <- function(n_obs = 2000,
                     seed,
                     alpha = 1,
                     gamma = 1,
                     a = c(-0.1, 0.05)) {
  set.seed(seed)

  if (length(a) != 2) stop("`a` must be a vector of numerical of length 2")

  x1 <- runif(n_obs)
  x2 <- runif(n_obs)
  x3 <- runif(n_obs)
  x4 <- runif(n_obs)

  a1 <- a[1]
  a2 <- a[2]
  # a3 <- a[3]
  # a4 <- a[4]

  # epsilon_p <- rnorm(n_obs, mean = 0, sd = .01)

  # True latent score
  eta <- a1*x1 + a2*x2 - (a1+a2)/2
  # a3*x3 + a4*x4 +
  # Transformed latent score
  eta_u <- gamma * eta

  # True probability
  p <- (1 / (1 + exp(-eta)))
  # Transformed probability
  p_u <- ((1 / (1 + exp(-eta_u))))^alpha

  # Observed event
  d <- rbinom(n_obs, size = 1, prob = p)

  tibble(
    # Event Probability
    p = p,
    p_u = p_u,
    # Binary outcome variable
    d = d,
    # Variables
    x1 = x1,
    x2 = x2,
    x3 = x3,
    x4 = x4
  )
}

#' Get calibration/test samples from the DGP
#'
#' @param n_obs number of desired observations
#' @param prop_train proportion of the data to be put in the train set (default
#'   to `.6`)
#' @param prop_calib proportion of the data to be put in the calibration set
#'   default to (`.2`)
#' @param seed seed to use to generate the data
#' @param alpha scale parameter for the latent probability (if different from 1,
#'   the probabilities are transformed and it may induce decalibration)
#' @param gamma scale parameter for the latent score (if different from 1, the
#'   probabilities are transformed and it may induce decalibration)
#' @param a vector of length 4 with values for \eqn{a}. Default to \eqn{(a_1,
#'   a_2) = (-0.1, 0.05)} in the DGP (see
#'   \code{\link{sim_data}})
#'
#' @importFrom dplyr tibble select slice
#'
#' @returns list with simulated data:
#'
#' \describe{
#'   \item{\code{data_all}}{A tibble with simulated data.}
#'   \item{\code{data}}{A tibble with simulated data, without the true probability.}
#'   \item{\code{tb_train}}{A subset of \code{data} that constitutes the train set.}
#'   \item{\code{tb_calib}}{A subset of \code{data} that constitutes the calibration set.}
#'   \item{\code{tb_test}}{A subset of \code{data} that constitutes the test set.}
#'   \item{\code{true_probas_train}}{True probability in the train set (tibble).}
#'   \item{\code{true_probas_calib}}{True probability in the calibration set (tibble).}
#'   \item{\code{true_probas_test}}{True probability in the test set (tibble).}
#'   \item{\code{train_index}}{Index in \code{data} that constitute the train set.}
#'   \item{\code{calib_index}}{Index in \code{data} that constitute the calibration set.}
#'   \item{\code{seed}}{Seed used to generate the data (using \code{sim_data()}).}
#'   \item{\code{n_obs}}{Number of generated observations.}
#'   \item{\code{alpha}}{Values used for \code{alpha}.}
#'   \item{\code{gamma}}{Values used for \code{gamma}.}
#'   \item{\code{a}}{Vector of values used of \code{a}.}
#' }
#'
#' @seealso [sim_data()] for the Data Generating Process
#' @examples
#' toy_data_samples <- get_samples(n_obs = 2000, seed = 1)
#' @export
get_samples <- function(n_obs = 2000,
                        prop_train = .6,
                        prop_calib = .2,
                        seed,
                        alpha = 1,
                        gamma = 1,
                        a = c(-0.1, 0.05)) {

  if (prop_train + prop_calib > 1)
    stop("Proportion in train + proportion in calib should be < 1.")

  set.seed(seed)
  data_all <- sim_data(
    n_obs = n_obs,
    seed = seed,
    alpha = alpha,
    gamma = gamma,
    a = a
  )

  # Train/calibration/test sets----
  data <- data_all |> select(d, x1:x4)
  true_probas <- data_all |> select(p)

  train_index <- sample(1:nrow(data), size = prop_train * nrow(data), replace = FALSE)
  tb_train <- data |> slice(train_index)
  tb_calib_test <- data |> slice(-train_index)
  true_probas_train <- true_probas |> slice(train_index)
  true_probas_calib_test <- true_probas |> slice(-train_index)

  calib_index <- sample(
    1:nrow(tb_calib_test), size = prop_calib * nrow(tb_calib_test), replace = FALSE
  )
  tb_calib <- tb_calib_test |> slice(calib_index)
  if (prop_calib > 0) {
    tb_test <- tb_calib_test |> slice(-calib_index)
    true_probas_test <- true_probas_calib_test |> slice(-calib_index)
  } else {
    tb_test <- tb_calib_test
    true_probas_test <- true_probas_calib_test
  }

  true_probas_calib <- true_probas_calib_test |> slice(calib_index)

  list(
    data_all = data_all,
    data = data,
    tb_train = tb_train,
    tb_calib = tb_calib,
    tb_test = tb_test,
    true_probas_train = true_probas_train,
    true_probas_calib = true_probas_calib,
    true_probas_test = true_probas_test,
    train_index = train_index,
    calib_index = calib_index,
    seed = seed,
    n_obs = n_obs,
    alpha = alpha,
    gamma = gamma,
    a = a
  )
}
