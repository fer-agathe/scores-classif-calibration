## 1. Quantile-based----

#' Confidence interval for binomial data, using quantile-defined bins
#'
#' @param obs vector of observed events
#' @param scores vector of predicted probabilities
#' @param k number of bins to create (quantiles, default to `10`)
#' @param threshold classification threshold (default to `.5`)
#' @param add_conf_int (logical) if TRUE, confidence intervals are computed
#' @param prob confidence interval level
#' @param method Which method to use to construct the interval. Any combination
#'   of c("exact", "ac", "asymptotic", "wilson", "prop.test", "bayes", "logit",
#'   "cloglog", "probit") is allowed. Default is "all".
#' @return A tibble with the following columns, where each row corresponds to a
#'   bin:
#' \itemize{
#'   \item \code{mean}: Estimation of \eqn{E(D | s(x) = p)} where \eqn{p} is the average score in bin \eqn{b}.
#'   \item \code{lower}: Lower bound of the confidence interval.
#'   \item \code{upper}: Upper bound of the confidence interval.
#'   \item \code{prediction}: Average of \code{s(x)} in bin \eqn{b}.
#'   \item \code{score_class}: Decile level of bin \eqn{b}.
#'   \item \code{nb}: Number of observations in bin \eqn{b}.
#' }
#' @importFrom binom binom.confint
#' @importFrom tibble tibble
#' @importFrom dplyr mutate
#' @export
calib_curve_quant <- function(obs,
                              scores,
                              k = 10,
                              threshold = .5,
                              add_conf_int = FALSE,
                              prob = .95,
                              method = "probit") {

  summary_bins_calib <- get_summary_bins(obs = obs, scores = scores, k = k)

  if (add_conf_int == TRUE) {
    new_k <- nrow(summary_bins_calib)
    prob_ic <- tibble(
      mean_obs = rep(NA, new_k),
      lower_obs = rep(NA, new_k),
      upper_obs = rep(NA, new_k)
    )
    for (i in 1:new_k) {
      prob_ic[i, 1:3] <- binom.confint(
        x = summary_bins_calib$sum_obs[i],
        n = summary_bins_calib$nb[i],
        conf.level = prob,
        methods = method
      )[, c("mean", "lower", "upper")]
    }

    summary_bins_calib <-
      summary_bins_calib |>
      mutate(
        lower_obs = prob_ic$lower_obs,
        upper_obs = prob_ic$upper_obs
      )

  }

  summary_bins_calib
}

#' Plot calibration curve obtained with quantile-defined bins
#'
#' @param calib_curve_quant tibble with calibration curve obtained with
#'    quantile-defined bins
#' @param title title of the plot
#' @param colour colour for the calibration curve
#' @param add_ci if `TRUE`, add error bars to the points (lower and upper
#'   bounds must then be found inside `calib_curve_locfit`)
#' @param add if `TRUE`, creates a new plot, else, add to an existing one
#'
#' @importFrom graphics plot lines segments arrows
#' @importFrom latex2exp TeX
#' @export
plot_calib_curve_quant <- function(calib_curve_quant,
                                   title,
                                   colour = "black",
                                   add_ci = FALSE,
                                   add = FALSE) {
  if (add == FALSE) {
    plot(
      0:1, 0:1,
      type = "l", col = NULL,
      xlim = 0:1, ylim = 0:1,
      xlab = latex2exp::TeX("Predicted score $\\hat{s}(x)$"),
      ylab = latex2exp::TeX("$E(D | \\hat{s}(x) = p)$"),
      main = title
    )
  }

  lines(
    calib_curve_quant$mean_score, calib_curve_quant$mean_obs,
    lwd = 2, col = colour, t = "b",
  )
  if (add_ci == TRUE) {
    arrows(
      x0 = calib_curve_quant$mean_score,
      y0 = calib_curve_quant$lower_obs,
      x1 = calib_curve_quant$mean_score,
      y1 = calib_curve_quant$upper_obs,
      angle = 90,length = .05, code = 3,
      col = colour
    )
  }
  segments(0, 0, 1, 1, col = "black", lty = 2)
}



#' Calibration curve obtained with local regression
#'
#' @param obs vector of observed events
#' @param scores vector of predicted probabilities
#' @param nn fraction of nearest neighbors
#' @param deg degree for the local regression
#' @param linspace vector of values at which to compute averages
#'
#' @returns a tibble with equally-spaced values on [0,1] at which the expected
#'   value for the event is estimated with local regression. The tibble contains
#'   two columns.
#' \itemize{
#'   \item \code{xlim}: value \eqn{p} at which the estimation is made
#'   \item \code{score_c}: Estimation of \eqn{E(D | s(x) = p)} where \eqn{p} is the average score estimated at \eqn{p}.
#' }
#'
#' @importFrom locfit locfit lp
#' @importFrom tibble tibble
#'
#' @export
calib_curve_locfit <- function(obs,
                               scores,
                               nn = 0.15,
                               deg = 0,
                               linspace = NULL) {
  if (is.null(linspace)) linspace <- seq(0, 1, length.out = 101)

  # Add a little noise (otherwise, R may crash...)
  scores <- scores + rnorm(length(scores), 0, .001)

  locfit_0 <- locfit(
    formula = d ~ lp(score, nn = nn, deg = deg),
    kern = "rect", maxk = 200,
    data = tibble(d = obs, score = scores)
  )

  # Predictions on [0,1]
  linspace_raw <- seq(0, 1, length.out = 100)

  # Restricting this space to the range of observed scores
  keep_linspace_calib <- which(
    linspace_raw >= min(scores) & linspace_raw <= max(scores)
  )
  linspace_restricted <- linspace_raw[keep_linspace_calib]

  score_locfit_0 <- predict(locfit_0, newdata = linspace_restricted)
  score_locfit_0[score_locfit_0 < 0] <- 0
  score_locfit_0[score_locfit_0 > 1] <- 1

  tibble(
    xlim = linspace_restricted,
    score_c = score_locfit_0
  )
}

#' Plot calibration curve obtained with local regression
#'
#' @param calib_curve_locfit tibble with calibration curve obtained with local
#'    regression
#' @param title title of the plot
#' @param colour colour for the calibration curve
#' @param add if `TRUE`, creates a new plot, else, add to an existing one
#'
#' @importFrom graphics plot lines segments arrows
#' @importFrom latex2exp TeX
#'
#' @export
plot_calib_curve_locfit <- function(calib_curve_locfit,
                                    title,
                                    colour,
                                    add = FALSE) {
  if (add == FALSE) {
    plot(
      0:1, 0:1,
      type = "l", col = NULL,
      xlim = 0:1, ylim = 0:1,
      xlab = latex2exp::TeX("Predicted score $\\hat{s}(x)$"),
      ylab = latex2exp::TeX("$E(D | \\hat{s}(x) = p)$"),
      main = title
    )
  }
  lines(
    calib_curve_locfit$xlim, calib_curve_locfit$score_c,
    lwd = 2, col = colour, type = "l",
  )
  segments(0, 0, 1, 1, col = "black", lty = 2)
}
