#' Theme for ggplot2
#'
#' @param ... arguments passed to the theme function
#' @export
#' @importFrom ggplot2 element_rect element_text element_blank element_line unit
#'   rel
theme_paper <- function (...) {
  ggthemes::theme_base() +
    theme(
      legend.background = element_rect(
        fill = "transparent", linetype="solid", colour ="black"),
      legend.position = "bottom",
      legend.direction = "horizontal",
      legend.box = "horizontal",
      legend.key = element_blank()
    )
}
