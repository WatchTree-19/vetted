# Independent oracle: every expected value in src/vetted/data/oracle.json comes from
# R PerformanceAnalytics, not from vetted's own Python code.
#
#   Rscript oracle/gen_metrics_oracle.R src/vetted/data/fixtures.json src/vetted/data/oracle.json
#
# Each entry records the exact R call, so any value can be re-derived by a
# reviewer who does not trust this project.

suppressMessages({
  library(jsonlite)
  library(PerformanceAnalytics)
  library(xts)
})

args <- commandArgs(trailingOnly = TRUE)
fixtures <- fromJSON(args[1], simplifyVector = TRUE)

safe <- function(expr) {
  v <- tryCatch(suppressWarnings(as.numeric(expr)), error = function(e) NA_real_)
  if (length(v) != 1 || !is.finite(v)) NA_real_ else v
}

out <- list()
for (name in names(fixtures)) {
  fx <- fixtures[[name]]
  r <- as.numeric(fx$returns)
  scale <- as.integer(fx$periods_per_year)
  # A regular date index so PerformanceAnalytics accepts the series.
  by <- if (scale == 12) "month" else "day"
  R <- xts(r, order.by = seq(as.Date("2000-01-31"), by = by, length.out = length(r)))

  v <- list(
    # call string -> value; the Python side maps (metric, convention) to these
    "mean"                                     = safe(mean(r)),
    "StdDev(R)"                                = safe(StdDev(R)),
    "StdDev.annualized(R, scale)"              = safe(StdDev.annualized(R, scale = scale)),
    "Return.annualized(R, scale, geometric=TRUE)"  = safe(Return.annualized(R, scale = scale, geometric = TRUE)),
    "Return.annualized(R, scale, geometric=FALSE)" = safe(Return.annualized(R, scale = scale, geometric = FALSE)),
    "SharpeRatio.annualized(R, 0, scale, geometric=TRUE)"  = safe(SharpeRatio.annualized(R, Rf = 0, scale = scale, geometric = TRUE)),
    "SharpeRatio.annualized(R, 0, scale, geometric=FALSE)" = safe(SharpeRatio.annualized(R, Rf = 0, scale = scale, geometric = FALSE)),
    "DownsideDeviation(R, MAR=0, method='full')"   = safe(DownsideDeviation(R, MAR = 0, method = "full")),
    "DownsideDeviation(R, MAR=0, method='subset')" = safe(DownsideDeviation(R, MAR = 0, method = "subset")),
    "SortinoRatio(R, MAR=0)"                   = safe(SortinoRatio(R, MAR = 0)),
    "maxDrawdown(R, geometric=TRUE)"           = safe(maxDrawdown(R, geometric = TRUE)),
    "maxDrawdown(R, geometric=FALSE)"          = safe(maxDrawdown(R, geometric = FALSE)),
    "CalmarRatio(R, scale)"                    = safe(CalmarRatio(R, scale = scale)),
    "VaR(R, p=0.95, method='historical')"      = safe(VaR(R, p = 0.95, method = "historical")),
    "VaR(R, p=0.95, method='gaussian')"        = safe(VaR(R, p = 0.95, method = "gaussian")),
    "VaR(R, p=0.95, method='modified')"        = safe(VaR(R, p = 0.95, method = "modified")),
    "ES(R, p=0.95, method='historical')"       = safe(ES(R, p = 0.95, method = "historical")),
    "ES(R, p=0.95, method='gaussian')"         = safe(ES(R, p = 0.95, method = "gaussian")),
    "Omega(R, L=0, method='simple')"           = safe(Omega(R, L = 0, method = "simple")),
    "UlcerIndex(R)"                            = safe(UlcerIndex(R)),
    "PainIndex(R)"                             = safe(PainIndex(R)),
    "skewness(R, method='moment')"             = safe(skewness(R, method = "moment")),
    "kurtosis(R, method='excess')"             = safe(kurtosis(R, method = "excess")),
    "kurtosis(R, method='sample_excess')"      = safe(kurtosis(R, method = "sample_excess")),
    "skewness(R, method='sample')"             = safe(skewness(R, method = "sample")),
    "skewness(R, method='fisher')"             = safe(skewness(R, method = "fisher"))
  )
  out[[name]] <- v
}

meta <- list(
  generator = "oracle/gen_metrics_oracle.R",
  R_version = R.version.string,
  PerformanceAnalytics = as.character(packageVersion("PerformanceAnalytics"))
)
write_json(list(meta = meta, values = out), args[2], auto_unbox = TRUE, digits = NA, pretty = TRUE, na = "null")
