# Independent oracle for vetted's backtests. Every value in
# tests/data/backtest_oracle.json comes from an established R package, not
# from vetted:
#
#   rugarch 1.5 VaRTest              Kupiec and Christoffersen LR statistics
#   GAS 0.3.4 BacktestVaR, FZLoss    Engle-Manganelli DQ test, FZ0 loss
#   esback cc_backtest               Nolde-Ziegel conditional calibration
#   forecast dm.test                 Diebold-Mariano with the HLN correction
#   pbo                              probability of backtest overfitting
#   PerformanceAnalytics             ProbSharpeRatio, MinTrackRecord
#
# Usage (from the repository root):
#   Rscript oracle/gen_backtest_oracle.R tests/data/backtest_fixtures.json tests/data/backtest_oracle.json

suppressMessages({
  library(jsonlite)
  library(rugarch)
  library(GAS)
  library(esback)
  library(forecast)
  library(pbo)
  library(PerformanceAnalytics)
})

args <- commandArgs(trailingOnly = TRUE)
fx <- fromJSON(args[1])
out <- list()

safe <- function(expr) tryCatch(expr, error = function(e) NULL)

var_block <- function(pnl, var_pos, alpha) {
  q <- -var_pos  # packages take VaR as a return quantile
  rg <- safe(VaRTest(alpha = alpha, actual = pnl, VaR = q))
  gas <- safe(BacktestVaR(pnl, q, alpha, Lags = 4))
  list(
    exceptions = sum(pnl < q),
    rugarch_uc_stat = if (is.null(rg)) NULL else rg$uc.LRstat,
    rugarch_uc_p = if (is.null(rg)) NULL else rg$uc.LRp,
    rugarch_cc_stat = if (is.null(rg)) NULL else rg$cc.LRstat,
    rugarch_cc_p = if (is.null(rg)) NULL else rg$cc.LRp,
    gas_dq_stat = if (is.null(gas)) NULL else unname(gas$DQ$stat),
    gas_dq_p = if (is.null(gas)) NULL else unname(gas$DQ$pvalue),
    gas_quantile_loss = if (is.null(gas)) NULL else unname(gas$Loss$Loss)
  )
}

for (name in c("correct_250", "underestimated_250", "static_250",
               "correct_1000", "underestimated_1000", "static_1000")) {
  f <- fx[[name]]
  cc <- safe(cc_backtest(r = f$pnl, q = -f$var975, e = -f$es975, s = f$scale, alpha = 0.025))
  out[[name]] <- list(
    var99 = var_block(f$pnl, f$var99, 0.01),
    var975 = var_block(f$pnl, f$var975, 0.025),
    fz0_loss_mean = mean(FZLoss(f$pnl, -f$var975, -f$es975, 0.025)),
    fz0_loss_first5 = head(FZLoss(f$pnl, -f$var975, -f$es975, 0.025), 5),
    esback_cc = cc,
    esback_er = safe(er_backtest(r = f$pnl, q = -f$var975, e = -f$es975, s = f$scale, B = 20000))
  )
}

# Diebold-Mariano on quantile losses of two VaR models
cmp <- fx[["compare_1000"]]
qloss <- function(y, v, a) (a - (y < -v)) * (y + v)
la <- qloss(cmp$pnl, cmp$var_garch, 0.01)
lb <- qloss(cmp$pnl, cmp$var_static, 0.01)
dm <- list()
for (h in c(1, 5)) {
  for (ve in c("acf", "bartlett")) {
    for (alt in c("two.sided", "less")) {
      t <- dm.test(la, lb, alternative = alt, h = h, power = 1, varestimator = ve)
      dm[[paste(h, ve, alt, sep = "_")]] <- list(statistic = unname(t$statistic), p = t$p.value)
    }
  }
}
out[["dm"]] <- dm

# PBO with 8 blocks (70 splits), Sharpe ratio as the metric
M <- as.matrix(fx[["trials_1000x12"]])
sharpe <- function(x) apply(x, 2, function(c) mean(c) / sd(c))
res <- pbo(as.data.frame(M), s = 8, f = sharpe, threshold = 0)
out[["pbo"]] <- list(
  phi = res$phi,
  os_rank = as.numeric(res$results[, "os_rank"]),
  n_star = as.numeric(res$results[, "n*"])
)

# Probabilistic Sharpe ratio and minimum track record on one column
r <- M[, 4]
n <- length(r)
sr <- mean(r) / sd(r)
m <- mean(r)
sk <- mean((r - m)^3) / mean((r - m)^2)^1.5
kr <- mean((r - m)^4) / mean((r - m)^2)^2
out[["psr"]] <- list(
  sr = sr, skew = sk, kurt = kr, n = n,
  psr_0 = as.numeric(ProbSharpeRatio(refSR = 0, n = n, sr = sr, sk = sk, kr = kr, ignore_kurtosis = FALSE)$sr_prob),
  psr_002 = as.numeric(ProbSharpeRatio(refSR = 0.02, n = n, sr = sr, sk = sk, kr = kr, ignore_kurtosis = FALSE)$sr_prob),
  mintrl_0 = as.numeric(MinTrackRecord(refSR = 0, n = n, sr = sr, sk = sk, kr = kr, ignore_kurtosis = FALSE)$min_TRL),
  mintrl_002_99 = as.numeric(MinTrackRecord(refSR = 0.02, p = 0.99, n = n, sr = sr, sk = sk, kr = kr, ignore_kurtosis = FALSE)$min_TRL)
)

out[["meta"]] <- list(
  generator = "oracle/gen_backtest_oracle.R",
  R_version = R.version.string,
  packages = setNames(lapply(c("rugarch", "GAS", "esback", "forecast", "pbo", "PerformanceAnalytics"),
                             function(p) as.character(packageVersion(p))),
                      c("rugarch", "GAS", "esback", "forecast", "pbo", "PerformanceAnalytics"))
)

writeLines(toJSON(out, auto_unbox = TRUE, digits = NA, null = "null"), args[2])
cat("wrote", args[2], "\n")
