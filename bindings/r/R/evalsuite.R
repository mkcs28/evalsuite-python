#' Compute an EvalSuite metric
#'
#' @param metric Metric name, e.g. "f1", "roc_auc", "rmse", "bleu" (see es_metrics()).
#' @param ... Arguments: y_true, y_pred, y_prob, average, sample_weight; references and predictions for text
#'   metrics; reference and current for drift ("psi", "ks").
#' @return The value (numeric scalar or vector), or a list for "ks".
#' @examples
#' es_metric("f1", y_true = c(0, 1, 1, 0), y_pred = c(0, 1, 0, 0))
es_metric <- function(metric, ...) {
  args <- list(...)
  json <- jsonlite::toJSON(args, auto_unbox = TRUE, digits = NA)
  out <- jsonlite::fromJSON(.Call(C_es_call, as.character(metric), as.character(json)))
  if (!is.null(out$values)) out$values else out$value
}

#' Names of the metrics available through the engine
es_metrics <- function() {
  c("accuracy", "balanced_accuracy", "precision", "recall", "f1", "specificity", "mcc", "roc_auc",
    "average_precision", "log_loss", "brier_score", "mae", "mse", "rmse", "r2", "mape",
    "median_absolute_error", "max_error", "bleu", "chrf", "ter", "rouge_1", "rouge_2", "rouge_l", "psi", "ks")
}

#' Engine version
es_version <- function() .Call(C_es_version)
