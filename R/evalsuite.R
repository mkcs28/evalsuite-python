#' Compute an EvalSuite metric
#'
#' @param metric Metric name, e.g. "f1", "roc_auc", "rmse", "bleu", "ndcg", "dice" (see es_metrics()).
#' @param ... Arguments: y_true, y_pred, y_prob, average, sample_weight, pos_label, threshold; references and
#'   predictions for text metrics; relevant, retrieved and k for retrieval; shape and num_classes for
#'   segmentation masks; reference and current for drift ("psi", "ks").
#' @param ci NULL (default), "analytic" (Wilson / DeLong where available, else bootstrap) or "bootstrap".
#' @param level,n_resamples,seed Interval settings.
#' @return The value (numeric scalar or vector). With `ci`, a list with value, ci_low, ci_high, ci_level and
#'   ci_method. A list for "ks" and "hosmer_lemeshow".
#' @examples
#' es_metric("f1", y_true = c(0, 1, 1, 0), y_pred = c(0, 1, 0, 0))
#' es_metric("roc_auc", y_true = c(0, 0, 1, 1), y_prob = c(0.1, 0.4, 0.35, 0.8), ci = "analytic")
es_metric <- function(metric, ..., ci = NULL, level = 0.95, n_resamples = 1000, seed = 0) {
  args <- list(...)
  if (!is.null(ci)) {
    args$ci <- ci
    args$level <- level
    args$n_resamples <- n_resamples
    args$seed <- seed
  }
  json <- jsonlite::toJSON(args, auto_unbox = TRUE, digits = NA)
  out <- jsonlite::fromJSON(.Call(C_es_call, as.character(metric), as.character(json)))
  value <- .es_num(if (!is.null(out$values)) out$values else out$value)
  if (!is.null(out$ci_low)) {
    return(list(value = value, ci_low = .es_num(out$ci_low), ci_high = .es_num(out$ci_high),
                ci_level = out$ci_level, ci_method = out$ci_method))
  }
  value
}

# JSON has no NaN / Inf: the engine sends "NaN", "Infinity" and "-Infinity" as strings.
.es_num <- function(x) {
  if (!is.character(x)) return(x)
  y <- suppressWarnings(as.numeric(sub("Infinity", "Inf", x, fixed = TRUE)))
  if (anyNA(y[x != "NaN"])) x else y
}

#' Names of the metrics available through the engine
es_metrics <- function() jsonlite::fromJSON(.Call(C_es_metrics))

#' Engine version
es_version <- function() .Call(C_es_version)
