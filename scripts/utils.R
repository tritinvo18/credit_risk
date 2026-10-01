# ==============================================================================
# Utility Functions for Credit Risk Modelling (MXN600 Assignment 2)
# ==============================================================================

suppressPackageStartupMessages({
  library(pROC)
})

#' Clean Benchmark and Extended Datasets
#' 
#' Removes residual row indices, imputes missing employment lengths to "Unknown",
#' and casts categorical columns to factors.
#' 
#' @param df Data frame to clean
#' @return Cleaned data frame
clean_benchmark <- function(df) {
  if ("Unnamed..0" %in% names(df)) df$Unnamed..0 <- NULL
  if ("X" %in% names(df)) df$X <- NULL
  df$emp_length[is.na(df$emp_length) | df$emp_length == "n/a"] <- "Unknown"
  nominal_cols <- c("term", "home_ownership", "verification_status", "purpose", "emp_length")
  for (col in nominal_cols) {
    if (col %in% names(df)) df[[col]] <- as.factor(df[[col]])
  }
  return(df)
}

#' Align Factor Levels
#' 
#' Enforces factor levels in new_data to strictly match the reference_data levels,
#' throwing an informative error if unseen levels are detected.
#' 
#' @param new_data Target data frame
#' @param reference_data Baseline data frame providing authoritative factor levels
#' @param factor_columns Character vector of factor column names
#' @return Harmonised data frame
align_factor_levels <- function(new_data, reference_data, factor_columns) {
  for (column in factor_columns) {
    if (column %in% names(new_data) && column %in% names(reference_data)) {
      unseen <- setdiff(unique(as.character(new_data[[column]])), levels(reference_data[[column]]))
      if (length(unseen) > 0) {
        stop(sprintf("Unseen levels in %s: %s", column, paste(unseen, collapse = ", ")))
      }
      new_data[[column]] <- factor(new_data[[column]], levels = levels(reference_data[[column]]))
    }
  }
  new_data
}

#' Calculate Model Gini on a Dataset
#' 
#' @param model Fitted model object supporting predict(..., type = "response")
#' @param data Validation or test data frame
#' @param target_var Name of binary outcome column (default: "repay_fail")
#' @return Gini coefficient (2 * AUC - 1)
calc_gini <- function(model, data, target_var = "repay_fail") {
  p <- predict(model, newdata = data, type = "response")
  2 * as.numeric(pROC::auc(pROC::roc(data[[target_var]], p, quiet = TRUE))) - 1
}

#' Calculate Gini from Predicted Probabilities
#' 
#' @param preds Numeric vector of predicted probabilities
#' @param actual Binary ground truth vector
#' @return Gini coefficient (2 * AUC - 1)
calc_pred_gini <- function(preds, actual) {
  2 * as.numeric(pROC::auc(pROC::roc(actual, preds, quiet = TRUE))) - 1
}

#' Base R Rank-Based AUC Calculation
#' 
#' @param actual Binary outcome (0/1)
#' @param predicted Predicted probabilities or scores
#' @return Area under ROC curve
auc_base <- function(actual, predicted) {
  r <- rank(predicted)
  n_pos <- sum(actual == 1)
  n_neg <- sum(actual == 0)
  (sum(r[actual == 1]) - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
}
