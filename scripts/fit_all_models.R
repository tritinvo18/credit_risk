#!/usr/bin/env Rscript
# Model Fitting & Export Pipeline
# Generates: assets/glmm_models.rds, assets/dharma_results.rds, assets/soap_data.rds
# This script fits or extracts all GLM and GLMM models used in Credit_Risk.Rmd
# and exports them into consolidated assets archives.

# Ensure working directory is repo root if invoked from scripts/ directory
if (!file.exists("data/benchmark_training_loan_data.csv") && file.exists("../data/benchmark_training_loan_data.csv")) {
  setwd("..")
}

suppressPackageStartupMessages({
  library(dplyr)
  library(lubridate)
  library(caret)
  library(lme4)
  library(DHARMa)
  library(ROCR)
  library(pROC)
})

# Source modular utility functions
source("scripts/utils.R")

cat("=== Starting GLM/GLMM Model Pre-Fitting Pipeline ===\n")
dir.create("assets", showWarnings = FALSE)

# Helper function to preprocess and standardize the extended dataset
load_ext_data <- function() {
  ext <- read.csv("data/extendend_version_loan_data.csv")
  ext <- clean_benchmark(ext)
  ext$issue_d_date <- lubridate::dmy(paste0("01-", ext$issue_d))
  ext$issue_year <- (lubridate::year(ext$issue_d_date) + (lubridate::month(ext$issue_d_date) - 1) / 12) - 2007
  ext$issue_d <- as.factor(ext$issue_d_date)
  ext$addr_state <- as.factor(ext$addr_state)
  
  set.seed(42)
  ext_idx <- caret::createDataPartition(ext$repay_fail, p = 0.8, list = FALSE)
  ext_train <- ext[ext_idx, ]
  ext_test  <- ext[-ext_idx, ]
  
  nom_ext <- c("term", "home_ownership", "verification_status", "purpose", "emp_length", "addr_state")
  ext_test <- align_factor_levels(ext_test, ext_train, nom_ext)
  
  cont_vars <- c("loan_amnt", "int_rate", "annual_inc", "inq_last_6mths", "pub_rec", "revol_bal", "revol_util")
  for (variable in cont_vars) {
    center_val <- mean(ext_train[[variable]], na.rm = TRUE)
    scale_val  <- sd(ext_train[[variable]], na.rm = TRUE)
    ext_train[[variable]] <- as.numeric((ext_train[[variable]] - center_val) / scale_val)
    ext_test[[variable]]  <- as.numeric((ext_test[[variable]] - center_val) / scale_val)
  }
  
  list(ext = ext, ext_train = ext_train, ext_test = ext_test, cont_vars = cont_vars, nom_ext = nom_ext)
}

# Benchmark Data Preprocessing & GLM Fitting
cat("Ingesting and cleaning benchmark training data...\n")
train <- read.csv("data/benchmark_training_loan_data.csv")
train <- clean_benchmark(train)

formula_sel <- repay_fail ~ term + int_rate + emp_length + annual_inc + purpose + 
  inq_last_6mths + pub_rec + revol_bal + revol_util

cat("Fitting Section 2 GLM models (Logit, Probit, Cloglog, Nonlinear)...\n")
fit_logit   <- glm(formula_sel, data = train, family = binomial(link = "logit"))
fit_probit  <- glm(formula_sel, data = train, family = binomial(link = "probit"))
fit_cloglog <- glm(formula_sel, data = train, family = binomial(link = "cloglog"))

form_nonlin <- repay_fail ~ term + int_rate + emp_length + log1p(annual_inc) + purpose + 
  inq_last_6mths + pub_rec + log1p(revol_bal) + revol_util
fit_nonlin  <- glm(form_nonlin, data = train, family = binomial(link = "logit"))

# Check for existing assets/glmm_models.rds
args <- commandArgs(trailingOnly = TRUE)
force_refit <- "--refit" %in% args

model_file <- "assets/glmm_models.rds"

if (!force_refit && file.exists(model_file)) {
  cat("Loading verified GLMM fits from existing assets/glmm_models.rds...\n")
  existing_mods <- readRDS(model_file)
  m1_glm        <- existing_mods$m1_glm
  m2_trend_glm  <- existing_mods$m2_trend_glm
  m3_state_glmm <- existing_mods$m3_state_glmm
  m4_st_glmm    <- existing_mods$m4_st_glmm
  m5_slope_glmm <- existing_mods$m5_slope_glmm
  th_glm        <- existing_mods$th_glm
  th_trend      <- existing_mods$th_trend
  th_glmm       <- existing_mods$th_glmm
} else {
  cat("Preprocessing extended dataset and fitting GLMM hierarchy from scratch...\n")
  ext_data  <- load_ext_data()
  ext       <- ext_data$ext
  ext_train <- ext_data$ext_train
  ext_test  <- ext_data$ext_test
  cont_vars <- ext_data$cont_vars
  nom_ext   <- ext_data$nom_ext
  
  cat("Fitting M1 (Flat GLM)...\n")
  m1_glm <- glm(formula_sel, data = ext_train, family = binomial(link = "logit"))
  
  cat("Fitting M2 (Trend GLM)...\n")
  m2_form <- update(formula_sel, ~ . + issue_year)
  m2_trend_glm <- glm(m2_form, data = ext_train, family = binomial(link = "logit"))
  
  cat("Fitting M3 (State GLMM)...\n")
  m3_form <- update(m2_form, ~ . + (1 | addr_state))
  m3_state_glmm <- glmer(
    m3_form, data = ext_train, family = binomial(link = "logit"),
    control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 1e5))
  )
  
  cat("Fitting M4 (Spatial-Temporal GLMM)...\n")
  m4_form <- update(m2_form, ~ . + (1 | addr_state) + (1 | issue_d))
  m4_st_glmm <- glmer(
    m4_form, data = ext_train, family = binomial(link = "logit"),
    control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 1e5))
  )
  
  cat("Fitting M5 (Random Slope GLMM)...\n")
  m5_form <- update(m2_form, ~ . + (1 + int_rate | addr_state) + (1 | issue_d))
  m5_slope_glmm <- glmer(
    m5_form, data = ext_train, family = binomial(link = "logit"),
    control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 1e5))
  )
  
  cat("Fitting Temporal Holdout models...\n")
  cutoff_date <- as.Date("2011-01-01")
  th_train <- ext[ext$issue_d_date < cutoff_date, ]
  th_test  <- ext[ext$issue_d_date >= cutoff_date, ]
  
  for (variable in cont_vars) {
    center_val <- mean(th_train[[variable]], na.rm = TRUE)
    scale_val  <- sd(th_train[[variable]], na.rm = TRUE)
    th_train[[variable]] <- as.numeric((th_train[[variable]] - center_val) / scale_val)
    th_test[[variable]]  <- as.numeric((th_test[[variable]] - center_val) / scale_val)
  }
  th_test <- align_factor_levels(th_test, th_train, nom_ext)
  
  th_glm <- glm(formula_sel, data = th_train, family = binomial(link = "logit"))
  th_trend <- glm(m2_form, data = th_train, family = binomial(link = "logit"))
  th_glmm <- glmer(
    m4_form, data = th_train, family = binomial(link = "logit"),
    control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 1e5))
  )
}

# Combine and save to assets/glmm_models.rds
glmm_models <- list(
  # Primary benchmark & link GLMs
  fit_logit     = fit_logit,
  fit_probit    = fit_probit,
  fit_cloglog   = fit_cloglog,
  fit_nonlin    = fit_nonlin,
  
  # Extended spatial-temporal hierarchy (ext_train)
  m1_glm        = m1_glm,
  m2_trend_glm  = m2_trend_glm,
  m3_state_glmm = m3_state_glmm,
  m4_st_glmm    = m4_st_glmm,
  m5_slope_glmm = m5_slope_glmm,
  
  # Temporal holdout models (th_train)
  th_glm        = th_glm,
  th_trend      = th_trend,
  th_glmm       = th_glmm
)

if (force_refit || !file.exists(model_file)) {
  cat("Consolidating and saving all models to assets/glmm_models.rds...\n")
  saveRDS(glmm_models, model_file, compress = "xz")
  file_sz_mb <- file.size(model_file) / (1024 * 1024)
  cat(sprintf("=== Successfully saved assets/glmm_models.rds (%.2f MB) ===\n", file_sz_mb))
} else {
  cat("assets/glmm_models.rds already verified and up to date.\n")
}

# Generate & Save DHARMa Diagnostics
if (!file.exists("assets/dharma_results.rds") || force_refit) {
  cat("\nStep 6: Computing conditional DHARMa simulations on Model 4 (1,000 simulations)...\n")
  if (!exists("ext_train")) {
    ext_data  <- load_ext_data()
    ext_train <- ext_data$ext_train
    ext_test  <- ext_data$ext_test
  }
  set.seed(42)
  sim_res <- DHARMa::simulateResiduals(
    fittedModel = m4_st_glmm,
    n = 1000,
    seed = 42,
    refit = FALSE,
    simulateREs = "conditional"
  )
  
  unif_test     <- DHARMa::testUniformity(sim_res, plot = FALSE)
  disp_test     <- DHARMa::testDispersion(sim_res, plot = FALSE)
  outlier_test  <- DHARMa::testOutliers(sim_res, plot = FALSE, type = "binomial")
  quantile_test <- DHARMa::testQuantiles(sim_res, predictor = sim_res$fittedPredictedResponse, plot = FALSE)
  
  n_obs <- length(sim_res$scaledResiduals)
  n_sim <- ncol(sim_res$simulatedResponse)
  observed_outliers <- as.integer(unname(outlier_test$statistic))
  outlier_flag <- DHARMa::outliers(sim_res, lowerQuantile = 1 / (n_sim + 1), upperQuantile = 1 - 1 / (n_sim + 1), return = "logical")
  
  risk_decile       <- dplyr::ntile(sim_res$fittedPredictedResponse, 10)
  decile_residuals  <- DHARMa::recalculateResiduals(sim_res, group = risk_decile)
  decile_uniformity <- DHARMa::testUniformity(decile_residuals, plot = FALSE)
  decile_dispersion <- DHARMa::testDispersion(decile_residuals, plot = FALSE)
  
  state_residuals   <- DHARMa::recalculateResiduals(sim_res, group = ext_train$addr_state)
  state_uniformity  <- DHARMa::testUniformity(state_residuals, plot = FALSE)
  state_dispersion  <- DHARMa::testDispersion(state_residuals, plot = FALSE)
  
  month_levels      <- sort(unique(ext_train$issue_d_date))
  month_group       <- factor(ext_train$issue_d_date, levels = month_levels, ordered = TRUE)
  month_residuals   <- DHARMa::recalculateResiduals(sim_res, group = month_group)
  month_uniformity  <- DHARMa::testUniformity(month_residuals, plot = FALSE)
  month_dispersion  <- DHARMa::testDispersion(month_residuals, plot = FALSE)
  
  temporal_test     <- DHARMa::testTemporalAutocorrelation(simulationOutput = month_residuals, time = seq_along(month_levels), plot = FALSE)
  
  dharma_results <- list(
    sim_res           = sim_res,
    unif_test         = unif_test,
    disp_test         = disp_test,
    outlier_test      = outlier_test,
    quantile_test     = quantile_test,
    outlier_flag      = outlier_flag,
    observed_outliers = observed_outliers,
    p_q25             = quantile_test$pvals[1],
    p_q50             = quantile_test$pvals[2],
    p_q75             = quantile_test$pvals[3],
    decile_residuals  = decile_residuals,
    decile_uniformity = decile_uniformity,
    decile_dispersion = decile_dispersion,
    state_residuals   = state_residuals,
    state_uniformity  = state_uniformity,
    state_dispersion  = state_dispersion,
    month_residuals   = month_residuals,
    month_uniformity  = month_uniformity,
    month_dispersion  = month_dispersion,
    temporal_test     = temporal_test
  )
  
  saveRDS(dharma_results, "assets/dharma_results.rds", compress = "xz")
  cat(sprintf("Successfully saved assets/dharma_results.rds (%.2f MB)\n", file.size("assets/dharma_results.rds") / (1024 * 1024)))
} else {
  cat(sprintf("assets/dharma_results.rds already exists (%.2f MB). Use --refit to recompute.\n", file.size("assets/dharma_results.rds") / (1024 * 1024)))
}

# Generate & Save SOAP Data
if (!file.exists("assets/soap_data.rds") || force_refit) {
  cat("\nConsolidating and exporting assets/soap_data.rds for executive summary...\n")
  nominal_cols <- c("term", "home_ownership", "verification_status", "purpose", "emp_length")
  test <- read.csv("data/benchmark_testing_loan_data.csv")
  test <- clean_benchmark(test)
  test <- align_factor_levels(test, train, nominal_cols)
  
  p_test <- predict(fit_logit, newdata = test, type = "response")
  roc_test <- pROC::roc(test$repay_fail, p_test, quiet = TRUE)
  test_gini <- 2 * as.numeric(pROC::auc(roc_test)) - 1
  
  threshold <- 0.15
  pred_class <- ifelse(p_test > threshold, 1, 0)
  n_total <- nrow(test)
  n_approved <- sum(pred_class == 0)
  approval_rate <- n_approved / n_total
  default_rate_of_approved <- sum(pred_class == 0 & test$repay_fail == 1) / n_approved
  
  pred_rocr <- ROCR::prediction(p_test, test$repay_fail)
  perf_ks <- ROCR::performance(pred_rocr, "tpr", "fpr")
  ks_stat <- max(attr(perf_ks, "y.values")[[1]] - attr(perf_ks, "x.values")[[1]])
  
  if (!exists("ext_test")) {
    ext_data  <- load_ext_data()
    ext_train <- ext_data$ext_train
    ext_test  <- ext_data$ext_test
  }
  
  p_m1 <- predict(m1_glm, newdata = ext_test, type = "response")
  p_m4 <- predict(m4_st_glmm, newdata = ext_test, type = "response", allow.new.levels = TRUE)
  gini_ext_glm  <- 2 * as.numeric(pROC::auc(pROC::roc(ext_test$repay_fail, p_m1, quiet = TRUE))) - 1
  gini_ext_glmm <- 2 * as.numeric(pROC::auc(pROC::roc(ext_test$repay_fail, p_m4, quiet = TRUE))) - 1
  
  vc <- as.data.frame(lme4::VarCorr(m4_st_glmm))
  var_state <- vc$vcov[vc$grp == "addr_state"]
  var_time  <- vc$vcov[vc$grp == "issue_d"]
  var_latent <- pi^2 / 3
  var_total <- var_state + var_time + var_latent
  icc_state <- var_state / var_total
  
  time_re_df <- lme4::ranef(m4_st_glmm)$issue_d
  time_re_df$Date <- lubridate::ymd(rownames(time_re_df))
  time_re_df <- time_re_df[order(time_re_df$Date), ]
  
  re_state_obj <- lme4::ranef(m4_st_glmm, condVar = TRUE)$addr_state
  pv_state <- attr(re_state_obj, "postVar")
  se_state <- sqrt(pv_state[1, 1, ])
  state_df <- data.frame(
    State = rownames(re_state_obj),
    Estimate = re_state_obj$`(Intercept)`,
    SE = se_state,
    Lower = re_state_obj$`(Intercept)` - 1.96 * se_state,
    Upper = re_state_obj$`(Intercept)` + 1.96 * se_state
  ) %>%
    dplyr::mutate(Significant = ifelse(Lower > 0 | Upper < 0, "Excludes Zero", "Overlaps Zero"))
  
  state_ranef <- data.frame(
    State = state_df$State,
    Estimate = state_df$Estimate,
    SE = state_df$SE,
    Lower = state_df$Lower,
    Upper = state_df$Upper,
    Significant = state_df$Significant,
    `(Intercept)` = state_df$Estimate,
    check.names = FALSE
  )
  
  export_data <- list(
    fit_glm                  = fit_logit,
    icc_state                = icc_state,
    approval_rate            = approval_rate,
    default_rate_of_approved = default_rate_of_approved,
    test_gini                = test_gini,
    ks_stat                  = ks_stat,
    gini_ext_glm             = gini_ext_glm,
    gini_ext_glmm            = gini_ext_glmm,
    time_ranef               = time_re_df,
    state_ranef              = state_ranef
  )
  
  saveRDS(export_data, "assets/soap_data.rds")
  cat(sprintf("Successfully saved assets/soap_data.rds (%.2f MB)\n", file.size("assets/soap_data.rds") / (1024 * 1024)))
} else {
  cat(sprintf("assets/soap_data.rds already exists (%.2f MB). Use --refit to recompute.\n", file.size("assets/soap_data.rds") / (1024 * 1024)))
}

cat("Pre-Fitting Pipeline Complete!\n")
