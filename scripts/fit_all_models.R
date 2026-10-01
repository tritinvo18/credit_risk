#!/usr/bin/env Rscript
# ==============================================================================
# Model Fitting & Export Pipeline for MXN600 Assignment 2
# Generates: assets/glmm_models.rds
#
# This script fits or extracts all GLM and GLMM models used in Credit_Risk.Rmd
# and exports them into a single consolidated assets/glmm_models.rds file.
# ==============================================================================

suppressPackageStartupMessages({
  library(dplyr)
  library(lubridate)
  library(caret)
  library(lme4)
  library(DHARMa)
})

# Source modular utility functions
source("scripts/utils.R")

cat("=== Starting GLM/GLMM Model Pre-Fitting Pipeline ===\n")
dir.create("assets", showWarnings = FALSE)

# 1. Benchmark Data Preprocessing & GLM Fitting
cat("Step 1: Ingesting and cleaning benchmark training data...\n")
train <- read.csv("data/benchmark_training_loan_data.csv")
train <- clean_benchmark(train)

formula_sel <- repay_fail ~ term + int_rate + emp_length + annual_inc + purpose + 
  inq_last_6mths + pub_rec + revol_bal + revol_util

cat("Step 2: Fitting Section 2 GLM models (Logit, Probit, Cloglog, Nonlinear)...\n")
fit_logit   <- glm(formula_sel, data = train, family = binomial(link = "logit"))
fit_probit  <- glm(formula_sel, data = train, family = binomial(link = "probit"))
fit_cloglog <- glm(formula_sel, data = train, family = binomial(link = "cloglog"))

form_nonlin <- repay_fail ~ term + int_rate + emp_length + log1p(annual_inc) + purpose + 
  inq_last_6mths + pub_rec + log1p(revol_bal) + revol_util
fit_nonlin  <- glm(form_nonlin, data = train, family = binomial(link = "logit"))

# 2. Check for existing assets/glmm_models.rds
args <- commandArgs(trailingOnly = TRUE)
force_refit <- "--refit" %in% args

model_file <- "assets/glmm_models.rds"

if (!force_refit && file.exists(model_file)) {
  cat("Step 3: Loading verified GLMM fits from existing assets/glmm_models.rds...\n")
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
  cat("Step 3: Preprocessing extended dataset and fitting GLMM hierarchy from scratch...\n")
  ext <- read.csv("data/extendend_version_loan_data.csv")
  if ("Unnamed..0" %in% names(ext)) ext$Unnamed..0 <- NULL
  if ("X" %in% names(ext)) ext$X <- NULL
  ext$emp_length[is.na(ext$emp_length) | ext$emp_length == "n/a"] <- "Unknown"
  for (col in c("term", "home_ownership", "verification_status", "purpose", "emp_length", "addr_state")) {
    ext[[col]] <- as.factor(ext[[col]])
  }
  ext$issue_d_date <- dmy(paste0("01-", ext$issue_d))
  ext$issue_year <- (year(ext$issue_d_date) + (month(ext$issue_d_date) - 1) / 12) - 2007
  ext$issue_d <- as.factor(ext$issue_d_date)
  
  set.seed(42)
  ext_idx <- createDataPartition(ext$repay_fail, p = 0.8, list = FALSE)
  ext_train <- ext[ext_idx, ]
  ext_test  <- ext[-ext_idx, ]
  
  cont_vars <- c("loan_amnt", "int_rate", "annual_inc", "inq_last_6mths", "pub_rec", "revol_bal", "revol_util")
  for (variable in cont_vars) {
    center_val <- mean(ext_train[[variable]], na.rm = TRUE)
    scale_val  <- sd(ext_train[[variable]], na.rm = TRUE)
    ext_train[[variable]] <- as.numeric((ext_train[[variable]] - center_val) / scale_val)
    ext_test[[variable]]  <- as.numeric((ext_test[[variable]] - center_val) / scale_val)
  }
  
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
  
  cat("Step 4: Fitting Temporal Holdout models...\n")
  cutoff_date <- as.Date("2011-01-01")
  th_train <- ext[ext$issue_d_date < cutoff_date, ]
  th_test  <- ext[ext$issue_d_date >= cutoff_date, ]
  
  for (variable in cont_vars) {
    center_val <- mean(th_train[[variable]], na.rm = TRUE)
    scale_val  <- sd(th_train[[variable]], na.rm = TRUE)
    th_train[[variable]] <- as.numeric((th_train[[variable]] - center_val) / scale_val)
    th_test[[variable]]  <- as.numeric((th_test[[variable]] - center_val) / scale_val)
  }
  
  th_glm <- glm(formula_sel, data = th_train, family = binomial(link = "logit"))
  th_trend <- glm(m2_form, data = th_train, family = binomial(link = "logit"))
  th_glmm <- glmer(
    m4_form, data = th_train, family = binomial(link = "logit"),
    control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 1e5))
  )
}

# 3. Assemble and Save to assets/glmm_models.rds
cat("Step 5: Consolidating and saving all models to assets/glmm_models.rds...\n")
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

saveRDS(glmm_models, "assets/glmm_models.rds", compress = "xz")

file_sz_mb <- file.size("assets/glmm_models.rds") / (1024 * 1024)
cat(sprintf("=== Successfully saved assets/glmm_models.rds (%.2f MB) ===\n", file_sz_mb))
cat("Models included:\n")
for (nm in names(glmm_models)) {
  cat(sprintf("  - %-15s [%s]\n", nm, class(glmm_models[[nm]])[1]))
}

# 4. Generate & Save DHARMa Diagnostics
if (!file.exists("assets/dharma_results.rds") || force_refit) {
  cat("\nStep 6: Computing conditional DHARMa simulations on Model 4 (1,000 simulations)...\n")
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
  cat(sprintf("=== Successfully saved assets/dharma_results.rds (%.2f MB) ===\n", file.size("assets/dharma_results.rds") / (1024 * 1024)))
} else {
  cat(sprintf("assets/dharma_results.rds already exists (%.2f MB). Use --refit to recompute.\n", file.size("assets/dharma_results.rds") / (1024 * 1024)))
}
cat("=== Pre-Fitting Pipeline Complete! ===\n")
