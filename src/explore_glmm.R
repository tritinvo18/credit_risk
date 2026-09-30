library(lme4)
ext <- read.csv('data/extendend_version_loan_data.csv')

if("Unnamed..0" %in% names(ext)) { ext$Unnamed..0 <- NULL }
if("X" %in% names(ext)) { ext$X <- NULL }

cat_vars <- c("term", "home_ownership", "verification_status", "purpose", "addr_state", "issue_d")
for(v in cat_vars) {
  ext[[v]] <- as.factor(ext[[v]])
}

# Scale continuous variables to help optimizer
cont_vars <- c("loan_amnt", "int_rate", "annual_inc", "inq_last_6mths", "pub_rec", "revol_bal", "revol_util")
for(v in cont_vars) {
  ext[[v]] <- scale(ext[[v]])
}

# Fit GLMM
start_time <- Sys.time()
fit_glmm <- glmer(repay_fail ~ loan_amnt + term + int_rate + annual_inc + 
                  inq_last_6mths + pub_rec + revol_bal + revol_util + 
                  (1 | addr_state), 
                  data=ext, family=binomial(link="logit"), 
                  control=glmerControl(optimizer="bobyqa", optCtrl=list(maxfun=1e5)))
end_time <- Sys.time()
print(end_time - start_time)

print(summary(fit_glmm))

# Gini Score on Training data for GLMM
pred_prob <- predict(fit_glmm, type="response")
r <- rank(pred_prob)
n_pos <- sum(ext$repay_fail == 1)
n_neg <- sum(ext$repay_fail == 0)
auc_val <- (sum(r[ext$repay_fail == 1]) - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
cat("GLMM Training AUC:", auc_val, "\n")
cat("GLMM Training Gini:", 2 * auc_val - 1, "\n")

# Fit baseline GLM on same data to compare
fit_glm <- glm(repay_fail ~ loan_amnt + term + int_rate + annual_inc + 
               inq_last_6mths + pub_rec + revol_bal + revol_util, 
               data=ext, family=binomial(link="logit"))
pred_prob_glm <- predict(fit_glm, type="response")
r2 <- rank(pred_prob_glm)
auc_val2 <- (sum(r2[ext$repay_fail == 1]) - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
cat("GLM Training Gini:", 2 * auc_val2 - 1, "\n")
