train <- read.csv('data/benchmark_training_loan_data.csv')
valid <- read.csv('data/benchmark_validation_loan_data.csv')

# Clean up row names column if present
if("Unnamed..0" %in% names(train)) { train$Unnamed..0 <- NULL }
if("X" %in% names(train)) { train$X <- NULL }
if("Unnamed..0" %in% names(valid)) { valid$Unnamed..0 <- NULL }
if("X" %in% names(valid)) { valid$X <- NULL }

# Convert categorical variables to factors
cat_vars <- c("term", "home_ownership", "verification_status", "purpose")
for(v in cat_vars) {
  train[[v]] <- as.factor(train[[v]])
  valid[[v]] <- as.factor(valid[[v]])
}

# Fit models
fit_logit <- glm(repay_fail ~ ., data=train, family=binomial(link="logit"))
fit_probit <- glm(repay_fail ~ ., data=train, family=binomial(link="probit"))
fit_cloglog <- glm(repay_fail ~ ., data=train, family=binomial(link="cloglog"))

print(paste("AIC logit:", AIC(fit_logit)))
print(paste("AIC probit:", AIC(fit_probit)))
print(paste("AIC cloglog:", AIC(fit_cloglog)))

# Predict on validation data (using logit)
# Remove rows with unseen factor levels if any
for(v in cat_vars) {
  valid <- valid[valid[[v]] %in% levels(train[[v]]), ]
}

pred_prob <- predict(fit_logit, newdata=valid, type="response")

# Base R AUC calculation
auc_base <- function(actual, predicted) {
  r <- rank(predicted)
  n_pos <- sum(actual == 1)
  n_neg <- sum(actual == 0)
  (sum(r[actual == 1]) - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
}

auc_val <- auc_base(valid$repay_fail, pred_prob)
gini <- 2 * auc_val - 1

print(paste("Validation AUC:", auc_val))
print(paste("Validation Gini:", gini))

print(summary(fit_logit))

