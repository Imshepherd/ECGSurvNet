
# Library

library(survival)
library(mxnet)
library(abind)
source('code/2. training process/1. iterator and predictor.R')

# Load data

load("data/processed_data/valid_data.RData")

# Set

model_name <- "model/ECGSurvNet/ECGSurvNet"

# Load ECGSurvNet

best_model <- mx.model.load(model_name, 0)

# Predict ecg_risk

valid_pred <- survival_risk_predict(model = best_model, data = valid_ecg, batch_size = 50, my_ctx = mx.gpu(4))
valid_label[["ecg_risk"]] <- valid_pred[["predict_out"]]

# Build traditional Cox model for comparison.

cox_age_sex <- coxph(Surv(time, event) ~ age + sex, data = valid_label)
cox_age_sex_ecg <- coxph(Surv(time, event) ~ age + sex + ecg_risk, data = valid_label)
cox_ecg <- coxph(Surv(time, event) ~ ecg_risk, data = valid_label)

# Evaluate traditional Cox in validaition set.

message("C-index of Cox model using age and sex as covariates: ", round(cox_age_sex[["concordance"]][6], digits = 4))
message("C-index of Cox model using the output of ECGSurvNet as covariates: ", round(cox_ecg[["concordance"]][6], digits = 4))
message("C-index of Cox model using age, sex, and the output of ECGSurvNet as covariates: ", round(cox_age_sex_ecg[["concordance"]][6], digits = 4))



