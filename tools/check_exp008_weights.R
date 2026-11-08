library(nanoparquet)
library(data.table)
library(glmnet)

out_dir <- "modeling_script/output"
oof_mvp <- as.data.table(read_parquet(file.path(out_dir, "oof_predictions_xgb_mvp.parquet")))
oof_e4  <- as.data.table(read_parquet(file.path(out_dir, "oof_predictions_exp_004.parquet")))
oof_e5  <- as.data.table(read_parquet(file.path(out_dir, "oof_predictions_exp_005.parquet")))
oof_e6  <- as.data.table(read_parquet(file.path(out_dir, "oof_predictions_exp_006.parquet")))
oof_e7  <- as.data.table(read_parquet(file.path(out_dir, "oof_predictions_exp_007.parquet")))

candidate_cols <- c(
  "M1_Baseline_MVP",
  "M2_Tomek_Cleaned",
  "M3_ExtraTrees",
  "M4_GLMnet_ElasticNet",
  "M5_Polarity_Stacking",
  "M6_Distilled_Student",
  "M7_Transductive_GBDT"
)

oof_meta <- cbind(
  M1_Baseline_MVP      = oof_mvp$oof_prob,
  M2_Tomek_Cleaned     = oof_e4$prob_xgb_tomek,
  M3_ExtraTrees        = oof_e4$prob_et_fe3,
  M4_GLMnet_ElasticNet = oof_e4$prob_glm_fe3,
  M5_Polarity_Stacking = oof_e5$oof_prob,
  M6_Distilled_Student = oof_e6$distilled_prob,
  M7_Transductive_GBDT = oof_e7$pseudo_prob
)

y_all <- as.integer(oof_e5$Y_true)

nnls_fit <- glmnet(
  x = oof_meta, y = y_all, family = "binomial",
  alpha = 0, lower.limits = 0, lambda = 0.0001
)
w_raw <- as.numeric(coef(nnls_fit)[-1])
w_norm <- w_raw / sum(w_raw)
names(w_norm) <- candidate_cols
print(round(w_norm, 4))
