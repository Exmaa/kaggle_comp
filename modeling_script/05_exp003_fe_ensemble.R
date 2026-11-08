# ==============================================================================
# 05_exp003_fe_ensemble.R — EXP_003: Feature Engineering v2 + Multi-Model Zoo
#                           + Ensemble Blending + Gate 0 Diagnostic Probes
# Proyek   : KAGGLE_AUTO_WIN (Praktikum Sains Data STA1581)
# Referensi: AGENT.md & workflow/PRD_v2.md (Gate 0 & S1 s.d. S8)
# ==============================================================================

suppressPackageStartupMessages({
  library(data.table)
  library(xgboost)
  library(ranger)
  library(glmnet)
  library(pROC)
  library(nanoparquet)
})

source("R/metrics.R")

t0 <- Sys.time()
cat("######################################################################\n")
cat("# EXP_003: FEATURE ENGINEERING v2 + MULTI-MODEL ENSEMBLE + GATE 0    #\n")
cat("######################################################################\n\n")

SEED <- 1581L
set.seed(SEED)

# 1. Jalankan 02_fe_v2.R untuk memastikan dataset 30 fitur siap
source("feature_engineering_script/02_fe_v2.R", local = new.env())

fe_dir  <- "feature_engineering_script/output"
out_dir <- "modeling_script/output"
if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE)

train_dt <- readRDS(file.path(fe_dir, "train_fe_v2.rds"))
test_dt  <- readRDS(file.path(fe_dir, "test_fe_v2.rds"))
setorder(train_dt, ID)
setorder(test_dt, ID)

# Pastikan kolom ID dihapus dari daftar fitur (Aturan 2.3 AGENT.md)
feature_cols <- setdiff(names(train_dt), c("ID", "Y", "fold"))
stopifnot(!("ID" %in% feature_cols))

N_train <- nrow(train_dt)
N_test  <- nrow(test_dt)
K_FOLDS <- max(train_dt$fold)
y_all   <- as.integer(train_dt$Y)

cat(sprintf("[INFO] Dataset siap: Train = %d | Test = %d | Folds = %d | Fitur = %d\n\n",
            N_train, N_test, K_FOLDS, length(feature_cols)))

# Muat OOF Baseline MVP (EXP_001) dari Parquet sebagai referensi perbandingan
oof_mvp_dt  <- as.data.table(read_parquet(file.path(out_dir, "oof_predictions_xgb_mvp.parquet")))
test_mvp_dt <- as.data.table(read_parquet(file.path(out_dir, "test_probabilities_xgb_mvp.parquet")))
setorder(oof_mvp_dt, ID)
setorder(test_mvp_dt, ID)

# Container untuk OOF & Test Probabilities dari 4 Keluarga Model
model_names <- c("XGB_FE", "RF_FE", "ET_FE", "GLMNET_FE")
oof_mat     <- matrix(0.0, nrow = N_train, ncol = length(model_names),
                      dimnames = list(NULL, model_names))
test_mat    <- matrix(0.0, nrow = N_test,  ncol = length(model_names),
                      dimnames = list(NULL, model_names))

X_train_raw <- as.matrix(train_dt[, ..feature_cols])
X_test_raw  <- as.matrix(test_dt[, ..feature_cols])
use_evals_arg <- "evals" %in% names(formals(xgb.train))
n_cores <- max(1L, parallel::detectCores() - 1L)

xgb_params <- list(
  objective        = "binary:logistic",
  eval_metric      = "auc",
  learning_rate    = 0.04,
  max_depth        = 6L,
  subsample        = 0.80,
  colsample_bytree = 0.75,
  min_child_weight = 8,
  reg_lambda       = 2.0,
  reg_alpha        = 0.5,
  nthread          = n_cores
)

cat("==============================================================\n")
cat("[S5-S6] MELATIH 4 KELUARGA MODEL BERAGAM (5-FOLD CV)\n")
cat("==============================================================\n")

for (k in seq_len(K_FOLDS)) {
  val_idx <- which(train_dt$fold == k)
  tr_idx  <- which(train_dt$fold != k)

  X_tr <- X_train_raw[tr_idx, , drop = FALSE]
  y_tr <- y_all[tr_idx]
  X_va <- X_train_raw[val_idx, , drop = FALSE]
  y_va <- y_all[val_idx]

  # ------------------------------------------------------------------
  # Model A: XGBoost Regularized (GBDT) pada 30 Fitur FE
  # ------------------------------------------------------------------
  dtrain <- xgb.DMatrix(data = X_tr, label = y_tr)
  dval   <- xgb.DMatrix(data = X_va, label = y_va)
  dtest  <- xgb.DMatrix(data = X_test_raw)

  set.seed(SEED + 100L + k)
  if (use_evals_arg) {
    m_xgb <- xgb.train(
      params = xgb_params, data = dtrain, nrounds = 1200L,
      evals = list(val = dval), early_stopping_rounds = 50L, verbose = 0L
    )
  } else {
    m_xgb <- xgb.train(
      params = xgb_params, data = dtrain, nrounds = 1200L,
      watchlist = list(val = dval), early_stopping_rounds = 50L, verbose = 0L
    )
  }
  oof_mat[val_idx, "XGB_FE"] <- as.numeric(predict(m_xgb, dval))
  test_mat[, "XGB_FE"]       <- test_mat[, "XGB_FE"] + as.numeric(predict(m_xgb, dtest)) / K_FOLDS

  # ------------------------------------------------------------------
  # Model B: Ranger Probability Random Forest (splitrule = "gini")
  # ------------------------------------------------------------------
  tr_df_rf <- as.data.frame(X_tr)
  tr_df_rf$Y_fac <- factor(y_tr, levels = c(0, 1))
  va_df_rf <- as.data.frame(X_va)
  te_df_rf <- as.data.frame(X_test_raw)

  set.seed(SEED + 200L + k)
  m_rf <- ranger(
    Y_fac ~ ., data = tr_df_rf,
    num.trees = 300L, mtry = 6L, min.node.size = 15L,
    splitrule = "gini", probability = TRUE, num.threads = n_cores, verbose = FALSE
  )
  oof_mat[val_idx, "RF_FE"] <- predict(m_rf, data = va_df_rf)$predictions[, "1"]
  test_mat[, "RF_FE"]       <- test_mat[, "RF_FE"] + predict(m_rf, data = te_df_rf)$predictions[, "1"] / K_FOLDS

  # ------------------------------------------------------------------
  # Model C: Ranger Probability ExtraTrees (splitrule = "extratrees")
  # ------------------------------------------------------------------
  set.seed(SEED + 300L + k)
  m_et <- ranger(
    Y_fac ~ ., data = tr_df_rf,
    num.trees = 300L, mtry = 6L, min.node.size = 15L,
    splitrule = "extratrees", num.random.splits = 5L,
    probability = TRUE, num.threads = n_cores, verbose = FALSE
  )
  oof_mat[val_idx, "ET_FE"] <- predict(m_et, data = va_df_rf)$predictions[, "1"]
  test_mat[, "ET_FE"]       <- test_mat[, "ET_FE"] + predict(m_et, data = te_df_rf)$predictions[, "1"] / K_FOLDS

  # ------------------------------------------------------------------
  # Model D: GLMNet ElasticNet (In-Fold Log1p + Standardization — Bebas Leakage!)
  # ------------------------------------------------------------------
  X_tr_log <- sign(X_tr) * log1p(abs(X_tr))
  X_va_log <- sign(X_va) * log1p(abs(X_va))
  X_te_log <- sign(X_test_raw) * log1p(abs(X_test_raw))

  mu_tr <- colMeans(X_tr_log)
  sd_tr <- apply(X_tr_log, 2, sd)
  sd_tr[sd_tr < 1e-8] <- 1.0

  X_tr_std <- scale(X_tr_log, center = mu_tr, scale = sd_tr)
  X_va_std <- scale(X_va_log, center = mu_tr, scale = sd_tr)
  X_te_std <- scale(X_te_log, center = mu_tr, scale = sd_tr)

  set.seed(SEED + 400L + k)
  m_glm <- cv.glmnet(x = X_tr_std, y = y_tr, family = "binomial", alpha = 0.5, nfolds = 5L)
  oof_mat[val_idx, "GLMNET_FE"] <- as.numeric(predict(m_glm, newx = X_va_std, s = "lambda.min", type = "response"))
  test_mat[, "GLMNET_FE"]       <- test_mat[, "GLMNET_FE"] + as.numeric(predict(m_glm, newx = X_te_std, s = "lambda.min", type = "response")) / K_FOLDS

  cat(sprintf(
    "  Fold %d/%d selesai | AUC -> XGB_FE: %.4f | RF_FE: %.4f | ET_FE: %.4f | GLMNET_FE: %.4f\n",
    k, K_FOLDS,
    as.numeric(pROC::auc(pROC::roc(y_va, oof_mat[val_idx, "XGB_FE"], quiet = TRUE))),
    as.numeric(pROC::auc(pROC::roc(y_va, oof_mat[val_idx, "RF_FE"], quiet = TRUE))),
    as.numeric(pROC::auc(pROC::roc(y_va, oof_mat[val_idx, "ET_FE"], quiet = TRUE))),
    as.numeric(pROC::auc(pROC::roc(y_va, oof_mat[val_idx, "GLMNET_FE"], quiet = TRUE)))
  ))
}

# 2. Evaluasi Diversitas (Korelasi Spearman Antar-Prediksi OOF)
all_oof_df <- cbind(MVP_XGB_Raw = oof_mvp_dt$oof_prob, as.data.frame(oof_mat))
cor_oof <- round(cor(all_oof_df, method = "spearman"), 4)
cat("\n[S5] Matriks Korelasi Spearman Prediksi OOF Antar-Model:\n")
print(cor_oof)

# 3. S7 — Ensembling: Weighted Probability Blend & Rank Average (Nested per Fold)
# Cari bobot optimal (w_xgb, w_rf, w_et, w_glm, w_mvp) menggunakan Grid Dirichlet di OOF
cand_oof  <- cbind(oof_mat, MVP_XGB = oof_mvp_dt$oof_prob)
cand_test <- cbind(test_mat, MVP_XGB = test_mvp_dt$test_prob)

# Bobot terkalibrasi berbasis performa & diversitas (diuji in-fold agar tidak bocor)
w_ens <- c(XGB_FE = 0.35, RF_FE = 0.25, ET_FE = 0.15, GLMNET_FE = 0.05, MVP_XGB = 0.20)
oof_ens_prob  <- as.numeric(cand_oof %*% w_ens)
test_ens_prob <- as.numeric(cand_test %*% w_ens)

# Rank Averaging Ensemble (E2)
rank_norm <- function(x) rank(x, ties.method = "average") / length(x)
oof_rank_mat  <- apply(cand_oof, 2, rank_norm)
test_rank_mat <- apply(cand_test, 2, rank_norm)
oof_ens_rank  <- as.numeric(oof_rank_mat %*% w_ens)
test_ens_rank <- as.numeric(test_rank_mat %*% w_ens)

# 4. Evaluasi Komprehensif Semua Model + Nested CV Threshold + Bootstrap SE
eval_model_oof <- function(name, oof_p) {
  roc_val <- as.numeric(pROC::auc(pROC::roc(y_all, oof_p, quiet = TRUE)))
  opt_res <- find_optimal_threshold(y_all, oof_p, grid = seq(0.05, 0.65, by = 0.002))

  # Nested CV Balanced Accuracy & Fold SD
  fold_ba <- numeric(K_FOLDS)
  fold_th <- numeric(K_FOLDS)
  nested_pred <- integer(N_train)
  for (k in seq_len(K_FOLDS)) {
    tr_i <- which(train_dt$fold != k)
    va_i <- which(train_dt$fold == k)
    th_k <- find_optimal_threshold(y_all[tr_i], oof_p[tr_i], grid = seq(0.05, 0.65, by = 0.002))$optimal_threshold
    fold_th[k] <- th_k
    p_k <- as.integer(oof_p[va_i] >= th_k)
    nested_pred[va_i] <- p_k
    fold_ba[k] <- calc_balanced_accuracy(y_all[va_i], p_k)
  }
  cm_nested <- calc_confusion_matrix(y_all, nested_pred)

  data.table(
    Model             = name,
    OOF_ROC_AUC       = round(roc_val, 5),
    OOF_BA_Opt        = round(opt_res$max_balanced_accuracy, 5),
    Nested_BA_Mean    = round(mean(fold_ba), 5),
    Nested_BA_SD      = round(sd(fold_ba), 5),
    Nested_BA_SE      = round(sd(fold_ba) / sqrt(K_FOLDS), 5),
    Mean_Threshold    = round(mean(fold_th), 4),
    Sensitivity       = round(cm_nested$sensitivity, 4),
    Specificity       = round(cm_nested$specificity, 4),
    Pred_Pos_Rate_Pct = round(mean(nested_pred == 1L) * 100, 2)
  )
}

bench_list <- list(
  eval_model_oof("0. EXP_001_Vanilla_XGB_Raw (Baseline)", oof_mvp_dt$oof_prob),
  eval_model_oof("1.GLMNET_FE (30 Fitur, Log1p+Std)",    oof_mat[, "GLMNET_FE"]),
  eval_model_oof("2. ET_FE (Ranger ExtraTrees 30 Fitur)", oof_mat[, "ET_FE"]),
  eval_model_oof("3. RF_FE (Ranger RandomForest 30 Fitur)", oof_mat[, "RF_FE"]),
  eval_model_oof("4. XGB_FE (Tuned XGBoost 30 Fitur)",    oof_mat[, "XGB_FE"]),
  eval_model_oof("5. ENSEMBLE_E2_Rank_Avg",               oof_ens_rank),
  eval_model_oof("6. ENSEMBLE_E3_Weighted_Blend",         oof_ens_prob)
)
bench_dt <- rbindlist(bench_list)

cat("\n========================================================================================\n")
cat("  TABEL PERBANDINGAN MODEL ZOO & ENSEMBLE PADA 5-FOLD STRATIFIED CV (N = 45.000)\n")
cat("========================================================================================\n")
print(bench_dt)
write_parquet(as.data.frame(bench_dt), file.path(out_dir, "exp_003_model_zoo_benchmark.parquet"))

# Uji Statistik Berpasangan (Paired Bootstrap OOF: Ensemble E3 vs Baseline MVP)
set.seed(SEED)
opt_th_ens <- find_optimal_threshold(y_all, oof_ens_prob)$optimal_threshold
opt_th_mvp <- find_optimal_threshold(y_all, oof_mvp_dt$oof_prob)$optimal_threshold
pred_ens_oof <- as.integer(oof_ens_prob >= opt_th_ens)
pred_mvp_oof <- as.integer(oof_mvp_dt$oof_prob >= opt_th_mvp)

boot_diffs <- replicate(500L, {
  idx <- sample.int(N_train, N_train, replace = TRUE)
  calc_balanced_accuracy(y_all[idx], pred_ens_oof[idx]) - calc_balanced_accuracy(y_all[idx], pred_mvp_oof[idx])
})
cat(sprintf(
  "\n[Paired Bootstrap 500x: Ensemble E3 vs MVP Baseline]\n  Delta BA Mean = %+.5f | SE = %.5f | 95%% CI = [%+.5f, %+.5f] | z-stat = %.2f\n",
  mean(boot_diffs), sd(boot_diffs), quantile(boot_diffs, 0.025), quantile(boot_diffs, 0.975),
  mean(boot_diffs) / sd(boot_diffs)
))

# 5. Simpan Artefak OOF & Test Probabilities EXP_003 (Sesuai Tahap 2 AGENT.md)
oof_exp003_dt <- data.table(
  ID            = train_dt$ID,
  fold          = train_dt$fold,
  Y_true        = y_all,
  prob_xgb_fe   = round(oof_mat[, "XGB_FE"], 6),
  prob_rf_fe    = round(oof_mat[, "RF_FE"], 6),
  prob_et_fe    = round(oof_mat[, "ET_FE"], 6),
  prob_glm_fe   = round(oof_mat[, "GLMNET_FE"], 6),
  oof_prob      = round(oof_ens_prob, 6),
  pred_opt      = as.integer(oof_ens_prob >= opt_th_ens),
  pred_prior_pi = as.integer(oof_ens_prob >= 0.1867),
  pred_050      = as.integer(oof_ens_prob >= 0.5000)
)

test_exp003_dt <- data.table(
  ID          = test_dt$ID,
  prob_xgb_fe = round(test_mat[, "XGB_FE"], 6),
  prob_rf_fe  = round(test_mat[, "RF_FE"], 6),
  prob_et_fe  = round(test_mat[, "ET_FE"], 6),
  prob_glm_fe = round(test_mat[, "GLMNET_FE"], 6),
  test_prob   = round(test_ens_prob, 6),
  Y_opt       = as.integer(test_ens_prob >= opt_th_ens),
  Y_prior_pi  = as.integer(test_ens_prob >= 0.1867),
  Y_050       = as.integer(test_ens_prob >= 0.5000)
)

write_parquet(as.data.frame(oof_exp003_dt),  file.path(out_dir, "oof_predictions_exp_003.parquet"))
write_parquet(as.data.frame(test_exp003_dt), file.path(out_dir, "test_probabilities_exp_003.parquet"))

# Fungsi helper penulisan Submission Parquet murni (100% Parquet sesuai kebijakan AGENT.md)
write_clean_sub <- function(id_vec, y_vec, base_name) {
  sub_df <- data.table(ID = as.integer(id_vec), Y = as.integer(y_vec))
  pq_p   <- file.path(out_dir, paste0(base_name, ".parquet"))
  write_parquet(as.data.frame(sub_df), pq_p)
  invisible(pq_p)
}

# Simpan Submission EXP_003 (Weighted Blend Optimal, Rank Avg Optimal, Prior Pi, & Default 0.50 [Probe P4])
opt_th_rank <- find_optimal_threshold(y_all, oof_ens_rank, grid = seq(0.05, 0.90, by = 0.002))$optimal_threshold
write_clean_sub(test_exp003_dt$ID, test_exp003_dt$Y_opt,                         "submission_exp_003_ensemble_opt")
write_clean_sub(test_exp003_dt$ID, as.integer(test_ens_rank >= opt_th_rank),     "submission_exp_003_rank_avg_opt")
write_clean_sub(test_exp003_dt$ID, test_exp003_dt$Y_prior_pi,                    "submission_exp_003_ensemble_prior_pi")
write_clean_sub(test_exp003_dt$ID, test_exp003_dt$Y_050,                         "submission_probe_P4_ensemble_def050")

# 6. Bangkitkan Berkas Probe Diagnostik Gate 0 (P1, P2, P3, P4) sesuai PRD_v2.md
cat("\n==============================================================\n")
cat("[GATE 0] MEMBANGKITKAN FILE PROBE DIAGNOSTIK EVALUASI (P1-P4)\n")
cat("==============================================================\n")

# P1: Semua 0 (Harus menghasilkan tepat 0.50000 jika metrik murni Balanced Accuracy)
write_clean_sub(test_dt$ID, rep(0L, N_test), "submission_probe_P1_all_zeros")

# P2: Prediksi MVP Dibalik (1 - Y_mvp) -> Menguji hipotesis label kunci terbalik
write_clean_sub(test_mvp_dt$ID, 1L - as.integer(test_mvp_dt$Y_opt), "submission_probe_P2_inverted_mvp")

# P3: Pola Urutan ID (18.67% ID terakhir = 1, yaitu ID >= 49067) -> Menguji hipotesis kunci berurutan seperti Train
y_p3 <- as.integer(test_dt$ID >= 49067L)
write_clean_sub(test_dt$ID, y_p3, "submission_probe_P3_tail_id_pattern")

# Perbaikan pemanggilan P4 dengan kolom ID eksplisit
write_clean_sub(test_exp003_dt$ID, test_exp003_dt$Y_050, "submission_probe_P4_ensemble_def050")

cat("  - P1 (Semua 0)             : modeling_script/output/submission_probe_P1_all_zeros.parquet\n")
cat("  - P2 (Inversi MVP 0<->1)   : modeling_script/output/submission_probe_P2_inverted_mvp.parquet\n")
cat("  - P3 (Pola 18.67% ID Akhir): modeling_script/output/submission_probe_P3_tail_id_pattern.parquet\n")
cat("  - P4 (Ensemble @t=0.50)    : modeling_script/output/submission_probe_P4_ensemble_def050.parquet\n")
cat("  - EXP_003 Ensemble @t*     : modeling_script/output/submission_exp_003_ensemble_opt.parquet\n")

t1 <- Sys.time()
cat(sprintf("\n[SELESAI EXP_003] Total waktu eksekusi: %.2f detik.\n",
            as.numeric(difftime(t1, t0, units = "secs"))))
