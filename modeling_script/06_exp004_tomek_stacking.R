# ==============================================================================
# 06_exp004_tomek_stacking.R — EXP_004: FE v3 (42 Fitur) + Tomek Boundary Cleaning
#                              + Multi-Model Zoo + Constrained NNLS Stacking
# Proyek   : KAGGLE_AUTO_WIN (Praktikum Sains Data STA1581)
# Referensi: AGENT.md, PRD_v2.md, dan Buku Besar Ledger
# ==============================================================================

suppressPackageStartupMessages({
  library(data.table)
  library(xgboost)
  library(ranger)
  library(glmnet)
  library(pROC)
  library(FNN)
  library(nanoparquet)
})

source("R/metrics.R")

t0 <- Sys.time()
cat("######################################################################\n")
cat("# EXP_004: FE v3 (42 FITUR) + TOMEK CLEANING + DIVERSE MODEL STACKING #\n")
cat("######################################################################\n\n")

SEED <- 1581L
set.seed(SEED)

fe_dir  <- "feature_engineering_script/output"
out_dir <- "modeling_script/output"
if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE)

# 1. Muat Dataset 42 Fitur FE v3
train_dt <- as.data.table(read_parquet(file.path(fe_dir, "train_fe_v3.parquet")))
test_dt  <- as.data.table(read_parquet(file.path(fe_dir, "test_fe_v3.parquet")))
setorder(train_dt, ID)
setorder(test_dt, ID)

feature_cols <- setdiff(names(train_dt), c("ID", "Y", "fold"))
stopifnot(!("ID" %in% feature_cols))

N_train <- nrow(train_dt)
N_test  <- nrow(test_dt)
K_FOLDS <- max(train_dt$fold)
y_all   <- as.integer(train_dt$Y)

cat(sprintf("[INFO] Dataset FE v3: Train = %d | Test = %d | Folds = %d | Fitur = %d\n\n",
            N_train, N_test, K_FOLDS, length(feature_cols)))

# Container untuk OOF & Test Probabilities dari 5 Model
model_names <- c("XGB_FE3", "XGB_Tomek", "RF_FE3", "ET_FE3", "GLMNET_FE3")
oof_mat     <- matrix(0.0, nrow = N_train, ncol = length(model_names),
                      dimnames = list(NULL, model_names))
test_mat    <- matrix(0.0, nrow = N_test,  ncol = length(model_names),
                      dimnames = list(NULL, model_names))

X_train_raw <- as.matrix(train_dt[, ..feature_cols])
X_test_raw  <- as.matrix(test_dt[, ..feature_cols])
use_evals_arg <- "evals" %in% names(formals(xgb.train))
n_cores <- max(1L, parallel::detectCores() - 1L)

# Parameter XGBoost Regularized
xgb_params <- list(
  objective        = "binary:logistic",
  eval_metric      = "auc",
  learning_rate    = 0.03,
  max_depth        = 6L,
  subsample        = 0.80,
  colsample_bytree = 0.70,
  min_child_weight = 8,
  gamma            = 0.5,
  reg_lambda       = 2.0,
  reg_alpha        = 0.5,
  nthread          = n_cores
)

cat("==============================================================\n")
cat("[EXP_004] MELATIH 5 MODEL DIVERGEN PADA 5-FOLD STRATIFIED CV\n")
cat("==============================================================\n")

for (k in seq_len(K_FOLDS)) {
  val_idx <- which(train_dt$fold == k)
  tr_idx  <- which(train_dt$fold != k)

  X_tr <- X_train_raw[tr_idx, , drop = FALSE]
  y_tr <- y_all[tr_idx]
  X_va <- X_train_raw[val_idx, , drop = FALSE]
  y_va <- y_all[val_idx]

  # ------------------------------------------------------------------
  # 1. Model 1: XGBoost Regularized pada Data Lengkap (XGB_FE3)
  # ------------------------------------------------------------------
  dtrain <- xgb.DMatrix(data = X_tr, label = y_tr)
  dval   <- xgb.DMatrix(data = X_va, label = y_va)
  dtest  <- xgb.DMatrix(data = X_test_raw)

  set.seed(SEED + 100L + k)
  if (use_evals_arg) {
    m_xgb <- xgb.train(
      params = xgb_params, data = dtrain, nrounds = 1500L,
      evals = list(val = dval), early_stopping_rounds = 60L, verbose = 0L
    )
  } else {
    m_xgb <- xgb.train(
      params = xgb_params, data = dtrain, nrounds = 1500L,
      watchlist = list(val = dval), early_stopping_rounds = 60L, verbose = 0L
    )
  }
  oof_mat[val_idx, "XGB_FE3"] <- as.numeric(predict(m_xgb, dval))
  test_mat[, "XGB_FE3"]       <- test_mat[, "XGB_FE3"] + as.numeric(predict(m_xgb, dtest)) / K_FOLDS

  # ------------------------------------------------------------------
  # 2. Model 2: XGBoost pada Training Set dengan Pembersihan Tomek Links (In-Fold!)
  # ------------------------------------------------------------------
  # Standarisasi X_tr untuk pencarian nearest neighbor
  mu_k <- colMeans(X_tr)
  sd_k <- apply(X_tr, 2, sd)
  sd_k[sd_k < 1e-6] <- 1.0
  X_tr_sc <- scale(X_tr, center = mu_k, scale = sd_k)
  X_tr_sc[is.nan(X_tr_sc)] <- 0

  nn_res <- get.knn(X_tr_sc, k = 1)
  nn_idx <- nn_res$nn.index[, 1]
  # Deteksi mutual Tomek link
  is_mutual_tomek <- (nn_idx[nn_idx] == seq_along(y_tr)) & (y_tr != y_tr[nn_idx])
  # Bersihkan HANYA sampel negatif (Y=0) yang ambigu di perbatasan
  remove_neg_idx  <- which(is_mutual_tomek & (y_tr == 0L))

  keep_idx <- setdiff(seq_along(y_tr), remove_neg_idx)
  X_tr_tomek <- X_tr[keep_idx, , drop = FALSE]
  y_tr_tomek <- y_tr[keep_idx]

  dtrain_tomek <- xgb.DMatrix(data = X_tr_tomek, label = y_tr_tomek)

  set.seed(SEED + 150L + k)
  if (use_evals_arg) {
    m_xgb_tomek <- xgb.train(
      params = xgb_params, data = dtrain_tomek, nrounds = 1500L,
      evals = list(val = dval), early_stopping_rounds = 60L, verbose = 0L
    )
  } else {
    m_xgb_tomek <- xgb.train(
      params = xgb_params, data = dtrain_tomek, nrounds = 1500L,
      watchlist = list(val = dval), early_stopping_rounds = 60L, verbose = 0L
    )
  }
  oof_mat[val_idx, "XGB_Tomek"] <- as.numeric(predict(m_xgb_tomek, dval))
  test_mat[, "XGB_Tomek"]       <- test_mat[, "XGB_Tomek"] + as.numeric(predict(m_xgb_tomek, dtest)) / K_FOLDS

  # ------------------------------------------------------------------
  # 3. Model 3: Ranger Random Forest (splitrule = "gini", mtry = 7)
  # ------------------------------------------------------------------
  tr_df_rf <- as.data.frame(X_tr)
  tr_df_rf$Y_fac <- factor(y_tr, levels = c(0, 1))
  va_df_rf <- as.data.frame(X_va)
  te_df_rf <- as.data.frame(X_test_raw)

  set.seed(SEED + 200L + k)
  m_rf <- ranger(
    Y_fac ~ ., data = tr_df_rf,
    num.trees = 300L, mtry = 7L, min.node.size = 15L,
    splitrule = "gini", probability = TRUE, num.threads = n_cores, verbose = FALSE
  )
  oof_mat[val_idx, "RF_FE3"] <- predict(m_rf, data = va_df_rf)$predictions[, "1"]
  test_mat[, "RF_FE3"]       <- test_mat[, "RF_FE3"] + predict(m_rf, data = te_df_rf)$predictions[, "1"] / K_FOLDS

  # ------------------------------------------------------------------
  # 4. Model 4: Ranger ExtraTrees (splitrule = "extratrees", mtry = 8)
  # ------------------------------------------------------------------
  set.seed(SEED + 300L + k)
  m_et <- ranger(
    Y_fac ~ ., data = tr_df_rf,
    num.trees = 300L, mtry = 8L, min.node.size = 15L,
    splitrule = "extratrees", num.random.splits = 5L,
    probability = TRUE, num.threads = n_cores, verbose = FALSE
  )
  oof_mat[val_idx, "ET_FE3"] <- predict(m_et, data = va_df_rf)$predictions[, "1"]
  test_mat[, "ET_FE3"]       <- test_mat[, "ET_FE3"] + predict(m_et, data = te_df_rf)$predictions[, "1"] / K_FOLDS

  # ------------------------------------------------------------------
  # 5. Model 5: GLMnet ElasticNet (Log1p + In-Fold Standardization)
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
  oof_mat[val_idx, "GLMNET_FE3"] <- as.numeric(predict(m_glm, newx = X_va_std, s = "lambda.min", type = "response"))
  test_mat[, "GLMNET_FE3"]       <- test_mat[, "GLMNET_FE3"] + as.numeric(predict(m_glm, newx = X_te_std, s = "lambda.min", type = "response")) / K_FOLDS

  cat(sprintf(
    "  Fold %d/%d (Tomek dibersihkan: %d baris) | AUC -> XGB: %.4f | XGB_Tomek: %.4f | RF: %.4f | ET: %.4f | GLM: %.4f\n",
    k, K_FOLDS, length(remove_neg_idx),
    as.numeric(pROC::auc(pROC::roc(y_va, oof_mat[val_idx, "XGB_FE3"], quiet = TRUE))),
    as.numeric(pROC::auc(pROC::roc(y_va, oof_mat[val_idx, "XGB_Tomek"], quiet = TRUE))),
    as.numeric(pROC::auc(pROC::roc(y_va, oof_mat[val_idx, "RF_FE3"], quiet = TRUE))),
    as.numeric(pROC::auc(pROC::roc(y_va, oof_mat[val_idx, "ET_FE3"], quiet = TRUE))),
    as.numeric(pROC::auc(pROC::roc(y_va, oof_mat[val_idx, "GLMNET_FE3"], quiet = TRUE)))
  ))
}

# ------------------------------------------------------------------------------
# 2. Meta-Learner Stacking: Non-Negative Least Squares (NNLS / Constrained Opt)
# ------------------------------------------------------------------------------
cat("\n==============================================================\n")
cat("[STACKING] OPTIMASI BOBOT NON-NEGATIVE LEAST SQUARES (NNLS)\n")
cat("==============================================================\n")

# Optimasi bobot menggunakan constrained logistic regression (koefisien >= 0)
nnls_fit <- glmnet(
  x = oof_mat, y = y_all, family = "binomial",
  alpha = 0, # Ridge penalty kecil untuk stabilitas
  lower.limits = 0, # Koefisien non-negatif
  lambda = 0.0001
)
raw_weights <- as.numeric(coef(nnls_fit)[-1]) # Hapus intercept
weights_norm <- raw_weights / sum(raw_weights)
names(weights_norm) <- model_names

cat("Bobot Stacking Terkalibrasi (Normalized NNLS):\n")
for (m in model_names) {
  cat(sprintf("  - %-12s : %.4f (%.2f%%)\n", m, weights_norm[m], weights_norm[m] * 100))
}

oof_stack_prob  <- as.numeric(oof_mat %*% weights_norm)
test_stack_prob <- as.numeric(test_mat %*% weights_norm)

# Rank-Averaging Stacking
rank_norm <- function(x) rank(x, ties.method = "average") / length(x)
oof_rank_mat   <- apply(oof_mat, 2, rank_norm)
test_rank_mat  <- apply(test_mat, 2, rank_norm)
oof_stack_rank  <- as.numeric(oof_rank_mat %*% weights_norm)
test_stack_rank <- as.numeric(test_rank_mat %*% weights_norm)

# ------------------------------------------------------------------------------
# 3. Evaluasi Komprehensif OOF & Nested CV (Bebas Leakage)
# ------------------------------------------------------------------------------
eval_model_oof <- function(name, oof_p) {
  roc_val <- as.numeric(pROC::auc(pROC::roc(y_all, oof_p, quiet = TRUE)))
  opt_res <- find_optimal_threshold(y_all, oof_p, grid = seq(0.05, 0.65, by = 0.002))

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

# Muat OOF Baseline EXP_001 dan EXP_003 untuk pembanding historis
oof_mvp_dt  <- as.data.table(read_parquet(file.path(out_dir, "oof_predictions_xgb_mvp.parquet")))
oof_e3_dt   <- as.data.table(read_parquet(file.path(out_dir, "oof_predictions_exp_003.parquet")))

bench_list <- list(
  eval_model_oof("0. EXP_001_Vanilla_XGB_Raw",      oof_mvp_dt$oof_prob),
  eval_model_oof("1. EXP_003_Ensemble_Zoo",          oof_e3_dt$oof_prob),
  eval_model_oof("2. EXP_004_GLMNET_FE3",            oof_mat[, "GLMNET_FE3"]),
  eval_model_oof("3. EXP_004_RF_FE3",                oof_mat[, "RF_FE3"]),
  eval_model_oof("4. EXP_004_ET_FE3",                oof_mat[, "ET_FE3"]),
  eval_model_oof("5. EXP_004_XGB_FE3",               oof_mat[, "XGB_FE3"]),
  eval_model_oof("6. EXP_004_XGB_Tomek",             oof_mat[, "XGB_Tomek"]),
  eval_model_oof("7. EXP_004_Stacking_NNLS",         oof_stack_prob),
  eval_model_oof("8. EXP_004_Stacking_RankAvg",      oof_stack_rank)
)
bench_dt <- rbindlist(bench_list)

cat("\n========================================================================================\n")
cat("  TABEL BENCHMARK EXP_004 vs EXP_003 vs BASELINE (5-FOLD STRATIFIED CV)\n")
cat("========================================================================================\n")
print(bench_dt)
write_parquet(as.data.frame(bench_dt), file.path(out_dir, "exp_004_model_zoo_benchmark.parquet"))

# ------------------------------------------------------------------------------
# 4. Simpan Artefak Probabilitas OOF & Test EXP_004
# ------------------------------------------------------------------------------
opt_th_stack <- find_optimal_threshold(y_all, oof_stack_prob)$optimal_threshold
opt_th_tomek <- find_optimal_threshold(y_all, oof_mat[, "XGB_Tomek"])$optimal_threshold

oof_exp004_dt <- data.table(
  ID             = train_dt$ID,
  fold           = train_dt$fold,
  Y_true         = y_all,
  prob_xgb_fe3   = round(oof_mat[, "XGB_FE3"], 6),
  prob_xgb_tomek = round(oof_mat[, "XGB_Tomek"], 6),
  prob_rf_fe3    = round(oof_mat[, "RF_FE3"], 6),
  prob_et_fe3    = round(oof_mat[, "ET_FE3"], 6),
  prob_glm_fe3   = round(oof_mat[, "GLMNET_FE3"], 6),
  oof_prob       = round(oof_stack_prob, 6),
  oof_rank       = round(oof_stack_rank, 6),
  pred_opt       = as.integer(oof_stack_prob >= opt_th_stack)
)

test_exp004_dt <- data.table(
  ID             = test_dt$ID,
  prob_xgb_fe3   = round(test_mat[, "XGB_FE3"], 6),
  prob_xgb_tomek = round(test_mat[, "XGB_Tomek"], 6),
  prob_rf_fe3    = round(test_mat[, "RF_FE3"], 6),
  prob_et_fe3    = round(test_mat[, "ET_FE3"], 6),
  prob_glm_fe3   = round(test_mat[, "GLMNET_FE3"], 6),
  test_prob      = round(test_stack_prob, 6),
  test_rank      = round(test_stack_rank, 6)
)

write_parquet(as.data.frame(oof_exp004_dt),  file.path(out_dir, "oof_predictions_exp_004.parquet"))
write_parquet(as.data.frame(test_exp004_dt), file.path(out_dir, "test_probabilities_exp_004.parquet"))

# ------------------------------------------------------------------------------
# 5. Pembangkitan Paket 5 Kandidat Submission Siap Submit (100% Parquet Policy)
# ------------------------------------------------------------------------------
cat("\n==============================================================\n")
cat("[SUBMISSIONS] MEMBANGKITKAN 5 KANDIDAT SUBMISSION PRESISI TINGGI\n")
cat("==============================================================\n")

write_clean_sub <- function(id_vec, y_vec, base_name) {
  sub_df <- data.table(ID = as.integer(id_vec), Y = as.integer(y_vec))
  pq_p   <- file.path(out_dir, paste0(base_name, ".parquet"))
  write_parquet(as.data.frame(sub_df), pq_p)
  cat(sprintf("  - %-48s : %d positif (%.2f%%)\n",
              paste0(base_name, ".parquet"), sum(y_vec == 1L), mean(y_vec == 1L) * 100))
  invisible(pq_p)
}

# Kandidat 1: EXP_004 Stacking pada Puncak Parabola Empiris (13.34% positif / 667 baris)
th_k1 <- quantile(test_stack_prob, probs = 1.0 - 0.1334)
y_k1  <- as.integer(test_stack_prob >= th_k1)
write_clean_sub(test_dt$ID, y_k1, "submission_exp_004_stacking_vertex1334")

# Kandidat 2: EXP_004 Stacking pada Proporsi Konservatif (12.00% positif / 600 baris)
th_k2 <- quantile(test_stack_prob, probs = 1.0 - 0.1200)
y_k2  <- as.integer(test_stack_prob >= th_k2)
write_clean_sub(test_dt$ID, y_k2, "submission_exp_004_stacking_pos1200")

# Kandidat 3: EXP_004 Stacking pada Proporsi Agresif (14.50% positif / 725 baris)
th_k3 <- quantile(test_stack_prob, probs = 1.0 - 0.1450)
y_k3  <- as.integer(test_stack_prob >= th_k3)
write_clean_sub(test_dt$ID, y_k3, "submission_exp_004_stacking_pos1450")

# Kandidat 4: EXP_004 Tomek-XGBoost Tunggal pada Puncak Parabola (13.34% positif / 667 baris)
th_k4 <- quantile(test_mat[, "XGB_Tomek"], probs = 1.0 - 0.1334)
y_k4  <- as.integer(test_mat[, "XGB_Tomek"] >= th_k4)
write_clean_sub(test_dt$ID, y_k4, "submission_exp_004_tomek_vertex1334")

# Kandidat 5: EXP_004 Rank-Averaging Stacking pada Puncak Parabola (13.34% positif / 667 baris)
th_k5 <- quantile(test_stack_rank, probs = 1.0 - 0.1334)
y_k5  <- as.integer(test_stack_rank >= th_k5)
write_clean_sub(test_dt$ID, y_k5, "submission_exp_004_rank_avg_vertex1334")

# ------------------------------------------------------------------------------
# 6. Catat ke Buku Besar Eksperimen (ledger/experiments.parquet)
# ------------------------------------------------------------------------------
ledger_path <- "ledger/experiments.parquet"
if (file.exists(ledger_path)) {
  exp_ledger <- as.data.table(read_parquet(ledger_path))
} else {
  exp_ledger <- data.table()
}

new_entry <- data.table(
  exp_id         = "EXP_004",
  timestamp      = format(Sys.time(), "%Y-%m-%d %H:%M:%S"),
  model_family   = "Stacking_NNLS (XGB+Tomek+RF+ET+GLM)",
  features_used  = "42 Fitur (11 Raw + 31 Domain FE v3)",
  cv_type        = "StratifiedKFold",
  cv_folds       = 5,
  cv_metric_name = "Balanced_Accuracy",
  cv_score_mean  = bench_dt[Model == "7. EXP_004_Stacking_NNLS", Nested_BA_Mean],
  cv_score_sd    = bench_dt[Model == "7. EXP_004_Stacking_NNLS", Nested_BA_SD],
  oof_roc_auc    = bench_dt[Model == "7. EXP_004_Stacking_NNLS", OOF_ROC_AUC],
  opt_threshold  = opt_th_stack,
  test_pos_pct   = 13.34,
  git_commit     = "exp_004_tomek_nnls",
  notes          = sprintf("FE v3 + Tomek Links Cleaning + NNLS Stacking. OOF AUC = %.5f, Nested BA = %.5f.",
                           bench_dt[Model == "7. EXP_004_Stacking_NNLS", OOF_ROC_AUC],
                           bench_dt[Model == "7. EXP_004_Stacking_NNLS", Nested_BA_Mean])
)

# Hindari duplikat EXP_004 jika dijalankan ulang
exp_ledger <- rbind(exp_ledger[exp_id != "EXP_004"], new_entry)
write_parquet(as.data.frame(exp_ledger), ledger_path)
cat("\n[LEDGER] ledger/experiments.parquet berhasil diperbarui dengan entri EXP_004.\n")

t1 <- Sys.time()
cat(sprintf("\n[SELESAI EXP_004] Total waktu eksekusi: %.2f detik.\n",
            as.numeric(difftime(t1, t0, units = "secs"))))
