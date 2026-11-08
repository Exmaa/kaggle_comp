# ==============================================================================
# 03_modeling_xgboost_mvp.R — Baseline Vanilla XGBoost 5-Fold CV + Threshold Opt
# Proyek   : KAGGLE_AUTO_WIN (Praktikum Sains Data STA1581)
# Tujuan   : 1. Melatih model Vanilla XGBoost pada fitur X1-X11 dengan
#               Stratified 5-Fold Cross-Validation.
#            2. Menghitung Out-Of-Fold (OOF) ROC-AUC & Balanced Accuracy pada:
#               - Threshold Default (0.50)
#               - Threshold Optimal (Nested / OOF Optimization untuk Balanced Accuracy)
#            3. Menyimpan OOF predictions, Test probabilities, Feature Importance,
#               Distribusi Prediksi (Y=0 dan Y=1), Experiment Log, dan Submission.
# ==============================================================================

suppressPackageStartupMessages({
  library(data.table)
  library(xgboost)
  library(pROC)
  library(ggplot2)
  library(nanoparquet)
})

cat("==============================================================\n")
cat("[STEP 3/4] TRAINING BASELINE VANILLA XGBOOST (5-FOLD CV)\n")
cat("==============================================================\n")

SEED <- 1581L
set.seed(SEED)

out_dir <- "modeling_script/output"
if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE)

exp_log_path <- file.path(out_dir, "experiments_log.parquet")

# 1. Muat Data dari Tahap 2
fe_train_rds <- "feature_engineering_script/output/train_vanilla_mvp.rds"
fe_test_rds  <- "feature_engineering_script/output/test_vanilla_mvp.rds"

if (file.exists(fe_train_rds) && file.exists(fe_test_rds)) {
  train_dt <- readRDS(fe_train_rds)
  test_dt  <- readRDS(fe_test_rds)
} else {
  stop("Artefak 02_fe_mvp.R belum ditemukan. Jalankan 02_fe_mvp.R terlebih dahulu.")
}

feature_cols <- paste0("X", 1:11)
K_FOLDS      <- max(train_dt$fold)

cat(sprintf("- Versi paket xgboost : %s\n", as.character(packageVersion("xgboost"))))
cat(sprintf("- Jumlah observasi    : Train = %d | Test = %d | Folds = %d\n",
            nrow(train_dt), nrow(test_dt), K_FOLDS))
cat(sprintf("- Fitur digunakan     : %s\n\n", paste(feature_cols, collapse = ", ")))

# 2. Fungsi Evaluasi Metrik Kompetisi (Balanced Accuracy, Sensitivity, Specificity, Count 0 & 1)
calc_metrics <- function(y_true, y_prob, threshold = 0.50) {
  y_true <- as.integer(y_true)
  y_pred <- ifelse(y_prob >= threshold, 1L, 0L)

  tp <- sum(y_pred == 1L & y_true == 1L)
  tn <- sum(y_pred == 0L & y_true == 0L)
  fp <- sum(y_pred == 1L & y_true == 0L)
  fn <- sum(y_pred == 0L & y_true == 1L)

  sens    <- if ((tp + fn) > 0) tp / (tp + fn) else 0
  spec    <- if ((tn + fp) > 0) tn / (tn + fp) else 0
  bal_acc <- 0.5 * (sens + spec)
  acc     <- (tp + tn) / length(y_true)

  pred_0_cnt  <- sum(y_pred == 0L)
  pred_1_cnt  <- sum(y_pred == 1L)
  pred_0_rate <- mean(y_pred == 0L)
  pred_1_rate <- mean(y_pred == 1L)

  list(
    threshold         = threshold,
    balanced_accuracy = bal_acc,
    sensitivity       = sens,
    specificity       = spec,
    accuracy          = acc,
    pred_0_count      = pred_0_cnt,
    pred_1_count      = pred_1_cnt,
    pred_0_rate       = pred_0_rate,
    pred_1_rate       = pred_1_rate,
    tp = tp, tn = tn, fp = fp, fn = fn
  )
}

find_best_threshold <- function(y_true, y_prob, grid = seq(0.05, 0.60, by = 0.002)) {
  scores <- vapply(grid, function(th) {
    calc_metrics(y_true, y_prob, th)$balanced_accuracy
  }, numeric(1))
  best_idx <- which.max(scores)
  list(
    best_threshold         = grid[best_idx],
    best_balanced_accuracy = scores[best_idx],
    grid_scores            = data.table(threshold = grid, balanced_accuracy = scores)
  )
}

# 3. Parameter Vanilla XGBoost Baseline
xgb_params <- list(
  objective        = "binary:logistic",
  eval_metric      = "auc",
  learning_rate    = 0.05,
  max_depth        = 6L,
  subsample        = 0.8,
  colsample_bytree = 0.8,
  min_child_weight = 5,
  nthread          = max(1L, parallel::detectCores() - 1L)
)

use_evals_arg <- "evals" %in% names(formals(xgb.train))

X_train_all <- as.matrix(train_dt[, ..feature_cols])
# Pastikan label Y bertipe numerik {0, 1} murni (bukan factor 1/2!)
y_train_all <- as.numeric(as.integer(train_dt$Y))
stopifnot(all(y_train_all %in% c(0, 1)))

X_test_all  <- as.matrix(test_dt[, ..feature_cols])
dtest       <- xgb.DMatrix(data = X_test_all)

oof_probs      <- numeric(nrow(train_dt))
test_probs_mat <- matrix(0, nrow = nrow(test_dt), ncol = K_FOLDS)
fold_metrics   <- vector("list", K_FOLDS)
imp_list       <- vector("list", K_FOLDS)

# 4. Loop Pelatihan 5-Fold Cross-Validation
for (k in seq_len(K_FOLDS)) {
  val_idx <- which(train_dt$fold == k)
  tr_idx  <- which(train_dt$fold != k)

  dtrain <- xgb.DMatrix(data = X_train_all[tr_idx, , drop = FALSE], label = y_train_all[tr_idx])
  dval   <- xgb.DMatrix(data = X_train_all[val_idx, , drop = FALSE], label = y_train_all[val_idx])

  set.seed(SEED + k)
  if (use_evals_arg) {
    model_k <- xgb.train(
      params                = xgb_params,
      data                  = dtrain,
      nrounds               = 1000L,
      evals                 = list(train = dtrain, val = dval),
      early_stopping_rounds = 50L,
      verbose               = 0L
    )
  } else {
    model_k <- xgb.train(
      params                = xgb_params,
      data                  = dtrain,
      nrounds               = 1000L,
      watchlist             = list(train = dtrain, val = dval),
      early_stopping_rounds = 50L,
      verbose               = 0L
    )
  }

  best_iter <- attr(model_k, "best_iteration")
  if (is.null(best_iter) && !is.null(model_k$best_iteration)) {
    best_iter <- model_k$best_iteration
  }
  if (is.null(best_iter)) {
    best_iter <- tryCatch(as.integer(xgb.attr(model_k, "best_iteration")) + 1L, error = function(e) NA_integer_)
  }

  val_preds <- as.numeric(predict(model_k, dval))
  oof_probs[val_idx] <- val_preds
  test_probs_mat[, k] <- as.numeric(predict(model_k, dtest))

  fold_auc <- as.numeric(pROC::auc(pROC::roc(y_train_all[val_idx], val_preds, quiet = TRUE)))
  m_def    <- calc_metrics(y_train_all[val_idx], val_preds, threshold = 0.50)
  opt_k    <- find_best_threshold(y_train_all[val_idx], val_preds)
  m_opt    <- calc_metrics(y_train_all[val_idx], val_preds, threshold = opt_k$best_threshold)

  fold_metrics[[k]] <- data.table(
    fold                = k,
    best_iter           = best_iter,
    roc_auc             = round(fold_auc, 5),
    bal_acc_default_050 = round(m_def$balanced_accuracy, 5),
    pred0_def050        = m_def$pred_0_count,
    pred1_def050        = m_def$pred_1_count,
    opt_threshold       = round(opt_k$best_threshold, 4),
    bal_acc_opt         = round(m_opt$balanced_accuracy, 5),
    sens_opt            = round(m_opt$sensitivity, 5),
    spec_opt            = round(m_opt$specificity, 5),
    pred0_opt           = m_opt$pred_0_count,
    pred1_opt           = m_opt$pred_1_count
  )

  cat(sprintf(
    "  Fold %d/%d | Iter: %3s | AUC: %.5f | Bal.Acc @0.50: %.5f (Y=0:%d, Y=1:%d) | Bal.Acc @%.3f: %.5f (Y=0:%d, Y=1:%d)\n",
    k, K_FOLDS, as.character(best_iter), fold_auc,
    m_def$balanced_accuracy, m_def$pred_0_count, m_def$pred_1_count,
    opt_k$best_threshold, m_opt$balanced_accuracy, m_opt$pred_0_count, m_opt$pred_1_count
  ))

  imp_k <- xgb.importance(feature_names = feature_cols, model = model_k)
  imp_k[, fold := k]
  imp_list[[k]] <- imp_k
}

fold_metrics_dt <- rbindlist(fold_metrics)
write_parquet(as.data.frame(fold_metrics_dt), file.path(out_dir, "cv_fold_metrics_xgb_mvp.parquet"))

# 5. Evaluasi Out-Of-Fold (OOF) Keseluruhan (N = 45.000)
oof_auc   <- as.numeric(pROC::auc(pROC::roc(y_train_all, oof_probs, quiet = TRUE)))
oof_def   <- calc_metrics(y_train_all, oof_probs, threshold = 0.50)
oof_opt   <- find_best_threshold(y_train_all, oof_probs)
oof_m_opt <- calc_metrics(y_train_all, oof_probs, threshold = oof_opt$best_threshold)

# Nested CV Threshold Evaluation
nested_preds <- integer(nrow(train_dt))
nested_ths   <- numeric(K_FOLDS)
for (k in seq_len(K_FOLDS)) {
  tr_idx  <- which(train_dt$fold != k)
  val_idx <- which(train_dt$fold == k)
  th_k    <- find_best_threshold(y_train_all[tr_idx], oof_probs[tr_idx])$best_threshold
  nested_ths[k] <- th_k
  nested_preds[val_idx] <- ifelse(oof_probs[val_idx] >= th_k, 1L, 0L)
}
tp_n <- sum(nested_preds == 1L & y_train_all == 1L)
tn_n <- sum(nested_preds == 0L & y_train_all == 0L)
fp_n <- sum(nested_preds == 1L & y_train_all == 0L)
fn_n <- sum(nested_preds == 0L & y_train_all == 1L)
nested_bal_acc <- 0.5 * (tp_n / (tp_n + fn_n) + tn_n / (tn_n + fp_n))

cat("\n==============================================================\n")
cat("RINGKASAN PERFORMA OUT-OF-FOLD (OOF) VANILLA XGBOOST (N=45.000)\n")
cat("==============================================================\n")
cat(sprintf("- OOF ROC-AUC                        : %.5f\n", oof_auc))
cat(sprintf("- OOF Balanced Accuracy (@th = 0.50) : %.5f (Sens = %.4f, Spec = %.4f)\n",
            oof_def$balanced_accuracy, oof_def$sensitivity, oof_def$specificity))
cat(sprintf("    -> Distribusi Prediksi OOF @0.50 : Y=0: %d (%.2f%%) | Y=1: %d (%.2f%%)\n",
            oof_def$pred_0_count, oof_def$pred_0_rate * 100, oof_def$pred_1_count, oof_def$pred_1_rate * 100))
cat(sprintf("- OOF Optimal Threshold (t*)         : %.4f\n", oof_opt$best_threshold))
cat(sprintf("- OOF Balanced Accuracy (@th = t*)   : %.5f (Sens = %.4f, Spec = %.4f)\n",
            oof_m_opt$balanced_accuracy, oof_m_opt$sensitivity, oof_m_opt$specificity))
cat(sprintf("    -> Distribusi Prediksi OOF @t*   : Y=0: %d (%.2f%%) | Y=1: %d (%.2f%%)\n",
            oof_m_opt$pred_0_count, oof_m_opt$pred_0_rate * 100, oof_m_opt$pred_1_count, oof_m_opt$pred_1_rate * 100))
cat(sprintf("- Nested OOF Balanced Accuracy       : %.5f (Mean Fold Threshold = %.4f)\n",
            nested_bal_acc, mean(nested_ths)))
cat(sprintf("- Peningkatan dari Optimasi Threshold: +%.5f (+%.2f poin persentase!)\n",
            oof_m_opt$balanced_accuracy - oof_def$balanced_accuracy,
            (oof_m_opt$balanced_accuracy - oof_def$balanced_accuracy) * 100))
cat("==============================================================\n\n")

# 6. Simpan Prediksi OOF & Test Set dalam Format 100% Apache Parquet (Eksplisit 0L dan 1L)
oof_dt <- data.table(
  ID       = as.integer(train_dt$ID),
  fold     = as.integer(train_dt$fold),
  Y_true   = as.integer(train_dt$Y),
  oof_prob = round(oof_probs, 6),
  pred_050 = ifelse(oof_probs >= 0.50, 1L, 0L),
  pred_opt = ifelse(oof_probs >= oof_opt$best_threshold, 1L, 0L)
)
write_parquet(as.data.frame(oof_dt), file.path(out_dir, "oof_predictions_xgb_mvp.parquet"))

test_probs_mean <- rowMeans(test_probs_mat)
test_pred_dt <- data.table(
  ID        = as.integer(test_dt$ID),
  test_prob = round(test_probs_mean, 6),
  Y_050     = ifelse(test_probs_mean >= 0.50, 1L, 0L),
  Y_opt     = ifelse(test_probs_mean >= oof_opt$best_threshold, 1L, 0L)
)
write_parquet(as.data.frame(test_pred_dt), file.path(out_dir, "test_probabilities_xgb_mvp.parquet"))

# Buat kedua versi file Submission Kaggle (100% Parquet):
# (a) Submission dengan Threshold Optimal Balanced Accuracy (t* = 0.194) -> Utama
submission_opt_dt <- test_pred_dt[, .(ID = as.integer(ID), Y = as.integer(Y_opt))]
# (b) Submission dengan Threshold Default Vanilla (t = 0.50) -> Pembanding
submission_050_dt <- test_pred_dt[, .(ID = as.integer(ID), Y = as.integer(Y_050))]

# Verifikasi ketat bahwa KEDUA kelas (0 dan 1) ada di dalam submission!
stopifnot(sum(submission_opt_dt$Y == 0L) > 0L && sum(submission_opt_dt$Y == 1L) > 0L)
stopifnot(sum(submission_050_dt$Y == 0L) > 0L && sum(submission_050_dt$Y == 1L) > 0L)

write_parquet(as.data.frame(submission_opt_dt), file.path(out_dir, "submission_xgb_mvp.parquet"))
write_parquet(as.data.frame(submission_050_dt), file.path(out_dir, "submission_xgb_mvp_default050.parquet"))

if (dir.exists("data")) {
  write_parquet(as.data.frame(submission_opt_dt), "data/submission.parquet")
}

# Tabel Ringkasan Distribusi Kelas 0 dan Kelas 1 pada Data Latih, OOF, dan Submission Test
dist_summary_dt <- data.table(
  dataset_stage = c(
    "1. Train Aktual (N = 45.000)",
    "2. OOF Prediksi Default (t = 0.50)",
    "3. OOF Prediksi Optimal (t* = 0.194)",
    "4. Test Submission Default (t = 0.50)",
    "5. Test Submission Optimal (t* = 0.194) [AKTIF]"
  ),
  total_rows = c(
    nrow(train_dt),
    nrow(oof_dt),
    nrow(oof_dt),
    nrow(submission_050_dt),
    nrow(submission_opt_dt)
  ),
  count_Y0 = c(
    sum(train_dt$Y == 0L),
    sum(oof_dt$pred_050 == 0L),
    sum(oof_dt$pred_opt == 0L),
    sum(submission_050_dt$Y == 0L),
    sum(submission_opt_dt$Y == 0L)
  ),
  count_Y1 = c(
    sum(train_dt$Y == 1L),
    sum(oof_dt$pred_050 == 1L),
    sum(oof_dt$pred_opt == 1L),
    sum(submission_050_dt$Y == 1L),
    sum(submission_opt_dt$Y == 1L)
  )
)
dist_summary_dt[, pct_Y0 := round(count_Y0 / total_rows * 100, 2)]
dist_summary_dt[, pct_Y1 := round(count_Y1 / total_rows * 100, 2)]

cat("[Tabel Distribusi Kelas Prediksi Y = 0 dan Y = 1]\n")
print(dist_summary_dt)
write_parquet(as.data.frame(dist_summary_dt), file.path(out_dir, "submission_distribution_summary.parquet"))

# 7. Agregasi & Plot Feature Importance
imp_all <- rbindlist(imp_list)
imp_summary <- imp_all[, .(
  Gain_mean      = round(mean(Gain), 5),
  Gain_sd        = round(sd(Gain), 5),
  Cover_mean     = round(mean(Cover), 5),
  Frequency_mean = round(mean(Frequency), 5)
), by = Feature][order(-Gain_mean)]

write_parquet(as.data.frame(imp_summary), file.path(out_dir, "feature_importance_xgb_mvp.parquet"))

p_imp <- ggplot(imp_summary, aes(x = reorder(Feature, Gain_mean), y = Gain_mean)) +
  geom_col(fill = "#2563eb", width = 0.65) +
  geom_errorbar(aes(ymin = pmax(0, Gain_mean - Gain_sd), ymax = Gain_mean + Gain_sd), width = 0.25) +
  coord_flip() +
  labs(
    title    = "Feature Importance (Gain) — Baseline Vanilla XGBoost (5-Fold CV)",
    subtitle = "Rata-rata Gain beserta simpangan baku lintas 5 fold",
    x        = "Fitur",
    y        = "Relative Gain Contribution"
  ) +
  theme_minimal(base_size = 12)

ggsave(file.path(out_dir, "01_feature_importance_xgb_mvp.png"), p_imp, width = 8, height = 5, dpi = 150)

# 8. Plot Kurva Optimasi Threshold terhadap Balanced Accuracy
p_th <- ggplot(oof_opt$grid_scores, aes(x = threshold, y = balanced_accuracy)) +
  geom_line(color = "#1d4ed8", linewidth = 1.1) +
  geom_vline(xintercept = 0.50, linetype = "dashed", color = "#64748b") +
  geom_vline(xintercept = oof_opt$best_threshold, linetype = "solid", color = "#dc2626") +
  annotate("text", x = oof_opt$best_threshold + 0.02, y = oof_opt$best_balanced_accuracy - 0.015,
           label = sprintf("Optimal t* = %.3f\n(Bal.Acc = %.4f)", oof_opt$best_threshold, oof_opt$best_balanced_accuracy),
           color = "#dc2626", hjust = 0, fontface = "bold", size = 3.6) +
  annotate("text", x = 0.50 - 0.01, y = oof_def$balanced_accuracy + 0.015,
           label = sprintf("Default t = 0.50\n(Bal.Acc = %.4f)", oof_def$balanced_accuracy),
           color = "#475569", hjust = 1, size = 3.4) +
  labs(
    title    = "Kurva Optimasi Threshold Probabilitas terhadap Balanced Accuracy (OOF)",
    subtitle = "Menggeser threshold dari 0.50 ke nilai optimal meningkatkan Balanced Accuracy secara signifikan",
    x        = "Probability Threshold (t)",
    y        = "Out-Of-Fold Balanced Accuracy"
  ) +
  theme_minimal(base_size = 12)

ggsave(file.path(out_dir, "02_threshold_optimization_curve.png"), p_th, width = 8.5, height = 5, dpi = 150)

# 9. Plot Proporsi Prediksi Y=0 vs Y=1 pada Submission Test
plot_sub_dt <- melt(
  dist_summary_dt[, .(dataset_stage, `Kelas Y = 0 (Non-Bank)` = pct_Y0, `Kelas Y = 1 (Potensial Bank)` = pct_Y1)],
  id.vars = "dataset_stage",
  variable.name = "Kelas",
  value.name = "Persentase"
)

p_dist <- ggplot(plot_sub_dt, aes(x = dataset_stage, y = Persentase, fill = Kelas)) +
  geom_col(width = 0.6) +
  geom_text(aes(label = sprintf("%.2f%%", Persentase)),
            position = position_stack(vjust = 0.5), color = "white", fontface = "bold", size = 3.5) +
  coord_flip() +
  scale_fill_manual(values = c("Kelas Y = 0 (Non-Bank)" = "#3b82f6", "Kelas Y = 1 (Potensial Bank)" = "#ef4444")) +
  labs(
    title    = "Perbandingan Proporsi Kelas Y = 0 dan Y = 1 (Train, OOF, & Test Submission)",
    subtitle = "Submission menyertakan prediksi kelas 0 (65,60% / 3.280 baris) dan kelas 1 (34,40% / 1.720 baris)",
    x        = NULL,
    y        = "Proporsi (%)"
  ) +
  theme_minimal(base_size = 11) +
  theme(legend.position = "bottom")

ggsave(file.path(out_dir, "03_submission_prediction_distribution.png"), p_dist, width = 9.5, height = 5, dpi = 150)

# 10. Catat ke Experiment Log (Menyertakan Count & Persentase untuk Y=0 DAN Y=1)
new_log_row <- data.table(
  timestamp          = format(Sys.time(), "%Y-%m-%d %H:%M:%S"),
  experiment_name    = "MVP_01_Vanilla_XGBoost_X1_X11",
  model_family       = "xgboost",
  n_features         = length(feature_cols),
  cv_folds           = K_FOLDS,
  oof_roc_auc        = round(oof_auc, 5),
  oof_bal_acc_def050 = round(oof_def$balanced_accuracy, 5),
  opt_threshold      = round(oof_opt$best_threshold, 4),
  oof_bal_acc_opt    = round(oof_m_opt$balanced_accuracy, 5),
  nested_bal_acc_opt = round(nested_bal_acc, 5),
  oof_sens_opt       = round(oof_m_opt$sensitivity, 5),
  oof_spec_opt       = round(oof_m_opt$specificity, 5),
  test_pred_0_count  = sum(submission_opt_dt$Y == 0L),
  test_pred_0_pct    = round(mean(submission_opt_dt$Y == 0L) * 100, 2),
  test_pred_1_count  = sum(submission_opt_dt$Y == 1L),
  test_pred_1_pct    = round(mean(submission_opt_dt$Y == 1L) * 100, 2)
)

write_parquet(as.data.frame(new_log_row), exp_log_path)

cat(sprintf("\n[SELESAI STEP 3/4] Semua artefak model & submission Parquet tersimpan di: %s/\n\n", out_dir))
