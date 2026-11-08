# ==============================================================================
# 08_exp006_distillation_temperature.R — EXP_006: In-Fold Knowledge Distillation
#                                        with Calibrated Temperature Scaling
# Proyek   : KAGGLE_AUTO_WIN (Praktikum Sains Data STA1581)
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
cat("# EXP_006: IN-FOLD KNOWLEDGE DISTILLATION (TEMPERATURE SCALING T=1.5) #\n")
cat("######################################################################\n\n")

SEED <- 1581L
set.seed(SEED)

fe_dir  <- "feature_engineering_script/output"
out_dir <- "modeling_script/output"
if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE)

# 1. Muat Dataset 46 Fitur FE v4
train_dt <- as.data.table(read_parquet(file.path(fe_dir, "train_fe_v4.parquet")))
test_dt  <- as.data.table(read_parquet(file.path(fe_dir, "test_fe_v4.parquet")))
setorder(train_dt, ID)
setorder(test_dt, ID)

feature_cols <- setdiff(names(train_dt), c("ID", "Y", "fold"))
stopifnot(!("ID" %in% feature_cols))

N_train <- nrow(train_dt)
N_test  <- nrow(test_dt)
K_FOLDS <- max(train_dt$fold)
y_all   <- as.integer(train_dt$Y)

X_train_raw <- as.matrix(train_dt[, ..feature_cols])
X_test_raw  <- as.matrix(test_dt[, ..feature_cols])
n_cores <- max(1L, parallel::detectCores() - 1L)

oof_student_prob <- numeric(N_train)
test_student_prob <- numeric(N_test)
oof_teacher_prob <- numeric(N_train)

TEMPERATURE <- 1.5

cat("==============================================================\n")
cat("[EXP_006] MELATIH DISTILASI IN-FOLD (5-FOLD CV STRICT ZERO LEAKAGE)\n")
cat("==============================================================\n")

for (k in seq_len(K_FOLDS)) {
  val_idx <- which(train_dt$fold == k)
  tr_idx  <- which(train_dt$fold != k)

  X_tr <- X_train_raw[tr_idx, , drop = FALSE]
  y_tr <- y_all[tr_idx]
  X_va <- X_train_raw[val_idx, , drop = FALSE]
  y_va <- y_all[val_idx]

  # --- A. Latih 3 Guru di In-Fold Train Data ---
  # Guru 1: XGBoost (depth=6, 400 rounds)
  dtrain_k <- xgb.DMatrix(X_tr, label = y_tr)
  dval_k   <- xgb.DMatrix(X_va, label = y_va)
  dtest    <- xgb.DMatrix(X_test_raw)

  set.seed(SEED + 100L + k)
  m_t_xgb <- xgb.train(
    params = list(objective = "binary:logistic", eval_metric = "auc",
                  learning_rate = 0.04, max_depth = 6L, subsample = 0.8, colsample_bytree = 0.7,
                  nthread = n_cores),
    data = dtrain_k, nrounds = 400L, verbose = 0L
  )
  p_g_xgb_tr <- predict(m_t_xgb, dtrain_k)
  p_g_xgb_va <- predict(m_t_xgb, dval_k)
  p_g_xgb_te <- predict(m_t_xgb, dtest)

  # Guru 2: Random Forest (300 trees)
  tr_df_rf <- as.data.frame(X_tr)
  tr_df_rf$Y_fac <- factor(y_tr, levels = c(0, 1))
  va_df_rf <- as.data.frame(X_va)
  te_df_rf <- as.data.frame(X_test_raw)

  set.seed(SEED + 200L + k)
  m_t_rf <- ranger(
    Y_fac ~ ., data = tr_df_rf, num.trees = 300L, mtry = 7L, min.node.size = 15L,
    probability = TRUE, num.threads = n_cores, verbose = FALSE
  )
  p_g_rf_tr <- predict(m_t_rf, data = tr_df_rf)$predictions[, "1"]
  p_g_rf_va <- predict(m_t_rf, data = va_df_rf)$predictions[, "1"]
  p_g_rf_te <- predict(m_t_rf, data = te_df_rf)$predictions[, "1"]

  # Guru 3: GLMnet ElasticNet
  X_tr_log <- sign(X_tr) * log1p(abs(X_tr))
  X_va_log <- sign(X_va) * log1p(abs(X_va))
  X_te_log <- sign(X_test_raw) * log1p(abs(X_test_raw))

  mu_tr <- colMeans(X_tr_log); sd_tr <- apply(X_tr_log, 2, sd); sd_tr[sd_tr < 1e-8] <- 1.0
  X_tr_std <- scale(X_tr_log, center = mu_tr, scale = sd_tr)
  X_va_std <- scale(X_va_log, center = mu_tr, scale = sd_tr)
  X_te_std <- scale(X_te_log, center = mu_tr, scale = sd_tr)

  set.seed(SEED + 300L + k)
  m_t_glm <- cv.glmnet(x = X_tr_std, y = y_tr, family = "binomial", alpha = 0.5, nfolds = 5L)
  p_g_glm_tr <- as.numeric(predict(m_t_glm, newx = X_tr_std, s = "lambda.min", type = "response"))
  p_g_glm_va <- as.numeric(predict(m_t_glm, newx = X_va_std, s = "lambda.min", type = "response"))
  p_g_glm_te <- as.numeric(predict(m_t_glm, newx = X_te_std, s = "lambda.min", type = "response"))

  # Ensemble Guru Lembut di In-Fold Train
  p_teacher_tr <- 0.45 * p_g_xgb_tr + 0.35 * p_g_rf_tr + 0.20 * p_g_glm_tr
  p_teacher_va <- 0.45 * p_g_xgb_va + 0.35 * p_g_rf_va + 0.20 * p_g_glm_va
  p_teacher_te <- 0.45 * p_g_xgb_te + 0.35 * p_g_rf_te + 0.20 * p_g_glm_te

  oof_teacher_prob[val_idx] <- p_teacher_va

  # --- B. Temperature Scaling Target Lembut ---
  # z = logit(p), p_temp = sigmoid(z / T)
  p_clamped <- pmin(pmax(p_teacher_tr, 1e-6), 1 - 1e-6)
  logits    <- log(p_clamped / (1 - p_clamped))
  soft_target_temp <- 1 / (1 + exp(-logits / TEMPERATURE))

  # --- C. Latih Model Murid Distilasi (Student GBDT) ---
  dtrain_soft <- xgb.DMatrix(X_tr, label = soft_target_temp)

  set.seed(SEED + 500L + k)
  m_student_k <- xgb.train(
    params = list(
      objective        = "reg:logistic",
      eval_metric      = "rmse",
      learning_rate    = 0.03,
      max_depth        = 6L,
      subsample        = 0.85,
      colsample_bytree = 0.75,
      min_child_weight = 8,
      reg_lambda       = 2.0,
      reg_alpha        = 0.5,
      nthread          = n_cores
    ),
    data = dtrain_soft, nrounds = 550L, verbose = 0L
  )

  p_student_va <- predict(m_student_k, dval_k)
  p_student_te <- predict(m_student_k, dtest)

  oof_student_prob[val_idx] <- p_student_va
  test_student_prob         <- test_student_prob + p_student_te / K_FOLDS

  cat(sprintf("  Fold %d/%d selesai | OOF AUC vs Y Asli: %.4f | OOF AUC vs Guru: %.4f\n",
              k, K_FOLDS,
              as.numeric(pROC::auc(pROC::roc(y_va, p_student_va, quiet = TRUE))),
              as.numeric(pROC::auc(pROC::roc(as.integer(p_teacher_va >= 0.1867), p_student_va, quiet = TRUE)))))
}

# 2. Evaluasi Komprehensif OOF & Nested CV
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

oof_e5_dt <- as.data.table(read_parquet(file.path(out_dir, "oof_predictions_exp_005.parquet")))
oof_e4_dt <- as.data.table(read_parquet(file.path(out_dir, "oof_predictions_exp_004.parquet")))

bench_list <- list(
  eval_model_oof("1. EXP_004_Stacking_NNLS",  oof_e4_dt$oof_prob),
  eval_model_oof("2. EXP_005_Golden_Stacking", oof_e5_dt$oof_prob),
  eval_model_oof("3. EXP_006_Teacher_Ensemble", oof_teacher_prob),
  eval_model_oof("4. EXP_006_Distilled_Student", oof_student_prob)
)
bench_dt <- rbindlist(bench_list)

cat("\n========================================================================================\n")
cat("  TABEL BENCHMARK EXP_006 (DISTILASI GURU T=1.5) vs EXP_005 & EXP_004\n")
cat("========================================================================================\n")
print(bench_dt)
write_parquet(as.data.frame(bench_dt), file.path(out_dir, "exp_006_model_zoo_benchmark.parquet"))

# 3. Simpan Artefak OOF & Test Probabilities EXP_006
oof_exp006_dt <- data.table(
  ID             = train_dt$ID,
  fold           = train_dt$fold,
  Y_true         = y_all,
  teacher_prob   = round(oof_teacher_prob, 6),
  distilled_prob = round(oof_student_prob, 6)
)

test_exp006_dt <- data.table(
  ID             = test_dt$ID,
  distilled_prob = round(test_student_prob, 6)
)

write_parquet(as.data.frame(oof_exp006_dt),  file.path(out_dir, "oof_predictions_exp_006.parquet"))
write_parquet(as.data.frame(test_exp006_dt), file.path(out_dir, "test_probabilities_exp_006.parquet"))

# 4. Bangkitkan Submission EXP_006 pada Puncak Parabola 13.34%
th_k6 <- quantile(test_student_prob, probs = 1.0 - 0.1334)
y_k6  <- as.integer(test_student_prob >= th_k6)
sub_p6 <- file.path(out_dir, "submission_exp_006_distilled_vertex1334.parquet")
write_parquet(as.data.frame(data.table(ID = as.integer(test_dt$ID), Y = y_k6)), sub_p6)

cat(sprintf("\n[SUBMISSION] Berkas Parquet EXP_006 dibangkitkan: %s\n", sub_p6))
cat(sprintf("             Positif = %d (%.2f%%) | Negatif = %d (%.2f%%)\n",
            sum(y_k6), mean(y_k6)*100, sum(y_k6 == 0), mean(y_k6 == 0)*100))

# 5. Catat ke Buku Besar
ledger_path <- "ledger/experiments.parquet"
if (file.exists(ledger_path)) {
  exp_ledger <- as.data.table(read_parquet(ledger_path))
} else {
  exp_ledger <- data.table()
}

new_entry <- data.table(
  exp_id         = "EXP_006",
  timestamp      = format(Sys.time(), "%Y-%m-%d %H:%M:%S"),
  model_family   = "Knowledge_Distillation_Student (T=1.5)",
  features_used  = "46 Fitur FE v4 (Soft-Target Manifold)",
  cv_type        = "StratifiedKFold",
  cv_folds       = 5,
  cv_metric_name = "Balanced_Accuracy",
  cv_score_mean  = bench_dt[Model == "4. EXP_006_Distilled_Student", Nested_BA_Mean],
  cv_score_sd    = bench_dt[Model == "4. EXP_006_Distilled_Student", Nested_BA_SD],
  oof_roc_auc    = bench_dt[Model == "4. EXP_006_Distilled_Student", OOF_ROC_AUC],
  opt_threshold  = bench_dt[Model == "4. EXP_006_Distilled_Student", Mean_Threshold],
  test_pos_pct   = 13.34,
  git_commit     = "exp_006_distillation",
  notes          = sprintf("Student GBDT trained on in-fold teacher soft targets with T=1.5. OOF AUC = %.5f, Nested BA = %.5f.",
                           bench_dt[Model == "4. EXP_006_Distilled_Student", OOF_ROC_AUC],
                           bench_dt[Model == "4. EXP_006_Distilled_Student", Nested_BA_Mean])
)

exp_ledger <- rbind(exp_ledger[exp_id != "EXP_006"], new_entry)
write_parquet(as.data.frame(exp_ledger), ledger_path)
cat("[LEDGER] ledger/experiments.parquet diperbarui dengan entri EXP_006.\n")

t1 <- Sys.time()
cat(sprintf("\n[SELESAI EXP_006] Total waktu: %.2f detik.\n", as.numeric(difftime(t1, t0, units="secs"))))
