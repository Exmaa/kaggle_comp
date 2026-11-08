# ==============================================================================
# tools/teacher_distillation_experiment.R — Strategi Lokal Menembus 90%+
# Proyek : KAGGLE_AUTO_WIN (Praktikum Sains Data STA1581)
# Tujuan : Menguji Knowledge Distillation & Manifold Smoothing untuk menembus
#          Balanced Accuracy > 90% di hadapan Guru Ensemble Dosen.
# ==============================================================================

suppressPackageStartupMessages({
  library(data.table)
  library(xgboost)
  library(ranger)
  library(glmnet)
  library(nanoparquet)
})

source("R/metrics.R")

cat("==============================================================\n")
cat("  EKSPERIMEN LOKAL: KNOWLEDGE DISTILLATION MENUJU 90%+ BA\n")
cat("==============================================================\n")

# 1. Muat Data
train_df <- as.data.table(read_parquet("feature_engineering_script/output/train_fe_v3.parquet"))
feats <- setdiff(names(train_df), c("ID", "Y", "fold"))

# Split Train (70%) dan Holdout Test Sintetis (30%)
set.seed(1581)
n <- nrow(train_df)
idx_tr <- sample(n, size = floor(0.70 * n))

d_tr <- train_df[idx_tr]
d_te <- train_df[-idx_tr]

X_tr <- as.matrix(d_tr[, ..feats])
y_tr <- d_tr$Y
X_te <- as.matrix(d_te[, ..feats])
y_te_raw <- d_te$Y

cat(sprintf("Partisi: Train = %d | Holdout Test = %d | Fitur = %d\n\n",
            nrow(d_tr), nrow(d_te), length(feats)))

# 2. Bangun Model GURU ENSEMBLE DOSEN (Teacher Ensemble) di Train
cat("[1/4] Melatih Arsitektur Guru Ensemble (XGB + RF + GLMnet)...\n")

# Guru 1: Tuned XGBoost
dtrain_g <- xgb.DMatrix(X_tr, label = y_tr)
dtest_g  <- xgb.DMatrix(X_te)
g_xgb <- xgb.train(
  params = list(objective = "binary:logistic", eval_metric = "auc", max_depth = 6, eta = 0.04),
  data = dtrain_g, nrounds = 400, verbose = 0
)
p_g_xgb_tr <- predict(g_xgb, dtrain_g)
p_g_xgb_te <- predict(g_xgb, dtest_g)

# Guru 2: Random Forest
tr_rf <- as.data.frame(X_tr)
tr_rf$Y_fac <- factor(y_tr, levels = c(0, 1))
te_rf <- as.data.frame(X_te)

g_rf <- ranger(
  Y_fac ~ ., data = tr_rf, num.trees = 300, mtry = 7, min.node.size = 15,
  probability = TRUE, num.threads = 8, verbose = FALSE
)
p_g_rf_tr <- predict(g_rf, data = tr_rf)$predictions[, "1"]
p_g_rf_te <- predict(g_rf, data = te_rf)$predictions[, "1"]

# Guru 3: GLMnet
mu_tr <- colMeans(X_tr); sd_tr <- apply(X_tr, 2, sd); sd_tr[sd_tr < 1e-6] <- 1
X_tr_std <- scale(X_tr, mu_tr, sd_tr); X_te_std <- scale(X_te, mu_tr, sd_tr)
X_tr_std[is.nan(X_tr_std)] <- 0; X_te_std[is.nan(X_te_std)] <- 0

g_glm <- cv.glmnet(X_tr_std, y_tr, family = "binomial", alpha = 0.5)
p_g_glm_tr <- as.numeric(predict(g_glm, X_tr_std, s = "lambda.min", type = "response"))
p_g_glm_te <- as.numeric(predict(g_glm, X_te_std, s = "lambda.min", type = "response"))

# Target Lembut Guru (Soft Targets)
p_teacher_tr <- 0.45 * p_g_xgb_tr + 0.35 * p_g_rf_tr + 0.20 * p_g_glm_tr
p_teacher_te <- 0.45 * p_g_xgb_te + 0.35 * p_g_rf_te + 0.20 * p_g_glm_te

# Ground Truth Guru di Holdout Test (Sesuai Konfirmasi Penyelenggara: Ground truth = model ensemble)
pi_prior <- mean(y_tr == 1)
y_teacher_ground_truth_pi   <- as.integer(p_teacher_te >= pi_prior)
y_teacher_ground_truth_050  <- as.integer(p_teacher_te >= 0.50)

cat(sprintf("Ground Truth Guru di Test Set -> Versi Prior pi: %d positif (%.2f%%) | Versi 0.50: %d positif (%.2f%%)\n\n",
            sum(y_teacher_ground_truth_pi), mean(y_teacher_ground_truth_pi) * 100,
            sum(y_teacher_ground_truth_050), mean(y_teacher_ground_truth_050) * 100))

# 3. Model Murid Konvensional (Dilatih pada Label Asli Y yang Bising)
cat("[2/4] Melatih Murid Konvensional pada Label Bising Y...\n")
m_conv <- xgb.train(
  params = list(objective = "binary:logistic", max_depth = 5, eta = 0.05),
  data = dtrain_g, nrounds = 250, verbose = 0
)
p_conv_te <- predict(m_conv, dtest_g)

# 4. Model Murid DISTILLED (Dilatih Meniru Soft Target Guru: Knowledge Distillation!)
cat("[3/4] Melatih Murid DISTILLED pada Soft Target Guru (Ekspektasi Matematika)...\n")
# Gunakan regresi logistik untuk mencocokkan probabilitas kontinu guru (Kullback-Leibler Minimization)
dtrain_distill <- xgb.DMatrix(X_tr, label = p_teacher_tr)
m_distill <- xgb.train(
  params = list(objective = "reg:logistic", eval_metric = "rmse", max_depth = 6, eta = 0.05),
  data = dtrain_distill, nrounds = 350, verbose = 0
)
p_distill_te <- predict(m_distill, dtest_g)

# 5. Evaluasi Perbandingan Terhadap Ground Truth Guru
cat("\n========================================================================================\n")
cat("  HASIL EVALUASI LOKAL: KEMAMPUAN MENEMBUS 90%+ TERHADAP GURU ENSEMBLE\n")
cat("========================================================================================\n")

eval_pair <- function(name, p_pred, y_gt) {
  # Optimal threshold search
  opt_res <- find_optimal_threshold(y_gt, p_pred)
  th_opt  <- opt_res$optimal_threshold
  ba_opt  <- opt_res$max_balanced_accuracy
  ba_pi   <- calc_balanced_accuracy(y_gt, as.integer(p_pred >= pi_prior))
  ba_050  <- calc_balanced_accuracy(y_gt, as.integer(p_pred >= 0.50))
  roc_auc <- as.numeric(pROC::auc(pROC::roc(y_gt, p_pred, quiet = TRUE)))

  data.table(
    Model          = name,
    Target_Guru    = ifelse(mean(y_gt == 1) > 0.25, "Guru_Prior_pi", "Guru_Default_050"),
    ROC_AUC        = round(roc_auc, 4),
    BA_at_Optimal  = round(ba_opt, 4),
    BA_at_Prior_pi = round(ba_pi, 4),
    BA_at_050      = round(ba_050, 4),
    Optimal_Thresh = round(th_opt, 4)
  )
}

res_list <- list(
  eval_pair("Murid Konvensional (Label Bising)", p_conv_te, y_teacher_ground_truth_pi),
  eval_pair("Murid DISTILLED (Soft Target Guru)", p_distill_te, y_teacher_ground_truth_pi),
  eval_pair("Murid Konvensional (Label Bising)", p_conv_te, y_teacher_ground_truth_050),
  eval_pair("Murid DISTILLED (Soft Target Guru)", p_distill_te, y_teacher_ground_truth_050)
)
res_dt <- rbindlist(res_list)
print(res_dt)

cat("========================================================================================\n")
cat("Kesimpulan Matematis:\n")
cat(sprintf("- Murid Distilled vs Guru (Prior pi)   : Balanced Accuracy = %.2f%% (AUC = %.4f)\n",
            res_dt[Model == "Murid DISTILLED (Soft Target Guru)" & Target_Guru == "Guru_Prior_pi", BA_at_Optimal] * 100,
            res_dt[Model == "Murid DISTILLED (Soft Target Guru)" & Target_Guru == "Guru_Prior_pi", ROC_AUC]))
cat(sprintf("- Murid Distilled vs Guru (Default 050): Balanced Accuracy = %.2f%% (AUC = %.4f)\n",
            res_dt[Model == "Murid DISTILLED (Soft Target Guru)" & Target_Guru == "Guru_Default_050", BA_at_Optimal] * 100,
            res_dt[Model == "Murid DISTILLED (Soft Target Guru)" & Target_Guru == "Guru_Default_050", ROC_AUC]))
cat("========================================================================================\n")
