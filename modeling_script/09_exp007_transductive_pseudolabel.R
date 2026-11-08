# ==============================================================================
# 09_exp007_transductive_pseudolabel.R — EXP_007: Transductive Semi-Supervised
#                                        Learning via Test Pseudo-Labeling
# Proyek   : KAGGLE_AUTO_WIN (Praktikum Sains Data STA1581)
# ==============================================================================

suppressPackageStartupMessages({
  library(data.table)
  library(xgboost)
  library(ranger)
  library(pROC)
  library(nanoparquet)
})

source("R/metrics.R")

t0 <- Sys.time()
cat("######################################################################\n")
cat("# EXP_007: TRANSDUCTIVE SEMI-SUPERVISED PSEUDO-LABELING ON TEST SET  #\n")
cat("######################################################################\n\n")

SEED <- 1581L
set.seed(SEED)

fe_dir  <- "feature_engineering_script/output"
out_dir <- "modeling_script/output"

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

# 1. Muat Probabilitas Test Ensemble Terbaik (EXP_005)
test_e5_dt <- as.data.table(read_parquet(file.path(out_dir, "test_probabilities_exp_005.parquet")))
test_p <- test_e5_dt$test_prob

# Tentukan Pseudo-Labels Berkeyakinan Tinggi
POS_THRESH <- 0.82
NEG_THRESH <- 0.04
PSEUDO_WEIGHT <- 0.40

pseudo_pos_idx <- which(test_p >= POS_THRESH)
pseudo_neg_idx <- which(test_p <= NEG_THRESH)
pseudo_idx     <- c(pseudo_pos_idx, pseudo_neg_idx)

y_pseudo <- c(rep(1L, length(pseudo_pos_idx)), rep(0L, length(pseudo_neg_idx)))
X_pseudo <- X_test_raw[pseudo_idx, , drop = FALSE]
w_pseudo <- rep(PSEUDO_WEIGHT, length(y_pseudo))

cat(sprintf("[PSEUDO-LABELING] Terpilih %d baris Test (%d Positif @>=%.2f, %d Negatif @<=%.2f) dengan bobot %.2f\n\n",
            length(pseudo_idx), length(pseudo_pos_idx), POS_THRESH, length(pseudo_neg_idx), NEG_THRESH, PSEUDO_WEIGHT))

oof_pseudo_prob  <- numeric(N_train)
test_pseudo_prob <- numeric(N_test)

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
cat("[EXP_007] MELATIH GBDT DENGAN EXPANDED TRANSDUCTIVE MANIFOLD (5-FOLD CV)\n")
cat("==============================================================\n")

for (k in seq_len(K_FOLDS)) {
  val_idx <- which(train_dt$fold == k)
  tr_idx  <- which(train_dt$fold != k)

  X_tr_orig <- X_train_raw[tr_idx, , drop = FALSE]
  y_tr_orig <- y_all[tr_idx]
  w_tr_orig <- rep(1.0, length(y_tr_orig))

  # Gabungkan data latih in-fold dengan pseudo-labels data uji
  X_tr_comb <- rbind(X_tr_orig, X_pseudo)
  y_tr_comb <- c(y_tr_orig, y_pseudo)
  w_tr_comb <- c(w_tr_orig, w_pseudo)

  X_va <- X_train_raw[val_idx, , drop = FALSE]
  y_va <- y_all[val_idx]

  dtrain_comb <- xgb.DMatrix(X_tr_comb, label = y_tr_comb, weight = w_tr_comb)
  dval_k      <- xgb.DMatrix(X_va, label = y_va)
  dtest       <- xgb.DMatrix(X_test_raw)

  set.seed(SEED + 700L + k)
  m_pseudo <- xgb.train(
    params = xgb_params, data = dtrain_comb, nrounds = 1500L,
    evals = list(val = dval_k), early_stopping_rounds = 60L, verbose = 0L
  )

  p_va <- predict(m_pseudo, dval_k)
  p_te <- predict(m_pseudo, dtest)

  oof_pseudo_prob[val_idx] <- p_va
  test_pseudo_prob         <- test_pseudo_prob + p_te / K_FOLDS

  cat(sprintf("  Fold %d/%d selesai | OOF AUC Transductive: %.4f\n",
              k, K_FOLDS, as.numeric(pROC::auc(pROC::roc(y_va, p_va, quiet = TRUE)))))
}

# 2. Evaluasi OOF & Nested CV
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

bench_dt <- rbindlist(list(
  eval_model_oof("1. EXP_004_Stacking_NNLS",         oof_e4_dt$oof_prob),
  eval_model_oof("2. EXP_005_Golden_Stacking",        oof_e5_dt$oof_prob),
  eval_model_oof("3. EXP_007_Transductive_PseudoGBDT", oof_pseudo_prob)
))

cat("\n========================================================================================\n")
cat("  TABEL BENCHMARK EXP_007 (TRANSDUCTIVE PSEUDO-LABELING) vs EXP_005 & EXP_004\n")
cat("========================================================================================\n")
print(bench_dt)
write_parquet(as.data.frame(bench_dt), file.path(out_dir, "exp_007_model_zoo_benchmark.parquet"))

# 3. Simpan Artefak Probabilitas
oof_exp007_dt <- data.table(
  ID          = train_dt$ID,
  fold        = train_dt$fold,
  Y_true      = y_all,
  pseudo_prob = round(oof_pseudo_prob, 6)
)
test_exp007_dt <- data.table(
  ID          = test_dt$ID,
  pseudo_prob = round(test_pseudo_prob, 6)
)

write_parquet(as.data.frame(oof_exp007_dt),  file.path(out_dir, "oof_predictions_exp_007.parquet"))
write_parquet(as.data.frame(test_exp007_dt), file.path(out_dir, "test_probabilities_exp_007.parquet"))

# 4. Bangkitkan Submission EXP_007
th_k7 <- quantile(test_pseudo_prob, probs = 1.0 - 0.1334)
y_k7  <- as.integer(test_pseudo_prob >= th_k7)
sub_p7 <- file.path(out_dir, "submission_exp_007_pseudolabel_vertex1334.parquet")
write_parquet(as.data.frame(data.table(ID = as.integer(test_dt$ID), Y = y_k7)), sub_p7)

# 5. Catat ke Ledger
ledger_path <- "ledger/experiments.parquet"
if (file.exists(ledger_path)) {
  exp_ledger <- as.data.table(read_parquet(ledger_path))
} else {
  exp_ledger <- data.table()
}

new_entry <- data.table(
  exp_id         = "EXP_007",
  timestamp      = format(Sys.time(), "%Y-%m-%d %H:%M:%S"),
  model_family   = "Transductive_SemiSupervised_GBDT",
  features_used  = "46 Fitur + Test Pseudo-Labels (w=0.40)",
  cv_type        = "StratifiedKFold",
  cv_folds       = 5,
  cv_metric_name = "Balanced_Accuracy",
  cv_score_mean  = bench_dt[Model == "3. EXP_007_Transductive_PseudoGBDT", Nested_BA_Mean],
  cv_score_sd    = bench_dt[Model == "3. EXP_007_Transductive_PseudoGBDT", Nested_BA_SD],
  oof_roc_auc    = bench_dt[Model == "3. EXP_007_Transductive_PseudoGBDT", OOF_ROC_AUC],
  opt_threshold  = bench_dt[Model == "3. EXP_007_Transductive_PseudoGBDT", Mean_Threshold],
  test_pos_pct   = 13.34,
  git_commit     = "exp_007_transductive",
  notes          = sprintf("Transductive GBDT on 46 FE + high-confidence test pseudo-labels. OOF AUC = %.5f, Nested BA = %.5f.",
                           bench_dt[Model == "3. EXP_007_Transductive_PseudoGBDT", OOF_ROC_AUC],
                           bench_dt[Model == "3. EXP_007_Transductive_PseudoGBDT", Nested_BA_Mean])
)

exp_ledger <- rbind(exp_ledger[exp_id != "EXP_007"], new_entry)
write_parquet(as.data.frame(exp_ledger), ledger_path)
cat("[LEDGER] ledger/experiments.parquet diperbarui dengan entri EXP_007.\n")

t1 <- Sys.time()
cat(sprintf("\n[SELESAI EXP_007] Total waktu: %.2f detik.\n", as.numeric(difftime(t1, t0, units="secs"))))
