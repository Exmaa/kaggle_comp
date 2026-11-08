# ==============================================================================
# 12_blend_exp008_exp009.R — Super-Grand-Master Ensemble:
#                            All Tree/Distill/Transductive Paradigms (EXP_001–008)
#                            + Pretrained LLM & Tabular Deep Learning GPU (EXP_009)
# Proyek   : KAGGLE_AUTO_WIN (Praktikum Sains Data STA1581 — pecutAI)
# Fungsi   : Menggabungkan 10 arsitektur lintas paradigma (GBDT, Tomek, ExtraTrees,
#            GLMnet, Polarity Stacking, Knowledge Distillation, Transductive GBDT,
#            LLM Semantic Head, PLR-FT-Transformer, TabResNet-GLU) secara 100%
#            bebas data leakage (In-Fold Constrained Ridge NNLS + Rank Simplex),
#            menguji Paired Bootstrap vs EXP_008 & EXP_004, serta menyimpan
#            submission Parquet (mode Vertex 13.34% & Nested OOF Optimal).
# ==============================================================================

suppressPackageStartupMessages({
  library(data.table)
  library(glmnet)
  library(nanoparquet)
  library(pROC)
})

source("R/metrics.R")
source("R/prediction_stability.R")

cat("======================================================================\n")
cat("  SUPER-GRAND-MASTER BLENDER: EXP_001..008 + LLM & TABULAR DL (EXP_009)\n")
cat("======================================================================\n\n")

out_dir <- "modeling_script/output"
oof_e9_path  <- file.path(out_dir, "oof_predictions_exp_009_dl_llm.parquet")
test_e9_path <- file.path(out_dir, "test_probabilities_exp_009_dl_llm.parquet")

if (!file.exists(oof_e9_path) || !file.exists(test_e9_path)) {
  stop(paste0(
    "Artefak EXP_009 belum tersedia di 'modeling_script/output/'.\n",
    "Jalankan terlebih dahulu di Kaggle Cloud GPU via:\n",
    "  1) powershell -ExecutionPolicy Bypass -File tools/kaggle_run.ps1 -ScriptPath modeling_script/11_exp009_dl_llm_kaggle_gpu.R -EnableGpu\n",
    "  2) powershell -ExecutionPolicy Bypass -File tools/kaggle_pull.ps1"
  ))
}

oof_mvp <- as.data.table(read_parquet(file.path(out_dir, "oof_predictions_xgb_mvp.parquet")))
te_mvp  <- as.data.table(read_parquet(file.path(out_dir, "test_probabilities_xgb_mvp.parquet")))
oof_e4  <- as.data.table(read_parquet(file.path(out_dir, "oof_predictions_exp_004.parquet")))
te_e4   <- as.data.table(read_parquet(file.path(out_dir, "test_probabilities_exp_004.parquet")))
oof_e5  <- as.data.table(read_parquet(file.path(out_dir, "oof_predictions_exp_005.parquet")))
te_e5   <- as.data.table(read_parquet(file.path(out_dir, "test_probabilities_exp_005.parquet")))
oof_e6  <- as.data.table(read_parquet(file.path(out_dir, "oof_predictions_exp_006.parquet")))
te_e6   <- as.data.table(read_parquet(file.path(out_dir, "test_probabilities_exp_006.parquet")))
oof_e7  <- as.data.table(read_parquet(file.path(out_dir, "oof_predictions_exp_007.parquet")))
te_e7   <- as.data.table(read_parquet(file.path(out_dir, "test_probabilities_exp_007.parquet")))
oof_e8  <- as.data.table(read_parquet(file.path(out_dir, "oof_predictions_exp_008.parquet")))
te_e8   <- as.data.table(read_parquet(file.path(out_dir, "test_probabilities_exp_008.parquet")))
oof_e9  <- as.data.table(read_parquet(oof_e9_path))
te_e9   <- as.data.table(read_parquet(test_e9_path))

for (dt_ref in list(oof_mvp, te_mvp, oof_e4, te_e4, oof_e5, te_e5, oof_e6, te_e6, oof_e7, te_e7, oof_e8, te_e8, oof_e9, te_e9)) {
  setorder(dt_ref, ID)
}

stopifnot(identical(oof_e5$ID, oof_e9$ID), identical(as.integer(oof_e5$fold), as.integer(oof_e9$fold)))
stopifnot(identical(te_e5$ID, te_e9$ID))

y_all   <- as.integer(oof_e5$Y_true)
folds   <- as.integer(oof_e5$fold)
K_FOLDS <- max(folds)
N_train <- length(y_all)

# Matriks 11 Keluarga Model Lintas Paradigma (8 Tree/Linear/Distill + 3 Neural/LLM)
OOF_M <- cbind(
  M1_Baseline_MVP       = oof_mvp$oof_prob,
  M2_EXP004_Stacking    = oof_e4$oof_prob,
  M3_Tomek_XGB          = oof_e4$prob_xgb_tomek,
  M4_ExtraTrees         = oof_e4$prob_et_fe3,
  M5_GLMnet_ElasticNet  = oof_e4$prob_glm_fe3,
  M6_Polarity_Stacking  = oof_e5$oof_prob,
  M7_Distilled_Student  = oof_e6$distilled_prob,
  M8_Transductive_GBDT  = oof_e7$pseudo_prob,
  M9_LLM_Semantic_Head  = oof_e9$prob_llm_head,
  M10_PLR_FT_Trans      = oof_e9$prob_ft_transformer,
  M11_TabResNet_GLU     = oof_e9$prob_tabresnet
)

TEST_M <- cbind(
  M1_Baseline_MVP       = te_mvp$test_prob,
  M2_EXP004_Stacking    = te_e4$test_prob,
  M3_Tomek_XGB          = te_e4$prob_xgb_tomek,
  M4_ExtraTrees         = te_e4$prob_et_fe3,
  M5_GLMnet_ElasticNet  = te_e4$prob_glm_fe3,
  M6_Polarity_Stacking  = te_e5$test_prob,
  M7_Distilled_Student  = te_e6$distilled_prob,
  M8_Transductive_GBDT  = te_e7$pseudo_prob,
  M9_LLM_Semantic_Head  = te_e9$prob_llm_head,
  M10_PLR_FT_Trans      = te_e9$prob_ft_transformer,
  M11_TabResNet_GLU     = te_e9$prob_tabresnet
)

cat("[1/4] Matriks Korelasi Spearman Antar-Prediksi OOF (Tree vs Pretrained LLM vs DL):\n")
print(round(cor(OOF_M, method = "spearman"), 4))

# In-Fold ECDF Rank-Normalization + Constrained Non-Negative Ridge Meta-Learner (0% Leakage!)
fit_ecdf_mappers <- function(M_tr) {
  lapply(seq_len(ncol(M_tr)), function(j) stats::ecdf(M_tr[, j]))
}
apply_ecdf_mappers <- function(mappers, M_new) {
  out <- matrix(0.0, nrow = nrow(M_new), ncol = ncol(M_new), dimnames = list(NULL, colnames(M_new)))
  for (j in seq_along(mappers)) out[, j] <- mappers[[j]](M_new[, j])
  out
}

cat("\n[2/4] Melatih Constrained Ridge NNLS Meta-Learner Strictly In-Fold...\n")
oof_super  <- numeric(N_train)
fold_w_mat <- matrix(0.0, nrow = K_FOLDS, ncol = ncol(OOF_M),
                     dimnames = list(paste0("Fold_", 1:K_FOLDS), colnames(OOF_M)))
test_fold_preds <- matrix(0.0, nrow = nrow(TEST_M), ncol = K_FOLDS)

for (k in seq_len(K_FOLDS)) {
  tr_i <- which(folds != k)
  va_i <- which(folds == k)

  ecdf_k <- fit_ecdf_mappers(OOF_M[tr_i, , drop = FALSE])
  R_tr_k <- apply_ecdf_mappers(ecdf_k, OOF_M[tr_i, , drop = FALSE])
  R_va_k <- apply_ecdf_mappers(ecdf_k, OOF_M[va_i, , drop = FALSE])
  R_te_k <- apply_ecdf_mappers(ecdf_k, TEST_M)

  set.seed(1581L + k)
  nnls_k <- cv.glmnet(
    x = R_tr_k, y = y_all[tr_i], family = "binomial",
    alpha = 0, lower.limits = 0, nfolds = 5L
  )
  w_raw <- as.numeric(coef(nnls_k, s = "lambda.1se")[-1])
  if (sum(w_raw) <= 0) w_raw <- as.numeric(coef(nnls_k, s = "lambda.min")[-1])
  if (sum(w_raw) <= 0) w_raw <- rep(1.0, ncol(OOF_M))
  w_norm <- w_raw / sum(w_raw)
  fold_w_mat[k, ] <- w_norm

  oof_super[va_i]       <- as.numeric(R_va_k %*% w_norm)
  test_fold_preds[, k]  <- as.numeric(R_te_k %*% w_norm)
}

final_weights <- colMeans(fold_w_mat)
final_weights <- final_weights / sum(final_weights)
test_super    <- rowMeans(test_fold_preds)

cat("Bobot Terkalibrasi Super-Grand-Master (Tree + LLM + Deep Learning):\n")
for (cn in colnames(OOF_M)) {
  cat(sprintf("  - %-22s : %.4f (%.2f%%)\n", cn, final_weights[cn], final_weights[cn] * 100))
}

# Evaluasi Nested CV Ketat
fold_ba <- numeric(K_FOLDS)
fold_th <- numeric(K_FOLDS)
nested_super_pred <- integer(N_train)
for (k in seq_len(K_FOLDS)) {
  tr_i <- which(folds != k)
  va_i <- which(folds == k)
  th_k <- find_optimal_threshold(y_all[tr_i], oof_super[tr_i], grid = seq(0.15, 0.85, by = 0.002))$optimal_threshold
  fold_th[k] <- th_k
  nested_super_pred[va_i] <- as.integer(oof_super[va_i] >= th_k)
  fold_ba[k] <- calc_balanced_accuracy(y_all[va_i], nested_super_pred[va_i])
}

opt_super <- find_optimal_threshold(y_all, oof_super, grid = seq(0.15, 0.85, by = 0.002))
auc_super <- as.numeric(pROC::auc(pROC::roc(y_all, oof_super, quiet = TRUE)))

# Paired Bootstrap vs EXP_008 Grand Master
th_e8   <- find_optimal_threshold(y_all, oof_e8$grand_prob, grid = seq(0.05, 0.65, by = 0.002))$optimal_threshold
pred_e8 <- as.integer(oof_e8$grand_prob >= th_e8)
set.seed(1581L)
boot_vs_e8 <- replicate(500L, {
  idx <- sample.int(N_train, N_train, replace = TRUE)
  calc_balanced_accuracy(y_all[idx], nested_super_pred[idx]) - calc_balanced_accuracy(y_all[idx], pred_e8[idx])
})

cat("\n======================================================================\n")
cat("  RINGKASAN PERFORMA SUPER-GRAND-MASTER (EXP_008 + EXP_009 DL/LLM)\n")
cat("======================================================================\n")
cat(sprintf("  - OOF ROC-AUC                  : %.5f (vs EXP_008: %.5f | EXP_004: 0.77653)\n",
            auc_super, as.numeric(pROC::auc(pROC::roc(y_all, oof_e8$grand_prob, quiet = TRUE)))))
cat(sprintf("  - OOF Balanced Accuracy (@t*)  : %.5f (t* = %.4f)\n",
            opt_super$max_balanced_accuracy, opt_super$optimal_threshold))
cat(sprintf("  - Nested CV Balanced Accuracy  : %.5f ± %.5f\n", mean(fold_ba), sd(fold_ba)))
cat(sprintf("  - Paired Bootstrap vs EXP_008  : Delta = %+.5f (SE = %.5f, z = %.2f)\n",
            mean(boot_vs_e8), sd(boot_vs_e8), mean(boot_vs_e8) / pmax(sd(boot_vs_e8), 1e-9)))
cat("======================================================================\n\n")

# Simpan 2 Varian Submission Parquet:
# (A) Vertex 13.34% (667 positif — sesuai puncak parabola empiris Public LB 0.53525)
th_vtx_1334 <- stats::quantile(test_super, probs = 1.0 - 0.1334, names = FALSE)
sub_vtx <- data.table(ID = as.integer(te_e5$ID), Y = as.integer(test_super >= th_vtx_1334))
vtx_path <- file.path(out_dir, "submission_exp_009_super_grand_master_vertex1334.parquet")
write_parquet(as.data.frame(sub_vtx), vtx_path)

# (B) Nested OOF Optimal (~36% positif — cadangan Bayes-optimal jika Private LB murni prior pi)
sub_opt <- data.table(ID = as.integer(te_e5$ID), Y = as.integer(test_super >= mean(fold_th)))
opt_path <- file.path(out_dir, "submission_exp_009_super_grand_master_opt.parquet")
write_parquet(as.data.frame(sub_opt), opt_path)

cat("[3/4] Uji Stabilitas Prediksi (EXP_008 Grand Master vs EXP_009 Super-Grand-Master @ 13.34%):\n")
compare_submissions(c(
  file.path(out_dir, "submission_exp_008_grand_master_vertex1334.parquet"),
  vtx_path
))

# Catat ke ledger/experiments.parquet
ledger_path <- "ledger/experiments.parquet"
if (file.exists(ledger_path)) {
  exp_ledger <- as.data.table(read_parquet(ledger_path))
  new_row <- data.table(
    exp_id         = "EXP_009",
    timestamp      = format(Sys.time(), "%Y-%m-%d %H:%M:%S"),
    model_family   = "Super_Grand_Master_DL_LLM_Tree",
    features_used  = "46 FE v4 + BGE-Small LLM (384d) + PLR-FT-Transformer + TabResNet-GLU + M1..M7",
    cv_type        = "StratifiedKFold",
    cv_folds       = 5,
    cv_metric_name = "Balanced_Accuracy",
    cv_score_mean  = round(mean(fold_ba), 5),
    cv_score_sd    = round(sd(fold_ba), 5),
    oof_roc_auc    = round(auc_super, 5),
    opt_threshold  = round(mean(fold_th), 4),
    test_pos_pct   = round(mean(sub_vtx$Y == 1L) * 100, 2),
    git_commit     = "exp_009_dl_llm_super",
    notes          = sprintf("Super-Grand-Master unifying M1..M7 with Kaggle GPU LLM+FT-Transformer+TabResNet. OOF AUC=%.5f, Nested BA=%.5f.",
                             auc_super, mean(fold_ba))
  )
  exp_ledger <- rbind(exp_ledger[exp_id != "EXP_009"], new_row, fill = TRUE)
  write_parquet(as.data.frame(exp_ledger), ledger_path)
}

cat(sprintf("\n[4/4] Selesai! Artefak submission tersimpan:\n  1) %s (Positif = %d / %.2f%%)\n  2) %s (Positif = %d / %.2f%%)\n",
            vtx_path, sum(sub_vtx$Y == 1L), mean(sub_vtx$Y == 1L) * 100,
            opt_path, sum(sub_opt$Y == 1L), mean(sub_opt$Y == 1L) * 100))
