# ==============================================================================
# R/prediction_stability.R — Analisis Stabilitas & Kesepakatan Antar Prediksi (.parquet)
# Proyek : KAGGLE_AUTO_WIN (Praktikum Sains Data STA1581)
# ==============================================================================

suppressPackageStartupMessages({
  library(data.table)
  library(nanoparquet)
})

source("R/metrics.R")

compare_submissions <- function(file_paths, labels = NULL) {
  if (is.null(labels)) {
    labels <- basename(file_paths)
  }

  n_files <- length(file_paths)
  pred_list <- vector("list", n_files)

  for (i in seq_len(n_files)) {
    path <- file_paths[i]
    if (!file.exists(path)) stop(sprintf("File tidak ditemukan: %s", path))
    dt <- as.data.table(read_parquet(path))

    if (!identical(names(dt), c("ID", "Y"))) stop(sprintf("Header file %s tidak valid: %s", path, paste(names(dt), collapse = ",")))
    if (nrow(dt) != 5000L) stop(sprintf("Jumlah baris %s bukan 5000: %d", path, nrow(dt)))

    setorder(dt, ID)
    pred_list[[i]] <- as.integer(dt$Y)
  }

  summary_dt <- data.table(
    Model     = labels,
    N_Pred1   = sapply(pred_list, function(x) sum(x == 1L)),
    Pct_Pred1 = sprintf("%.2f%%", sapply(pred_list, function(x) mean(x == 1L) * 100)),
    N_Pred0   = sapply(pred_list, function(x) sum(x == 0L)),
    Pct_Pred0 = sprintf("%.2f%%", sapply(pred_list, function(x) mean(x == 0L) * 100))
  )

  agree_mat <- matrix(1.0, nrow = n_files, ncol = n_files, dimnames = list(labels, labels))
  kappa_mat <- matrix(1.0, nrow = n_files, ncol = n_files, dimnames = list(labels, labels))
  diff_mat  <- matrix(0L,  nrow = n_files, ncol = n_files, dimnames = list(labels, labels))

  for (i in 1:(n_files - 1)) {
    for (j in (i + 1):n_files) {
      a <- pred_list[[i]]
      b <- pred_list[[j]]
      agree  <- mean(a == b)
      kap    <- calc_cohen_kappa(a, b)
      n_diff <- sum(a != b)

      agree_mat[i, j] <- agree_mat[j, i] <- round(agree, 4)
      kappa_mat[i, j] <- kappa_mat[j, i] <- round(kap, 4)
      diff_mat[i, j]  <- diff_mat[j, i]  <- n_diff
    }
  }

  cat("==============================================================\n")
  cat("  RINGKASAN DISTRIBUSI KELAS PREDIKSI TEST PARQUET (N = 5.000)\n")
  cat("==============================================================\n")
  print(summary_dt)

  cat("\n==============================================================\n")
  cat("  MATRIKS PERSENTASE KESEPAKATAN (AGREEMENT RATE)\n")
  cat("==============================================================\n")
  print(round(agree_mat * 100, 2))

  cat("\n==============================================================\n")
  cat("  MATRIKS COHEN'S KAPPA (INTER-PREDICTOR STABILITY)\n")
  cat("==============================================================\n")
  print(kappa_mat)

  cat("\n==============================================================\n")
  cat("  JUMLAH BARIS PREDIKSI BERBEDA (DIFFERENCE COUNT / 5000)\n")
  cat("==============================================================\n")
  print(diff_mat)
  cat("==============================================================\n")

  invisible(list(
    summary     = summary_dt,
    agreement   = agree_mat,
    kappa       = kappa_mat,
    differences = diff_mat
  ))
}

if (!interactive() && identical(environment(), globalenv())) {
  candidates <- c(
    "modeling_script/output/submission_xgb_mvp.parquet",
    "modeling_script/output/submission_exp_003_ensemble_opt.parquet",
    "modeling_script/output/submission_exp_003_ensemble_prior_pi.parquet",
    "modeling_script/output/submission_probe_P4_ensemble_def050.parquet"
  )
  valid_files <- candidates[file.exists(candidates)]
  if (length(valid_files) >= 2) {
    compare_submissions(valid_files)
  }
}
