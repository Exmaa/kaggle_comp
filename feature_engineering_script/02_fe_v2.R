# ==============================================================================
# 02_fe_v2.R — Feature Engineering Bank (F1 s.d. F6 sesuai PRD_v2.md)
# Proyek   : KAGGLE_AUTO_WIN (Praktikum Sains Data STA1581)
# Tujuan   : Membangun 18 fitur rekayasa domain telekomunikasi & perbankan
#            tanpa data leakage (pure row-wise + out-of-fold safe).
# ==============================================================================

suppressPackageStartupMessages({
  library(data.table)
  library(nanoparquet)
})

cat("==============================================================\n")
cat("[FE v2] MEMBANGUN BANK FITUR DOMAIN TELEKOMUNIKASI (F1 - F6)\n")
cat("==============================================================\n")

build_features_v2 <- function(dt) {
  d <- copy(dt)

  # F1: Hitungan Bulat Asli 3-Bulanan (Menghapus Galat Pembulatan .0 / .3 / .7)
  d[, exact_3m_recharges  := round(X10 * 3)]
  d[, exact_3m_regions    := round(X11 * 3)]
  d[, is_integer_recharge := as.integer(abs(X10 - round(X10)) < 1e-4)]

  # F2: Total ARPU & Nominal Isi Ulang per Transaksi (Indikator Kanal Bank vs Konter)
  d[, total_arpu               := X2 + X6 + X9]
  d[, avg_nominal_per_recharge := total_arpu / (X10 + 0.1)]
  d[, data_rev_per_recharge    := X9 / (X10 + 0.1)]
  d[, voice_rev_per_recharge   := X2 / (X10 + 0.1)]
  d[, recharge_per_region      := X10 / (X11 + 0.1)]

  # F3: Komposisi Belanja Digital vs Legacy (Share of Wallet)
  d[, share_rev_voice := X2 / (total_arpu + 1.0)]
  d[, share_rev_sms   := X6 / (total_arpu + 1.0)]
  d[, share_rev_data  := X9 / (total_arpu + 1.0)]

  # F4: Unit Economics Panggilan & Data
  d[, avg_call_duration := X1 / (X3 + 0.1)]
  d[, rev_per_call_min  := X2 / (X1 + 0.1)]
  d[, rev_per_call_freq := X2 / (X3 + 0.1)]
  d[, rev_per_mb        := X9 / (X8 + 0.1)]

  # F5: Volume Aktivitas Jam Kerja & Lintas Operator (Off-Net)
  d[, workhour_call_mins := X1 * X4]
  d[, offnet_call_mins   := X1 * (1.0 - X5)]
  d[, offnet_sms_rev     := X6 * (1.0 - X7)]

  # F6: Indikator Segmen Diskrit & Anomali Pembulatan
  d[, is_zero_data       := as.integer(X8 == 0)]
  d[, flag_rounding_call := as.integer(X3 == 0 & (X1 > 0 | X2 > 0))]
  d[, flag_short_call    := as.integer(X1 == 0 & X3 > 0)]
  d[, flag_free_data     := as.integer(X9 == 0 & X8 > 0)]

  return(d)
}

out_dir <- "feature_engineering_script/output"
if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE)

train_raw <- readRDS(file.path(out_dir, "train_vanilla_mvp.rds"))
test_raw  <- readRDS(file.path(out_dir, "test_vanilla_mvp.rds"))

train_fe <- build_features_v2(train_raw)
test_fe  <- build_features_v2(test_raw)

raw_cols <- paste0("X", 1:11)
fe_cols  <- setdiff(names(train_fe), c("ID", "Y", "fold", raw_cols))
all_feature_cols <- c(raw_cols, fe_cols)

# Pastikan tidak ada NA / Inf pada seluruh fitur baru
for (col in all_feature_cols) {
  stopifnot(sum(is.na(train_fe[[col]])) == 0L)
  stopifnot(sum(is.infinite(train_fe[[col]])) == 0L)
  stopifnot(sum(is.na(test_fe[[col]])) == 0L)
  stopifnot(sum(is.infinite(test_fe[[col]])) == 0L)
}

saveRDS(train_fe, file.path(out_dir, "train_fe_v2.rds"))
saveRDS(test_fe,  file.path(out_dir, "test_fe_v2.rds"))
write_parquet(as.data.frame(train_fe), file.path(out_dir, "train_fe_v2.parquet"))
write_parquet(as.data.frame(test_fe),  file.path(out_dir, "test_fe_v2.parquet"))

# Hitung korelasi Spearman setiap fitur baru terhadap Y
cor_scores <- sapply(all_feature_cols, function(col) {
  cor(train_fe[[col]], train_fe$Y, method = "spearman")
})
fe_rank_dt <- data.table(
  feature      = all_feature_cols,
  group        = ifelse(all_feature_cols %in% raw_cols, "Raw (X1-X11)", "Engineered (F1-F6)"),
  spearman_rho = round(cor_scores, 4),
  abs_rho      = round(abs(cor_scores), 4)
)[order(-abs_rho)]

cat(sprintf("- Total fitur siap pakai : %d (%d Raw + %d Engineered)\n",
            length(all_feature_cols), length(raw_cols), length(fe_cols)))
cat("\n[Top 15 Fitur Berdasarkan Korelasi Peringkat Spearman |rho| terhadap Y]\n")
print(fe_rank_dt[1:15])

write_parquet(as.data.frame(fe_rank_dt), file.path(out_dir, "fe_v2_feature_correlations.parquet"))
cat(sprintf("\n[SELESAI FE v2] Artefak disimpan di: %s/train_fe_v2.(rds|parquet)\n\n", out_dir))
