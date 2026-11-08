# ==============================================================================
# 03_fe_v3.R — Feature Engineering Generasi ke-3 (Top-Decile & Non-Linear)
# Proyek   : KAGGLE_AUTO_WIN (Praktikum Sains Data STA1581)
# Tujuan   : Memperkaya bank fitur ke 35+ prediktor dengan fokus khusus pada
#            pemisahan nasabah perbankan di desil teratas (top 13.34%).
# ==============================================================================

suppressPackageStartupMessages({
  library(data.table)
  library(nanoparquet)
})

cat("==============================================================\n")
cat("[FE v3] MEMBANGUN BANK FITUR GENERASI KE-3 (36 FITUR DOMAIN)\n")
cat("==============================================================\n")

build_features_v3 <- function(dt) {
  d <- copy(dt)

  # --- KELOMPOK F1: Rekonstruksi Nilai Bulat 3-Bulanan ---
  d[, exact_3m_recharges  := round(X10 * 3)]
  d[, exact_3m_regions    := round(X11 * 3)]
  d[, is_integer_recharge := as.integer(abs(X10 - round(X10)) < 1e-4)]

  # --- KELOMPOK F2: Total ARPU & Nominal Isi Ulang per Transaksi ---
  d[, total_arpu               := X2 + X6 + X9]
  d[, avg_nominal_per_recharge := total_arpu / (X10 + 0.1)]
  d[, data_rev_per_recharge    := X9 / (X10 + 0.1)]
  d[, voice_rev_per_recharge   := X2 / (X10 + 0.1)]
  d[, recharge_per_region      := X10 / (X11 + 0.1)]

  # --- KELOMPOK F3: Komposisi Belanja Digital vs Legacy (Share of Wallet) ---
  d[, share_rev_voice := X2 / (total_arpu + 1.0)]
  d[, share_rev_sms   := X6 / (total_arpu + 1.0)]
  d[, share_rev_data  := X9 / (total_arpu + 1.0)]

  # --- KELOMPOK F4: Unit Economics Panggilan & Data ---
  d[, avg_call_duration := X1 / (X3 + 0.1)]
  d[, rev_per_call_min  := X2 / (X1 + 0.1)]
  d[, rev_per_call_freq := X2 / (X3 + 0.1)]
  d[, rev_per_mb        := X9 / (X8 + 0.1)]

  # --- KELOMPOK F5: Volume Aktivitas Jam Kerja & Lintas Operator ---
  d[, workhour_call_mins := X1 * X4]
  d[, offnet_call_mins   := X1 * (1.0 - X5)]
  d[, offnet_sms_rev     := X6 * (1.0 - X7)]

  # --- KELOMPOK F6: Indikator Segmen Diskrit ---
  d[, is_zero_data       := as.integer(X8 == 0)]
  d[, flag_rounding_call := as.integer(X3 == 0 & (X1 > 0 | X2 > 0))]
  d[, flag_short_call    := as.integer(X1 == 0 & X3 > 0)]
  d[, flag_free_data     := as.integer(X9 == 0 & X8 > 0)]

  # --- KELOMPOK F7: FITUR BARU GENERASI KE-3 (Top-Decile Banking Discriminators) ---
  # 1. Transformasi non-linear stabil untuk prediktor terkuat (rho = 0.3271)
  d[, log_avg_nominal := log1p(pmax(0, avg_nominal_per_recharge))]
  d[, log_total_arpu  := log1p(pmax(0, total_arpu))]
  d[, log_data_rev    := log1p(pmax(0, X9))]

  # 2. Rasio Dominansi Digital vs Voice
  d[, digital_dominance := (pmax(0, X8) * pmax(0, X9)) / ((pmax(0, X1) * pmax(0, X2)) + 1.0)]
  d[, voice_to_data_ratio := (pmax(0, X2) + 0.1) / (pmax(0, X9) + 0.1)]

  # 3. Kepadatan Belanja per Siklus Isi Ulang Bulat
  d[, recharge_density := total_arpu / (pmax(1L, exact_3m_recharges) + 0.1)]

  # 4. Potensi Belanja Lintas Operator (Off-Net Financial Flexibility)
  d[, cross_operator_rev := (1.0 - pmin(1.0, pmax(0, X5))) * (1.0 - pmin(1.0, pmax(0, X7))) * total_arpu]

  # 5. Flag High-Value Banking Profile (Segmen VIP yang mengisi ulang lewat M-Banking)
  # Di Indonesia, nasabah perbankan memiliki ARPU > median, nominal/recharge > 15rb, dan regional aktif
  d[, flag_vip_banking := as.integer(total_arpu >= 40000 & avg_nominal_per_recharge >= 15000 & X11 >= 2)]

  # 6. Interaksi ARPU terhadap durasi bicara
  d[, arpu_per_call_minute := total_arpu / (pmax(0, X1) + 1.0)]

  return(d)
}

out_dir <- "feature_engineering_script/output"
if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE)

# Baca dataset mentah dari Parquet
train_raw <- as.data.table(read_parquet("data/training_data_tugas_praktikum_DataScience.parquet"))
test_raw  <- as.data.table(read_parquet("data/testing_data_tugas_praktikum_DataScience.parquet"))

# Baca fold cross-validation tetap
cv_folds <- as.data.table(read_parquet("data/cv_folds.parquet"))
train_raw <- merge(train_raw, cv_folds[, .(ID, fold)], by = "ID", all.x = TRUE)

train_fe3 <- build_features_v3(train_raw)
test_fe3  <- build_features_v3(test_raw)

raw_cols <- paste0("X", 1:11)
fe_cols  <- setdiff(names(train_fe3), c("ID", "Y", "fold", raw_cols))
all_feature_cols <- c(raw_cols, fe_cols)

# Verifikasi integritas data (Zero NA & Zero Inf)
for (col in all_feature_cols) {
  stopifnot(sum(is.na(train_fe3[[col]])) == 0L)
  stopifnot(sum(is.infinite(train_fe3[[col]])) == 0L)
  stopifnot(sum(is.na(test_fe3[[col]])) == 0L)
  stopifnot(sum(is.infinite(test_fe3[[col]])) == 0L)
}

# Simpan artefak Parquet murni (100% Parquet Policy)
write_parquet(as.data.frame(train_fe3), file.path(out_dir, "train_fe_v3.parquet"))
write_parquet(as.data.frame(test_fe3),  file.path(out_dir, "test_fe_v3.parquet"))

# Hitung korelasi peringkat Spearman terhadap target Y
cor_scores <- sapply(all_feature_cols, function(col) {
  cor(train_fe3[[col]], train_fe3$Y, method = "spearman")
})

fe_rank_dt <- data.table(
  feature      = all_feature_cols,
  group        = ifelse(all_feature_cols %in% raw_cols, "Raw (X1-X11)", "Engineered (F1-F7)"),
  spearman_rho = round(cor_scores, 4),
  abs_rho      = round(abs(cor_scores), 4)
)[order(-abs_rho)]

cat(sprintf("- Total fitur FE v3 siap pakai : %d (%d Raw + %d Engineered)\n",
            length(all_feature_cols), length(raw_cols), length(fe_cols)))
cat("\n[Top 20 Fitur FE v3 Berdasarkan Korelasi Peringkat Spearman |rho| terhadap Y]\n")
print(fe_rank_dt[1:20])

write_parquet(as.data.frame(fe_rank_dt), file.path(out_dir, "fe_v3_feature_correlations.parquet"))
cat(sprintf("\n[SELESAI FE v3] Artefak disimpan di: %s/train_fe_v3.parquet\n\n", out_dir))
