# ==============================================================================
# 04_fe_v4_underlooked.R — Feature Engineering Generasi ke-4
#                          (Injeksi Sinyal Tersembunyi: Polaritas & Dual-Regime)
# Proyek   : KAGGLE_AUTO_WIN (Praktikum Sains Data STA1581)
# Tujuan   : Memperkaya bank fitur ke 46 prediktor dengan sinyal underlooked
#            yang terbukti secara empiris memiliki korelasi tertinggi (rho = 0.3306).
# ==============================================================================

suppressPackageStartupMessages({
  library(data.table)
  library(nanoparquet)
})

cat("==============================================================\n")
cat("[FE v4] INJEKSI SINYAL UNDERLOOKED (POLARITAS & DUAL-REGIME)\n")
cat("==============================================================\n")

# 1. Baca FE v3 yang sudah ada
fe_dir <- "feature_engineering_script/output"
train_fe3 <- as.data.table(read_parquet(file.path(fe_dir, "train_fe_v3.parquet")))
test_fe3  <- as.data.table(read_parquet(file.path(fe_dir, "test_fe_v3.parquet")))

setorder(train_fe3, ID)
setorder(test_fe3, ID)

build_underlooked_features <- function(dt, med_arpu_ref = NULL, med_rec_ref = NULL) {
  d <- copy(dt)

  # Hitung referensi median dari data training jika belum ada (Bebas Leakage!)
  if (is.null(med_arpu_ref)) med_arpu_ref <- median(d$total_arpu)
  if (is.null(med_rec_ref))  med_rec_ref  <- median(d$X10)

  # --- SINYAL UNDERLOOKED 1: Rasio Polaritas Moneter Non-Linear (rho = 0.3306!) ---
  d[, monetary_polarity := (pmax(0, total_arpu)^1.5) / (pmax(0, X10) + 0.1)]

  # --- SINYAL UNDERLOOKED 2: Dual-Regime Segment Flags ---
  # Segmen ATM/M-Banking (High ARPU & Low Recharge Frequency: 36.37% Positif)
  d[, seg_atm_profile := as.integer(total_arpu >= med_arpu_ref & X10 <= med_rec_ref)]

  # Segmen Konter Pulsa Eceran (Low ARPU & High Recharge Frequency: 4.61% Positif)
  d[, seg_kiosk_profile := as.integer(total_arpu < med_arpu_ref & X10 > med_rec_ref)]

  # --- SINYAL UNDERLOOKED 3: Share Belanja Digital Murni (Safe from Integer Overflow) ---
  d[, digital_expenditure_share := (as.numeric(pmax(0, X8)) * as.numeric(pmax(0, X9))) /
        ((as.numeric(pmax(0, X1)) * as.numeric(pmax(0, X2))) +
         (as.numeric(pmax(0, X6)) * as.numeric(pmax(0, X7))) + 1.0)]

  return(list(dt = d, med_arpu = med_arpu_ref, med_rec = med_rec_ref))
}

res_train <- build_underlooked_features(train_fe3)
train_fe4 <- res_train$dt

res_test <- build_underlooked_features(test_fe3,
                                      med_arpu_ref = res_train$med_arpu,
                                      med_rec_ref  = res_train$med_rec)
test_fe4 <- res_test$dt

raw_cols <- paste0("X", 1:11)
fe_cols  <- setdiff(names(train_fe4), c("ID", "Y", "fold", raw_cols))
all_feature_cols <- c(raw_cols, fe_cols)

# Verifikasi integritas data (Zero NA & Zero Inf)
for (col in all_feature_cols) {
  stopifnot(sum(is.na(train_fe4[[col]])) == 0L)
  stopifnot(sum(is.infinite(train_fe4[[col]])) == 0L)
  stopifnot(sum(is.na(test_fe4[[col]])) == 0L)
  stopifnot(sum(is.infinite(test_fe4[[col]])) == 0L)
}

# Simpan artefak Parquet murni (100% Parquet Policy)
write_parquet(as.data.frame(train_fe4), file.path(fe_dir, "train_fe_v4.parquet"))
write_parquet(as.data.frame(test_fe4),  file.path(fe_dir, "test_fe_v4.parquet"))

# Hitung korelasi Spearman
cor_scores <- sapply(all_feature_cols, function(col) {
  cor(train_fe4[[col]], train_fe4$Y, method = "spearman")
})

fe_rank_dt <- data.table(
  feature      = all_feature_cols,
  group        = ifelse(all_feature_cols %in% raw_cols, "Raw", "Engineered"),
  spearman_rho = round(cor_scores, 4),
  abs_rho      = round(abs(cor_scores), 4)
)[order(-abs_rho)]

cat(sprintf("- Total fitur FE v4 siap pakai : %d (%d Raw + %d Engineered)\n",
            length(all_feature_cols), length(raw_cols), length(fe_cols)))
cat("\n[Top 10 Fitur FE v4 dengan Korelasi Spearman Tertinggi]:\n")
print(head(fe_rank_dt, 10))

write_parquet(as.data.frame(fe_rank_dt), file.path(fe_dir, "fe_v4_feature_correlations.parquet"))
cat(sprintf("\n[SELESAI FE v4] Disimpan di: %s/train_fe_v4.parquet\n\n", fe_dir))
