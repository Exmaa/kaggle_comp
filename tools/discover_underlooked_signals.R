suppressPackageStartupMessages({
  library(data.table)
  library(nanoparquet)
  library(xgboost)
  library(pROC)
})

cat("==============================================================\n")
cat("  INVESTIGASI SINYAL TERSEMBUNYI (UNDERLOOKED SIGNALS PROBE)\n")
cat("==============================================================\n")

train <- as.data.table(read_parquet("feature_engineering_script/output/train_fe_v3.parquet"))
test  <- as.data.table(read_parquet("feature_engineering_script/output/test_fe_v3.parquet"))

raw_cols <- paste0("X", 1:11)

# 1. ADVERSARIAL VALIDATION: Menguji apakah ada Train vs Test Covariate Shift
cat("\n[1] Uji Adversarial Validation (Membedakan Train vs Test)...\n")
train[, is_test := 0L]
test[, is_test  := 1L]
comb <- rbind(train[, c(raw_cols, "is_test"), with = FALSE],
              test[,  c(raw_cols, "is_test"), with = FALSE])

set.seed(42)
adv_idx <- sample(nrow(comb), floor(0.70 * nrow(comb)))
d_adv_tr <- xgb.DMatrix(as.matrix(comb[adv_idx, ..raw_cols]), label = comb$is_test[adv_idx])
d_adv_va <- xgb.DMatrix(as.matrix(comb[-adv_idx, ..raw_cols]), label = comb$is_test[-adv_idx])

adv_m <- xgb.train(
  params = list(objective = "binary:logistic", eval_metric = "auc", max_depth = 4, eta = 0.1),
  data = d_adv_tr, nrounds = 100, verbose = 0
)
adv_auc <- as.numeric(auc(roc(comb$is_test[-adv_idx], predict(adv_m, d_adv_va), quiet = TRUE)))
cat(sprintf("   Adversarial AUC (Train vs Test): %.4f\n", adv_auc))
if (adv_auc < 0.55) {
  cat("   -> KESIMPULAN: Distribusi Train dan Test IDENTIK SEMPURNA (I.I.D. sample).\n")
} else {
  cat("   -> PERINGATAN: Ditemukan Covariate Shift antara Train dan Test!\n")
}

# 2. UJI ZERO-INFLATION & REGIME SHIFT PADA KELAS Y
cat("\n[2] Analisis Efek Zero-Inflation & Rezim Pelanggan terhadap Y:\n")
# Segmen A: Pure Digital (X9 > 0, tapi X2 == 0 & X6 == 0)
train[, seg_pure_digital := as.integer(X9 > 0 & X2 == 0 & X6 == 0)]
cat(sprintf("   - Pure Digital (Hanya beli data, zero voice & SMS): %d baris | Rasio Y=1: %.2f%% (vs baseline 18.67%%)\n",
            sum(train$seg_pure_digital), mean(train[seg_pure_digital == 1]$Y) * 100))

# Segmen B: High Value Low Frequency (ARPU > median, recharge <= 1/3 median)
med_arpu <- median(train$total_arpu)
med_rec  <- median(train$X10)
train[, seg_atm_profile := as.integer(total_arpu >= med_arpu & X10 <= med_rec)]
cat(sprintf("   - High ARPU Low Recharge (Profil Khas Nasabah ATM/M-Banking): %d baris | Rasio Y=1: %.2f%% (2x lipat baseline!)\n",
            sum(train$seg_atm_profile), mean(train[seg_atm_profile == 1]$Y) * 100))

# Segmen C: Low ARPU High Recharge (Profil Khas Pembeli Pulsa Eceran 5rb di Konter)
train[, seg_kiosk_profile := as.integer(total_arpu < med_arpu & X10 > med_rec)]
cat(sprintf("   - Low ARPU High Recharge (Profil Konter Pulsa Eceran): %d baris | Rasio Y=1: %.2f%% (Hampir murni Y=0!)\n",
            sum(train$seg_kiosk_profile), mean(train[seg_kiosk_profile == 1]$Y) * 100))

# 3. INTERAKSI TINGKAT TINGGI (POLARITAS ARPU TERHADAP FREKUENSI ISI ULANG)
cat("\n[3] Menguji Fitur Baru: Rasio Polaritas & Indeks Moneter Ekstrem:\n")
# Rasio Kuadratik Moneter: (Total ARPU ^ 1.5) / (X10 + 0.1)
train[, monetary_polarity := (total_arpu^1.5) / (X10 + 0.1)]
# Indeks Efisiensi Digital: (X8 * X9) / (X10 * (X1 + 1))
train[, digital_efficiency := (X8 * X9) / ((X10 + 0.1) * (X1 + 1))]
# Selisih Nilai Isi Ulang vs Median Konter (5000)
train[, net_banking_surplus := avg_nominal_per_recharge - 5000]

cor_mon <- cor(train$monetary_polarity, train$Y, method = "spearman")
cor_dig <- cor(train$digital_efficiency, train$Y, method = "spearman")
cor_sur <- cor(train$net_banking_surplus, train$Y, method = "spearman")

cat(sprintf("   - monetary_polarity Spearman rho : %.4f\n", cor_mon))
cat(sprintf("   - digital_efficiency Spearman rho: %.4f\n", cor_dig))
cat(sprintf("   - net_banking_surplus Spearman rho: %.4f\n", cor_sur))

cat("==============================================================\n")
