# ==============================================================================
# 02_fe_mvp.R — Stratified CV Split & Preprocessing / Feature Engineering MVP
# Proyek   : KAGGLE_AUTO_WIN (Praktikum Sains Data STA1581)
# Tujuan   : 1. Membuat Stratified 5-Fold Cross-Validation tetap (seed = 1581)
#               agar eksperimen Lokal (Antigravity) & Kaggle 100% konsisten.
#            2. Menyiapkan matriks data Vanilla (X1-X11) untuk MVP Baseline
#               serta menyediakan fungsi modular untuk ekspansi fitur tahap 2.
# ==============================================================================

suppressPackageStartupMessages({
  library(data.table)
  library(caret)
  library(nanoparquet)
})

cat("==============================================================\n")
cat("[STEP 2/4] STRATIFIED 5-FOLD SPLIT & PREPROCESSING MVP\n")
cat("==============================================================\n")

SEED <- 1581L
K_FOLDS <- 5L

# 1. Resolusi Path (100% Apache Parquet)
train_path <- if (file.exists("data/training_data_tugas_praktikum_DataScience.parquet")) {
  "data/training_data_tugas_praktikum_DataScience.parquet"
} else {
  list.files("/kaggle/input", pattern = "training_data.*\\.parquet$", recursive = TRUE, full.names = TRUE)[1]
}

test_path <- if (file.exists("data/testing_data_tugas_praktikum_DataScience.parquet")) {
  "data/testing_data_tugas_praktikum_DataScience.parquet"
} else {
  list.files("/kaggle/input", pattern = "testing_data.*\\.parquet$", recursive = TRUE, full.names = TRUE)[1]
}

out_dir <- "feature_engineering_script/output"
if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE)

train_dt <- as.data.table(read_parquet(train_path))
test_dt  <- as.data.table(read_parquet(test_path))

# Pastikan terurut berdasarkan ID
setorder(train_dt, ID)
setorder(test_dt, ID)

# 2. Pembentukan Stratified 5-Fold CV
set.seed(SEED)
train_dt[, fold := createFolds(factor(train_dt$Y), k = K_FOLDS, list = FALSE)]

fold_check <- train_dt[, .(
  n_rows   = .N,
  n_pos    = sum(Y == 1),
  pos_rate = round(mean(Y == 1) * 100, 3)
), by = fold][order(fold)]

cat("\n[Verifikasi Stratified 5-Fold Cross-Validation (Seed = 1581)]\n")
print(fold_check)

# Simpan indeks fold ke data/cv_folds.parquet & output directory
folds_out <- train_dt[, .(ID, Y, fold)]
if (dir.exists("data")) {
  write_parquet(as.data.frame(folds_out), "data/cv_folds.parquet")
}
write_parquet(as.data.frame(folds_out), file.path(out_dir, "cv_folds.parquet"))

# 3. Validasi Skema Fitur Vanilla (X1 - X11)
vanilla_features <- paste0("X", 1:11)
stopifnot(all(vanilla_features %in% names(train_dt)))
stopifnot(all(vanilla_features %in% names(test_dt)))
stopifnot(sum(is.na(train_dt[, ..vanilla_features])) == 0)
stopifnot(sum(is.na(test_dt[, ..vanilla_features])) == 0)

# 4. Simpan Artefak Data Siap Modeling MVP (Vanilla X1-X11 dalam RDS & Parquet)
saveRDS(train_dt, file.path(out_dir, "train_vanilla_mvp.rds"))
saveRDS(test_dt,  file.path(out_dir, "test_vanilla_mvp.rds"))
write_parquet(as.data.frame(train_dt), file.path(out_dir, "train_vanilla_mvp.parquet"))
write_parquet(as.data.frame(test_dt),  file.path(out_dir, "test_vanilla_mvp.parquet"))

cat(sprintf("\n- Indeks CV disimpan ke : data/cv_folds.parquet (%d baris)\n", nrow(folds_out)))
cat(sprintf("- Data Vanilla MVP      : %d fitur (%s)\n",
            length(vanilla_features), paste(vanilla_features, collapse = ", ")))
cat(sprintf("[SELESAI STEP 2/4] Artefak preprocessing tersimpan di: %s/\n\n", out_dir))
