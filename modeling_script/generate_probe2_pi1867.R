# ==============================================================================
# modeling_script/generate_probe2_pi1867.R — Rate-Matching Quantile Prior (18.67%)
# Proyek : KAGGLE_AUTO_WIN (Praktikum Sains Data STA1581)
# ==============================================================================

suppressPackageStartupMessages({
  library(data.table)
  library(nanoparquet)
})

prob_path <- "modeling_script/output/test_probabilities_xgb_mvp.parquet"
if (!file.exists(prob_path)) {
  stop(sprintf("File probabilitas tidak ditemukan: %s", prob_path))
}

prob_dt <- as.data.table(read_parquet(prob_path))
# prob_dt memiliki kolom ID dan probabilitas (atau P_1 / prob)
prob_col <- intersect(c("prob_Y1", "prob", "P_1", "prob_y1", "probability"), names(prob_dt))[1]
if (is.na(prob_col)) {
  # Ambil kolom numerik selain ID
  prob_col <- setdiff(names(prob_dt), "ID")[1]
}

cat(sprintf("Menggunakan kolom probabilitas: %s\n", prob_col))

n_total <- nrow(prob_dt)
target_n1 <- round(0.1867 * n_total) # 934 baris

# Urutkan berdasarkan probabilitas menurun
setorderv(prob_dt, prob_col, order = -1L)

# 934 probabilitas teratas bernilai 1, sisanya bernilai 0
prob_dt[, Y := 0L]
prob_dt[1:target_n1, Y := 1L]

threshold_value <- prob_dt[[prob_col]][target_n1]
cat(sprintf("Threshold kuantil pada rank %d: %.5f\n", target_n1, threshold_value))

# Kembalikan ke urutan ID awal
setorder(prob_dt, ID)

sub_dt <- prob_dt[, .(ID = as.integer(ID), Y = as.integer(Y))]

out_path <- "modeling_script/output/submission_xgb_mvp_quantile_pi1867.parquet"
write_parquet(as.data.frame(sub_dt), out_path)

cat(sprintf("Sukses membuat %s!\n", out_path))
cat(sprintf("  - Total Baris : %d\n", nrow(sub_dt)))
cat(sprintf("  - Positif (Y=1): %d (%.2f%%)\n", sum(sub_dt$Y == 1L), mean(sub_dt$Y == 1L) * 100))
cat(sprintf("  - Negatif (Y=0): %d (%.2f%%)\n", sum(sub_dt$Y == 0L), mean(sub_dt$Y == 0L) * 100))
