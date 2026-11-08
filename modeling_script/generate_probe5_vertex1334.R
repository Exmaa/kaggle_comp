# ==============================================================================
# modeling_script/generate_probe5_vertex1334.R
# Proyek : KAGGLE_AUTO_WIN (Praktikum Sains Data STA1581)
# ==============================================================================

suppressPackageStartupMessages({
  library(data.table)
  library(nanoparquet)
})

prob_path <- "modeling_script/output/test_probabilities_exp_003.parquet"
prob_dt <- as.data.table(read_parquet(prob_path))

prob_col <- "test_prob"
n_total <- nrow(prob_dt)
target_n1 <- 667L # 13.34% puncak parabola empiris

# Urutkan berdasarkan probabilitas ensemble menurun
setorderv(prob_dt, prob_col, order = -1L)

prob_dt[, Y := 0L]
prob_dt[1:target_n1, Y := 1L]

threshold_value <- prob_dt[[prob_col]][target_n1]
cat(sprintf("Threshold kuantil ensemble puncak pada rank %d: %.5f\n", target_n1, threshold_value))

# Kembalikan ke urutan ID awal
setorder(prob_dt, ID)

sub_dt <- prob_dt[, .(ID = as.integer(ID), Y = as.integer(Y))]

out_path <- "modeling_script/output/submission_exp_003_ensemble_vertex1334.parquet"
write_parquet(as.data.frame(sub_dt), out_path)

cat(sprintf("Sukses membuat %s!\n", out_path))
cat(sprintf("  - Total Baris : %d\n", nrow(sub_dt)))
cat(sprintf("  - Positif (Y=1): %d (%.2f%%)\n", sum(sub_dt$Y == 1L), mean(sub_dt$Y == 1L) * 100))
cat(sprintf("  - Negatif (Y=0): %d (%.2f%%)\n", sum(sub_dt$Y == 0L), mean(sub_dt$Y == 0L) * 100))
