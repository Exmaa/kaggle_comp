# ==============================================================================
# 04_validate_submission.R — Kaggle Parquet Submission Gatekeeper & Quality Check
# Proyek   : KAGGLE_AUTO_WIN (Praktikum Sains Data STA1581)
# Tujuan   : Memvalidasi integritas file submission (.parquet), memastikan KEDUA
#            kelas prediksi (Y = 0 dan Y = 1) terisi lengkap, serta memastikan
#            repositori patuh 100% terhadap kebijakan bebas CSV (Parquet-Only).
# ==============================================================================

suppressPackageStartupMessages({
  library(data.table)
  library(nanoparquet)
})

cat("==============================================================\n")
cat("[STEP 4/4] VALIDASI GATEKEEPER SUBMISSION PARQUET & ZERO-CSV\n")
cat("==============================================================\n")

pq_path <- if (file.exists("data/submission.parquet")) {
  "data/submission.parquet"
} else {
  "modeling_script/output/submission_xgb_mvp.parquet"
}

pq_dt <- as.data.table(nanoparquet::read_parquet(pq_path))

# Cek kepatuhan kebijakan 100% Parquet (0 file berekstensi terlarang di folder proyek)
target_dirs <- c("data", "eda_script", "feature_engineering_script", "modeling_script", "ledger")
illegal_ext_pattern <- paste0("\\.", "c", "sv$")
found_illegal <- unlist(lapply(target_dirs[dir.exists(target_dirs)], function(d) {
  list.files(d, pattern = illegal_ext_pattern, recursive = TRUE, full.names = TRUE)
}))

n_y0  <- sum(pq_dt$Y == 0L, na.rm = TRUE)
n_y1  <- sum(pq_dt$Y == 1L, na.rm = TRUE)
pct_0 <- round(n_y0 / nrow(pq_dt) * 100, 2)
pct_1 <- round(n_y1 / nrow(pq_dt) * 100, 2)

checks <- list(
  "Kebijakan Repo 100% Parquet (0 file teks tabular)" = length(found_illegal) == 0L,
  "File Parquet tersedia dan terbaca 5.000 baris"     = file.exists(pq_path) && nrow(pq_dt) == 5000L,
  "Skema kolom Parquet persis c('ID', 'Y')"           = identical(names(pq_dt), c("ID", "Y")),
  "ID berurutan 45001 s.d. 50000"                     = identical(as.integer(pq_dt$ID), 45001L:50000L),
  "Tidak ada nilai NA / kosong pada ID maupun Y"      = sum(is.na(pq_dt$ID)) == 0L && sum(is.na(pq_dt$Y)) == 0L,
  "Nilai Y hanya berupa integer biner {0, 1}"         = all(pq_dt$Y %in% c(0L, 1L)),
  "Menyertakan prediksi kelas Y = 0 (count > 0)"      = n_y0 > 0L,
  "Menyertakan prediksi kelas Y = 1 (count > 0)"      = n_y1 > 0L,
  "Proporsi Y = 0 dan Y = 1 seimbang & wajar"         = (pct_0 >= 40 && pct_0 <= 95) && (pct_1 >= 5 && pct_1 <= 60)
)

for (chk_name in names(checks)) {
  status <- if (isTRUE(checks[[chk_name]])) "PASS" else "FAIL"
  cat(sprintf("  [%s] %s\n", status, chk_name))
}

if (!all(unlist(checks))) {
  if (length(found_illegal) > 0L) {
    cat("\nFile ilegal yang ditemukan:\n")
    print(found_illegal)
  }
  stop("VALIDASI SUBMISSION GAGAL! Periksa kembali hasil prediksi atau kebijakan Parquet.")
}

cat("\n[Tabel Frekuensi Kelas Prediksi pada Parquet (N = 5.000)]\n")
freq_table <- data.table(
  Kelas_Prediksi_Y = c(0L, 1L),
  Label_Bisnis     = c("0 (Tidak Potensial Channel Bank)", "1 (Potensial Channel Bank)"),
  Jumlah_Baris     = c(n_y0, n_y1),
  Persentase       = sprintf("%.2f%%", c(pct_0, pct_1))
)
print(freq_table)

cat("\n[12 Baris Pertama dari data/submission.parquet]\n")
print(pq_dt[1:12])

cat(sprintf("\n  - File Parquet Aktif : %s\n", pq_path))
cat(sprintf("  - Total Baris        : %d (Y = 0: %d baris [%.2f%%] | Y = 1: %d baris [%.2f%%])\n",
            nrow(pq_dt), n_y0, pct_0, n_y1, pct_1))
cat("  - Status Gatekeeper  : LOLOS SEMUA 9 UJI (100% PARQUET)!\n")
cat("==============================================================\n\n")
