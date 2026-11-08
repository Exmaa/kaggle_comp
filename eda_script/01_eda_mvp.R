# ==============================================================================
# 01_eda_mvp.R — Exploratory Data Analysis (EDA) MVP Pipeline
# Proyek   : KAGGLE_AUTO_WIN (Praktikum Sains Data STA1581)
# Tujuan   : Audit skema data, ketimpangan kelas Y, distribusi X1-X11,
#            deteksi anomali pembulatan 3-bulanan, dan ekspor grafik EDA.
# ==============================================================================

suppressPackageStartupMessages({
  library(data.table)
  library(ggplot2)
  library(nanoparquet)
})

cat("==============================================================\n")
cat("[STEP 1/4] MEMULAI EXPLORATORY DATA ANALYSIS (EDA) MVP\n")
cat("==============================================================\n")

# 1. Resolusi Path (100% Apache Parquet — Kompatibel Lokal & Kaggle Kernel)
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

out_dir <- "eda_script/output"
if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE)

train_dt <- as.data.table(read_parquet(train_path))
test_dt  <- as.data.table(read_parquet(test_path))

cat(sprintf("- Data Latih dimuat : %d baris x %d kolom\n", nrow(train_dt), ncol(train_dt)))
cat(sprintf("- Data Uji dimuat   : %d baris x %d kolom\n", nrow(test_dt), ncol(test_dt)))

# 2. Kamus Label Fitur
feature_cols <- paste0("X", 1:11)
feature_descriptions <- c(
  X1  = "Durasi Panggilan / bln (menit)",
  X2  = "Penghasilan Panggilan / bln (Rp)",
  X3  = "Frekuensi Panggilan / bln",
  X4  = "Persentase Panggilan Jam Kerja",
  X5  = "Persentase Panggilan Sesama Operator",
  X6  = "Penghasilan SMS / bln (Rp)",
  X7  = "Persentase SMS Sesama Operator",
  X8  = "Pemakaian Data / bln (MB)",
  X9  = "Penghasilan Data / bln (Rp)",
  X10 = "Frekuensi Recharge Pulsa / bln",
  X11 = "Banyaknya Wilayah Dikunjungi / bln"
)

# 3. Analisis Proporsi Kelas Target (Y)
class_summary <- train_dt[, .(count = .N), by = Y][
  order(Y)
][, prop_pct := round(count / sum(count) * 100, 2)]

cat("\n[Distribusi Kelas Target Y]\n")
print(class_summary)
write_parquet(as.data.frame(class_summary), file.path(out_dir, "eda_class_balance.parquet"))

# 4. Ringkasan Statistik Fitur X1-X11 (Train vs Test & Y=0 vs Y=1)
summary_list <- lapply(feature_cols, function(col) {
  v_tr <- train_dt[[col]]
  v_te <- test_dt[[col]]
  v_y0 <- train_dt[Y == 0, get(col)]
  v_y1 <- train_dt[Y == 1, get(col)]
  wt   <- suppressWarnings(wilcox.test(v_y1, v_y0))

  data.table(
    feature         = col,
    description     = feature_descriptions[[col]],
    train_missing   = sum(is.na(v_tr)),
    test_missing    = sum(is.na(v_te)),
    train_zeros_pct = round(mean(v_tr == 0) * 100, 2),
    test_zeros_pct  = round(mean(v_te == 0) * 100, 2),
    train_min       = min(v_tr),
    train_median    = median(v_tr),
    train_mean      = round(mean(v_tr), 3),
    train_sd        = round(sd(v_tr), 3),
    train_max       = max(v_tr),
    test_mean       = round(mean(v_te), 3),
    mean_Y0         = round(mean(v_y0), 3),
    mean_Y1         = round(mean(v_y1), 3),
    ratio_Y1_to_Y0  = round(mean(v_y1) / (mean(v_y0) + 1e-9), 3),
    wilcox_p_value  = signif(wt$p.value, 4)
  )
})
eda_summary_dt <- rbindlist(summary_list)

cat("\n[Ringkasan Statistik Fitur & Perbandingan Kelas]\n")
print(eda_summary_dt[, .(feature, train_zeros_pct, mean_Y0, mean_Y1, ratio_Y1_to_Y0, wilcox_p_value)])
write_parquet(as.data.frame(eda_summary_dt), file.path(out_dir, "eda_feature_summary.parquet"))

# 5. Audit Anomali & Karakteristik Agregasi 3 Bulanan
anomalies_dt <- data.table(
  check_name = c(
    "X3 == 0 tapi (X1 > 0 atau X2 > 0) [Efek Pembulatan Frekuensi Panggilan]",
    "X1 == 0 tapi X3 > 0 [Efek Pembulatan Durasi Panggilan < 30 dtk]",
    "X9 == 0 tapi X8 > 0 [Kuota Data Gratis / Paket Bundling]",
    "X8 == 0 (Pelanggan Non-Pengguna Internet Seluler)",
    "X10 == 0 (Tidak Pernah Recharge di Jendela Observasi)",
    "X11 == 0 (Tidak Terdeteksi Kunjungan Wilayah)"
  ),
  train_count = c(
    train_dt[X3 == 0 & (X1 > 0 | X2 > 0), .N],
    train_dt[X1 == 0 & X3 > 0, .N],
    train_dt[X9 == 0 & X8 > 0, .N],
    train_dt[X8 == 0, .N],
    train_dt[X10 == 0, .N],
    train_dt[X11 == 0, .N]
  ),
  test_count = c(
    test_dt[X3 == 0 & (X1 > 0 | X2 > 0), .N],
    test_dt[X1 == 0 & X3 > 0, .N],
    test_dt[X9 == 0 & X8 > 0, .N],
    test_dt[X8 == 0, .N],
    test_dt[X10 == 0, .N],
    test_dt[X11 == 0, .N]
  )
)
anomalies_dt[, train_pct := round(train_count / nrow(train_dt) * 100, 2)]
anomalies_dt[, test_pct  := round(test_count / nrow(test_dt) * 100, 2)]

cat("\n[Audit Anomali & Zero-Inflation]\n")
print(anomalies_dt)
write_parquet(as.data.frame(anomalies_dt), file.path(out_dir, "eda_anomalies_report.parquet"))

# 6. Visualisasi 1: Distribusi Kelas Target Y
p1 <- ggplot(class_summary, aes(x = factor(Y, labels = c("0 (Non-Potensial)", "1 (Potensial Bank)")),
                                y = count, fill = factor(Y))) +
  geom_col(width = 0.55, show.legend = FALSE) +
  geom_text(aes(label = sprintf("%s (%s%%)", format(count, big.mark = "."), prop_pct)),
            vjust = -0.5, fontface = "bold", size = 4) +
  scale_fill_manual(values = c("#3b82f6", "#ef4444")) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.12))) +
  labs(
    title    = "Distribusi Kelas Target (Y) pada Data Latih (N = 45.000)",
    subtitle = "Ketimpangan kelas 81,33% vs 18,67% menuntut optimasi threshold untuk metrik Balanced Accuracy",
    x        = "Kelas Target (Y)",
    y        = "Jumlah Pelanggan"
  ) +
  theme_minimal(base_size = 12)

ggsave(file.path(out_dir, "01_target_distribution.png"), p1, width = 8, height = 5, dpi = 150)

# 7. Visualisasi 2: Distribusi Log1p Fitur berdasarkan Kelas Y
long_dt <- melt(
  train_dt[, c("ID", "Y", feature_cols), with = FALSE],
  id.vars = c("ID", "Y"),
  variable.name = "Feature",
  value.name = "Value"
)
long_dt[, LogValue := log10(Value + 1)]
long_dt[, Kelas := ifelse(Y == 1, "Y = 1 (Potensial)", "Y = 0 (Non-Potensial)")]

p2 <- ggplot(long_dt, aes(x = Kelas, y = LogValue, fill = Kelas)) +
  geom_boxplot(outlier.size = 0.3, outlier.alpha = 0.2, alpha = 0.85) +
  facet_wrap(~ Feature, scales = "free_y", ncol = 4) +
  scale_fill_manual(values = c("Y = 0 (Non-Potensial)" = "#60a5fa", "Y = 1 (Potensial)" = "#f87171")) +
  labs(
    title    = "Perbandingan Distribusi Fitur X1-X11 Berdasarkan Kelas Target Y",
    subtitle = "Transformasi log10(X + 1) untuk menangani kecondongan (right-skewness)",
    x        = NULL,
    y        = "log10(Nilai + 1)"
  ) +
  theme_minimal(base_size = 10) +
  theme(legend.position = "bottom", axis.text.x = element_blank())

ggsave(file.path(out_dir, "02_feature_boxplots_by_target.png"), p2, width = 12, height = 8, dpi = 150)

# 8. Visualisasi 3: Matriks Korelasi Spearman
cor_mat <- cor(train_dt[, c(feature_cols, "Y"), with = FALSE], method = "spearman")
cor_dt  <- as.data.table(as.table(cor_mat))
setnames(cor_dt, c("Var1", "Var2", "Correlation"))

p3 <- ggplot(cor_dt, aes(x = Var1, y = Var2, fill = Correlation)) +
  geom_tile(color = "white") +
  geom_text(aes(label = sprintf("%.2f", Correlation)), size = 2.8) +
  scale_fill_gradient2(low = "#2563eb", mid = "white", high = "#dc2626", midpoint = 0, limits = c(-1, 1)) +
  labs(
    title    = "Matriks Korelasi Spearman Antar-Fitur (X1-X11) dan Target (Y)",
    subtitle = "Korelasi kuat terlihat pada klaster Panggilan (X1-X3), SMS (X6-X7), dan Data (X8-X9)",
    x        = NULL,
    y        = NULL
  ) +
  theme_minimal(base_size = 11)

ggsave(file.path(out_dir, "03_correlation_heatmap.png"), p3, width = 9, height = 7.5, dpi = 150)

cat(sprintf("\n[SELESAI STEP 1/4] Seluruh tabel & plot EDA tersimpan di: %s/\n\n", out_dir))
