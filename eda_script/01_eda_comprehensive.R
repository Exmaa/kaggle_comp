# ==============================================================================
# KAGGLE_AUTO_WIN — Project Praktikum Sains Data (STA1581)
# Script      : eda_script/01_eda_comprehensive.R
# Deskripsi   : Exploratory Data Analysis (EDA) Komprehensif pada Data Training
#               (45.000 baris) dan Data Testing (5.000 baris)
# Output      : - Tabel statistik & uji hipotesis di eda_script/output/
#               - Visualisasi beresolusi tinggi (presentasi-ready) di eda_script/plots/
# ==============================================================================

suppressPackageStartupMessages({
  library(tidyverse)
  library(patchwork)
  library(scales)
  library(nanoparquet)
})

# Set working directory otomatis ke root project jika dijalankan dari subfolder
if (basename(getwd()) == "eda_script") {
  setwd("..")
}

# Buat folder output dan plots di dalam eda_script/
dir.create("eda_script/output", recursive = TRUE, showWarnings = FALSE)
dir.create("eda_script/plots", recursive = TRUE, showWarnings = FALSE)

cat("======================================================================\n")
cat("  1. MEMUAT DATA TRAINING & TESTING (PARQUET)\n")
cat("======================================================================\n")

train_df <- as.data.frame(read_parquet("data/training_data_tugas_praktikum_DataScience.parquet"))
test_df  <- as.data.frame(read_parquet("data/testing_data_tugas_praktikum_DataScience.parquet"))

cat(sprintf("Dimensi Train : %d baris x %d kolom\n", nrow(train_df), ncol(train_df)))
cat(sprintf("Dimensi Test  : %d baris x %d kolom\n", nrow(test_df), ncol(test_df)))

feature_cols <- paste0("X", 1:11)
feature_labels <- c(
  X1  = "X1: Durasi Panggilan (menit/bln)",
  X2  = "X2: Revenue Panggilan (Rp/bln)",
  X3  = "X3: Frekuensi Panggilan (kali/bln)",
  X4  = "X4: Proporsi Panggilan Jam Kerja",
  X5  = "X5: Proporsi Panggilan On-Net",
  X6  = "X6: Revenue SMS (Rp/bln)",
  X7  = "X7: Proporsi SMS On-Net",
  X8  = "X8: Pemakaian Data (MB/bln)",
  X9  = "X9: Revenue Data (Rp/bln)",
  X10 = "X10: Frekuensi Recharge (kali/bln)",
  X11 = "X11: Jumlah Wilayah Dikunjungi (/bln)"
)

# ==============================================================================
# 2. PEMERIKSAAN KUALITAS DATA, DUPLIKAT & JEBAKAN ID (TARGET LEAKAGE)
# ==============================================================================
cat("\n======================================================================\n")
cat("  2. AUDIT KUALITAS DATA, DUPLIKAT & URUTAN ID\n")
cat("======================================================================\n")

na_train <- sum(is.na(train_df))
na_test  <- sum(is.na(test_df))
dup_train_all  <- sum(duplicated(train_df[, c(feature_cols, "Y")]))
dup_train_feat <- sum(duplicated(train_df[, feature_cols]))

# Cek rentang ID untuk masing-masing kelas Y
id_y0 <- range(train_df$ID[train_df$Y == 0])
id_y1 <- range(train_df$ID[train_df$Y == 1])

cat(sprintf("Total Missing Values (Train / Test) : %d / %d\n", na_train, na_test))
cat(sprintf("Duplikat Fitur Identik di Train     : %d baris (dari 45.000)\n", dup_train_feat))
cat(sprintf("Rentang ID Kelas Y=0 di Train       : %d s.d. %d (N = %d)\n",
            id_y0[1], id_y0[2], sum(train_df$Y == 0)))
cat(sprintf("Rentang ID Kelas Y=1 di Train       : %d s.d. %d (N = %d)\n",
            id_y1[1], id_y1[2], sum(train_df$Y == 1)))

# ==============================================================================
# 3. TABEL RINGKASAN STATISTIK TRAIN VS TEST & DRIFT TEST (KS-TEST)
# ==============================================================================
cat("\n======================================================================\n")
cat("  3. STATISTIK DESKRIPTIF & UJI KESTABILAN DISTRIBUSI (TRAIN VS TEST)\n")
cat("======================================================================\n")

calc_summary <- function(df, dataset_name) {
  map_dfr(feature_cols, function(col) {
    x <- df[[col]]
    tibble(
      Dataset   = dataset_name,
      Variabel  = col,
      Deskripsi = unname(feature_labels[col]),
      N         = length(x),
      Missing   = sum(is.na(x)),
      Zero_Cnt  = sum(x == 0),
      Zero_Pct  = round(mean(x == 0) * 100, 2),
      Mean      = round(mean(x), 4),
      SD        = round(sd(x), 4),
      Min       = round(min(x), 4),
      Q25       = round(quantile(x, 0.25, names = FALSE), 4),
      Median    = round(median(x), 4),
      Q75       = round(quantile(x, 0.75, names = FALSE), 4),
      Max       = round(max(x), 4),
      IQR       = round(IQR(x), 4),
      Skewness  = round(mean((x - mean(x))^3) / (sd(x)^3), 4)
    )
  })
}

summary_train <- calc_summary(train_df, "Train")
summary_test  <- calc_summary(test_df, "Test")
summary_all   <- bind_rows(summary_train, summary_test)
write_parquet(as.data.frame(summary_all), "eda_script/output/01_summary_statistics_train_test.parquet")

# Perbandingan Kelas Y=0 vs Y=1 + Uji Statistik (Wilcoxon, Korelasi, KS Train-Test)
class_comparison <- map_dfr(feature_cols, function(col) {
  x  <- train_df[[col]]
  y  <- train_df$Y
  x0 <- x[y == 0]
  x1 <- x[y == 1]
  xt <- test_df[[col]]
  
  wt <- suppressWarnings(wilcox.test(x1, x0))
  ks_class <- suppressWarnings(ks.test(x1, x0))
  ks_drift <- suppressWarnings(ks.test(x, xt))
  
  tibble(
    Variabel          = col,
    Deskripsi         = unname(feature_labels[col]),
    Mean_Y0           = round(mean(x0), 4),
    Mean_Y1           = round(mean(x1), 4),
    Diff_Mean_Pct     = round((mean(x1) - mean(x0)) / pmax(abs(mean(x0)), 1e-9) * 100, 2),
    Median_Y0         = round(median(x0), 4),
    Median_Y1         = round(median(x1), 4),
    Q25_Y0            = round(quantile(x0, 0.25, names = FALSE), 4),
    Q75_Y0            = round(quantile(x0, 0.75, names = FALSE), 4),
    Q25_Y1            = round(quantile(x1, 0.25, names = FALSE), 4),
    Q75_Y1            = round(quantile(x1, 0.75, names = FALSE), 4),
    Pearson_r         = round(cor(x, y, method = "pearson"), 4),
    Spearman_rho      = round(cor(x, y, method = "spearman"), 4),
    Wilcoxon_p_value  = signif(wt$p.value, 4),
    KS_Class_Stat     = round(unname(ks_class$statistic), 4),
    KS_TrainTest_Stat = round(unname(ks_drift$statistic), 4),
    KS_TrainTest_pval = round(ks_drift$p.value, 4)
  )
}) %>%
  arrange(desc(abs(Spearman_rho)))

write_parquet(as.data.frame(class_comparison), "eda_script/output/02_class_comparison_and_stat_tests.parquet")
print(class_comparison %>% select(Variabel, Mean_Y0, Mean_Y1, Median_Y0, Median_Y1, Pearson_r, Spearman_rho, KS_Class_Stat))

# ==============================================================================
# 4. AUDIT ZERO-INFLATION, POLA DESIMAL 1/3, & ANOMALI DOMAIN
# ==============================================================================
cat("\n======================================================================\n")
cat("  4. AUDIT ANOMALI DOMAIN & SEGMEN KHUSUS\n")
cat("======================================================================\n")

segment_audit <- bind_rows(
  train_df %>%
    summarise(
      Segmen = "1. Non-Data Users (X8 == 0 & X9 == 0)",
      Train_N = sum(X8 == 0 & X9 == 0),
      Train_Pct = round(mean(X8 == 0 & X9 == 0) * 100, 2),
      Test_N = sum(test_df$X8 == 0 & test_df$X9 == 0),
      Test_Pct = round(mean(test_df$X8 == 0 & test_df$X9 == 0) * 100, 2),
      Y1_Rate_Pct = round(mean(Y[X8 == 0 & X9 == 0]) * 100, 2)
    ),
  train_df %>%
    summarise(
      Segmen = "2. Active Data Users (X8 > 0 | X9 > 0)",
      Train_N = sum(X8 > 0 | X9 > 0),
      Train_Pct = round(mean(X8 > 0 | X9 > 0) * 100, 2),
      Test_N = sum(test_df$X8 > 0 | test_df$X9 > 0),
      Test_Pct = round(mean(test_df$X8 > 0 | test_df$X9 > 0) * 100, 2),
      Y1_Rate_Pct = round(mean(Y[X8 > 0 | X9 > 0]) * 100, 2)
    ),
  train_df %>%
    summarise(
      Segmen = "3. Durasi Bulat 0 tapi Ada Panggilan (X1 == 0 & X3 > 0)",
      Train_N = sum(X1 == 0 & X3 > 0),
      Train_Pct = round(mean(X1 == 0 & X3 > 0) * 100, 2),
      Test_N = sum(test_df$X1 == 0 & test_df$X3 > 0),
      Test_Pct = round(mean(test_df$X1 == 0 & test_df$X3 > 0) * 100, 2),
      Y1_Rate_Pct = round(mean(Y[X1 == 0 & X3 > 0]) * 100, 2)
    ),
  train_df %>%
    summarise(
      Segmen = "4. Frekuensi Bulat 0 tapi Durasi > 0 (X3 == 0 & X1 > 0)",
      Train_N = sum(X3 == 0 & X1 > 0),
      Train_Pct = round(mean(X3 == 0 & X1 > 0) * 100, 2),
      Test_N = sum(test_df$X3 == 0 & test_df$X1 > 0),
      Test_Pct = round(mean(test_df$X3 == 0 & test_df$X1 > 0) * 100, 2),
      Y1_Rate_Pct = round(mean(Y[X3 == 0 & X1 > 0]) * 100, 2)
    ),
  train_df %>%
    summarise(
      Segmen = "5. Tanpa Panggilan sama sekali (X1 == 0 & X2 == 0 & X3 == 0)",
      Train_N = sum(X1 == 0 & X2 == 0 & X3 == 0),
      Train_Pct = round(mean(X1 == 0 & X2 == 0 & X3 == 0) * 100, 2),
      Test_N = sum(test_df$X1 == 0 & test_df$X2 == 0 & test_df$X3 == 0),
      Test_Pct = round(mean(test_df$X1 == 0 & test_df$X2 == 0 & test_df$X3 == 0) * 100, 2),
      Y1_Rate_Pct = round(mean(Y[X1 == 0 & X2 == 0 & X3 == 0]) * 100, 2)
    ),
  train_df %>%
    summarise(
      Segmen = "6. Tanpa Panggilan tapi X4 == 1 & X5 == 1 (Anomali Encoding)",
      Train_N = sum(X3 == 0 & X4 == 1 & X5 == 1),
      Train_Pct = round(mean(X3 == 0 & X4 == 1 & X5 == 1) * 100, 2),
      Test_N = sum(test_df$X3 == 0 & test_df$X4 == 1 & test_df$X5 == 1),
      Test_Pct = round(mean(test_df$X3 == 0 & test_df$X4 == 1 & test_df$X5 == 1) * 100, 2),
      Y1_Rate_Pct = round(mean(Y[X3 == 0 & X4 == 1 & X5 == 1]) * 100, 2)
    ),
  train_df %>%
    summarise(
      Segmen = "7. Tanpa SMS sama sekali (X6 == 0)",
      Train_N = sum(X6 == 0),
      Train_Pct = round(mean(X6 == 0) * 100, 2),
      Test_N = sum(test_df$X6 == 0),
      Test_Pct = round(mean(test_df$X6 == 0) * 100, 2),
      Y1_Rate_Pct = round(mean(Y[X6 == 0]) * 100, 2)
    ),
  train_df %>%
    summarise(
      Segmen = "8. Pemakaian Data Gratis / Bonus (X8 > 0 & X9 == 0)",
      Train_N = sum(X8 > 0 & X9 == 0),
      Train_Pct = round(mean(X8 > 0 & X9 == 0) * 100, 2),
      Test_N = sum(test_df$X8 > 0 & test_df$X9 == 0),
      Test_Pct = round(mean(test_df$X8 > 0 & test_df$X9 == 0) * 100, 2),
      Y1_Rate_Pct = round(mean(Y[X8 > 0 & X9 == 0]) * 100, 2)
    )
)

write_parquet(as.data.frame(segment_audit), "eda_script/output/03_zero_inflation_and_anomalies.parquet")
print(segment_audit)

# Simpan matriks korelasi
cor_mat <- round(cor(train_df[, c(feature_cols, "Y")], method = "spearman"), 4)
cor_df  <- cbind(Variabel = rownames(cor_mat), as.data.frame(cor_mat))
write_parquet(cor_df, "eda_script/output/04_spearman_correlation_matrix.parquet")

# ==============================================================================
# 5. VISUALISASI EDA (PRESENTATION-READY PLOTS)
# ==============================================================================
cat("\n======================================================================\n")
cat("  5. MEMBUAT VISUALISASI EDA BERESOLUSI TINGGI\n")
cat("======================================================================\n")

theme_eda <- theme_minimal(base_size = 11) +
  theme(
    plot.title       = element_text(face = "bold", size = 13, color = "#1f2937"),
    plot.subtitle    = element_text(size = 10, color = "#4b5563"),
    strip.text       = element_text(face = "bold", size = 10, color = "#111827"),
    strip.background = element_rect(fill = "#f3f4f6", color = NA),
    panel.grid.minor = element_blank(),
    legend.position  = "top"
  )

# ------------------------------------------------------------------------------
# FIG 01: Target Class Imbalance & Jebakan Urutan ID (Target Leakage)
# ------------------------------------------------------------------------------
p1a <- train_df %>%
  count(Y) %>%
  mutate(
    Label = ifelse(Y == 1, "Y = 1 (Potensial Bank)", "Y = 0 (Non-Bank)"),
    Pct   = n / sum(n) * 100
  ) %>%
  ggplot(aes(x = Label, y = n, fill = factor(Y))) +
  geom_col(width = 0.55, show.legend = FALSE) +
  geom_text(aes(label = sprintf("%s\n(%.2f%%)", comma(n), Pct)),
            vjust = -0.3, fontface = "bold", size = 3.8) +
  scale_fill_manual(values = c("0" = "#3b82f6", "1" = "#ef4444")) +
  scale_y_continuous(labels = comma, expand = expansion(mult = c(0, 0.18))) +
  labs(
    title = "A. Proporsi Kelas Target (Y) pada Data Training",
    subtitle = "Class Imbalance: Rasio Kelas 0 vs Kelas 1 adalah 4,36 : 1",
    x = NULL, y = "Jumlah Pelanggan"
  ) +
  theme_eda

p1b <- train_df %>%
  mutate(ID_Bin = cut(ID, breaks = seq(0, 45000, by = 4500), include.lowest = TRUE,
                      labels = paste0("D", 1:10, "\n", seq(0, 40.5, 4.5), "k-", seq(4.5, 45, 4.5), "k"))) %>%
  group_by(ID_Bin) %>%
  summarise(Y_Rate = mean(Y) * 100, .groups = "drop") %>%
  ggplot(aes(x = ID_Bin, y = Y_Rate, fill = Y_Rate > 0)) +
  geom_col(width = 0.65, show.legend = FALSE) +
  geom_text(aes(label = sprintf("%.1f%%", Y_Rate)), vjust = -0.4, fontface = "bold", size = 3.3) +
  scale_fill_manual(values = c("FALSE" = "#3b82f6", "TRUE" = "#ef4444")) +
  scale_y_continuous(labels = function(x) paste0(x, "%"), limits = c(0, 115)) +
  labs(
    title = "B. Temuan Kritis: Urutan ID Training Terurut Penuh pada Y!",
    subtitle = "ID 1–36.598 bernilai Y=0 (100%), ID 36.599–45.000 bernilai Y=1 (100%) -> Wajib Drop ID",
    x = "Desil Nomor Urut ID (Data Training)", y = "Persentase Y = 1 (%)"
  ) +
  theme_eda

fig01 <- p1a + p1b + plot_layout(widths = c(1, 1.55))
ggsave("eda_script/plots/fig01_target_distribution_and_id_leakage.png", fig01, width = 13.5, height = 5.4, dpi = 300)

# ------------------------------------------------------------------------------
# FIG 02: Distribusi 11 Fitur Prediktor (X1-X11) Berdasarkan Kelas Y
# ------------------------------------------------------------------------------
# Gunakan transformasi log10(x + 1) untuk fitur uang/volume yang sangat skewed
skewed_cols <- c("X1", "X2", "X3", "X6", "X8", "X9", "X10", "X11")

train_long_plot <- train_df %>%
  mutate(Kelas = factor(ifelse(Y == 1, "Y=1 (Bank)", "Y=0 (Non-Bank)"),
                        levels = c("Y=0 (Non-Bank)", "Y=1 (Bank)"))) %>%
  select(Kelas, all_of(feature_cols)) %>%
  pivot_longer(cols = all_of(feature_cols), names_to = "Fitur", values_to = "Nilai") %>%
  mutate(
    Fitur_Label = factor(feature_labels[Fitur], levels = unname(feature_labels)),
    Nilai_Trans = ifelse(Fitur %in% skewed_cols, log10(Nilai + 1), Nilai)
  )

fig02 <- ggplot(train_long_plot, aes(x = Kelas, y = Nilai_Trans, fill = Kelas)) +
  geom_violin(alpha = 0.35, color = NA, trim = TRUE) +
  geom_boxplot(width = 0.24, outlier.size = 0.3, outlier.alpha = 0.2, alpha = 0.85) +
  facet_wrap(~ Fitur_Label, scales = "free_y", ncol = 4) +
  scale_fill_manual(values = c("Y=0 (Non-Bank)" = "#3b82f6", "Y=1 (Bank)" = "#ef4444")) +
  labs(
    title = "Perbandingan Distribusi Fitur X1–X11 antara Kelas Y=0 vs Y=1",
    subtitle = "Catatan: Fitur X1, X2, X3, X6, X8, X9, X10, X11 ditampilkan dalam skala log10(x + 1) karena berdistribusi right-skewed ekstrem",
    x = NULL, y = "Nilai (atau log10(Nilai + 1))", fill = "Target"
  ) +
  theme_eda

ggsave("eda_script/plots/fig02_feature_distributions_by_class.png", fig02, width = 15, height = 10, dpi = 300)

# ------------------------------------------------------------------------------
# FIG 03: Korelasi Fitur dengan Target Y & Heatmap Korelasi Antar-Fitur
# ------------------------------------------------------------------------------
cor_long <- class_comparison %>%
  select(Variabel, Deskripsi, Pearson_r, Spearman_rho) %>%
  pivot_longer(cols = c(Pearson_r, Spearman_rho), names_to = "Metode", values_to = "Korelasi") %>%
  mutate(
    Metode = ifelse(Metode == "Pearson_r", "Pearson (Linear)", "Spearman (Rank Monotonik)"),
    Deskripsi = fct_reorder(Deskripsi, abs(Korelasi), .fun = max)
  )

p3a <- ggplot(cor_long, aes(x = Korelasi, y = Deskripsi, fill = Metode)) +
  geom_col(position = position_dodge(width = 0.75), width = 0.68) +
  geom_vline(xintercept = 0, linetype = "dashed", color = "gray40") +
  geom_text(aes(label = sprintf("%+.3f", Korelasi),
                hjust = ifelse(Korelasi >= 0, -0.15, 1.15)),
            position = position_dodge(width = 0.75), size = 3.0, fontface = "bold") +
  scale_fill_manual(values = c("Pearson (Linear)" = "#2563eb", "Spearman (Rank Monotonik)" = "#dc2626")) +
  scale_x_continuous(limits = c(-0.32, 0.35)) +
  labs(
    title = "A. Korelasi Fitur terhadap Target Y",
    subtitle = "X9 & X8 berkorelasi positif kuat; X10 & X11 berkorelasi negatif",
    x = "Koefisien Korelasi", y = NULL, fill = "Metode Korelasi"
  ) +
  theme_eda

cor_df_heat <- as.data.frame(as.table(cor_mat)) %>%
  rename(Var1 = Var1, Var2 = Var2, Corr = Freq)

p3b <- ggplot(cor_df_heat, aes(x = Var1, y = fct_rev(Var2), fill = Corr)) +
  geom_tile(color = "white") +
  geom_text(aes(label = sprintf("%.2f", Corr)), size = 2.8, fontface = "bold",
            color = ifelse(abs(cor_df_heat$Corr) > 0.5, "white", "#111827")) +
  scale_fill_gradient2(low = "#2563eb", mid = "white", high = "#dc2626", midpoint = 0, limits = c(-1, 1)) +
  labs(
    title = "B. Matriks Korelasi Spearman Antar-Variabel",
    subtitle = "Multikolinearitas tinggi pada (X1, X2, X3), (X8, X9), dan (X10, X11)",
    x = NULL, y = NULL, fill = "Spearman r"
  ) +
  theme_eda

fig03 <- p3a + p3b + plot_layout(widths = c(1.15, 1))
ggsave("eda_script/plots/fig03_correlation_and_feature_associations.png", fig03, width = 15.5, height = 6.4, dpi = 300)

# ------------------------------------------------------------------------------
# FIG 04: Analisis Konversi (Y=1 Rate) Berdasarkan Segmen Perilaku Utama
# ------------------------------------------------------------------------------
seg_data <- train_df %>%
  mutate(
    Seg_Data = case_when(
      X8 == 0 & X9 == 0 ~ "Non-Data (0 MB)",
      X9 <= 10000       ~ "Data Ringan (<=Rp10rb)",
      X9 <= 30000       ~ "Data Sedang (Rp10rb-30rb)",
      X9 <= 60000       ~ "Data Tinggi (Rp30rb-60rb)",
      TRUE              ~ "Data Heavy (>Rp60rb)"
    ),
    Seg_Data = factor(Seg_Data, levels = c(
      "Non-Data (0 MB)", "Data Ringan (<=Rp10rb)", "Data Sedang (Rp10rb-30rb)",
      "Data Tinggi (Rp30rb-60rb)", "Data Heavy (>Rp60rb)"
    )),
    Seg_Recharge = case_when(
      X10 <= 3   ~ "Jarang (<=3x/bln)",
      X10 <= 6   ~ "Sedang (3.3-6x/bln)",
      X10 <= 10  ~ "Sering (6.3-10x/bln)",
      TRUE       ~ "Eceran Ekstrem (>10x/bln)"
    ),
    Seg_Recharge = factor(Seg_Recharge, levels = c(
      "Jarang (<=3x/bln)", "Sedang (3.3-6x/bln)",
      "Sering (6.3-10x/bln)", "Eceran Ekstrem (>10x/bln)"
    )),
    Seg_Mobility = case_when(
      X11 <= 2   ~ "Menetap (<=2 wil/bln)",
      X11 <= 4   ~ "Mobilitas Rendah (2.3-4)",
      X11 <= 7   ~ "Mobilitas Sedang (4.3-7)",
      TRUE       ~ "Mobilitas Tinggi (>7)"
    ),
    Seg_Mobility = factor(Seg_Mobility, levels = c(
      "Menetap (<=2 wil/bln)", "Mobilitas Rendah (2.3-4)",
      "Mobilitas Sedang (4.3-7)", "Mobilitas Tinggi (>7)"
    ))
  )

plot_seg <- function(df, col_sym, title_str, fill_col) {
  df %>%
    group_by({{ col_sym }}) %>%
    summarise(N = n(), Rate = mean(Y) * 100, .groups = "drop") %>%
    ggplot(aes(x = {{ col_sym }}, y = Rate)) +
    geom_col(fill = fill_col, width = 0.6) +
    geom_hline(yintercept = mean(train_df$Y) * 100, linetype = "dashed", color = "#111827") +
    geom_text(aes(label = sprintf("%.1f%%\n(n=%s)", Rate, comma(N))),
              vjust = -0.25, size = 3.2, fontface = "bold") +
    scale_y_continuous(limits = c(0, 48), labels = function(x) paste0(x, "%")) +
    labs(title = title_str, x = NULL, y = "Proporsi Y = 1 (%)") +
    theme_eda +
    theme(axis.text.x = element_text(angle = 12, hjust = 1))
}

p4a <- plot_seg(seg_data, Seg_Data, "A. Belanja Data (X9)", "#dc2626")
p4b <- plot_seg(seg_data, Seg_Recharge, "B. Frekuensi Isi Ulang (X10)", "#2563eb")
p4c <- plot_seg(seg_data, Seg_Mobility, "C. Mobilitas Wilayah (X11)", "#059669")

fig04 <- p4a + p4b + p4c +
  plot_annotation(
    title = "Profil Peluang Adopsi Channel Bank (Y=1) Berdasarkan Segmen Perilaku Kunci",
    subtitle = "Garis putus-putus menunjukkan rata-rata populasi (Baseline Y=1 = 18,67%)",
    theme = theme(
      plot.title = element_text(face = "bold", size = 14),
      plot.subtitle = element_text(size = 11, color = "#4b5563")
    )
  )
ggsave("eda_script/plots/fig04_zero_inflation_and_segment_conversion.png", fig04, width = 15, height = 5.5, dpi = 300)

# ------------------------------------------------------------------------------
# FIG 05: Perbandingan Distribusi Data Training vs Testing (Covariate Shift Check)
# ------------------------------------------------------------------------------
combined_long <- bind_rows(
  train_df %>% select(all_of(feature_cols)) %>% mutate(Dataset = "Train (N=45.000)"),
  test_df  %>% select(all_of(feature_cols)) %>% mutate(Dataset = "Test (N=5.000)")
) %>%
  pivot_longer(cols = all_of(feature_cols), names_to = "Fitur", values_to = "Nilai") %>%
  mutate(
    Fitur_Label = factor(feature_labels[Fitur], levels = unname(feature_labels)),
    Nilai_Trans = ifelse(Fitur %in% skewed_cols, log10(Nilai + 1), Nilai)
  )

fig05 <- ggplot(combined_long, aes(x = Nilai_Trans, color = Dataset, fill = Dataset)) +
  geom_density(alpha = 0.25, linewidth = 0.8) +
  facet_wrap(~ Fitur_Label, scales = "free", ncol = 4) +
  scale_color_manual(values = c("Train (N=45.000)" = "#1d4ed8", "Test (N=5.000)" = "#d97706")) +
  scale_fill_manual(values  = c("Train (N=45.000)" = "#1d4ed8", "Test (N=5.000)" = "#d97706")) +
  labs(
    title = "Pemeriksaan Covariate Shift: Distribusi Fitur Data Training vs Data Testing",
    subtitle = "Kurva densitas Train dan Test berhimpit sangat rapat di seluruh 11 variabel (Uji KS p-value > 0.05 pada mayoritas fitur)",
    x = "Nilai (atau log10(Nilai + 1))", y = "Densitas", color = "Dataset", fill = "Dataset"
  ) +
  theme_eda

ggsave("eda_script/plots/fig05_train_vs_test_distribution_check.png", fig05, width = 15, height = 9.5, dpi = 300)

# ------------------------------------------------------------------------------
# FIG 06: Insight Multivariat untuk Feature Engineering (ARPU & Ticket Size)
# ------------------------------------------------------------------------------
fe_preview <- train_df %>%
  mutate(
    Total_Spend = X2 + X6 + X9,
    Est_Ticket_Size = Total_Spend / pmax(X10, 0.3333),
    Data_Share = ifelse(Total_Spend > 0, X9 / Total_Spend, 0),
    Kelas = factor(ifelse(Y == 1, "Y=1 (Bank)", "Y=0 (Non-Bank)"),
                   levels = c("Y=0 (Non-Bank)", "Y=1 (Bank)"))
  )

p6a <- ggplot(fe_preview, aes(x = Kelas, y = log10(Est_Ticket_Size + 1), fill = Kelas)) +
  geom_violin(alpha = 0.35, color = NA) +
  geom_boxplot(width = 0.25, outlier.alpha = 0.15) +
  scale_fill_manual(values = c("Y=0 (Non-Bank)" = "#3b82f6", "Y=1 (Bank)" = "#ef4444")) +
  labs(
    title = "A. Estimasi Nominal / Isi Ulang",
    subtitle = "Rumus: (X2 + X6 + X9) / X10",
    x = NULL, y = "log10(Nominal per Recharge + 1)"
  ) +
  theme_eda + theme(legend.position = "none")

p6b <- ggplot(fe_preview, aes(x = Kelas, y = Data_Share * 100, fill = Kelas)) +
  geom_violin(alpha = 0.35, color = NA) +
  geom_boxplot(width = 0.25, outlier.alpha = 0.15) +
  scale_fill_manual(values = c("Y=0 (Non-Bank)" = "#3b82f6", "Y=1 (Bank)" = "#ef4444")) +
  labs(
    title = "B. Porsi Pengeluaran Data (%)",
    subtitle = "Rumus: X9 / (X2 + X6 + X9)",
    x = NULL, y = "Porsi Belanja Data (%)"
  ) +
  theme_eda + theme(legend.position = "none")

p6c <- fe_preview %>%
  mutate(Ticket_Bin = cut(Est_Ticket_Size,
                          breaks = c(-Inf, 3000, 6000, 10000, 20000, 50000, Inf),
                          labels = c("<=Rp3rb", "Rp3rb-6rb", "Rp6rb-10rb", "Rp10rb-20rb", "Rp20rb-50rb", ">Rp50rb"))) %>%
  group_by(Ticket_Bin) %>%
  summarise(N = n(), Rate = mean(Y) * 100, .groups = "drop") %>%
  ggplot(aes(x = Ticket_Bin, y = Rate)) +
  geom_col(fill = "#7c3aed", width = 0.6) +
  geom_text(aes(label = sprintf("%.1f%%\n(n=%s)", Rate, comma(N))),
            vjust = -0.25, size = 3.1, fontface = "bold") +
  scale_y_continuous(limits = c(0, 55), labels = function(x) paste0(x, "%")) +
  labs(
    title = "C. Konversi Y=1 per Kategori Ticket Size",
    subtitle = "Memisahkan peluang Y=1 dari 3,8% hingga 42,2%!",
    x = "Estimasi Nominal Sekali Isi Ulang (Rp)", y = "Proporsi Y = 1 (%)"
  ) +
  theme_eda

fig06 <- p6a + p6b + p6c +
  plot_layout(widths = c(1, 1, 1.35)) +
  plot_annotation(
    title = "Sintesis Insight Multivariat: Daya Pisah Fitur Turunan (Feature Engineering Preview)",
    subtitle = "Pelanggan Y=1 berbelanja paket data dominan dan mengisi ulang pulsa dengan nominal per transaksi jauh lebih tinggi",
    theme = theme(
      plot.title = element_text(face = "bold", size = 14),
      plot.subtitle = element_text(size = 11, color = "#4b5563")
    )
  )
ggsave("eda_script/plots/fig06_multivariate_arpu_vs_recharge.png", fig06, width = 15.5, height = 5.6, dpi = 300)

# ==============================================================================
# 6. GENERATE STANDALONE HTML REPORT (DENGAN PENJELASAN SINTAKS, TABEL & PLOT)
# ==============================================================================
cat("\n======================================================================\n")
cat("  6. MEMBUAT LAPORAN HTML STANDALONE (eda_script/01_eda_report.html)\n")
cat("======================================================================\n")

img_uri <- function(path) base64enc::dataURI(file = path, mime = "image/png")

html_content <- paste0(
  "<!DOCTYPE html><html lang='id'><head><meta charset='UTF-8'>",
  "<title>Laporan EDA & Sintaks R — KAGGLE_AUTO_WIN</title>",
  "<style>",
  "body { font-family: 'Segoe UI', Roboto, Helvetica, sans-serif; line-height: 1.65; color: #1f2937; max-width: 1180px; margin: 0 auto; padding: 32px 24px; background: #f9fafb; }",
  ".card { background: #ffffff; border-radius: 10px; padding: 26px 30px; margin-bottom: 28px; box-shadow: 0 2px 8px rgba(0,0,0,0.06); border: 1px solid #e5e7eb; }",
  "h1 { color: #111827; font-size: 28px; margin-bottom: 6px; border-bottom: 3px solid #2563eb; padding-bottom: 10px; }",
  "h2 { color: #1d4ed8; font-size: 21px; margin-top: 0; border-left: 4px solid #2563eb; padding-left: 12px; }",
  "h3 { color: #374151; font-size: 17px; margin-top: 18px; }",
  "pre { background: #1e293b; color: #f8fafc; padding: 14px 18px; border-radius: 8px; overflow-x: auto; font-size: 13px; line-height: 1.5; }",
  "code { font-family: Consolas, Monaco, monospace; }",
  "table { border-collapse: collapse; width: 100%; font-size: 13px; margin: 16px 0; }",
  "th, td { border: 1px solid #d1d5db; padding: 8px 10px; text-align: right; }",
  "th { background: #eff6ff; color: #1e3a8a; font-weight: 600; text-align: center; }",
  "td:first-child, td:nth-child(2) { text-align: left; font-weight: 500; }",
  "tr:nth-child(even) { background: #f8fafc; }",
  "img { max-width: 100%; height: auto; border-radius: 8px; border: 1px solid #e5e7eb; margin: 14px 0; }",
  ".callout { background: #fef2f2; border-left: 4px solid #ef4444; padding: 12px 16px; border-radius: 6px; margin: 14px 0; }",
  ".insight { background: #eff6ff; border-left: 4px solid #2563eb; padding: 12px 16px; border-radius: 6px; margin: 14px 0; }",
  "</style></head><body>",
  "<div class='card'>",
  "<h1>Exploratory Data Analysis (EDA) — Prediksi Potensi Recharge Channel Perbankan</h1>",
  "<p><strong>Proyek:</strong> Praktikum Sains Data (STA1581) — Tim <code>KAGGLE_AUTO_WIN</code><br>",
  "<strong>Dataset:</strong> <code>training_data_tugas_praktikum_DataScience.parquet</code> (45.000 baris) &amp; <code>testing_data_tugas_praktikum_DataScience.parquet</code> (5.000 baris)<br>",
  "<strong>Metrik Evaluasi Kompetisi:</strong> <em>Balanced Accuracy</em></p>",
  "</div>",

  "<div class='card'>",
  "<h2>Tahap 1: Pemuatan Data &amp; Pemeriksaan Kualitas Data (Data Integrity Audit)</h2>",
  "<p>Tahap pertama membaca dataset <em>training</em> dan <em>testing</em> dalam format <code>.parquet</code>, memeriksa nilai hilang (<em>missing values</em>), baris duplikat, serta memeriksa struktur pengurutan kolom <code>ID</code> terhadap variabel respon <code>Y</code>.</p>",
  "<pre><code>train_df &lt;- nanoparquet::read_parquet('data/training_data_tugas_praktikum_DataScience.parquet')\ntest_df  &lt;- nanoparquet::read_parquet('data/testing_data_tugas_praktikum_DataScience.parquet')\nsum(is.na(train_df)); sum(is.na(test_df))\nrange(train_df$ID[train_df$Y == 0]); range(train_df$ID[train_df$Y == 1])</code></pre>",
  "<div class='callout'><strong>Temuan Kritis #1 (Target Leakage pada ID &amp; Class Imbalance):</strong><br>",
  "<ul><li><strong>Tidak ada Missing Values (NA = 0)</strong> di seluruh 45.000 baris data <em>training</em> maupun 5.000 baris data <em>testing</em>.</li>",
  "<li><strong>Class Imbalance (4,36 : 1):</strong> Kelas <code>Y = 0</code> berjumlah <strong>36.598 pelanggan (81,33%)</strong> dan kelas <code>Y = 1</code> berjumlah <strong>8.402 pelanggan (18,67%)</strong>.</li>",
  "<li><strong>Urutan Baris Data Training Terurut Penuh Berdasarkan Y:</strong> Baris <code>ID 1–36.598</code> semuanya bernilai <code>Y = 0</code>, sedangkan baris <code>ID 36.599–45.000</code> semuanya bernilai <code>Y = 1</code>. Sebaliknya, data <em>testing</em> (<code>ID 45.001–50.000</code>) sudah diacak. Oleh karena itu, <strong>kolom ID wajib dibuang dari pemodelan</strong> dan validasi lokal wajib memakai <strong>Stratified K-Fold Cross-Validation yang diacak (shuffled)</strong>.</li></ul></div>",
  "<img src='", img_uri("eda_script/plots/fig01_target_distribution_and_id_leakage.png"), "' alt='Fig 01'>",
  "</div>",

  "<div class='card'>",
  "<h2>Tahap 2: Statistik Deskriptif &amp; Perbandingan Kelas (Y = 0 vs Y = 1)</h2>",
  "<p>Kita menghitung rata-rata, median, kuartil (Q25, Q75), kecondongan (<em>skewness</em>), uji non-parametrik <em>Wilcoxon Rank-Sum</em>, serta statistik <em>Kolmogorov-Smirnov (KS)</em> untuk mengukur daya pisah tiap fitur terhadap kelas <code>Y</code>.</p>",
  "<pre><code># Contoh perhitungan uji perbandingan distribusi kelas dan korelasi\nwilcox.test(x[y == 1], x[y == 0])\nks.test(x[y == 1], x[y == 0])\ncor(x, y, method = 'spearman')</code></pre>",
  knitr::kable(class_comparison, format = "html"),
  "<img src='", img_uri("eda_script/plots/fig02_feature_distributions_by_class.png"), "' alt='Fig 02'>",
  "<div class='insight'><strong>Interpretasi Distribusi Fitur:</strong><br>",
  "<ul><li><strong>X9 (Revenue Data) &amp; X8 (Pemakaian Data MB):</strong> Pembeda terkuat antara kedua kelas. Median belanja data pelanggan <code>Y = 1</code> mencapai <strong>Rp 20.000/bulan</strong> (dibandingkan <code>Y = 0</code> yang hanya <strong>Rp 1.010/bulan</strong> — hampir 20x lipat lebih tinggi!).</li>",
  "<li><strong>X10 (Frekuensi Recharge) &amp; X11 (Wilayah Dikunjungi):</strong> Memiliki hubungan negatif yang kuat dengan <code>Y</code>. Pelanggan <code>Y = 1</code> lebih jarang mengisi pulsa (median <code>4,3 kali/bulan</code> vs <code>6,3 kali/bulan</code>) dan memiliki mobilitas wilayah yang lebih stabil (median <code>2,3 wilayah</code> vs <code>3,3 wilayah</code>).</li>",
  "<li><strong>X2 (Revenue Panggilan) vs X5 (Proporsi On-Net):</strong> Walaupun durasi telepon (<code>X1</code>) pelanggan <code>Y = 1</code> sedikit lebih rendah, pengeluaran telepon (<code>X2</code>) mereka <strong>22% lebih tinggi</strong> karena mereka lebih sering menelepon lintas operator (<em>off-net</em>, <code>1 - X5</code> lebih besar).</li></ul></div>",
  "</div>",

  "<div class='card'>",
  "<h2>Tahap 3: Analisis Korelasi &amp; Multikolinearitas Antar-Variabel</h2>",
  "<p>Mengingat sebagian besar fitur berdistribusi menceng kanan ekstrem (<em>skewness</em> &gt; 3), korelasi peringkat <em>Spearman</em> digunakan bersama korelasi linear <em>Pearson</em> untuk mendeteksi hubungan monotonik dan multikolinearitas.</p>",
  "<pre><code>cor_mat &lt;- round(cor(train_df[, c(paste0('X', 1:11), 'Y')], method = 'spearman'), 4)</code></pre>",
  "<img src='", img_uri("eda_script/plots/fig03_correlation_and_feature_associations.png"), "' alt='Fig 03'>",
  "<div class='insight'><strong>Struktur Multikolinearitas:</strong><br>",
  "Terdapat 3 klaster fitur yang saling berkorelasi tinggi: (1) Klaster Suara <code>X1, X2, X3</code> (&rho; = 0,68–0,82), (2) Klaster Data <code>X8, X9</code> (&rho; = 0,90), dan (3) Klaster Isi Ulang &amp; Mobilitas <code>X10, X11</code> (&rho; = 0,82). Pada model linear/regresi logistik, multikolinearitas ini perlu ditangani dengan regularisasi (<em>Lasso/Ridge/ElasticNet</em>) atau transformasi rasio.</div>",
  "</div>",

  "<div class='card'>",
  "<h2>Tahap 4: Audit Zero-Inflation, Pola Desimal 1/3, &amp; Segmen Perilaku</h2>",
  "<p>Pemeriksaan mendalam menemukan adanya kelompok pelanggan bernilai nol (<em>zero-inflated</em>) serta artefak pembulatan rata-rata 3 bulan (di mana <code>X10</code> dan <code>X11</code> selalu berakhir dengan <code>.0</code>, <code>.3</code>, atau <code>.7</code>).</p>",
  knitr::kable(segment_audit, format = "html"),
  "<img src='", img_uri("eda_script/plots/fig04_zero_inflation_and_segment_conversion.png"), "' alt='Fig 04'>",
  "</div>",

  "<div class='card'>",
  "<h2>Tahap 5: Pemeriksaan Covariate Shift (Data Training vs Data Testing)</h2>",
  "<p>Untuk memastikan model tidak mengalami penurunan performa saat diuji pada data <em>testing</em>, kita membandingkan fungsi kepadatan distribusi (<em>density plot</em>) dan uji <em>Kolmogorov-Smirnov</em> antara <code>Train</code> (45.000 baris) dan <code>Test</code> (5.000 baris).</p>",
  "<img src='", img_uri("eda_script/plots/fig05_train_vs_test_distribution_check.png"), "' alt='Fig 05'>",
  "<div class='insight'><strong>Kesimpulan Kestabilan Distribusi:</strong><br>",
  "Seluruh 11 fitur menunjukkan kurva distribusi yang berhimpit sangat rapat antara <em>Train</em> dan <em>Test</em> (statistik <code>KS_TrainTest</code> sangat kecil, berkisar <code>0,0069</code>–<code>0,0210</code>). Tidak ditemukan adanya <em>covariate shift</em>.</div>",
  "</div>",

  "<div class='card'>",
  "<h2>Tahap 6: Sintesis Insight untuk Feature Engineering &amp; Strategi Pemodelan</h2>",
  "<p>Menggabungkan total pengeluaran bulanan (<code>Total_Spend = X2 + X6 + X9</code>) dengan frekuensi isi ulang (<code>X10</code>) menghasilkan fitur kunci <strong>Estimasi Nominal per Sekali Isi Ulang (<em>Estimated Ticket Size</em>)</strong> serta <strong>Porsi Pengeluaran Data (<em>Data Share</em>)</strong>.</p>",
  "<pre><code>Total_Spend     = X2 + X6 + X9\nEst_Ticket_Size = Total_Spend / pmax(X10, 0.3333)\nData_Share      = ifelse(Total_Spend &gt; 0, X9 / Total_Spend, 0)</code></pre>",
  "<img src='", img_uri("eda_script/plots/fig06_multivariate_arpu_vs_recharge.png"), "' alt='Fig 06'>",
  "<div class='insight'><strong>Rekomendasi Langsung untuk Tahap Selanjutnya:</strong><br>",
  "<ol><li><strong>Feature Engineering (<code>feature_engineering_script/</code>):</strong> Buat fitur rasio ekonomi (<em>Ticket Size per Recharge</em>, <em>Total ARPU</em>, <em>Share Voice/SMS/Data</em>, <em>Tarif Efektif per Menit/Panggilan/MB</em>, <em>Off-Net Call &amp; SMS Volume/Spend</em>, rekonstruksi angka bulat 3 bulan <code>round(X10*3)</code> dan <code>round(X11*3)</code>, serta <em>flag</em> indikator <em>Zero-Data</em> dan anomali pembulatan).</li>",
  "<li><strong>Modeling &amp; Hyperparameter Tuning (<code>modeling_script/</code>):</strong> Gunakan <em>Stratified K-Fold CV</em> (tanpa kolom <code>ID</code>), bandingkan <code>caret</code> baseline &amp; <em>Gradient Boosted Trees</em> (<code>xgboost</code>, <code>randomForest</code>, <code>glmnet</code>), serta lakukan <strong>Threshold Tuning</strong> untuk memaksimalkan <strong>Balanced Accuracy</strong>.</li></ol></div>",
  "</div>",
  "</body></html>"
)

writeLines(html_content, "eda_script/01_eda_report.html", useBytes = TRUE)

cat("\n======================================================================\n")
cat("  SELESAI! Seluruh tabel, plot, dan HTML tersimpan di eda_script/\n")
cat("======================================================================\n")
