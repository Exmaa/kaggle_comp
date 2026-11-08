suppressPackageStartupMessages({
  library(data.table)
  library(nanoparquet)
})

train <- as.data.table(read_parquet("feature_engineering_script/output/train_fe_v3.parquet"))

# 1. Interaksi X10 vs X11
train[, recharges_per_region := (as.numeric(exact_3m_recharges) + 0.1) / (as.numeric(exact_3m_regions) + 0.1)]
cat(sprintf("recharges_per_region rho: %.4f\n", cor(train$recharges_per_region, train$Y, method="spearman")))

# 2. Durasi per panggilan
train[, min_per_call := (as.numeric(X1) + 0.1) / (as.numeric(X3) + 0.1)]
cat(sprintf("min_per_call rho: %.4f\n", cor(train$min_per_call, train$Y, method="spearman")))

# 3. Rasio Pulsa terhadap Nominal Minimal (Indikator Transaksi Bank)
# Di bank, nominal transaksi selalu kelipatan 25k, 50k, 100k
train[, mod_25k_proximity := abs(total_arpu - round(total_arpu / 25000) * 25000)]
cat(sprintf("mod_25k_proximity rho: %.4f\n", cor(train$mod_25k_proximity, train$Y, method="spearman")))

# 4. K-Means Clustering Pseudo-Feature (Unsupervised Behavioral Latent Space)
set.seed(1581)
km_data <- scale(as.matrix(train[, .(X1, X2, X3, X4, X5, X6, X7, X8, X9, X10, X11)]))
km_res  <- kmeans(km_data, centers = 5, nstart = 10)
train[, cluster_id := factor(km_res$cluster)]
cat("\nRasio Y=1 per Cluster K-Means:\n")
print(train[, .(n = .N, y1_pct = round(mean(Y) * 100, 2)), by = cluster_id][order(-y1_pct)])
