suppressPackageStartupMessages({
  library(FNN)
  library(data.table)
  library(nanoparquet)
})

df <- as.data.table(read_parquet("feature_engineering_script/output/train_fe_v3.parquet"))
feats <- setdiff(names(df), c("ID", "Y", "fold"))

# Test on Fold 1 train (36,000 rows)
train_subset <- df[fold != 1]
X <- scale(as.matrix(train_subset[, ..feats]))
# replace any NA/NaN in scale with 0
X[is.nan(X)] <- 0
y <- train_subset$Y

t0 <- Sys.time()
nn <- get.knn(X, k = 1)
t1 <- Sys.time()
cat(sprintf("1-NN search took: %.2f sec\n", as.numeric(difftime(t1, t0, units="secs"))))

nn_idx <- nn$nn.index[, 1]
# Fast vectorized check:
# For each i, neighbor is j = nn_idx[i]
# Check if nn_idx[j] == i AND y[i] != y[j]
j_vec <- nn_idx
is_mutual <- (nn_idx[j_vec] == seq_along(y)) & (y != y[j_vec])

cat(sprintf("Total mutual Tomek points: %d (pairs: %d)\n", sum(is_mutual), sum(is_mutual)/2))
remove_neg_idx <- which(is_mutual & y == 0)
cat(sprintf("Borderline negatives (Y=0) to clean: %d (%.2f%% of negatives)\n",
            length(remove_neg_idx), length(remove_neg_idx) / sum(y == 0) * 100))
