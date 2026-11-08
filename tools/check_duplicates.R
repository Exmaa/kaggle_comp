suppressPackageStartupMessages({
  library(data.table)
  library(nanoparquet)
})

train <- as.data.table(read_parquet("data/training_data_tugas_praktikum_DataScience.parquet"))
test  <- as.data.table(read_parquet("data/testing_data_tugas_praktikum_DataScience.parquet"))
raw_cols <- paste0("X", 1:11)

cat("Checking exact duplicates in Train:\n")
dup_groups <- train[, .(n = .N, n_y1 = sum(Y == 1), n_y0 = sum(Y == 0)), by = raw_cols][n > 1]
cat(sprintf("Unique feature patterns with >1 occurrence: %d\n", nrow(dup_groups)))
conflicting <- dup_groups[n_y1 > 0 & n_y0 > 0]
cat(sprintf("Patterns with CONFLICTING labels (both Y=0 and Y=1): %d\n", nrow(conflicting)))
cat(sprintf("Total rows involved in conflicting exact duplicates: %d\n", sum(conflicting$n)))

cat("\nChecking Train vs Test exact feature overlap:\n")
overlap <- merge(train[, .(ID_train = ID, Y, .SD), .SDcols = raw_cols],
                 test[, .(ID_test = ID, .SD), .SDcols = raw_cols],
                 by = raw_cols)
cat(sprintf("Number of Test rows that have EXACT feature match in Train: %d\n", length(unique(overlap$ID_test))))
