suppressPackageStartupMessages({
  library(data.table)
  library(xgboost)
  library(nanoparquet)
  library(pROC)
})

train <- as.data.table(read_parquet("feature_engineering_script/output/train_fe_v3.parquet"))
test  <- as.data.table(read_parquet("feature_engineering_script/output/test_fe_v3.parquet"))
feats <- setdiff(names(train), c("ID", "Y", "fold"))

cat(sprintf("Total fitur: %d\n", length(feats)))

# Cek training fit vs OOF fit pada berbagai kedalaman model (max_depth 3, 6, 10, 15)
val_idx <- which(train$fold == 1)
tr_idx  <- which(train$fold != 1)

X_tr <- as.matrix(train[tr_idx, ..feats])
y_tr <- train$Y[tr_idx]
X_va <- as.matrix(train[val_idx, ..feats])
y_va <- train$Y[val_idx]

dtrain <- xgb.DMatrix(X_tr, label = y_tr)
dval   <- xgb.DMatrix(X_va, label = y_va)

for (depth in c(4, 6, 8, 12)) {
  m <- xgb.train(
    params = list(
      objective = "binary:logistic",
      eval_metric = "auc",
      max_depth = depth,
      learning_rate = 0.05
    ),
    data = dtrain,
    nrounds = 300,
    watchlist = list(train = dtrain, val = dval),
    verbose = 0
  )
  p_tr <- predict(m, dtrain)
  p_va <- predict(m, dval)
  auc_tr <- as.numeric(auc(roc(y_tr, p_tr, quiet = TRUE)))
  auc_va <- as.numeric(auc(roc(y_va, p_va, quiet = TRUE)))
  cat(sprintf("Depth %2d -> Train AUC: %.4f | Val (Fold 1) AUC: %.4f\n", depth, auc_tr, auc_va))
}
