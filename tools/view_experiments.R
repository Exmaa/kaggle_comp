library(nanoparquet)
exp <- read_parquet('ledger/experiments.parquet')
cols <- intersect(c('exp_id', 'model_family', 'n_features', 'oof_auc', 'oof_ba_fixed_pi', 'nested_ba_mean', 'nested_ba_sd'), names(exp))
print(exp[, cols])
