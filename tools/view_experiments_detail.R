library(nanoparquet)
exp <- read_parquet('ledger/experiments.parquet')
for (i in 1:nrow(exp)) {
  cat(sprintf("[%s] %s\n  Score: %.5f +- %.5f | AUC: %.5f | Thresh: %.4f | PosPct: %.2f%%\n  Notes: %s\n\n",
              exp$exp_id[i], exp$model_family[i],
              as.numeric(exp$cv_score_mean[i]), as.numeric(exp$cv_score_sd[i]),
              as.numeric(exp$oof_roc_auc[i]), as.numeric(exp$opt_threshold[i]),
              as.numeric(exp$test_pos_pct[i]) * 100, exp$notes[i]))
}
