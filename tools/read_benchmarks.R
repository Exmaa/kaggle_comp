suppressPackageStartupMessages({
  library(data.table)
  library(nanoparquet)
})

cat("=== EXP_005 BENCHMARK (CV PADA Y ASLI) ===\n")
b5 <- as.data.table(read_parquet("modeling_script/output/exp_005_model_zoo_benchmark.parquet"))
print(b5[, .(Model, OOF_ROC_AUC, OOF_BA_Opt, Nested_BA_Mean, Nested_BA_SD, Nested_BA_SE, Mean_Threshold, Sensitivity, Specificity)])

cat("\n=== EXP_006 PUSH90 BENCHMARK (TERHADAP GURU ENSEMBLE) ===\n")
b6 <- as.data.table(read_parquet("modeling_script/output/exp_006_push90_benchmark.parquet"))
print(b6)
