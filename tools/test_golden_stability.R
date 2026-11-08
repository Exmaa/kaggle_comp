suppressPackageStartupMessages({
  library(data.table)
  library(nanoparquet)
})

source("R/prediction_stability.R")

files <- c(
  "modeling_script/output/submission_exp_003_ensemble_vertex1334.parquet",
  "modeling_script/output/submission_exp_004_stacking_vertex1334.parquet",
  "modeling_script/output/submission_exp_005_golden_shot_vertex1334.parquet"
)
lbls <- c("EXP003_Kaggle_Rank2", "EXP004_Tomek_NNLS", "EXP005_Golden_Shot")

compare_submissions(files, lbls)
