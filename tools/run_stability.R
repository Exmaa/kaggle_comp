suppressPackageStartupMessages({
  library(data.table)
  library(nanoparquet)
})

source("R/prediction_stability.R")

files <- c(
  "data/submission.parquet",
  "modeling_script/output/submission_exp_004_stacking_vertex1334.parquet",
  "modeling_script/output/submission_exp_004_tomek_vertex1334.parquet",
  "modeling_script/output/submission_exp_004_rank_avg_vertex1334.parquet",
  "modeling_script/output/submission_exp_004_stacking_pos1200.parquet",
  "modeling_script/output/submission_exp_004_stacking_pos1450.parquet"
)
lbls <- c("EXP003_Rank2", "EXP004_NNLS", "EXP004_Tomek", "EXP004_RankAvg", "EXP004_12pct", "EXP004_14.5pct")

compare_submissions(files, lbls)
