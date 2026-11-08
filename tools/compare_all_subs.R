library(nanoparquet)
source("R/prediction_stability.R")

sub_files <- c(
  "EXP_003_ENS" = "modeling_script/output/submission_exp_003_ensemble_vertex1334.parquet",
  "EXP_004_STK" = "modeling_script/output/submission_exp_004_stacking_vertex1334.parquet",
  "EXP_005_GLD" = "modeling_script/output/submission_exp_005_golden_shot_vertex1334.parquet",
  "EXP_006_DST" = "modeling_script/output/submission_exp_006_distilled_vertex1334.parquet",
  "EXP_007_TRN" = "modeling_script/output/submission_exp_007_pseudolabel_vertex1334.parquet",
  "EXP_008_GND" = "modeling_script/output/submission_exp_008_grand_master_vertex1334.parquet"
)

res <- compare_submissions(sub_files)
