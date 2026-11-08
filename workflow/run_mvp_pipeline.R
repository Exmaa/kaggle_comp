# ==============================================================================
# run_mvp_pipeline.R — End-to-End Orchestrator Script (Local & Kaggle Ready)
# Proyek   : KAGGLE_AUTO_WIN (Praktikum Sains Data STA1581)
# Cara Run : Rscript workflow/run_mvp_pipeline.R
# ==============================================================================

t0 <- Sys.time()
cat("##############################################################\n")
cat("#   KAGGLE_AUTO_WIN — END-TO-END MVP PIPELINE (VANILLA XGB)  #\n")
cat("##############################################################\n")
cat(sprintf("Waktu Mulai : %s\n\n", format(t0, "%Y-%m-%d %H:%M:%S")))

source("eda_script/01_eda_mvp.R", local = new.env())
source("feature_engineering_script/02_fe_mvp.R", local = new.env())
source("modeling_script/03_modeling_xgboost_mvp.R", local = new.env())
source("modeling_script/04_validate_submission.R", local = new.env())

t1 <- Sys.time()
elapsed <- round(as.numeric(difftime(t1, t0, units = "secs")), 2)

cat("##############################################################\n")
cat(sprintf("# PIPELINE MVP SELESAI DENGAN SUKSES DALAM %.2f DETIK!      #\n", elapsed))
cat("##############################################################\n")
