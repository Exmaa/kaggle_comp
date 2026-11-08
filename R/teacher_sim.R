# ==============================================================================
# R/teacher_sim.R — Teacher-Simulation Validation Framework
# Proyek : KAGGLE_AUTO_WIN (Praktikum Sains Data STA1581)
# ==============================================================================
# Tujuan : Mensimulasikan skenario di mana ground truth data uji dibangkitkan oleh
#          model ensemble guru (H2/H3). Mengukur resiliensi model kandidat (student)
#          terhadap berbagai variasi arsitektur guru dan threshold rule.
# ==============================================================================

suppressPackageStartupMessages({
  library(data.table)
  library(nanoparquet)
  library(xgboost)
  library(randomForest)
  library(glmnet)
  library(rpart)
})

source("R/metrics.R")

run_teacher_simulation <- function(
  train_data_path = "data/training_data_tugas_praktikum_DataScience.parquet",
  split_seed = 1581,
  sample_size = 30000, # Subsampling untuk efisiensi komputasi jika diperlukan
  teachers_to_run = c("rf", "xgb", "glmnet", "tree", "ensemble")
) {
  t0 <- Sys.time()
  cat("==============================================================\n")
  cat("  MEMULAI TEACHER-SIMULATION VALIDATION (H2/H3 ROBUSTNESS)\n")
  cat("==============================================================\n")
  
  # 1. Load Data (.parquet)
  cat("[1/5] Membaca data training Parquet...\n")
  raw_df <- as.data.table(read_parquet(train_data_path))
  if ("ID" %in% names(raw_df)) raw_df[, ID := NULL]
  
  # Prior kelas positif
  pi_prior <- mean(raw_df$Y == 1L)
  cat(sprintf("      Total baris: %d | Kelas positif (pi): %.4f (%.2f%%)\n", 
              nrow(raw_df), pi_prior, pi_prior * 100))
  
  if (!is.null(sample_size) && nrow(raw_df) > sample_size) {
    set.seed(split_seed)
    idx_sample <- sample(nrow(raw_df), sample_size)
    raw_df <- raw_df[idx_sample]
    cat(sprintf("      Subsampled ke %d baris untuk kecepatan eksekusi.\n", sample_size))
  }
  
  # 2. Split Partition A (70% Train) & Partition B (30% Simulated Test)
  set.seed(split_seed)
  n_total <- nrow(raw_df)
  idx_a <- sample(n_total, size = floor(0.70 * n_total))
  
  df_a <- raw_df[idx_a]
  df_b <- raw_df[-idx_a]
  
  feature_cols <- paste0("X", 1:11)
  
  X_a <- as.matrix(df_a[, ..feature_cols])
  y_a <- df_a$Y
  X_b <- as.matrix(df_b[, ..feature_cols])
  y_b_actual <- df_b$Y
  
  cat(sprintf("[2/5] Partisi: Set A (Latih Guru & Murid): %d baris | Set B (Simulasi Test): %d baris\n",
              nrow(df_a), nrow(df_b)))
  
  # 3. Latih Perpustakaan Guru (Teacher Models) di Partisi A
  cat("[3/5] Melatih keluarga model Guru (Teachers) di Set A...\n")
  teacher_probs <- list()
  
  # A. Random Forest Teacher
  if ("rf" %in% teachers_to_run) {
    cat("      - Melatih Teacher: Random Forest (ntree=150)...\n")
    set.seed(split_seed + 1)
    rf_model <- randomForest(x = X_a, y = as.factor(y_a), ntree = 150)
    rf_prob_b <- predict(rf_model, newdata = X_b, type = "prob")[, "1"]
    teacher_probs[["RF"]] <- rf_prob_b
  }
  
  # B. XGBoost Teacher
  if ("xgb" %in% teachers_to_run) {
    cat("      - Melatih Teacher: XGBoost (depth=5, eta=0.1, nrounds=120)...\n")
    set.seed(split_seed + 2)
    dtrain_a <- xgb.DMatrix(data = X_a, label = y_a)
    dtest_b  <- xgb.DMatrix(data = X_b)
    xgb_params <- list(
      objective = "binary:logistic",
      eval_metric = "auc",
      max_depth = 5,
      eta = 0.1,
      nthread = 4
    )
    xgb_model <- xgb.train(params = xgb_params, data = dtrain_a, nrounds = 120, verbose = 0)
    xgb_prob_b <- predict(xgb_model, dtest_b)
    teacher_probs[["XGB"]] <- xgb_prob_b
  }
  
  # C. ElasticNet GLMnet Teacher
  if ("glmnet" %in% teachers_to_run) {
    cat("      - Melatih Teacher: GLMnet ElasticNet (alpha=0.5)...\n")
    set.seed(split_seed + 3)
    cv_glm <- cv.glmnet(x = X_a, y = y_a, family = "binomial", alpha = 0.5)
    glm_prob_b <- as.numeric(predict(cv_glm, newx = X_b, s = "lambda.1se", type = "response"))
    teacher_probs[["GLMNET"]] <- glm_prob_b
  }
  
  # D. Decision Tree CART Teacher
  if ("tree" %in% teachers_to_run) {
    cat("      - Melatih Teacher: Rpart Decision Tree...\n")
    tree_model <- rpart(as.factor(Y) ~ ., data = df_a, method = "class", cp = 0.005)
    tree_prob_b <- predict(tree_model, newdata = df_b, type = "prob")[, "1"]
    teacher_probs[["CART"]] <- tree_prob_b
  }
  
  # E. Teacher Ensemble (Average of Teachers)
  if ("ensemble" %in% teachers_to_run && length(teacher_probs) >= 2) {
    cat("      - Membangun Teacher: Multi-Model Ensemble (Rata-rata Probabilitas)...\n")
    ens_prob_b <- rowMeans(do.call(cbind, teacher_probs))
    teacher_probs[["ENSEMBLE_AVG"]] <- ens_prob_b
  }
  
  # Buat variasi threshold untuk setiap guru:
  # - Versi default (t = 0.50)
  # - Versi prior / balanced (t = pi_prior)
  synthetic_ground_truths <- list()
  for (t_name in names(teacher_probs)) {
    prob <- teacher_probs[[t_name]]
    synthetic_ground_truths[[paste0(t_name, "@0.50")]] <- as.integer(prob >= 0.50)
    synthetic_ground_truths[[paste0(t_name, "@pi")]]   <- as.integer(prob >= pi_prior)
  }
  
  cat(sprintf("      Terbentuk %d variasi Ground Truth Guru sintetis di Set B.\n", length(synthetic_ground_truths)))
  
  # 4. Latih Model Murid Kandidat (Candidate Students) di Partisi A
  cat("[4/5] Melatih model Murid Kandidat (Students) di Set A...\n")
  student_preds <- list()
  
  # Student S1: Vanilla XGBoost dengan threshold t* (OOF optimal)
  set.seed(split_seed + 10)
  dtrain_s <- xgb.DMatrix(data = X_a, label = y_a)
  s_xgb <- xgb.train(
    params = list(objective = "binary:logistic", eval_metric = "auc", max_depth = 4, eta = 0.05, nthread = 4),
    data = dtrain_s, nrounds = 150, verbose = 0
  )
  s_prob_b <- predict(s_xgb, xgb.DMatrix(data = X_b))
  student_preds[["Student_XGB_at_0.50"]]   <- as.integer(s_prob_b >= 0.50)
  student_preds[["Student_XGB_at_pi"]]     <- as.integer(s_prob_b >= pi_prior)
  student_preds[["Student_XGB_at_opt0.194"]] <- as.integer(s_prob_b >= 0.194)
  
  # Student S2: Random Forest
  set.seed(split_seed + 20)
  s_rf <- randomForest(x = X_a, y = as.factor(y_a), ntree = 100)
  s_rf_prob_b <- predict(s_rf, newdata = X_b, type = "prob")[, "1"]
  student_preds[["Student_RF_at_0.50"]] <- as.integer(s_rf_prob_b >= 0.50)
  student_preds[["Student_RF_at_pi"]]   <- as.integer(s_rf_prob_b >= pi_prior)
  
  # Student S3: Blend (Average XGB + RF)
  s_blend_prob_b <- 0.6 * s_prob_b + 0.4 * s_rf_prob_b
  student_preds[["Student_Blend_at_pi"]] <- as.integer(s_blend_prob_b >= pi_prior)
  student_preds[["Student_Blend_at_opt"]] <- as.integer(s_blend_prob_b >= 0.194)
  
  # 5. Evaluasi Matriks Kesepakatan (Agreement Matrix)
  cat("[5/5] Menghitung matriks kesepakatan Balanced Accuracy (Student vs Teachers)...\n\n")
  
  teacher_names <- names(synthetic_ground_truths)
  student_names <- names(student_preds)
  
  eval_matrix <- matrix(NA_real_, nrow = length(student_names), ncol = length(teacher_names),
                        dimnames = list(student_names, teacher_names))
  
  for (s_i in seq_along(student_names)) {
    s_name <- student_names[s_i]
    pred_s <- student_preds[[s_name]]
    
    for (t_j in seq_along(teacher_names)) {
      t_name <- teacher_names[t_j]
      y_teacher <- synthetic_ground_truths[[t_name]]
      ba <- calc_balanced_accuracy(y_teacher, pred_s)
      eval_matrix[s_name, t_name] <- round(ba, 4)
    }
  }
  
  summary_results <- data.table(
    Student_Model = student_names,
    Mean_BA_vs_Teachers = round(rowMeans(eval_matrix), 4),
    WorstCase_BA = round(apply(eval_matrix, 1, min), 4),
    BestCase_BA = round(apply(eval_matrix, 1, max), 4),
    SD_BA = round(apply(eval_matrix, 1, sd), 4),
    Agreement_vs_True_Y = round(sapply(student_preds, function(p) calc_balanced_accuracy(y_b_actual, p)), 4)
  )
  setorder(summary_results, -WorstCase_BA, -Mean_BA_vs_Teachers)
  
  cat("========================================================================================\n")
  cat("  HASIL EVALUASI TEACHER SIMULATION (DIURUTKAN BERDASARKAN ROBUSTNESS / WORST-CASE BA)\n")
  cat("========================================================================================\n")
  print(summary_results)
  
  cat("\n========================================================================================\n")
  cat("  RINCIAN MATRIKS SKOR BALANCED ACCURACY (STUDENTS x TEACHERS)\n")
  cat("========================================================================================\n")
  print(round(eval_matrix, 4))
  cat("========================================================================================\n")
  
  t1 <- Sys.time()
  cat(sprintf("\nTeacher-Simulation selesai dalam %.2f detik!\n", as.numeric(difftime(t1, t0, units = "secs"))))
  
  invisible(list(
    summary = summary_results,
    matrix = eval_matrix,
    teachers = synthetic_ground_truths,
    students = student_preds
  ))
}

if (!interactive() && identical(environment(), globalenv())) {
  res <- run_teacher_simulation(sample_size = 20000)
}
