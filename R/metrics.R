# ==============================================================================
# R/metrics.R — Standard Evaluation Metrics & Threshold Optimization
# Proyek : KAGGLE_AUTO_WIN (Praktikum Sains Data STA1581)
# ==============================================================================

calc_confusion_matrix <- function(y_true, y_pred) {
  y_true <- as.integer(y_true)
  y_pred <- as.integer(y_pred)
  
  tp <- sum(y_true == 1L & y_pred == 1L)
  tn <- sum(y_true == 0L & y_pred == 0L)
  fp <- sum(y_true == 0L & y_pred == 1L)
  fn <- sum(y_true == 1L & y_pred == 0L)
  
  sens <- if ((tp + fn) > 0) tp / (tp + fn) else 0.0
  spec <- if ((tn + fp) > 0) tn / (tn + fp) else 0.0
  bal_acc <- 0.5 * (sens + spec)
  acc <- (tp + tn) / length(y_true)
  
  list(
    tp = tp, tn = tn, fp = fp, fn = fn,
    sensitivity = sens,
    specificity = spec,
    balanced_accuracy = bal_acc,
    accuracy = acc
  )
}

calc_balanced_accuracy <- function(y_true, y_pred) {
  calc_confusion_matrix(y_true, y_pred)$balanced_accuracy
}

find_optimal_threshold <- function(y_true, prob, grid = seq(0.05, 0.95, by = 0.002)) {
  y_true <- as.integer(y_true)
  best_t <- 0.5
  best_ba <- -1.0
  
  for (t in grid) {
    pred <- as.integer(prob >= t)
    ba <- calc_balanced_accuracy(y_true, pred)
    if (ba > best_ba) {
      best_ba <- ba
      best_t <- t
    }
  }
  
  list(optimal_threshold = best_t, max_balanced_accuracy = best_ba)
}

calc_cohen_kappa <- function(pred_a, pred_b) {
  a <- as.integer(pred_a)
  b <- as.integer(pred_b)
  n <- length(a)
  if (n != length(b)) stop("Panjang kedua vektor prediksi harus sama!")
  
  agree <- sum(a == b) / n
  p_a1 <- mean(a == 1L)
  p_b1 <- mean(b == 1L)
  p_expected <- (p_a1 * p_b1) + ((1 - p_a1) * (1 - p_b1))
  
  if (p_expected == 1.0) return(1.0)
  (agree - p_expected) / (1 - p_expected)
}
