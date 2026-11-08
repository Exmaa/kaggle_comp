# ==============================================================================
# R/submission_helper.R — Helper Operasional Validasi, Serialisasi & Ledger
# Proyek : KAGGLE_AUTO_WIN (Praktikum Sains Data STA1581)
# ==============================================================================

suppressPackageStartupMessages({
  library(data.table)
  library(nanoparquet)
  library(jsonlite)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) == 0) {
  stop("Argumen aksi diperlukan: validate, serialize, atau record")
}

action <- args[1]

if (action == "validate") {
  # Argumen: validate <sub_parquet_path> <ledger_parquet_path> <today_str> <sub_hash>
  sub_path    <- args[2]
  ledger_path <- args[3]
  today_str   <- args[4]
  sub_hash    <- args[5]
  
  today_cnt <- 0L
  total_rows_ledger <- 0L
  dup_id <- ""
  
  if (file.exists(ledger_path)) {
    ldg <- read_parquet(ledger_path)
    total_rows_ledger <- nrow(ldg)
    if ("timestamp" %in% names(ldg) && "status" %in% names(ldg)) {
      today_cnt <- sum(startsWith(as.character(ldg$timestamp), today_str) & ldg$status == "evaluated", na.rm = TRUE)
    }
    if ("sha256" %in% names(ldg) && "status" %in% names(ldg)) {
      hits <- which(ldg$sha256 == sub_hash & ldg$status == "evaluated")
      if (length(hits) > 0L) dup_id <- as.character(ldg$submission_id[hits[1]])
    }
  }
  
  if (!file.exists(sub_path)) {
    cat(toJSON(list(error = "File tidak ditemukan"), auto_unbox = TRUE))
    quit(status = 0)
  }
  
  dt <- read_parquet(sub_path)
  nr <- nrow(dt)
  cols_ok <- identical(names(dt), c("ID", "Y"))
  na_ok   <- cols_ok && (sum(is.na(dt$ID)) == 0L) && (sum(is.na(dt$Y)) == 0L)
  id_ok   <- cols_ok && na_ok && (nr == 5000L) && identical(as.integer(dt$ID), 45001L:50000L)
  bin_ok  <- cols_ok && na_ok && all(dt$Y %in% c(0, 1))
  c1      <- if (cols_ok) sum(as.integer(dt$Y) == 1L, na.rm = TRUE) else 0L
  c0      <- if (cols_ok) sum(as.integer(dt$Y) == 0L, na.rm = TRUE) else 0L
  pct1    <- if (nr > 0L) round(c1 / nr * 100, 2) else 0.0
  
  res <- list(
    today_cnt   = today_cnt,
    ledger_rows = total_rows_ledger,
    dup_id      = dup_id,
    rows        = nr,
    cols_ok     = cols_ok,
    na_ok       = na_ok,
    id_ok       = id_ok,
    bin_ok      = bin_ok,
    count1      = c1,
    count0      = c0,
    pct1        = pct1
  )
  cat(toJSON(res, auto_unbox = TRUE))

} else if (action == "budget") {
  # Argumen: budget <ledger_path> <today_str>
  lp <- args[2]
  today_str <- args[3]
  today_cnt <- 0L
  if (file.exists(lp)) {
    ldg <- read_parquet(lp)
    if ("timestamp" %in% names(ldg) && "status" %in% names(ldg)) {
      today_cnt <- sum(grepl(today_str, as.character(ldg$timestamp)) & ldg$status == "complete", na.rm = TRUE)
    }
  }
  cat(today_cnt)

} else if (action == "serialize") {
  # Argumen: serialize <sub_parquet_path>
  sub_path <- args[2]
  dt <- read_parquet(sub_path)
  cat("ID,Y\n")
  cat(sprintf("%d,%d\n", as.integer(dt$ID), as.integer(dt$Y)), sep = "")

} else if (action == "record") {
  # Argumen: record <ledger_path> <sub_id> <ts> <fname> <hash> <model> <thr> <c1> <pct1> <notes>
  lp     <- args[2]
  sub_id <- args[3]
  ts     <- args[4]
  fname  <- args[5]
  hash   <- args[6]
  model  <- args[7]
  thr    <- as.numeric(args[8])
  c1     <- as.integer(args[9])
  pct1   <- as.numeric(args[10])
  notes  <- args[11]
  
  new_row <- data.table(
    submission_id  = sub_id,
    timestamp      = ts,
    file_name      = fname,
    sha256         = hash,
    model_name     = model,
    threshold      = thr,
    positive_count = c1,
    positive_pct   = pct1,
    public_score   = NA_real_,
    status         = "evaluated",
    notes          = notes
  )
  
  if (file.exists(lp)) {
    old_dt <- as.data.table(read_parquet(lp))
    out_dt <- rbindlist(list(old_dt, new_row), use.names = TRUE, fill = TRUE, ignore.attr = TRUE)
  } else {
    out_dt <- new_row
  }
  write_parquet(as.data.frame(out_dt), lp)
  cat("OK")

} else if (action == "budget") {
  # Argumen: budget <ledger_path> <today_str>
  lp        <- args[2]
  today_str <- args[3]
  n_today   <- 0L
  if (file.exists(lp)) {
    dt <- read_parquet(lp)
    if ("timestamp" %in% names(dt) && "status" %in% names(dt)) {
      n_today <- sum(startsWith(as.character(dt$timestamp), today_str) & dt$status == "evaluated", na.rm = TRUE)
    }
  }
  cat(n_today)

} else if (action == "audit") {
  # Argumen: audit <pipe_separated_paths>
  paths <- strsplit(args[2], "\\|")[[1]]
  for (p in paths) {
    dt <- tryCatch(read_parquet(p), error = function(e) NULL)
    if (is.null(dt)) {
      cat(sprintf("%s\tFAIL\t0\t0\t0.00%%\tGagal baca Parquet\n", p))
      next
    }
    nr <- nrow(dt)
    cols_ok <- identical(names(dt), c("ID", "Y"))
    na_ok   <- cols_ok && sum(is.na(dt$ID)) == 0L && sum(is.na(dt$Y)) == 0L
    id_ok   <- cols_ok && na_ok && nr == 5000L && identical(as.integer(dt$ID), 45001L:50000L)
    y_ok    <- cols_ok && na_ok && all(dt$Y %in% c(0, 1))
    c1      <- if (cols_ok) sum(dt$Y == 1, na.rm = TRUE) else 0L
    pct1    <- if (nr > 0) round(c1 / nr * 100, 2) else 0
    issues  <- character(0)
    if (nr != 5000L) issues <- c(issues, sprintf("Rows=%d", nr))
    if (!cols_ok)    issues <- c(issues, "Kolom bukan ID,Y")
    if (!na_ok)      issues <- c(issues, "Ada nilai NA")
    if (!id_ok)      issues <- c(issues, "ID bukan 45001:50000")
    if (!y_ok)       issues <- c(issues, "Y bukan biner {0,1}")
    status  <- if (length(issues) == 0L) "PASS" else "FAIL"
    iss_str <- if (length(issues) == 0L) "OK" else paste(issues, collapse = "; ")
    cat(sprintf("%s\t%s\t%d\t%d\t%.2f%%\t%s\n", p, status, nr, c1, pct1, iss_str))
  }
}
