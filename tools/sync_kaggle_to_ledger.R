# ==============================================================================
# tools/sync_kaggle_to_ledger.R — Sinkronisasi Riwayat Kaggle ke Ledger Parquet
# Proyek : KAGGLE_AUTO_WIN (Praktikum Sains Data STA1581)
# ==============================================================================

suppressPackageStartupMessages({
  library(data.table)
  library(httr)
  library(jsonlite)
  library(nanoparquet)
})

read_env <- function(file_path = ".env") {
  if (!file.exists(file_path)) return(list())
  lines <- readLines(file_path, warn = FALSE)
  env_list <- list()
  for (line in lines) {
    line <- trimws(line)
    if (startsWith(line, "#") || !grepl("=", line)) next
    parts <- strsplit(line, "=", fixed = TRUE)[[1]]
    env_list[[trimws(parts[1])]] <- trimws(paste(parts[-1], collapse = "="))
  }
  env_list
}

cfg <- read_env()
user <- cfg$KAGGLE_USERNAME
key  <- cfg$KAGGLE_KEY
slug <- cfg$COMPETITION_SLUG

url <- sprintf("https://www.kaggle.com/api/v1/competitions/submissions/list/%s", slug)
res <- GET(url, authenticate(user, key, type = "basic"))

if (status_code(res) != 200) {
  stop(sprintf("Gagal memanggil Kaggle API: HTTP %d", status_code(res)))
}

data <- fromJSON(content(res, "text", encoding = "UTF-8"), flatten = TRUE)

if (!is.data.frame(data) || nrow(data) == 0) {
  cat("Tidak ada data submission di Kaggle.\n")
  quit(status = 0)
}

desc_clean <- as.character(data$description)
missing_desc <- is.na(desc_clean) | desc_clean == ""
desc_clean[missing_desc] <- as.character(data$fileName)[missing_desc]

synced_dt <- data.table(
  submission_id  = paste0("KAG_", data$ref),
  timestamp      = as.character(data$date),
  file_name      = as.character(data$fileName),
  sha256         = rep(NA_character_, nrow(data)),
  model_name     = desc_clean,
  threshold      = rep(NA_real_, nrow(data)),
  positive_count = rep(NA_integer_, nrow(data)),
  positive_pct   = rep(NA_real_, nrow(data)),
  public_score   = suppressWarnings(as.numeric(as.character(data$publicScore))),
  status         = as.character(data$status),
  notes          = sprintf("Kaggle Ref: %s | Status: %s", data$ref, data$status)
)

setorder(synced_dt, timestamp)
ledger_path <- "ledger/submissions.parquet"
write_parquet(as.data.frame(synced_dt), ledger_path)
cat(sprintf("Sukses menyinkronkan %d riwayat resmi Kaggle ke %s!\n", nrow(synced_dt), ledger_path))
