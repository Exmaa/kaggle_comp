# ==============================================================================
# tools/fetch_kaggle_submissions.R — Sinkronisasi Riwayat Submission dari Kaggle
# Proyek : KAGGLE_AUTO_WIN (Praktikum Sains Data STA1581)
# ==============================================================================

suppressPackageStartupMessages({
  library(httr)
  library(jsonlite)
  library(nanoparquet)
})

# 1. Baca kredensial dari .env (Zero-Leakage: tidak pernah dicetak)
read_env <- function(file_path = ".env") {
  if (!file.exists(file_path)) return(list())
  lines <- readLines(file_path, warn = FALSE)
  env_list <- list()
  for (line in lines) {
    line <- trimws(line)
    if (startsWith(line, "#") || !grepl("=", line)) next
    parts <- strsplit(line, "=", fixed = TRUE)[[1]]
    key <- trimws(parts[1])
    val <- trimws(paste(parts[-1], collapse = "="))
    env_list[[key]] <- val
  }
  env_list
}

cfg <- read_env()
user <- cfg$KAGGLE_USERNAME
key  <- cfg$KAGGLE_KEY
slug <- cfg$COMPETITION_SLUG

if (is.null(user) || is.null(key) || is.null(slug) || slug == "") {
  stop("Kredensial atau COMPETITION_SLUG tidak lengkap di .env")
}

url <- sprintf("https://www.kaggle.com/api/v1/competitions/submissions/list/%s", slug)
res <- GET(url, authenticate(user, key, type = "basic"))

if (status_code(res) != 200) {
  stop(sprintf("Gagal memanggil Kaggle API: HTTP %d - %s", status_code(res), content(res, "text", encoding = "UTF-8")))
}

data <- fromJSON(content(res, "text", encoding = "UTF-8"), flatten = TRUE)

if (length(data) == 0 || nrow(data) == 0) {
  cat("Tidak ada riwayat submission ditemukan di Kaggle untuk kompetisi ini.\n")
} else {
  cols_to_show <- intersect(c("id", "date", "fileName", "description", "status", "publicScore", "privateScore"), names(data))
  cat("========================================================================================\n")
  cat(sprintf("  RIWAYAT RESMI SUBMISSION KAGGLE — %s (Total: %d)\n", slug, nrow(data)))
  cat("========================================================================================\n")
  print(data[, cols_to_show, drop = FALSE])
  cat("========================================================================================\n")
}
