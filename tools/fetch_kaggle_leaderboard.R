# ==============================================================================
# tools/fetch_kaggle_leaderboard.R — Ambil Leaderboard Kaggle
# ==============================================================================

suppressPackageStartupMessages({
  library(httr)
  library(jsonlite)
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

url <- sprintf("https://www.kaggle.com/api/v1/competitions/%s/leaderboard/view", slug)
res <- GET(url, authenticate(user, key, type = "basic"))

if (status_code(res) == 200) {
  lb <- fromJSON(content(res, "text", encoding = "UTF-8"), flatten = TRUE)
  cat("========================================================================================\n")
  cat(sprintf("  LEADERBOARD RESMI KAGGLE — %s\n", slug))
  cat("========================================================================================\n")
  if (!is.null(lb$submissions)) {
    sub_df <- lb$submissions
    sub_df <- sub_df[order(as.numeric(sub_df$score), decreasing = TRUE), ]
    print(head(sub_df[, intersect(c("teamId", "teamName", "submissionDate", "score"), names(sub_df))], 30))
  } else {
    print(head(lb, 20))
  }
} else {
  # Coba endpoint alternatif
  url2 <- sprintf("https://www.kaggle.com/api/v1/competitions/data-science-competition-2026/leaderboard")
  res2 <- GET(url2, authenticate(user, key, type = "basic"))
  cat(sprintf("HTTP %d: %s\n", status_code(res2), content(res2, "text", encoding = "UTF-8")))
}
