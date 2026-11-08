# ==============================================================================
# tools/extract_rpubs_details.R — Analisis Detail Laporan RStudio Pubs
# ==============================================================================

suppressPackageStartupMessages({
  library(rvest)
  library(httr)
  library(stringr)
})

url <- "https://rstudio-pubs-static.s3.amazonaws.com/1266987_054fc539d0ac4aa88eecb5a4266f12ec.html"
res <- GET(url, user_agent("Mozilla/5.0"))
h <- read_html(content(res, "text", encoding = "UTF-8"))

# Ekstrak semua kode R dan output
code_nodes <- html_elements(h, "pre")
cat(sprintf("Ditemukan %d blok kode/output pre.\n\n", length(code_nodes)))

for (i in seq_along(code_nodes)) {
  txt <- html_text2(code_nodes[i])
  # Cari blok penting: recipe, step_, tune_grid, workflow, predict, submission, roc_auc
  if (grepl("recipe|step_|tune_grid|workflow|boost_tree|logistic_reg|themis|smote|themis|downsample|predict|threshold", txt, ignore.case = TRUE)) {
    cat(sprintf("=== BLOK KODE %d ===\n", i))
    cat(substr(txt, 1, 1000), "\n\n")
  }
}
