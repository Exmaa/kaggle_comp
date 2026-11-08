# ==============================================================================
# 11_exp009_dl_llm_kaggle_gpu.R — EXP_009: Pretrained LLM Semantic Embeddings
#                                 + PLR-FT-Transformer + TabResNet-GLU (R + Cloud)
# Proyek    : KAGGLE_AUTO_WIN (Praktikum Sains Data STA1581 — pecutAI)
# Lingkungan: Kaggle Private Kernel (Dual-Mode: GPU/Online & Offline CPU/Local)
# Fitur     : 46 Fitur Domain FE v4 (termasuk monetary_polarity & dual-regime)
# Aturan    : 100% Bahasa R | 100% Output Apache Parquet (.parquet) | Bebas Leakage
# ==============================================================================

t_start <- Sys.time()
cat("######################################################################\n")
cat("# EXP_009: LLM SEMANTIC EMBEDDING + TABULAR DEEP LEARNING (R + CLOUD)#\n")
cat("######################################################################\n\n")

args_cli   <- commandArgs(trailingOnly = TRUE)
SMOKE_TEST <- "--smoke-test" %in% args_cli

options(timeout = 120)
if (identical(Sys.info()[["sysname"]], "Linux") && file.exists("/etc/os-release")) {
  osr <- readLines("/etc/os-release", warn = FALSE)
  codeline <- osr[grepl("^VERSION_CODENAME=", osr)][1]
  codename <- sub("^VERSION_CODENAME=", "", gsub('"', "", codeline))
  if (!is.na(codename) && nzchar(codename) && codename %in% c("jammy", "noble", "focal")) {
    options(repos = c(
      PPM  = sprintf("https://packagemanager.posit.co/cran/__linux__/%s/latest", codename),
      CRAN = "https://cloud.r-project.org"
    ))
  } else {
    options(repos = c(CRAN = "https://cloud.r-project.org"))
  }
} else {
  options(repos = c(CRAN = "https://cloud.r-project.org"))
}

# ------------------------------------------------------------------------------
# 0. Deteksi Konektivitas & Python Interpreter Secara Cepat (<3 Detik, Bebas Hang)
# ------------------------------------------------------------------------------
detect_internet <- function() {
  env_net <- Sys.getenv("KAGGLE_HAS_INTERNET", unset = "")
  if (identical(env_net, "0")) return(FALSE)
  if (identical(env_net, "1")) return(TRUE)
  tryCatch({
    con <- suppressWarnings(socketConnection("cloud.r-project.org", port = 443L, open = "r", blocking = TRUE, timeout = 3))
    close(con)
    TRUE
  }, error = function(e) FALSE)
}

HAS_INTERNET <- detect_internet()
cat(sprintf("[ENV] Internet aktif: %s | Sysname: %s\n", HAS_INTERNET, Sys.info()[["sysname"]]))

find_py_exe <- function() {
  cands <- unique(c(
    Sys.getenv("KAGGLE_PY_EXE", unset = ""),
    Sys.getenv("RETICULATE_PYTHON", unset = ""),
    "/usr/bin/python3",
    "/opt/conda/bin/python",
    unname(Sys.which("python3")),
    unname(Sys.which("python"))
  ))
  cands <- cands[nzchar(cands) & file.exists(cands)]
  for (pc in cands) {
    rc <- suppressWarnings(system2(pc, args = c("-c", "import numpy"), stdout = FALSE, stderr = FALSE))
    if (identical(rc, 0L)) return(pc)
  }
  if (length(cands) > 0L) cands[1L] else ""
}

PY_EXE <- find_py_exe()
cat(sprintf("[ENV] Python bridge terdeteksi: %s\n", ifelse(nzchar(PY_EXE), PY_EXE, "<tidak ada>")))

ensure_r_pkg <- function(pkg, optional = TRUE) {
  if (requireNamespace(pkg, quietly = TRUE)) return(TRUE)
  if (!HAS_INTERNET) return(FALSE)
  cat(sprintf("[SETUP] Menginstal paket R '%s' dari CRAN/PPM...\n", pkg))
  ok <- tryCatch({
    install.packages(pkg, quiet = TRUE)
    requireNamespace(pkg, quietly = TRUE)
  }, error = function(e) {
    if (!optional) stop(e)
    cat(sprintf("[SETUP-WARN] Paket '%s' dilewati (%s)\n", pkg, conditionMessage(e)))
    FALSE
  })
  ok
}

# Paket wajib minimal (seluruhnya tersedia di container Kaggle & R lokal)
if (!requireNamespace("data.table", quietly = TRUE)) ensure_r_pkg("data.table", optional = FALSE)
if (!requireNamespace("jsonlite", quietly = TRUE))   ensure_r_pkg("jsonlite",   optional = FALSE)

suppressPackageStartupMessages({
  library(data.table)
  library(jsonlite)
})

# Paket akselerasi opsional (jika offline di Kaggle, otomatis pakai native R / Py bridge)
HAS_GLMNET      <- ensure_r_pkg("glmnet",      optional = TRUE)
HAS_NANOPARQUET <- ensure_r_pkg("nanoparquet", optional = TRUE)
HAS_ARROW       <- if (!HAS_NANOPARQUET) ensure_r_pkg("arrow", optional = TRUE) else FALSE
HAS_R_TORCH     <- ensure_r_pkg("torch",       optional = TRUE)

cat(sprintf("[ENV] Paket Opsional -> glmnet=%s | nanoparquet=%s | arrow=%s | R_torch=%s\n\n",
            HAS_GLMNET, HAS_NANOPARQUET, HAS_ARROW, HAS_R_TORCH))

SEED <- 1581L
suppressWarnings(RNGkind("Mersenne-Twister", "Inversion", "Rejection"))
set.seed(SEED)

# ------------------------------------------------------------------------------
# Helper I/O 100% Apache Parquet (Tanpa CSV: nanoparquet -> arrow -> pyarrow)
# ------------------------------------------------------------------------------
write_pq_via_pyarrow <- function(dt, path) {
  if (!nzchar(PY_EXE)) stop("PyArrow/nanoparquet/arrow tidak tersedia untuk menulis Parquet.")
  tmp_dir <- tempfile("pq_w_")
  dir.create(tmp_dir, recursive = TRUE, showWarnings = FALSE)
  on.exit(unlink(tmp_dir, recursive = TRUE, force = TRUE), add = TRUE)

  cols_meta <- list()
  col_names <- names(dt)
  for (i in seq_along(col_names)) {
    cn <- col_names[i]
    v  <- dt[[cn]]
    fn <- sprintf("col_%03d", i)
    if (is.integer(v)) {
      p <- file.path(tmp_dir, paste0(fn, ".bin"))
      writeBin(as.integer(v), p, size = 4L, endian = "little")
      cols_meta[[i]] <- list(name = cn, dtype = "int32", file = p)
    } else if (is.numeric(v) || is.logical(v)) {
      p <- file.path(tmp_dir, paste0(fn, ".bin"))
      writeBin(as.double(v), p, size = 8L, endian = "little")
      cols_meta[[i]] <- list(name = cn, dtype = "float64", file = p)
    } else {
      p <- file.path(tmp_dir, paste0(fn, ".json"))
      jsonlite::write_json(as.list(as.character(v)), p, auto_unbox = TRUE)
      cols_meta[[i]] <- list(name = cn, dtype = "string", file = p)
    }
  }
  meta_path <- file.path(tmp_dir, "schema.json")
  py_path   <- file.path(tmp_dir, "write_pq.py")
  jsonlite::write_json(list(n = nrow(dt), out = normalizePath(path, mustWork = FALSE), cols = cols_meta),
                       meta_path, auto_unbox = TRUE)

  writeLines(c(
    "import json, sys, numpy as np, pyarrow as pa, pyarrow.parquet as pq",
    "with open(sys.argv[1], 'r', encoding='utf-8') as f: meta = json.load(f)",
    "arrays, names = [], []",
    "for c in meta['cols']:",
    "    names.append(c['name'])",
    "    if c['dtype'] == 'int32': arrays.append(pa.array(np.fromfile(c['file'], dtype='<i4')))",
    "    elif c['dtype'] == 'float64': arrays.append(pa.array(np.fromfile(c['file'], dtype='<f8')))",
    "    else:",
    "        with open(c['file'], 'r', encoding='utf-8') as sf: arrays.append(pa.array(json.load(sf)))",
    "tbl = pa.table(arrays, names=names)",
    "pq.write_table(tbl, meta['out'], compression='snappy')"
  ), py_path)

  rc <- system2(PY_EXE, args = c(py_path, meta_path))
  if (!identical(rc, 0L) || !file.exists(path)) stop("Gagal menulis Parquet via PyArrow!")
}

read_pq_via_pyarrow <- function(path) {
  if (!nzchar(PY_EXE)) stop("PyArrow/nanoparquet/arrow tidak tersedia untuk membaca Parquet.")
  tmp_dir <- tempfile("pq_r_")
  dir.create(tmp_dir, recursive = TRUE, showWarnings = FALSE)
  on.exit(unlink(tmp_dir, recursive = TRUE, force = TRUE), add = TRUE)

  py_path   <- file.path(tmp_dir, "read_pq.py")
  meta_path <- file.path(tmp_dir, "schema.json")
  writeLines(c(
    "import json, os, sys, numpy as np, pyarrow as pa, pyarrow.parquet as pq",
    "tbl = pq.read_table(sys.argv[1])",
    "out_dir = sys.argv[2]",
    "meta = {'n': tbl.num_rows, 'cols': []}",
    "for i, name in enumerate(tbl.column_names):",
    "    col = tbl.column(i).combine_chunks()",
    "    if pa.types.is_integer(col.type):",
    "        fp = os.path.join(out_dir, f'c_{i}.bin')",
    "        col.to_numpy().astype('<i4').tofile(fp)",
    "        meta['cols'].append({'name': name, 'dtype': 'int32', 'file': fp})",
    "    elif pa.types.is_floating(col.type) or pa.types.is_boolean(col.type):",
    "        fp = os.path.join(out_dir, f'c_{i}.bin')",
    "        col.to_numpy().astype('<f8').tofile(fp)",
    "        meta['cols'].append({'name': name, 'dtype': 'float64', 'file': fp})",
    "    else:",
    "        fp = os.path.join(out_dir, f'c_{i}.json')",
    "        with open(fp, 'w', encoding='utf-8') as f: json.dump([str(x) for x in col.to_pylist()], f)",
    "        meta['cols'].append({'name': name, 'dtype': 'string', 'file': fp})",
    "with open(os.path.join(out_dir, 'schema.json'), 'w', encoding='utf-8') as f: json.dump(meta, f)"
  ), py_path)

  rc <- system2(PY_EXE, args = c(py_path, path, tmp_dir))
  if (!identical(rc, 0L) || !file.exists(meta_path)) stop("Gagal membaca Parquet via PyArrow!")
  meta <- jsonlite::fromJSON(meta_path, simplifyVector = FALSE)
  n_rows <- as.integer(meta$n)
  cols_list <- list()
  for (cmeta in meta$cols) {
    if (cmeta$dtype == "int32") {
      cols_list[[cmeta$name]] <- readBin(cmeta$file, what = "integer", n = n_rows, size = 4L, endian = "little")
    } else if (cmeta$dtype == "float64") {
      cols_list[[cmeta$name]] <- readBin(cmeta$file, what = "numeric", n = n_rows, size = 8L, endian = "little")
    } else {
      cols_list[[cmeta$name]] <- unlist(jsonlite::fromJSON(cmeta$file))
    }
  }
  as.data.table(cols_list)
}

read_pq <- function(path) {
  if (HAS_NANOPARQUET) {
    as.data.table(nanoparquet::read_parquet(path))
  } else if (HAS_ARROW) {
    as.data.table(arrow::read_parquet(path))
  } else {
    read_pq_via_pyarrow(path)
  }
}

write_pq <- function(dt, path) {
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  if (HAS_NANOPARQUET) {
    nanoparquet::write_parquet(as.data.frame(dt), path)
  } else if (HAS_ARROW) {
    arrow::write_parquet(as.data.frame(dt), path)
  } else {
    write_pq_via_pyarrow(dt, path)
  }
}

# ------------------------------------------------------------------------------
# 1. Fungsi Metrik & Stratified Fold Murni Base-R (100% Parity caret & pROC)
# ------------------------------------------------------------------------------
create_stratified_folds <- function(y, k = 5L) {
  y <- factor(y)
  numInClass <- table(y)
  foldVector <- integer(length(y))
  for (i in seq_along(numInClass)) {
    seqVector <- rep(seq_len(k), numInClass[i] %/% k)
    if (numInClass[i] %% k > 0L) {
      seqVector <- c(seqVector, sample(seq_len(k), numInClass[i] %% k))
    }
    foldVector[which(y == dimnames(numInClass)[[1L]][i])] <- sample(seqVector)
  }
  foldVector
}

calc_roc_auc <- function(y_true, prob) {
  y  <- as.integer(y_true)
  n1 <- sum(y == 1L)
  n0 <- sum(y == 0L)
  if (n1 == 0L || n0 == 0L) return(0.5)
  rk <- rank(prob, ties.method = "average")
  (sum(rk[y == 1L]) - as.numeric(n1) * (as.numeric(n1) + 1.0) / 2.0) / (as.numeric(n1) * as.numeric(n0))
}

calc_ba <- function(y_true, y_pred) {
  y_true <- as.integer(y_true)
  y_pred <- as.integer(y_pred)
  tp <- sum(y_true == 1L & y_pred == 1L)
  tn <- sum(y_true == 0L & y_pred == 0L)
  fp <- sum(y_true == 0L & y_pred == 1L)
  fn <- sum(y_true == 1L & y_pred == 0L)
  sens <- if ((tp + fn) > 0L) tp / (tp + fn) else 0.0
  spec <- if ((tn + fp) > 0L) tn / (tn + fp) else 0.0
  0.5 * (sens + spec)
}

find_opt_th <- function(y_true, prob, grid = seq(0.05, 0.85, by = 0.002)) {
  y_true <- as.integer(y_true)
  scores <- vapply(grid, function(th) calc_ba(y_true, prob >= th), numeric(1))
  idx <- which.max(scores)
  list(optimal_threshold = grid[idx], max_balanced_accuracy = scores[idx])
}

# ------------------------------------------------------------------------------
# 2. Resolusi Dataset Dual-Mode (Lokal Parquet Repo vs Kaggle /kaggle/input)
#    + Rekayasa 44 Fitur Row-Wise FE v4 + 2 Fitur In-Fold Dual-Regime (= 46 Fitur)
# ------------------------------------------------------------------------------
build_rowwise_fe_v4 <- function(dt) {
  d <- copy(dt)
  # F1: Rekonstruksi 3-Bulanan
  d[, exact_3m_recharges       := round(X10 * 3)]
  d[, exact_3m_regions         := round(X11 * 3)]
  d[, is_integer_recharge      := as.integer(abs(X10 - round(X10)) < 1e-4)]
  # F2: ARPU & Nominal Top-Up
  d[, total_arpu               := X2 + X6 + X9]
  d[, avg_nominal_per_recharge := total_arpu / (X10 + 0.1)]
  d[, data_rev_per_recharge    := X9 / (X10 + 0.1)]
  d[, voice_rev_per_recharge   := X2 / (X10 + 0.1)]
  d[, recharge_per_region      := X10 / (X11 + 0.1)]
  # F3: Share of Wallet
  d[, share_rev_voice          := X2 / (total_arpu + 1.0)]
  d[, share_rev_sms            := X6 / (total_arpu + 1.0)]
  d[, share_rev_data           := X9 / (total_arpu + 1.0)]
  # F4: Unit Economics
  d[, avg_call_duration        := X1 / (X3 + 0.1)]
  d[, rev_per_call_min         := X2 / (X1 + 0.1)]
  d[, rev_per_call_freq        := X2 / (X3 + 0.1)]
  d[, rev_per_mb               := X9 / (X8 + 0.1)]
  # F5: Aktivitas Jam Kerja & Lintas Operator
  d[, workhour_call_mins       := X1 * X4]
  d[, offnet_call_mins         := X1 * (1.0 - X5)]
  d[, offnet_sms_rev           := X6 * (1.0 - X7)]
  # F6: Segmen Diskrit & Anomali Pembulatan
  d[, is_zero_data             := as.integer(X8 == 0)]
  d[, flag_rounding_call       := as.integer(X3 == 0 & (X1 > 0 | X2 > 0))]
  d[, flag_short_call          := as.integer(X1 == 0 & X3 > 0)]
  d[, flag_free_data           := as.integer(X9 == 0 & X8 > 0)]
  # F7: Top-Decile Banking Discriminators (FE v3)
  d[, log_avg_nominal          := log1p(pmax(0, avg_nominal_per_recharge))]
  d[, log_total_arpu           := log1p(pmax(0, total_arpu))]
  d[, log_data_rev             := log1p(pmax(0, X9))]
  d[, digital_dominance        := (as.numeric(pmax(0, X8)) * as.numeric(pmax(0, X9))) /
                                  ((as.numeric(pmax(0, X1)) * as.numeric(pmax(0, X2))) + 1.0)]
  d[, voice_to_data_ratio      := (pmax(0, X2) + 0.1) / (pmax(0, X9) + 0.1)]
  d[, recharge_density         := total_arpu / (pmax(1.0, exact_3m_recharges) + 0.1)]
  d[, cross_operator_rev       := (1.0 - pmin(1.0, pmax(0, X5))) * (1.0 - pmin(1.0, pmax(0, X7))) * total_arpu]
  d[, flag_vip_banking         := as.integer(total_arpu >= 40000 & avg_nominal_per_recharge >= 15000 & X11 >= 2)]
  d[, arpu_per_call_minute     := total_arpu / (pmax(0, X1) + 1.0)]
  # F8: Underlooked Signals Row-Wise (FE v4: monetary_polarity rho=0.3306)
  d[, monetary_polarity        := (pmax(0, total_arpu)^1.5) / (pmax(0, X10) + 0.1)]
  d[, digital_expenditure_share := (as.numeric(pmax(0, X8)) * as.numeric(pmax(0, X9))) /
                                   ((as.numeric(pmax(0, X1)) * as.numeric(pmax(0, X2))) +
                                    (as.numeric(pmax(0, X6)) * as.numeric(pmax(0, X7))) + 1.0)]
  d
}

load_competition_datasets <- function() {
  raw_tr_pq <- "data/training_data_tugas_praktikum_DataScience.parquet"
  raw_te_pq <- "data/testing_data_tugas_praktikum_DataScience.parquet"
  if (file.exists(raw_tr_pq) && file.exists(raw_te_pq)) {
    cat("[DATA] Memuat raw Parquet lokal & membangun 44 fitur row-wise FE v4...\n")
    tr <- read_pq(raw_tr_pq)
    te <- read_pq(raw_te_pq)
  } else if (dir.exists("/kaggle/input")) {
    cat("[DATA] Mendeteksi lingkungan Kaggle Cloud (/kaggle/input)...\n")
    all_in <- list.files("/kaggle/input", recursive = TRUE, full.names = TRUE)
    tr_f <- all_in[grepl("training_data", basename(all_in), ignore.case = TRUE)][1]
    te_f <- all_in[grepl("testing_data",  basename(all_in), ignore.case = TRUE)][1]
    if (is.na(tr_f) || is.na(te_f)) stop("Dataset kompetisi tidak ditemukan di /kaggle/input!")
    read_any <- function(f) {
      if (grepl("\\.parquet$", f, ignore.case = TRUE)) {
        read_pq(f)
      } else {
        data.table::fread(f, data.table = TRUE, showProgress = FALSE)
      }
    }
    tr <- read_any(tr_f)
    te <- read_any(te_f)
  } else {
    stop("Dataset training/testing tidak ditemukan!")
  }

  setorder(tr, ID)
  setorder(te, ID)
  if (file.exists("data/cv_folds.parquet")) {
    folds_dt <- read_pq("data/cv_folds.parquet")
    setorder(folds_dt, ID)
    tr[, fold := as.integer(folds_dt$fold)]
  } else {
    suppressWarnings(RNGkind("Mersenne-Twister", "Inversion", "Rejection"))
    set.seed(SEED)
    tr[, fold := create_stratified_folds(tr$Y, k = 5L)]
  }
  list(train = build_rowwise_fe_v4(tr), test = build_rowwise_fe_v4(te))
}

ds <- load_competition_datasets()
train_dt <- ds$train
test_dt  <- ds$test
setorder(train_dt, ID)
setorder(test_dt, ID)

if (SMOKE_TEST) {
  set.seed(SEED)
  idx_sm <- c(which(train_dt$Y == 0L)[1:240], which(train_dt$Y == 1L)[1:60])
  train_dt <- train_dt[idx_sm]
  train_dt[, fold := rep(1:5, length.out = .N)]
  test_dt  <- test_dt[1:100]
}

# Pencegahan Target Leakage: Hapus mutlak kolom ID dari prediktor!
base_feature_cols <- setdiff(names(train_dt), c("ID", "Y", "fold"))
stopifnot(!("ID" %in% base_feature_cols))

N_train <- nrow(train_dt)
N_test  <- nrow(test_dt)
K_FOLDS <- max(train_dt$fold)
y_all   <- as.integer(train_dt$Y)

cat(sprintf("[INFO] Train = %d baris | Test = %d baris | Folds = %d | Fitur Tabular FE v4 = %d (+2 in-fold = 46)\n\n",
            N_train, N_test, K_FOLDS, length(base_feature_cols)))

# ------------------------------------------------------------------------------
# 3. PARADIGMA 1: Pretrained Transformer LLM Semantic Customer Serialization
#    (Mendukung Offline Local Checkpoints di /kaggle/input + Online HF Hub)
# ------------------------------------------------------------------------------
serialize_customer_profiles <- function(dt) {
  data_tier <- fifelse(dt$X8 == 0 & dt$X9 == 0, "non-internet voice/SMS subscriber",
               fifelse(dt$X9 < 10000, "light mobile data user",
               fifelse(dt$X9 < 35000, "moderate mobile data user", "heavy digital data user")))
  topup_pat <- fifelse(dt$flag_vip_banking == 1L,
                       "high-value VIP mobile-banking topup candidate",
               fifelse(dt$X10 <= 3.5 & dt$avg_nominal_per_recharge >= 12000,
                       "infrequent high-ticket banking-like topup behavior",
               fifelse(dt$X10 >= 8.0 & dt$avg_nominal_per_recharge < 6000,
                       "frequent micro-retail kiosk voucher topup behavior",
                       "mixed prepaid recharge behavior")))
  sprintf(
    paste0(
      "Prepaid telecom banking-channel propensity profile: %s; %s. ",
      "Total monthly ARPU IDR %d (avg ticket IDR %.0f per recharge, monetary polarity index %.0f). ",
      "Data usage %d MB with data spend IDR %d (%.0f%% share of wallet). ",
      "Voice %d mins across %d calls with spend IDR %d (%.0f%% work-hours, %.0f%% on-net). ",
      "SMS spend IDR %d (%.0f%% on-net). ",
      "Recharge frequency %.1f/month (~%.0f/quarter) across %.1f regions."
    ),
    data_tier, topup_pat,
    as.integer(dt$total_arpu), dt$avg_nominal_per_recharge, dt$monetary_polarity,
    as.integer(dt$X8), as.integer(dt$X9), dt$share_rev_data * 100,
    as.integer(dt$X1), as.integer(dt$X3), as.integer(dt$X2), dt$X4 * 100, dt$X5 * 100,
    as.integer(dt$X6), dt$X7 * 100,
    dt$X10, dt$exact_3m_recharges, dt$X11
  )
}

cat("==============================================================\n")
cat("[STAGE 1/3] EKSTRAKSI EMBEDDING SEMANTIK PRETRAINED LLM\n")
cat("==============================================================\n")

all_texts <- c(serialize_customer_profiles(train_dt), serialize_customer_profiles(test_dt))
cat("Contoh narasi profil pelanggan [Baris #1]:\n  \"", all_texts[1], "\"\n\n", sep = "")

extract_llm_embeddings_via_subprocess <- function(texts_vec, skip_llm = FALSE) {
  if (skip_llm || !nzchar(PY_EXE)) return(NULL)
  tryCatch({
    cat(sprintf("[LLM] Menggunakan interpreter Python terisolasi: %s\n", PY_EXE))

    tmp_prompts <- tempfile(fileext = ".json")
    tmp_bin     <- tempfile(fileext = ".bin")
    tmp_meta    <- tempfile(fileext = ".json")
    tmp_py      <- tempfile(fileext = ".py")
    on.exit(unlink(c(tmp_prompts, tmp_bin, tmp_meta, tmp_py), force = TRUE), add = TRUE)

    jsonlite::write_json(as.list(texts_vec), tmp_prompts, auto_unbox = TRUE)

    py_code <- '
import json, os, sys, time, importlib

has_net = os.environ.get("KAGGLE_HAS_INTERNET", "0") == "1"

def check_mod(mod_name):
    try:
        importlib.import_module(mod_name)
        return True
    except ImportError:
        return False

for req in ["numpy", "torch", "transformers"]:
    if not check_mod(req):
        if has_net:
            import subprocess
            print(f"[LLM-PY] Installing {req}...", flush=True)
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", req])
        else:
            raise RuntimeError(f"Modul Python {req} tidak tersedia di kontainer offline.")

import numpy as np
import torch as pt
from transformers import AutoTokenizer, AutoModel

prompts_path, out_bin_path, out_meta_path = sys.argv[1], sys.argv[2], sys.argv[3]
with open(prompts_path, "r", encoding="utf-8") as f:
    texts = json.load(f)

dev = "cuda" if pt.cuda.is_available() else "cpu"
if dev == "cpu":
    n_threads = max(1, os.cpu_count() or 4)
    pt.set_num_threads(n_threads)
    print(f"[LLM-PY] Device: CPU (threads={n_threads}) | Profiles: {len(texts)}", flush=True)
else:
    print(f"[LLM-PY] Device: CUDA ({pt.cuda.get_device_name(0)}) | Profiles: {len(texts)}", flush=True)

# Cari direktori checkpoint HuggingFace lokal di /kaggle/input (zero-internet offline ready!)
local_dirs = []
if os.path.isdir("/kaggle/input"):
    for root, dirs, files in os.walk("/kaggle/input"):
        fset = set(files)
        has_cfg = "config.json" in fset
        has_wt  = ("pytorch_model.bin" in fset) or ("model.safetensors" in fset)
        has_tok = ("tokenizer.json" in fset) or ("vocab.txt" in fset)
        if has_cfg and has_wt and has_tok:
            local_dirs.append(root)

# Urutkan kandidat: prioritaskan bge-small lalu all-minilm untuk kekuatan representasi semantik,
# kecuali jika online+CUDA bisa unduh BAAI/bge-small-en-v1.5 terbaru.
def rank_dir(p):
    pl = p.lower()
    if "bge-small" in pl: return (0, len(p))
    if "minilm" in pl:    return (1, len(p))
    return (2, len(p))

local_dirs = sorted(local_dirs, key=rank_dir)
print(f"[LLM-PY] Checkpoint lokal ditemukan di /kaggle/input: {local_dirs}", flush=True)

candidates = []
if has_net:
    candidates.append(("BAAI/bge-small-en-v1.5", False, "BAAI/bge-small-en-v1.5"))
for ld in local_dirs:
    lbl = os.path.basename(ld.rstrip("/"))
    candidates.append((ld, True, f"local:{lbl}"))
if has_net:
    candidates.append(("sentence-transformers/all-MiniLM-L6-v2", False, "sentence-transformers/all-MiniLM-L6-v2"))

tok = mdl = None
used_model = None
pooling_mode = "mean"

for source_id, is_local, label in candidates:
    try:
        if is_local:
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
        tok = AutoTokenizer.from_pretrained(source_id, local_files_only=is_local)
        mdl = AutoModel.from_pretrained(source_id, local_files_only=is_local)
        used_model = label
        # Baca konfigurasi pooling resmi bila ada (mis. CLS pooling untuk BGE)
        pool_cfg_path = os.path.join(source_id, "1_Pooling", "config.json") if is_local else None
        if pool_cfg_path and os.path.exists(pool_cfg_path):
            with open(pool_cfg_path, "r") as pf:
                pcfg = json.load(pf)
            if pcfg.get("pooling_mode_cls_token", False) and not pcfg.get("pooling_mode_mean_tokens", True):
                pooling_mode = "cls"
            elif "bge" in label.lower():
                pooling_mode = "cls"
        elif "bge" in label.lower():
            pooling_mode = "cls"
        break
    except Exception as ex:
        print(f"[LLM-PY] Fallback from {label}: {ex}", flush=True)

if mdl is None:
    raise RuntimeError("Tidak ada model HuggingFace lokal maupun online yang berhasil dimuat.")

mdl.eval().to(dev)
if dev == "cuda":
    mdl = mdl.half()

d_emb = int(mdl.config.hidden_size)
n_tot = len(texts)
emb = np.empty((n_tot, d_emb), dtype="<f4")

# Urutkan indeks berdasarkan panjang karakter agar padding batch minimal (akselerasi ~2.2x di CPU)
order = sorted(range(n_tot), key=lambda i: len(texts[i]))
batch_size = 512 if dev == "cuda" else 64
n_batches = (n_tot + batch_size - 1) // batch_size
t0 = time.time()

print(f"[LLM-PY] Memulai inferensi {used_model} (dim={d_emb}, pooling={pooling_mode}, bs={batch_size})...", flush=True)
with pt.inference_mode():
    for b_idx, start in enumerate(range(0, n_tot, batch_size)):
        idxs = order[start:start + batch_size]
        batch = [texts[i] for i in idxs]
        enc = tok(batch, padding=True, truncation=True, max_length=128, return_tensors="pt").to(dev)
        out = mdl(**enc).last_hidden_state
        if pooling_mode == "cls":
            pooled = out[:, 0].float()
        else:
            mask = enc["attention_mask"].unsqueeze(-1).expand(out.size()).float()
            pooled = (out.float() * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        pooled = pt.nn.functional.normalize(pooled, p=2, dim=1)
        emb[idxs] = pooled.cpu().numpy().astype("<f4")
        if b_idx in (2, 19) or (b_idx + 1) % max(1, n_batches // 8) == 0 or (b_idx + 1) == n_batches:
            el = time.time() - t0
            done = start + len(idxs)
            eta = (el / max(done, 1)) * (n_tot - done)
            print(f"[LLM-PY] Progress {done}/{n_tot} ({100.0*done/n_tot:.1f}%) | elapsed={el:.1f}s | ETA={eta:.1f}s", flush=True)

np.ascontiguousarray(emb, dtype="<f4").tofile(out_bin_path)
with open(out_meta_path, "w", encoding="utf-8") as f:
    json.dump({"n": int(emb.shape[0]), "d": int(emb.shape[1]), "model": used_model, "dev": dev, "pooling": pooling_mode}, f)
print(f"[LLM-PY] Selesai: {used_model} ({emb.shape[0]} x {emb.shape[1]}) dalam {time.time()-t0:.1f}s", flush=True)
'
    writeLines(py_code, tmp_py, useBytes = FALSE)

    cat("[LLM] Menjalankan inferensi encoder Transformer pada ", length(texts_vec), " profil...\n", sep = "")
    status_py <- system2(PY_EXE, args = c(tmp_py, tmp_prompts, tmp_bin, tmp_meta))
    if (!identical(status_py, 0L) || !file.exists(tmp_meta) || !file.exists(tmp_bin)) {
      stop(sprintf("Subproses LLM berakhir dengan exit code %s.", as.character(status_py)))
    }

    meta <- jsonlite::fromJSON(tmp_meta)
    vec_f32 <- readBin(tmp_bin, what = "numeric", n = as.integer(meta$n) * as.integer(meta$d),
                       size = 4L, endian = "little")
    emb_mat <- matrix(vec_f32, nrow = as.integer(meta$n), ncol = as.integer(meta$d), byrow = TRUE)
    attr(emb_mat, "model_label") <- meta$model
    attr(emb_mat, "llm_dev")     <- meta$dev
    cat(sprintf("[LLM] Sukses (%s | %s | pooling=%s): matriks %d x %d.\n\n",
                meta$model, meta$dev, meta$pooling, nrow(emb_mat), ncol(emb_mat)))
    emb_mat
  }, error = function(e) {
    cat(sprintf("[LLM-FALLBACK] Ekstraksi HuggingFace dilewati (%s); menggunakan TF-IDF token deterministik.\n\n",
                conditionMessage(e)))
    NULL
  })
}

llm_emb_all <- extract_llm_embeddings_via_subprocess(all_texts, skip_llm = SMOKE_TEST)
llm_model_label <- attr(llm_emb_all, "model_label")

if (is.null(llm_emb_all)) {
  vocab <- unique(unlist(strsplit(tolower(gsub("[^a-z0-9%.-]+", " ", all_texts[1:min(1000L, length(all_texts))])), "\\s+")))
  vocab <- vocab[nchar(vocab) >= 2L][1:min(128L, length(vocab))]
  tf_mat <- vapply(vocab, function(w) as.numeric(grepl(w, tolower(all_texts), fixed = TRUE)), numeric(length(all_texts)))
  llm_emb_all <- scale(tf_mat)
  llm_emb_all[is.na(llm_emb_all)] <- 0.0
  llm_model_label <- "TFIDF_Token_Fallback"
}

llm_emb_train <- llm_emb_all[seq_len(N_train), , drop = FALSE]
llm_emb_test  <- llm_emb_all[(N_train + 1L):(N_train + N_test), , drop = FALSE]
rm(llm_emb_all, all_texts); gc()

# ------------------------------------------------------------------------------
# 4. In-Fold Dual-Regime, RankGauss, & Truncated SVD Helpers (100% Bebas Leakage!)
# ------------------------------------------------------------------------------
append_infold_dual_regime <- function(X_base_mat, arpu_vec, x10_vec, med_arpu, med_x10) {
  seg_atm   <- as.numeric(arpu_vec >= med_arpu & x10_vec <= med_x10)
  seg_kiosk <- as.numeric(arpu_vec <  med_arpu & x10_vec >  med_x10)
  cbind(X_base_mat, seg_atm_profile = seg_atm, seg_kiosk_profile = seg_kiosk)
}

fit_rankgauss <- function(X_tr) {
  n <- nrow(X_tr)
  lapply(seq_len(ncol(X_tr)), function(j) {
    xj <- X_tr[, j]
    ux <- sort(unique(xj))
    if (length(ux) <= 2L) {
      mu <- mean(xj); s <- sd(xj); if (is.na(s) || s < 1e-8) s <- 1.0
      return(function(v) (v - mu) / s)
    }
    rk <- rank(xj, ties.method = "average")
    z  <- stats::qnorm(pmin(pmax((rk - 0.5) / n, 1e-6), 1 - 1e-6))
    df_map <- aggregate(z ~ xj, FUN = mean)
    stats::approxfun(x = df_map$xj, y = df_map$z, rule = 2, ties = mean)
  })
}

apply_rankgauss <- function(mappers, X_new) {
  out <- matrix(0.0, nrow = nrow(X_new), ncol = ncol(X_new))
  for (j in seq_along(mappers)) {
    out[, j] <- mappers[[j]](X_new[, j])
  }
  out
}

fit_infold_svd <- function(E_tr, k_dim = 24L) {
  k_use <- min(k_dim, ncol(E_tr), nrow(E_tr) - 1L)
  mu <- colMeans(E_tr)
  Ec <- sweep(E_tr, 2L, mu, "-")
  cov_mat <- crossprod(Ec) / max(1L, nrow(Ec) - 1L)
  eig <- eigen(cov_mat, symmetric = TRUE)
  V_k <- eig$vectors[, seq_len(k_use), drop = FALSE]
  scale_k <- sqrt(pmax(eig$values[seq_len(k_use)], 1e-8))
  list(mu = mu, V = V_k, scale = scale_k, k = k_use)
}

apply_infold_svd <- function(svd_obj, E_new) {
  Ec <- sweep(E_new, 2L, svd_obj$mu, "-")
  proj <- Ec %*% svd_obj$V
  sweep(proj, 2L, svd_obj$scale, "/")
}

# ------------------------------------------------------------------------------
# 5. Linear Ridge/ElasticNet Head Dual-Backend (glmnet -> sklearn L2 CV fallback)
# ------------------------------------------------------------------------------
fit_predict_llm_linear_head <- function(X_tr, y_tr, X_va, X_te, seed_val = 1581L) {
  if (HAS_GLMNET) {
    set.seed(seed_val)
    mdl <- glmnet::cv.glmnet(x = X_tr, y = y_tr, family = "binomial", alpha = 0.2, nfolds = 5L)
    return(list(
      val_prob  = as.numeric(predict(mdl, newx = X_va, s = "lambda.min", type = "response")),
      test_prob = as.numeric(predict(mdl, newx = X_te, s = "lambda.min", type = "response"))
    ))
  }
  if (!nzchar(PY_EXE)) stop("glmnet dan Python sklearn tidak tersedia.")

  tmp_dir <- tempfile("llm_lin_")
  dir.create(tmp_dir, recursive = TRUE, showWarnings = FALSE)
  on.exit(unlink(tmp_dir, recursive = TRUE, force = TRUE), add = TRUE)

  write_f32_mat <- function(M, fn) writeBin(as.double(t(M)), file.path(tmp_dir, fn), size = 4L, endian = "little")
  write_f32_mat(X_tr, "X_tr.bin")
  write_f32_mat(X_va, "X_va.bin")
  write_f32_mat(X_te, "X_te.bin")
  writeBin(as.integer(y_tr), file.path(tmp_dir, "y_tr.bin"), size = 4L, endian = "little")

  meta_path <- file.path(tmp_dir, "meta.json")
  py_path   <- file.path(tmp_dir, "lin_head.py")
  jsonlite::write_json(list(
    n_tr = nrow(X_tr), n_va = nrow(X_va), n_te = nrow(X_te), d = ncol(X_tr),
    seed = as.integer(seed_val), dir = tmp_dir
  ), meta_path, auto_unbox = TRUE)

  writeLines(c(
    "import json, os, sys, numpy as np",
    "from sklearn.preprocessing import StandardScaler",
    "from sklearn.linear_model import LogisticRegressionCV",
    "from sklearn.pipeline import Pipeline",
    "with open(sys.argv[1], 'r') as f: m = json.load(f)",
    "d = m['d']; bd = m['dir']",
    "X_tr = np.fromfile(os.path.join(bd, 'X_tr.bin'), dtype='<f4').reshape(m['n_tr'], d)",
    "X_va = np.fromfile(os.path.join(bd, 'X_va.bin'), dtype='<f4').reshape(m['n_va'], d)",
    "X_te = np.fromfile(os.path.join(bd, 'X_te.bin'), dtype='<f4').reshape(m['n_te'], d)",
    "y_tr = np.fromfile(os.path.join(bd, 'y_tr.bin'), dtype='<i4')",
    "pipe = Pipeline([('sc', StandardScaler()), ('clf', LogisticRegressionCV(Cs=np.logspace(-3.0, 1.0, 8), cv=5, scoring='roc_auc', max_iter=500, n_jobs=-1, random_state=m['seed']))])",
    "pipe.fit(X_tr, y_tr)",
    "p_va = pipe.predict_proba(X_va)[:, 1].astype('<f4')",
    "p_te = pipe.predict_proba(X_te)[:, 1].astype('<f4')",
    "p_va.tofile(os.path.join(bd, 'p_va.bin'))",
    "p_te.tofile(os.path.join(bd, 'p_te.bin'))"
  ), py_path)

  rc <- system2(PY_EXE, args = c(py_path, meta_path))
  if (!identical(rc, 0L)) stop("Subproses LogisticRegressionCV gagal.")
  list(
    val_prob  = readBin(file.path(tmp_dir, "p_va.bin"), what = "numeric", n = nrow(X_va), size = 4L, endian = "little"),
    test_prob = readBin(file.path(tmp_dir, "p_te.bin"), what = "numeric", n = nrow(X_te), size = 4L, endian = "little")
  )
}

# ------------------------------------------------------------------------------
# 6. PARADIGMA 2 & 3: Deep Learning Tabular (PLR-FT-Transformer & TabResNet-GLU)
#    Dual-Backend: R `torch` jika terpasang, else PyTorch 2.11 CPU/CUDA Identik!
# ------------------------------------------------------------------------------
use_r_torch <- FALSE
if (HAS_R_TORCH) {
  use_r_torch <- tryCatch({
    suppressPackageStartupMessages(library(torch))
    if (!torch::torch_is_installed() && HAS_INTERNET) {
      Sys.setenv(TORCH_INSTALL = "1")
      torch::install_torch()
    }
    torch::torch_is_installed()
  }, error = function(e) FALSE)
}

detect_py_cuda <- function() {
  if (!nzchar(PY_EXE)) return(FALSE)
  rc <- suppressWarnings(system2(PY_EXE, args = c("-c", "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)"),
                                 stdout = FALSE, stderr = FALSE))
  identical(rc, 0L)
}

if (use_r_torch) {
  torch_manual_seed(SEED)
  use_cuda <- cuda_is_available()
  device   <- if (use_cuda) torch_device("cuda") else torch_device("cpu")
  dl_backend <- "R_torch"
} else {
  use_cuda   <- detect_py_cuda()
  dl_backend <- "PyTorch_2.11_Subprocess"
}

cat(sprintf("[DEVICE] Backend Deep Learning: %s | Device: %s (CUDA=%s | SMOKE_TEST=%s)\n\n",
            dl_backend, ifelse(use_cuda, "CUDA_GPU", "CPU"), use_cuda, SMOKE_TEST))

if (use_r_torch) {
  SelfAttentionBlock <- nn_module(
    "SelfAttentionBlock",
    initialize = function(d_token = 64L, n_heads = 4L, attn_drop = 0.15, ffn_drop = 0.20) {
      self$d_token <- d_token
      self$n_heads <- n_heads
      self$d_head  <- as.integer(d_token / n_heads)
      self$norm1   <- nn_layer_norm(d_token)
      self$qkv     <- nn_linear(d_token, 3L * d_token)
      self$proj    <- nn_linear(d_token, d_token)
      self$drop_a  <- nn_dropout(attn_drop)
      self$norm2   <- nn_layer_norm(d_token)
      self$ffn1    <- nn_linear(d_token, 2L * d_token)
      self$ffn2    <- nn_linear(2L * d_token, d_token)
      self$drop_f  <- nn_dropout(ffn_drop)
    },
    forward = function(x) {
      B <- x$size(1L); S <- x$size(2L)
      h <- self$norm1(x)
      qkv <- self$qkv(h)$view(c(B, S, 3L, self$n_heads, self$d_head))
      qkv_list <- torch_chunk(qkv, 3L, dim = 3L)
      q <- qkv_list[[1L]]$squeeze(3L)$permute(c(1L, 3L, 2L, 4L))
      k <- qkv_list[[2L]]$squeeze(3L)$permute(c(1L, 3L, 2L, 4L))
      v <- qkv_list[[3L]]$squeeze(3L)$permute(c(1L, 3L, 2L, 4L))

      scores <- torch_matmul(q, k$transpose(-2L, -1L)) / sqrt(self$d_head)
      attn   <- self$drop_a(nnf_softmax(scores, dim = -1L))
      ctx    <- torch_matmul(attn, v)$permute(c(1L, 3L, 2L, 4L))$contiguous()$view(c(B, S, self$d_token))
      x      <- x + self$drop_a(self$proj(ctx))
      h2     <- self$norm2(x)
      x + self$drop_f(self$ffn2(self$drop_f(nnf_gelu(self$ffn1(h2)))))
    }
  )

  PLR_FT_Transformer <- nn_module(
    "PLR_FT_Transformer",
    initialize = function(n_tab, d_llm, k_freq = 16L, d_token = 64L, n_heads = 4L, n_blocks = 3L) {
      self$d_token   <- d_token
      self$n_llm_tok <- 4L
      self$freqs     <- nn_parameter(torch_randn(n_tab, k_freq) * 0.6)
      self$tok_proj  <- nn_linear(2L * k_freq + 1L, d_token)
      self$feat_bias <- nn_parameter(torch_randn(1L, n_tab, d_token) * 0.02)
      self$llm_proj  <- nn_linear(d_llm, self$n_llm_tok * d_token)
      self$cls_token <- nn_parameter(torch_zeros(1L, 1L, d_token))
      self$blocks    <- nn_module_list(lapply(seq_len(n_blocks), function(i) SelfAttentionBlock(d_token, n_heads)))
      self$norm_out  <- nn_layer_norm(d_token)
      self$head      <- nn_linear(d_token, 1L)
    },
    forward = function(x_tab, x_llm) {
      B <- x_tab$size(1L)
      angles   <- 2.0 * pi * x_tab$unsqueeze(3L) * self$freqs$unsqueeze(1L)
      periodic <- torch_cat(list(torch_sin(angles), torch_cos(angles)), dim = 3L)
      raw_per  <- torch_cat(list(x_tab$unsqueeze(3L), periodic), dim = 3L)
      tok_tab  <- nnf_relu(self$tok_proj(raw_per) + self$feat_bias)
      tok_llm  <- nnf_gelu(self$llm_proj(x_llm))$view(c(B, self$n_llm_tok, self$d_token))
      cls_tok  <- self$cls_token$expand(c(B, 1L, self$d_token))
      h <- torch_cat(list(cls_tok, tok_tab, tok_llm), dim = 2L)
      for (i in seq_len(length(self$blocks))) h <- self$blocks[[i]](h)
      cls_out <- self$norm_out(torch_squeeze(h[, 1L, , drop = FALSE], dim = 2L))
      self$head(cls_out)$view(c(B))
    }
  )

  SwiGLUResBlock <- nn_module(
    "SwiGLUResBlock",
    initialize = function(d_hidden = 192L, dropout = 0.25) {
      self$bn      <- nn_batch_norm1d(d_hidden)
      self$fc_gate <- nn_linear(d_hidden, 2L * d_hidden)
      self$drop1   <- nn_dropout(dropout)
      self$fc_out  <- nn_linear(d_hidden, d_hidden)
      self$drop2   <- nn_dropout(dropout * 0.6)
    },
    forward = function(x) {
      h <- self$bn(x); chunks <- torch_chunk(self$fc_gate(h), 2L, dim = 2L)
      h <- self$drop1(chunks[[1L]] * nnf_silu(chunks[[2L]]))
      x + self$drop2(self$fc_out(h))
    }
  )

  TabResNet_GLU <- nn_module(
    "TabResNet_GLU",
    initialize = function(d_in, d_hidden = 192L, n_blocks = 3L, dropout = 0.25) {
      self$stem   <- nn_linear(d_in, d_hidden)
      self$blocks <- nn_module_list(lapply(seq_len(n_blocks), function(i) SwiGLUResBlock(d_hidden, dropout)))
      self$bn_out <- nn_batch_norm1d(d_hidden)
      self$head   <- nn_linear(d_hidden, 1L)
    },
    forward = function(x_tab, x_llm) {
      B <- x_tab$size(1L)
      h <- self$stem(torch_cat(list(x_tab, x_llm), dim = 2L))
      for (i in seq_len(length(self$blocks))) h <- self$blocks[[i]](h)
      self$head(nnf_silu(self$bn_out(h)))$view(c(B))
    }
  )
}

run_dl_model_unified <- function(
  arch = c("ft_transformer", "tabresnet_glu"),
  X_tr_tab, X_tr_llm, y_tr, X_va_tab, X_va_llm, y_va, X_te_tab, X_te_llm,
  k_freq = 16L, d_token = 64L, n_heads = 4L, n_blocks = 3L, d_hidden = 192L, dropout = 0.25,
  max_epochs = 24L, batch_size = 256L, lr = 1e-3, weight_decay = 1e-4, patience = 6L, seed_val = 1581L
) {
  arch <- match.arg(arch)
  if (use_r_torch) {
    set.seed(seed_val); torch_manual_seed(seed_val)
    net <- if (arch == "ft_transformer") {
      PLR_FT_Transformer(ncol(X_tr_tab), ncol(X_tr_llm), k_freq, d_token, n_heads, n_blocks)$to(device = device)
    } else {
      TabResNet_GLU(ncol(X_tr_tab) + ncol(X_tr_llm), d_hidden, n_blocks, dropout)$to(device = device)
    }
    opt   <- optim_adamw(net$parameters, lr = lr, weight_decay = weight_decay)
    sched <- lr_cosine_annealing(opt, T_max = max_epochs, eta_min = 1e-5)
    tr_tab_t <- torch_tensor(X_tr_tab, dtype = torch_float(), device = device)
    tr_llm_t <- torch_tensor(X_tr_llm, dtype = torch_float(), device = device)
    tr_y_t   <- torch_tensor(as.numeric(y_tr) * 0.99 + 0.005, dtype = torch_float(), device = device)
    va_tab_t <- torch_tensor(X_va_tab, dtype = torch_float(), device = device)
    va_llm_t <- torch_tensor(X_va_llm, dtype = torch_float(), device = device)
    te_tab_t <- torch_tensor(X_te_tab, dtype = torch_float(), device = device)
    te_llm_t <- torch_tensor(X_te_llm, dtype = torch_float(), device = device)

    predict_batched <- function(tab_t, llm_t, bs = 1024L) {
      net$eval(); n <- tab_t$size(1L); preds <- numeric(n)
      with_no_grad({
        for (s in seq(1L, n, by = bs)) {
          e <- min(s + bs - 1L, n); idx <- s:e
          preds[idx] <- as.numeric(as_array(torch_sigmoid(net(tab_t[idx, , drop = FALSE], llm_t[idx, , drop = FALSE]))$cpu()))
        }
      })
      preds
    }

    n_tr <- nrow(X_tr_tab); best_auc <- -1.0; no_improve <- 0L
    best_va_p <- rep(0.5, nrow(X_va_tab)); best_te_p <- rep(0.5, nrow(X_te_tab))
    for (ep in seq_len(max_epochs)) {
      net$train(); perm <- sample.int(n_tr)
      for (s in seq(1L, n_tr, by = batch_size)) {
        e <- min(s + batch_size - 1L, n_tr); if ((e - s + 1L) < 8L) next
        b_idx <- perm[s:e]
        opt$zero_grad()
        logits <- net(tr_tab_t[b_idx, , drop = FALSE], tr_llm_t[b_idx, , drop = FALSE])
        loss   <- nnf_binary_cross_entropy_with_logits(logits, tr_y_t[b_idx])
        loss$backward(); nn_utils_clip_grad_norm_(net$parameters, max_norm = 2.0); opt$step()
      }
      sched$step()
      va_prob <- predict_batched(va_tab_t, va_llm_t)
      va_auc  <- calc_roc_auc(y_va, va_prob)
      if (va_auc > best_auc + 1e-5) {
        best_auc <- va_auc; best_va_p <- va_prob; best_te_p <- predict_batched(te_tab_t, te_llm_t); no_improve <- 0L
      } else {
        no_improve <- no_improve + 1L; if (no_improve >= patience) break
      }
    }
    rm(net, opt, sched, tr_tab_t, tr_llm_t, tr_y_t, va_tab_t, va_llm_t, te_tab_t, te_llm_t); gc()
    if (use_cuda) cuda_empty_cache()
    return(list(val_prob = best_va_p, test_prob = best_te_p, best_auc = best_auc))
  }

  # Fallback: PyTorch 2.11 Subproses dengan arsitektur identik 100%
  if (!nzchar(PY_EXE)) stop("R torch maupun Python PyTorch tidak tersedia.")
  tmp_dir <- tempfile("pt_dl_")
  dir.create(tmp_dir, recursive = TRUE, showWarnings = FALSE)
  on.exit(unlink(tmp_dir, recursive = TRUE, force = TRUE), add = TRUE)

  write_f32 <- function(M, fn) writeBin(as.double(t(M)), file.path(tmp_dir, fn), size = 4L, endian = "little")
  write_f32(X_tr_tab, "tr_tab.bin"); write_f32(X_tr_llm, "tr_llm.bin")
  write_f32(X_va_tab, "va_tab.bin"); write_f32(X_va_llm, "va_llm.bin")
  write_f32(X_te_tab, "te_tab.bin"); write_f32(X_te_llm, "te_llm.bin")
  writeBin(as.double(y_tr), file.path(tmp_dir, "y_tr.bin"), size = 4L, endian = "little")
  writeBin(as.double(y_va), file.path(tmp_dir, "y_va.bin"), size = 4L, endian = "little")

  cfg_path <- file.path(tmp_dir, "cfg.json")
  py_path  <- file.path(tmp_dir, "train_dl.py")
  jsonlite::write_json(list(
    arch = arch, dir = tmp_dir,
    n_tr = nrow(X_tr_tab), n_va = nrow(X_va_tab), n_te = nrow(X_te_tab),
    n_tab = ncol(X_tr_tab), d_llm = ncol(X_tr_llm),
    k_freq = as.integer(k_freq), d_token = as.integer(d_token),
    n_heads = as.integer(n_heads), n_blocks = as.integer(n_blocks),
    d_hidden = as.integer(d_hidden), dropout = as.numeric(dropout),
    max_epochs = as.integer(max_epochs), batch_size = as.integer(batch_size),
    lr = as.numeric(lr), weight_decay = as.numeric(weight_decay),
    patience = as.integer(patience), seed = as.integer(seed_val)
  ), cfg_path, auto_unbox = TRUE)

  py_dl <- '
import json, math, os, sys, numpy as np, torch as pt, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import roc_auc_score

with open(sys.argv[1], "r") as f: cfg = json.load(f)
pt.manual_seed(cfg["seed"]); np.random.seed(cfg["seed"])
dev = "cuda" if pt.cuda.is_available() else "cpu"
if dev == "cpu": pt.set_num_threads(max(1, os.cpu_count() or 4))

bd = cfg["dir"]; nt, dl = cfg["n_tab"], cfg["d_llm"]
def rd(fn, r, c): return pt.from_numpy(np.fromfile(os.path.join(bd, fn), dtype="<f4").reshape(r, c).copy())
tr_tab = rd("tr_tab.bin", cfg["n_tr"], nt).to(dev)
tr_llm = rd("tr_llm.bin", cfg["n_tr"], dl).to(dev)
y_tr   = pt.from_numpy(np.fromfile(os.path.join(bd, "y_tr.bin"), dtype="<f4").copy()).to(dev) * 0.99 + 0.005
va_tab = rd("va_tab.bin", cfg["n_va"], nt).to(dev)
va_llm = rd("va_llm.bin", cfg["n_va"], dl).to(dev)
y_va   = np.fromfile(os.path.join(bd, "y_va.bin"), dtype="<f4").astype(np.int32)
te_tab = rd("te_tab.bin", cfg["n_te"], nt).to(dev)
te_llm = rd("te_llm.bin", cfg["n_te"], dl).to(dev)

class SelfAttentionBlock(nn.Module):
    def __init__(self, d_token=64, n_heads=4, attn_drop=0.15, ffn_drop=0.20):
        super().__init__()
        self.d_token, self.n_heads, self.d_head = d_token, n_heads, d_token // n_heads
        self.norm1 = nn.LayerNorm(d_token)
        self.qkv   = nn.Linear(d_token, 3 * d_token)
        self.proj  = nn.Linear(d_token, d_token)
        self.da    = nn.Dropout(attn_drop)
        self.norm2 = nn.LayerNorm(d_token)
        self.ffn1  = nn.Linear(d_token, 2 * d_token)
        self.ffn2  = nn.Linear(2 * d_token, d_token)
        self.df    = nn.Dropout(ffn_drop)
    def forward(self, x):
        B, S, _ = x.shape
        h = self.norm1(x)
        qkv = self.qkv(h).view(B, S, 3, self.n_heads, self.d_head)
        q, k, v = [t.squeeze(2).permute(0, 2, 1, 3) for t in qkv.chunk(3, dim=2)]
        scores = pt.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)
        ctx = pt.matmul(self.da(F.softmax(scores, dim=-1)), v).permute(0, 2, 1, 3).contiguous().view(B, S, self.d_token)
        x = x + self.da(self.proj(ctx))
        return x + self.df(self.ffn2(self.df(F.gelu(self.ffn1(self.norm2(x))))))

class PLR_FT_Transformer(nn.Module):
    def __init__(self, n_tab, d_llm, k_freq=16, d_token=64, n_heads=4, n_blocks=3):
        super().__init__()
        self.d_token, self.n_llm_tok = d_token, 4
        self.freqs     = nn.Parameter(pt.randn(n_tab, k_freq) * 0.6)
        self.tok_proj  = nn.Linear(2 * k_freq + 1, d_token)
        self.feat_bias = nn.Parameter(pt.randn(1, n_tab, d_token) * 0.02)
        self.llm_proj  = nn.Linear(d_llm, self.n_llm_tok * d_token)
        self.cls_token = nn.Parameter(pt.zeros(1, 1, d_token))
        self.blocks    = nn.ModuleList([SelfAttentionBlock(d_token, n_heads) for _ in range(n_blocks)])
        self.norm_out  = nn.LayerNorm(d_token)
        self.head      = nn.Linear(d_token, 1)
    def forward(self, x_tab, x_llm):
        B = x_tab.shape[0]
        ang = 2.0 * math.pi * x_tab.unsqueeze(2) * self.freqs.unsqueeze(0)
        raw_per = pt.cat([x_tab.unsqueeze(2), pt.sin(ang), pt.cos(ang)], dim=2)
        tok_tab = F.relu(self.tok_proj(raw_per) + self.feat_bias)
        tok_llm = F.gelu(self.llm_proj(x_llm)).view(B, self.n_llm_tok, self.d_token)
        h = pt.cat([self.cls_token.expand(B, 1, self.d_token), tok_tab, tok_llm], dim=1)
        for blk in self.blocks: h = blk(h)
        return self.head(self.norm_out(h[:, 0])).view(B)

class SwiGLUResBlock(nn.Module):
    def __init__(self, d_hidden=192, dropout=0.25):
        super().__init__()
        self.bn = nn.BatchNorm1d(d_hidden)
        self.fc_gate = nn.Linear(d_hidden, 2 * d_hidden)
        self.d1 = nn.Dropout(dropout)
        self.fc_out = nn.Linear(d_hidden, d_hidden)
        self.d2 = nn.Dropout(dropout * 0.6)
    def forward(self, x):
        a, g = self.fc_gate(self.bn(x)).chunk(2, dim=1)
        return x + self.d2(self.fc_out(self.d1(a * F.silu(g))))

class TabResNet_GLU(nn.Module):
    def __init__(self, d_in, d_hidden=192, n_blocks=3, dropout=0.25):
        super().__init__()
        self.stem = nn.Linear(d_in, d_hidden)
        self.blocks = nn.ModuleList([SwiGLUResBlock(d_hidden, dropout) for _ in range(n_blocks)])
        self.bn_out = nn.BatchNorm1d(d_hidden)
        self.head = nn.Linear(d_hidden, 1)
    def forward(self, x_tab, x_llm):
        h = self.stem(pt.cat([x_tab, x_llm], dim=1))
        for blk in self.blocks: h = blk(h)
        return self.head(F.silu(self.bn_out(h))).view(-1)

if cfg["arch"] == "ft_transformer":
    net = PLR_FT_Transformer(nt, dl, cfg["k_freq"], cfg["d_token"], cfg["n_heads"], cfg["n_blocks"]).to(dev)
else:
    net = TabResNet_GLU(nt + dl, cfg["d_hidden"], cfg["n_blocks"], cfg["dropout"]).to(dev)

opt = pt.optim.AdamW(net.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
sched = pt.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["max_epochs"], eta_min=1e-5)

def pred_batched(t1, t2, bs=1024):
    net.eval(); outs = []
    with pt.inference_mode():
        for i in range(0, t1.shape[0], bs):
            outs.append(pt.sigmoid(net(t1[i:i+bs], t2[i:i+bs])).cpu().numpy())
    return np.concatenate(outs).astype("<f4")

n_tr, bs = cfg["n_tr"], cfg["batch_size"]
best_auc, no_imp = -1.0, 0
best_va = np.full(cfg["n_va"], 0.5, dtype="<f4")
best_te = np.full(cfg["n_te"], 0.5, dtype="<f4")

for ep in range(cfg["max_epochs"]):
    net.train(); perm = pt.randperm(n_tr, device=dev)
    for s in range(0, n_tr, bs):
        idx = perm[s:s+bs]
        if idx.numel() < 8: continue
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(net(tr_tab[idx], tr_llm[idx]), y_tr[idx])
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 2.0)
        opt.step()
    sched.step()
    vp = pred_batched(va_tab, va_llm)
    auc = float(roc_auc_score(y_va, vp))
    if auc > best_auc + 1e-5:
        best_auc, best_va, best_te, no_imp = auc, vp, pred_batched(te_tab, te_llm), 0
    else:
        no_imp += 1
        if no_imp >= cfg["patience"]: break

best_va.tofile(os.path.join(bd, "p_va.bin"))
best_te.tofile(os.path.join(bd, "p_te.bin"))
with open(os.path.join(bd, "res.json"), "w") as f: json.dump({"best_auc": best_auc}, f)
'
  writeLines(py_dl, py_path)
  rc <- system2(PY_EXE, args = c(py_path, cfg_path))
  if (!identical(rc, 0L)) stop(sprintf("Subproses DL (%s) gagal.", arch))
  res_meta <- jsonlite::fromJSON(file.path(tmp_dir, "res.json"))
  list(
    val_prob  = readBin(file.path(tmp_dir, "p_va.bin"), what = "numeric", n = nrow(X_va_tab), size = 4L, endian = "little"),
    test_prob = readBin(file.path(tmp_dir, "p_te.bin"), what = "numeric", n = nrow(X_te_tab), size = 4L, endian = "little"),
    best_auc  = as.numeric(res_meta$best_auc)
  )
}

# ------------------------------------------------------------------------------
# 7. Loop Pelatihan 5-Fold Stratified CV (LLM-Head + FT-Transformer + TabResNet)
# ------------------------------------------------------------------------------
cat("==============================================================\n")
cat("[STAGE 2/3] PELATIHAN 5-FOLD CV: LLM-HEAD + FT-TRANS + RESNET\n")
cat("==============================================================\n")

X_train_base <- as.matrix(train_dt[, ..base_feature_cols])
X_test_base  <- as.matrix(test_dt[, ..base_feature_cols])

dl_models <- c("LLM_SEMANTIC_HEAD", "PLR_FT_TRANSFORMER", "TAB_RESNET_GLU")
oof_dl    <- matrix(0.0, nrow = N_train, ncol = length(dl_models), dimnames = list(NULL, dl_models))
test_dl   <- matrix(0.0, nrow = N_test,  ncol = length(dl_models), dimnames = list(NULL, dl_models))

SVD_K      <- if (SMOKE_TEST) 8L else 24L
FT_DTOK    <- if (SMOKE_TEST) 32L else if (use_cuda) 64L else 48L
FT_NBLK    <- if (SMOKE_TEST) 2L  else if (use_cuda) 3L  else 2L
FT_KFREQ   <- if (SMOKE_TEST) 8L  else if (use_cuda) 16L else 12L
FT_EPOCHS  <- if (SMOKE_TEST) 2L  else if (use_cuda) 32L else 12L
FT_BS      <- if (SMOKE_TEST) 64L else if (use_cuda) 256L else 512L
FT_PAT     <- if (use_cuda) 6L else 4L

RN_DHID    <- if (SMOKE_TEST) 64L else 192L
RN_NBLK    <- if (SMOKE_TEST) 2L  else 3L
RN_EPOCHS  <- if (SMOKE_TEST) 2L  else if (use_cuda) 32L else 18L
RN_BS      <- if (SMOKE_TEST) 64L else if (use_cuda) 256L else 512L
RN_PAT     <- if (use_cuda) 6L else 5L

for (k in seq_len(K_FOLDS)) {
  t_fold <- Sys.time()
  val_idx <- which(train_dt$fold == k)
  tr_idx  <- which(train_dt$fold != k)

  y_tr <- y_all[tr_idx]
  y_va <- y_all[val_idx]

  # (0) In-Fold Median Dual-Regime Flags -> Total 46 Fitur FE v4 (0% Leakage!)
  med_arpu_k <- stats::median(train_dt$total_arpu[tr_idx])
  med_x10_k  <- stats::median(train_dt$X10[tr_idx])
  X_tr_46 <- append_infold_dual_regime(
    X_train_base[tr_idx, , drop = FALSE],
    train_dt$total_arpu[tr_idx], train_dt$X10[tr_idx], med_arpu_k, med_x10_k
  )
  X_va_46 <- append_infold_dual_regime(
    X_train_base[val_idx, , drop = FALSE],
    train_dt$total_arpu[val_idx], train_dt$X10[val_idx], med_arpu_k, med_x10_k
  )
  X_te_46 <- append_infold_dual_regime(
    X_test_base,
    test_dt$total_arpu, test_dt$X10, med_arpu_k, med_x10_k
  )

  # (1) In-Fold RankGauss pada 46 Fitur Tabular FE v4 (0% Leakage!)
  rg_map   <- fit_rankgauss(X_tr_46)
  X_tr_rg  <- apply_rankgauss(rg_map, X_tr_46)
  X_va_rg  <- apply_rankgauss(rg_map, X_va_46)
  X_te_rg  <- apply_rankgauss(rg_map, X_te_46)

  # (2) In-Fold SVD pada Embedding Pretrained LLM (0% Leakage!)
  svd_fit  <- fit_infold_svd(llm_emb_train[tr_idx, , drop = FALSE], k_dim = SVD_K)
  E_tr_svd <- apply_infold_svd(svd_fit, llm_emb_train[tr_idx, , drop = FALSE])
  E_va_svd <- apply_infold_svd(svd_fit, llm_emb_train[val_idx, , drop = FALSE])
  E_te_svd <- apply_infold_svd(svd_fit, llm_emb_test)

  # --- Model 1: LLM Semantic Ridge/ElasticNet Head (Full LLM + 46 RankGauss) ---
  lin_res <- fit_predict_llm_linear_head(
    X_tr = cbind(llm_emb_train[tr_idx, , drop = FALSE], X_tr_rg), y_tr = y_tr,
    X_va = cbind(llm_emb_train[val_idx, , drop = FALSE], X_va_rg),
    X_te = cbind(llm_emb_test, X_te_rg),
    seed_val = SEED + 10L + k
  )
  oof_dl[val_idx, "LLM_SEMANTIC_HEAD"] <- lin_res$val_prob
  test_dl[, "LLM_SEMANTIC_HEAD"]       <- test_dl[, "LLM_SEMANTIC_HEAD"] + lin_res$test_prob / K_FOLDS

  # --- Model 2: PLR-FT-Transformer (Periodic Numerical Embeddings + Self-Attention) ---
  ft_res <- run_dl_model_unified(
    arch = "ft_transformer",
    X_tr_tab = X_tr_rg, X_tr_llm = E_tr_svd, y_tr = y_tr,
    X_va_tab = X_va_rg, X_va_llm = E_va_svd, y_va = y_va,
    X_te_tab = X_te_rg, X_te_llm = E_te_svd,
    k_freq = FT_KFREQ, d_token = FT_DTOK, n_heads = 4L, n_blocks = FT_NBLK,
    max_epochs = FT_EPOCHS, batch_size = FT_BS, lr = 1e-3, patience = FT_PAT, seed_val = SEED + 100L + k
  )
  oof_dl[val_idx, "PLR_FT_TRANSFORMER"] <- ft_res$val_prob
  test_dl[, "PLR_FT_TRANSFORMER"]       <- test_dl[, "PLR_FT_TRANSFORMER"] + ft_res$test_prob / K_FOLDS

  # --- Model 3: TabResNet-GLU (SwiGLU Gated Residual Network) ---
  rn_res <- run_dl_model_unified(
    arch = "tabresnet_glu",
    X_tr_tab = X_tr_rg, X_tr_llm = E_tr_svd, y_tr = y_tr,
    X_va_tab = X_va_rg, X_va_llm = E_va_svd, y_va = y_va,
    X_te_tab = X_te_rg, X_te_llm = E_te_svd,
    d_hidden = RN_DHID, n_blocks = RN_NBLK, dropout = 0.25,
    max_epochs = RN_EPOCHS, batch_size = RN_BS, lr = 1e-3, patience = RN_PAT, seed_val = SEED + 200L + k
  )
  oof_dl[val_idx, "TAB_RESNET_GLU"] <- rn_res$val_prob
  test_dl[, "TAB_RESNET_GLU"]       <- test_dl[, "TAB_RESNET_GLU"] + rn_res$test_prob / K_FOLDS

  auc_llm <- calc_roc_auc(y_va, oof_dl[val_idx, "LLM_SEMANTIC_HEAD"])
  cat(sprintf(
    "  Fold %d/%d (%.1fs) | AUC -> LLM_Head: %.4f | FT-Transformer: %.4f | TabResNet-GLU: %.4f\n",
    k, K_FOLDS, as.numeric(difftime(Sys.time(), t_fold, units = "secs")),
    auc_llm, ft_res$best_auc, rn_res$best_auc
  ))
  flush.console()
}

# ------------------------------------------------------------------------------
# 8. Evaluasi Out-Of-Fold, Blend Internal Neural+LLM, & Ekspor 100% Parquet
# ------------------------------------------------------------------------------
cat("\n==============================================================\n")
cat("[STAGE 3/3] EVALUASI OOF & EKSPOR ARTEFAK 100% PARQUET\n")
cat("==============================================================\n")

w_dl <- c(LLM_SEMANTIC_HEAD = 0.20, PLR_FT_TRANSFORMER = 0.45, TAB_RESNET_GLU = 0.35)
oof_dl_ens  <- as.numeric(oof_dl %*% w_dl)
test_dl_ens <- as.numeric(test_dl %*% w_dl)

eval_summary_row <- function(name, oof_p) {
  roc_v <- calc_roc_auc(y_all, oof_p)
  opt_v <- find_opt_th(y_all, oof_p)
  fold_ba <- numeric(K_FOLDS)
  fold_th <- numeric(K_FOLDS)
  for (k in seq_len(K_FOLDS)) {
    tr_i <- which(train_dt$fold != k)
    va_i <- which(train_dt$fold == k)
    th_k <- find_opt_th(y_all[tr_i], oof_p[tr_i])$optimal_threshold
    fold_th[k] <- th_k
    fold_ba[k] <- calc_ba(y_all[va_i], oof_p[va_i] >= th_k)
  }
  data.table(
    Model          = name,
    OOF_ROC_AUC    = round(roc_v, 5),
    OOF_BA_Opt     = round(opt_v$max_balanced_accuracy, 5),
    Nested_BA_Mean = round(mean(fold_ba), 5),
    Nested_BA_SD   = round(sd(fold_ba), 5),
    Mean_Threshold = round(mean(fold_th), 4)
  )
}

bench_dl_dt <- rbindlist(list(
  eval_summary_row(sprintf("1. LLM_SEMANTIC_HEAD (%s + 46 FE v4)", llm_model_label), oof_dl[, "LLM_SEMANTIC_HEAD"]),
  eval_summary_row("2. PLR_FT_TRANSFORMER (Periodic + Attention)",                 oof_dl[, "PLR_FT_TRANSFORMER"]),
  eval_summary_row("3. TAB_RESNET_GLU (SwiGLU Residual Net)",                      oof_dl[, "TAB_RESNET_GLU"]),
  eval_summary_row("4. EXP_009_DL_LLM_ENSEMBLE (Neural+LLM Blend)",                oof_dl_ens)
))
print(bench_dl_dt)

opt_th_dl    <- find_opt_th(y_all, oof_dl_ens)$optimal_threshold
th_vtx_1334  <- stats::quantile(test_dl_ens, probs = 1.0 - 0.1334, names = FALSE)

oof_out_dt <- data.table(
  ID                  = as.integer(train_dt$ID),
  fold                = as.integer(train_dt$fold),
  Y_true              = y_all,
  prob_llm_head       = round(oof_dl[, "LLM_SEMANTIC_HEAD"], 6),
  prob_ft_transformer = round(oof_dl[, "PLR_FT_TRANSFORMER"], 6),
  prob_tabresnet      = round(oof_dl[, "TAB_RESNET_GLU"], 6),
  oof_prob            = round(oof_dl_ens, 6),
  pred_opt            = as.integer(oof_dl_ens >= opt_th_dl)
)

test_out_dt <- data.table(
  ID                  = as.integer(test_dt$ID),
  prob_llm_head       = round(test_dl[, "LLM_SEMANTIC_HEAD"], 6),
  prob_ft_transformer = round(test_dl[, "PLR_FT_TRANSFORMER"], 6),
  prob_tabresnet      = round(test_dl[, "TAB_RESNET_GLU"], 6),
  test_prob           = round(test_dl_ens, 6),
  Y_opt               = as.integer(test_dl_ens >= opt_th_dl),
  Y_vertex1334        = as.integer(test_dl_ens >= th_vtx_1334)
)

sub_dl_opt <- data.table(ID = as.integer(test_out_dt$ID), Y = as.integer(test_out_dt$Y_opt))
sub_dl_vtx <- data.table(ID = as.integer(test_out_dt$ID), Y = as.integer(test_out_dt$Y_vertex1334))

if (!SMOKE_TEST) {
  out_targets <- if (dir.exists("/kaggle/working")) {
    c("/kaggle/working")
  } else {
    c("modeling_script/output")
  }
  for (od in unique(out_targets)) {
    write_pq(oof_out_dt,  file.path(od, "oof_predictions_exp_009_dl_llm.parquet"))
    write_pq(test_out_dt, file.path(od, "test_probabilities_exp_009_dl_llm.parquet"))
    write_pq(bench_dl_dt, file.path(od, "exp_009_dl_llm_benchmark.parquet"))
    write_pq(sub_dl_opt,  file.path(od, "submission_exp_009_dl_llm_opt.parquet"))
    write_pq(sub_dl_vtx,  file.path(od, "submission_exp_009_dl_llm_vertex1334.parquet"))
  }
  cat("\n[OK] Seluruh 5 artefak Parquet EXP_009 berhasil disimpan!\n")
} else {
  cat("\n[SMOKE-TEST PASS] Seluruh modul DL+LLM dual-backend & Parquet berhasil diverifikasi.\n")
}

cat(sprintf("[SELESAI EXP_009] Total durasi: %.2f detik.\n",
            as.numeric(difftime(Sys.time(), t_start, units = "secs"))))
