# ==============================================================================
# tools/kaggle_run.ps1 — Remote R Kernel Execution on Kaggle Cloud (GPU Ready)
# Proyek    : KAGGLE_AUTO_WIN (Praktikum Sains Data STA1581)
# Fungsi    : Mengunggah skrip R lokal ke Kaggle Private Kernel via REST API
#             (mendukung akselerator GPU untuk Deep Learning & LLM Tabular).
# Catatan   : Skrip ini HANYA menjalankan komputasi privat di Kaggle Kernel,
#             TIDAK melakukan submit ke kompetisi Kaggle.
# ==============================================================================

param(
    [Parameter(Mandatory=$true)]
    [string]$ScriptPath,

    [string]$KernelSlug = "kaggle-auto-win-sta1581-dl-llm-gpu",
    [string]$Title      = "Kaggle Auto Win STA1581 DL LLM GPU",
    [switch]$EnableGpu  = $true
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

$KaggleUsername  = ""
$KaggleKey       = ""
$CompetitionSlug = ""
$EnvFile = Join-Path $RepoRoot ".env"
if (Test-Path $EnvFile) {
    Get-Content $EnvFile | ForEach-Object {
        if ($_ -match "^\s*KAGGLE_USERNAME=(.*)$")  { $KaggleUsername  = $matches[1].Trim().Trim('"').Trim("'") }
        if ($_ -match "^\s*KAGGLE_KEY=(.*)$")       { $KaggleKey       = $matches[1].Trim().Trim('"').Trim("'") }
        if ($_ -match "^\s*COMPETITION_SLUG=(.*)$") { $CompetitionSlug = $matches[1].Trim().Trim('"').Trim("'") }
    }
}

$KaggleJsonPath = Join-Path $env:USERPROFILE ".kaggle\kaggle.json"
if ((Test-Path $KaggleJsonPath) -and ([string]::IsNullOrWhiteSpace($KaggleUsername) -or [string]::IsNullOrWhiteSpace($KaggleKey))) {
    $kj = Get-Content $KaggleJsonPath -Raw | ConvertFrom-Json
    $KaggleUsername = $kj.username
    $KaggleKey      = $kj.key
}

if ([string]::IsNullOrWhiteSpace($KaggleUsername) -or [string]::IsNullOrWhiteSpace($KaggleKey) -or [string]::IsNullOrWhiteSpace($CompetitionSlug)) {
    Write-Error "KAGGLE_USERNAME, KAGGLE_KEY, dan COMPETITION_SLUG wajib diatur terlebih dahulu di .env atau ~/.kaggle/kaggle.json."
    exit 1
}

if (-not (Test-Path $ScriptPath)) {
    Write-Error "File skrip R tidak ditemukan: $ScriptPath"
    exit 1
}

$absScriptPath = (Resolve-Path $ScriptPath).Path
$rawRBytes     = [System.IO.File]::ReadAllBytes($absScriptPath)
$rB64           = [Convert]::ToBase64String($rawRBytes)

# Karena Kaggle API menonaktifkan native language='r', launcher kontainer Python
# ini disusun in-memory (tanpa file .py di repo) untuk mendeteksi konektivitas
# + GPU secara cepat (<3 detik), menginstal paket R hanya bila internet aktif,
# lalu mengeksekusi skrip R asli 100% di Kaggle Cloud.
$launcherTemplate = @'
import base64, os, shutil, socket, subprocess, sys

print("=== KAGGLE CONTAINER R BOOTSTRAP (IN-MEMORY LAUNCHER) ===", flush=True)
has_gpu = False
cuda_ver = None
pt_ver = "N/A"
try:
    import torch
    pt_ver = torch.__version__
    cuda_ver = torch.version.cuda
    has_gpu = bool(torch.cuda.is_available())
    if has_gpu:
        print(f"GPU Device: {torch.cuda.get_device_name(0)}", flush=True)
except Exception as ex:
    print(f"[WARN] PyTorch probe: {ex}", flush=True)

print(f"Python: {sys.executable} | PyTorch: {pt_ver} | CUDA: {cuda_ver} | GPU: {has_gpu} | CPUs: {os.cpu_count()}", flush=True)

def check_online(host="1.1.1.1", port=443, timeout_sec=2.5):
    res = [False]
    def _probe():
        try:
            with socket.create_connection(("cloud.r-project.org", 443), timeout=timeout_sec):
                res[0] = True
        except Exception:
            res[0] = False
    import threading
    t = threading.Thread(target=_probe, daemon=True)
    t.start()
    t.join(timeout_sec + 0.5)
    return res[0]

has_internet = check_online()
print(f"[NET] Internet aktif: {has_internet}", flush=True)

os.environ["DEBIAN_FRONTEND"]   = "noninteractive"
os.environ["RETICULATE_PYTHON"] = sys.executable
os.environ["KAGGLE_PY_EXE"]     = sys.executable
os.environ["KAGGLE_HAS_INTERNET"] = "1" if has_internet else "0"
os.environ["KAGGLE_HAS_GPU"]      = "1" if has_gpu else "0"
os.environ["TORCH_INSTALL"]       = "1" if has_internet else "0"
if cuda_ver:
    os.environ["CUDA"] = str(cuda_ver)

if shutil.which("Rscript") is None and has_internet:
    codename = "noble"
    if os.path.exists("/etc/os-release"):
        with open("/etc/os-release") as f:
            for line in f:
                if line.startswith("VERSION_CODENAME="):
                    codename = line.strip().split("=", 1)[1].strip('"')
                    break
    print(f"[SETUP] Menginstal R + r2u binary packages (Ubuntu {codename})...", flush=True)
    setup_sh = f'''
set -e
apt-get update -qq
apt-get install -y --no-install-recommends wget ca-certificates gnupg sudo lsb-release
if wget -qO /tmp/add_r2u.sh "https://raw.githubusercontent.com/eddelbuettel/r2u/master/inst/scripts/add_cranapt_{codename}.sh"; then
  sed -i 's/apt update/apt-get update/g; s/apt install/apt-get install/g' /tmp/add_r2u.sh
  bash /tmp/add_r2u.sh || true
fi
apt-get update -qq
apt-get install -y --no-install-recommends r-base-core r-base-dev libcurl4-openssl-dev libssl-dev libxml2-dev
'''
    subprocess.run(["bash", "-c", setup_sh], check=True)
else:
    print(f"[SETUP] Rscript siap: {shutil.which('Rscript')}", flush=True)

r_script_path = "/tmp/run_kaggle_r_pipeline.R"
with open(r_script_path, "wb") as f:
    f.write(base64.b64decode("__R_B64__"))

print(f"[RUN] Menjalankan skrip R asli di Kaggle Cloud: {r_script_path}", flush=True)
subprocess.run(["Rscript", "--vanilla", r_script_path], check=True)
'@
$launcherCode = $launcherTemplate.Replace("__R_B64__", $rB64)

$pair = "${KaggleUsername}:${KaggleKey}"
$b64  = [Convert]::ToBase64String([Text.Encoding]::ASCII.GetBytes($pair))
$headers = @{
    Authorization  = "Basic $b64"
    "Content-Type" = "application/json; charset=utf-8"
}

$payload = @{
    slug                   = "$KaggleUsername/$KernelSlug"
    newTitle               = $Title
    text                   = $launcherCode
    language               = "python"
    kernelType             = "script"
    isPrivate              = $true
    enableGpu              = [bool]$EnableGpu
    enableTpu              = $false
    enableInternet         = $true
    datasetDataSources     = @("donkeys/bge-small-en", "shinomoriaoshi/sentencetransformersallminilml6v2")
    competitionDataSources = @($CompetitionSlug)
    kernelDataSources      = @()
    categoryIds            = @()
} | ConvertTo-Json -Depth 4 -Compress
$bodyBytes = [System.Text.Encoding]::UTF8.GetBytes($payload)

Write-Host "==============================================================" -ForegroundColor Cyan
Write-Host "  KAGGLE CLOUD R KERNEL RUNNER (PRIVATE | GPU=$EnableGpu)" -ForegroundColor Cyan
Write-Host "==============================================================" -ForegroundColor Cyan
Write-Host "  Script Lokal : $ScriptPath"
Write-Host "  Target Kernel: $KaggleUsername/$KernelSlug"
Write-Host "  Kompetisi    : $CompetitionSlug"

try {
    $resp = Invoke-RestMethod -Uri "https://www.kaggle.com/api/v1/kernels/push" -Headers $headers -Method Post -Body $bodyBytes
} catch {
    $errBody = ""
    if ($_.Exception.Response) {
        $sr = New-Object System.IO.StreamReader($_.Exception.Response.GetResponseStream())
        $errBody = $sr.ReadToEnd()
    }
    Write-Host "[ERROR] Kaggle API Push Gagal: $($_.Exception.Message)" -ForegroundColor Red
    if ($errBody) { Write-Host "[ERROR-BODY] $errBody" -ForegroundColor Yellow }
    exit 1
}
Write-Host "[OK] Respon API Kaggle Push:" -ForegroundColor Green
Write-Host ($resp | ConvertTo-Json -Depth 4) -ForegroundColor Yellow
Write-Host "     Gunakan 'powershell -ExecutionPolicy Bypass -File tools/kaggle_pull.ps1' untuk cek status & unduh hasil .parquet." -ForegroundColor Cyan
Write-Host "==============================================================" -ForegroundColor Cyan
