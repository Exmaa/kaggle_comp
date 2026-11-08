# ==============================================================================
# tools/kaggle_pull.ps1 — Cek Status & Unduh Artefak Parquet dari Kaggle Kernel
# Proyek : KAGGLE_AUTO_WIN (Praktikum Sains Data STA1581 — Kebijakan 100% Parquet)
# Fungsi : Memeriksa status eksekusi Kaggle Private R GPU Kernel dan mengunduh
#          hanya berkas keluaran .parquet (dan .log) ke modeling_script/output/.
# ==============================================================================

param(
    [string]$KernelSlug = "kaggle-auto-win-sta1581-dl-llm-gpu",
    [string]$OutDir     = "modeling_script/output"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

$KaggleUsername = ""
$KaggleKey      = ""
$EnvFile = Join-Path $RepoRoot ".env"
if (Test-Path $EnvFile) {
    Get-Content $EnvFile | ForEach-Object {
        if ($_ -match "^\s*KAGGLE_USERNAME=(.*)$") { $KaggleUsername = $matches[1].Trim().Trim('"').Trim("'") }
        if ($_ -match "^\s*KAGGLE_KEY=(.*)$")      { $KaggleKey      = $matches[1].Trim().Trim('"').Trim("'") }
    }
}

$KaggleJsonPath = Join-Path $env:USERPROFILE ".kaggle\kaggle.json"
if ((Test-Path $KaggleJsonPath) -and ([string]::IsNullOrWhiteSpace($KaggleUsername) -or [string]::IsNullOrWhiteSpace($KaggleKey))) {
    $kj = Get-Content $KaggleJsonPath -Raw | ConvertFrom-Json
    $KaggleUsername = $kj.username
    $KaggleKey      = $kj.key
}

if ([string]::IsNullOrWhiteSpace($KaggleUsername) -or [string]::IsNullOrWhiteSpace($KaggleKey)) {
    Write-Error "KAGGLE_USERNAME dan KAGGLE_KEY belum diatur di .env atau ~/.kaggle/kaggle.json."
    exit 1
}

$pair = "${KaggleUsername}:${KaggleKey}"
$b64  = [Convert]::ToBase64String([Text.Encoding]::ASCII.GetBytes($pair))
$headers = @{ Authorization = "Basic $b64" }

Write-Host "==============================================================" -ForegroundColor Cyan
Write-Host "  KAGGLE KERNEL STATUS & PARQUET OUTPUT PULLER" -ForegroundColor Cyan
Write-Host "==============================================================" -ForegroundColor Cyan

$statusUrl = "https://www.kaggle.com/api/v1/kernels/status?userName=$KaggleUsername&kernelSlug=$KernelSlug"
$stResp = Invoke-RestMethod -Uri $statusUrl -Headers $headers -Method Get

Write-Host "  Kernel       : $KaggleUsername/$KernelSlug"
Write-Host "  Status       : $($stResp.status)" -ForegroundColor $(if ($stResp.status -match "complete") { "Green" } elseif ($stResp.status -match "error") { "Red" } else { "Yellow" })
if ($stResp.failureMessage) {
    Write-Host "  Pesan Error  : $($stResp.failureMessage)" -ForegroundColor Red
}

if ($stResp.status -match "error|cancel") {
    Write-Host "[ERROR] Kernel gagal/dibatalkan di Kaggle Cloud. Memeriksa log keluaran jika tersedia..." -ForegroundColor Red
    try {
        $errOut = Invoke-RestMethod -Uri "https://www.kaggle.com/api/v1/kernels/output?userName=$KaggleUsername&kernelSlug=$KernelSlug" -Headers $headers -Method Get
        if ($errOut.log) { Write-Host $errOut.log }
    } catch {}
    exit 1
}

if ($stResp.status -notmatch "complete") {
    Write-Host "[INFO] Kernel belum berstatus complete (Status: $($stResp.status)). Jalankan kembali perintah ini beberapa menit lagi." -ForegroundColor Yellow
    exit 0
}

if (-not (Test-Path $OutDir)) { New-Item -ItemType Directory -Path $OutDir -Force | Out-Null }

$outUrl = "https://www.kaggle.com/api/v1/kernels/output?userName=$KaggleUsername&kernelSlug=$KernelSlug"
$outResp = Invoke-RestMethod -Uri $outUrl -Headers $headers -Method Get

$downloaded = 0
foreach ($f in $outResp.files) {
    $fname = $f.fileName
    $ext   = [System.IO.Path]::GetExtension($fname).ToLower()
    # Patuhi kebijakan repo 100% Parquet: hanya unduh .parquet (dan .png jika ada plot)
    if ($ext -eq ".parquet" -or $ext -eq ".png") {
        $dest = Join-Path $OutDir ([System.IO.Path]::GetFileName($fname))
        Invoke-WebRequest -Uri $f.url -OutFile $dest -UseBasicParsing
        Write-Host "  [UNDUH] $dest" -ForegroundColor Green
        $downloaded++
    }
}

Write-Host "--------------------------------------------------------------"
Write-Host "[OK] Berhasil mengunduh $downloaded berkas Parquet/PNG ke $OutDir!" -ForegroundColor Green
Write-Host "     Langkah berikutnya: jalankan blending EXP_008 + EXP_009:" -ForegroundColor Cyan
Write-Host '     & "C:\Program Files\R\R-4.6.1\bin\Rscript.exe" modeling_script/12_blend_exp008_exp009.R' -ForegroundColor Yellow
Write-Host "==============================================================" -ForegroundColor Cyan
