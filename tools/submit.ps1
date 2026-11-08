# ==============================================================================
# tools/submit.ps1 — Gatekeeper Automated Submission to Kaggle (100% Parquet)
# Referensi : AGENT.md & AGENTS.md (Kebijakan Wajib 100% Parquet)
# Fungsi    :
#   1. Wajib menerima input file .parquet (menolak file teks/CSV di disk)
#   2. Cek sisa kuota harian (maks 5/hari) via ledger/submissions.parquet
#   3. Validasi integritas Parquet (5.000 baris, skema ID & Y, tanpa NA, biner {0,1})
#   4. Cek duplikasi SHA-256 terhadap riwayat di ledger/submissions.parquet
#   5. Serialisasi payload di memori (in-memory stream, 0 file teks di disk) &
#      catat otomatis ke ledger/submissions.parquet
# ==============================================================================

param(
    [Parameter(Mandatory=$true)]
    [string]$File,

    [Parameter(Mandatory=$true)]
    [string]$ModelName,

    [string]$Threshold = "0.1867",
    [string]$Message   = "",
    [string]$CompetitionSlug = "",
    [string]$LedgerPath = "ledger/submissions.parquet",
    [int]$DailyQuota = 5,
    [switch]$ForceResubmit,
    [switch]$DryRun,
    [switch]$UserConfirmed
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

function Get-RscriptPath {
    $cmd = Get-Command Rscript.exe -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    $candidates = Get-ChildItem "C:\Program Files\R\*\bin\Rscript.exe" -ErrorAction SilentlyContinue | Sort-Object FullName -Descending
    if ($candidates) { return $candidates[0].FullName }
    throw "Rscript.exe tidak ditemukan di PATH maupun C:\Program Files\R\"
}

# 1. Load konfigurasi dari .env & kaggle.json
$KaggleUsername = ""
$KaggleKey      = ""
$EnvFile = Join-Path $RepoRoot ".env"
if (Test-Path $EnvFile) {
    Get-Content $EnvFile | ForEach-Object {
        if ($_ -match "^\s*KAGGLE_USERNAME=(.*)$")  { $KaggleUsername  = $matches[1].Trim() }
        if ($_ -match "^\s*KAGGLE_KEY=(.*)$")       { $KaggleKey       = $matches[1].Trim() }
        if ($_ -match "^\s*COMPETITION_SLUG=(.*)$" -and [string]::IsNullOrWhiteSpace($CompetitionSlug)) {
            $CompetitionSlug = $matches[1].Trim()
        }
    }
}

$KaggleJsonPath = Join-Path $env:USERPROFILE ".kaggle\kaggle.json"
if ((Test-Path $KaggleJsonPath) -and ([string]::IsNullOrWhiteSpace($KaggleUsername) -or [string]::IsNullOrWhiteSpace($KaggleKey))) {
    $kj = Get-Content $KaggleJsonPath -Raw | ConvertFrom-Json
    $KaggleUsername = $kj.username
    $KaggleKey      = $kj.key
}

Write-Host "==============================================================" -ForegroundColor Cyan
Write-Host "  KAGGLE PARQUET GATEKEEPER SUBMISSION PIPELINE (STA1581)" -ForegroundColor Cyan
Write-Host "==============================================================" -ForegroundColor Cyan

if (-not (Test-Path $File)) {
    Write-Error "GAGAL [Gate 1]: File submission tidak ditemukan di '$File'."
    exit 1
}

if ([System.IO.Path]::GetExtension($File).ToLower() -ne ".parquet") {
    Write-Error "GAGAL [Gate 1b]: Kebijakan repositori mewajibkan 100% file .parquet! File '$File' ditolak."
    exit 1
}

$Rscript = Get-RscriptPath
$OutputEncoding = [System.Text.UTF8Encoding]::new($false)
$hash = (Get-FileHash -Path $File -Algorithm SHA256).Hash
$now  = Get-Date

# 2. Baca Ledger Parquet & Validasi File Submission Parquet via R
$resolvedSub = (Resolve-Path $File).Path
$resolvedLedger = if ([System.IO.Path]::IsPathRooted($LedgerPath)) { $LedgerPath } else { Join-Path $RepoRoot $LedgerPath }
$todayStr = $now.ToString("yyyy-MM-dd")

$metaRaw = & $Rscript --vanilla "R/submission_helper.R" "validate" $resolvedSub $resolvedLedger $todayStr $hash
$meta = $metaRaw | ConvertFrom-Json

$remainingQuota = [math]::Max(0, $DailyQuota - [int]$meta.today_cnt)
Write-Host "  Sisa Kuota Hari Ini : $remainingQuota / $DailyQuota submit" -ForegroundColor $(if ($remainingQuota -gt 0) { "Green" } else { "Red" })
if ($remainingQuota -le 0 -and -not $DryRun) {
    Remove-Item Env:\SUB_FILE -ErrorAction SilentlyContinue
    Write-Error "GAGAL [Gate 2]: Kuota harian ($DailyQuota submit/hari) telah habis! Tunggu reset pukul 07:00 WIB."
    exit 1
}

$pct1   = [double]$meta.pct1
$count1 = [int]$meta.count1
$count0 = [int]$meta.count0

$checks = [ordered]@{
    "Format artefak murni Apache Parquet (.parquet)"   = $true
    "Skema kolom persis c('ID', 'Y')"                  = [bool]$meta.cols_ok
    "Jumlah baris prediksi tepat 5.000"                = ([int]$meta.rows -eq 5000)
    "ID berurutan 45001 s.d. 50000"                    = [bool]$meta.id_ok
    "Bebas nilai NA & murni integer biner {0, 1}"      = ([bool]$meta.na_ok -and [bool]$meta.bin_ok)
    "Proporsi positif dalam rentang wajar atau probe"  = (($pct1 -ge 5 -and $pct1 -le 60) -or [bool]$ForceResubmit)
}

$allPass = $true
foreach ($k in $checks.Keys) {
    $ok = $checks[$k]
    $lbl = if ($ok) { "[PASS]" } else { "[FAIL]"; $allPass = $false }
    $col = if ($ok) { "Green" } else { "Red" }
    Write-Host ("  {0} {1}" -f $lbl, $k) -ForegroundColor $col
}

Write-Host "  Ringkasan Prediksi  : Y=1 sebanyak $count1 baris ($pct1%), Y=0 sebanyak $count0 baris"
Write-Host "  SHA-256 Parquet     : $hash"

if (-not $allPass) {
    Remove-Item Env:\SUB_FILE -ErrorAction SilentlyContinue
    Write-Error "GAGAL [Gate 3]: File Parquet tidak lolos pemeriksaan integritas Gatekeeper!"
    exit 1
}

if (-not [string]::IsNullOrWhiteSpace($meta.dup_id) -and -not $ForceResubmit) {
    Remove-Item Env:\SUB_FILE -ErrorAction SilentlyContinue
    Write-Error "GAGAL [Gate 4]: File Parquet dengan SHA-256 identik sudah pernah di-submit ($($meta.dup_id)). Gunakan -ForceResubmit jika sengaja melakukan Probe P5."
    exit 1
}

if ($DryRun) {
    Remove-Item Env:\SUB_FILE -ErrorAction SilentlyContinue
    Write-Host "[DRY-RUN] Seluruh gerbang validasi Parquet LOLOS. Tidak melakukan pengiriman ke Kaggle API." -ForegroundColor Yellow
    exit 0
}

if ([string]::IsNullOrWhiteSpace($CompetitionSlug) -or [string]::IsNullOrWhiteSpace($KaggleUsername) -or [string]::IsNullOrWhiteSpace($KaggleKey)) {
    Remove-Item Env:\SUB_FILE -ErrorAction SilentlyContinue
    Write-Error "GAGAL [Gate 5]: Kredensial Kaggle atau COMPETITION_SLUG belum diatur. Jalankan tools/setup_credentials.ps1 terlebih dahulu."
    exit 1
}

if ([string]::IsNullOrWhiteSpace($Message)) {
    $Message = "$ModelName | thr=$Threshold | pos=$pct1% | sha256=$($hash.Substring(0,12))"
}

# 4b. Pemeriksaan Wajib Persetujuan Pengguna (Security Gatekeeper)
if (-not $UserConfirmed) {
    Remove-Item Env:\SUB_FILE -ErrorAction SilentlyContinue
    Write-Host ""
    Write-Host "==============================================================" -ForegroundColor Red
    Write-Host "[SECURITY BLOCKER] SUBMISSION KE KAGGLE DIBATALKAN!" -ForegroundColor Red
    Write-Host "Aturan Mutlak Repositori (Rule 5 AGENTS.md):" -ForegroundColor Yellow
    Write-Host "DILARANG KERAS men-submit apapun ke Kaggle tanpa persetujuan eksplisit dari pengguna." -ForegroundColor Yellow
    Write-Host "Pengiriman hanya boleh dieksekusi setelah pengguna menyetujui secara tertulis" -ForegroundColor Yellow
    Write-Host "dan skrip dipanggil dengan parameter: -UserConfirmed" -ForegroundColor Cyan
    Write-Host "==============================================================" -ForegroundColor Red
    exit 1
}

# 5. Submit Artefak Parquet via Kaggle CLI
Write-Host "[INFO] Mengirim artefak Parquet ke kompetisi Kaggle '$CompetitionSlug'..." -ForegroundColor Cyan

& python -m kaggle competitions submit -f $resolvedSub -m $Message $CompetitionSlug
if ($LASTEXITCODE -ne 0) {
    Write-Error "GAGAL [Gate 5b]: Pengiriman ke Kaggle API gagal dengan exit code $LASTEXITCODE."
    exit 1
}

# 6. Catat ke ledger/submissions.parquet
$nextNum = [int]$meta.ledger_rows + 1
$subId   = ("SUB_{0:D3}" -f $nextNum)
$ts      = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")

$recRes = & $Rscript --vanilla "R/submission_helper.R" "record" $resolvedLedger $subId $ts $fileInfo.Name $hash $ModelName ([string]$Threshold) ([string]$count1) ([string]$pct1) $Message

Write-Host "[OK] Submission berhasil dikirim ke Kaggle dan dicatat ke $LedgerPath sebagai $subId!" -ForegroundColor Green

Write-Host "[INFO] Menunggu 10 detik untuk scoring leaderboard Kaggle..." -ForegroundColor Cyan
Start-Sleep -Seconds 10
& $Rscript --vanilla "tools/sync_kaggle_to_ledger.R"
& $Rscript --vanilla "tools/fetch_kaggle_submissions.R"

Write-Host "==============================================================" -ForegroundColor Cyan
