# ==============================================================================
# tools/audit_submissions.ps1 — Parquet Submission Quality & SHA-256 Hash Audit
# Proyek : KAGGLE_AUTO_WIN (Praktikum Sains Data STA1581)
# Aturan : 100% Apache Parquet (.parquet) — Bebas file .csv di seluruh repo
# ==============================================================================

param(
    [string]$TargetDir = "modeling_script/output",
    [string]$LedgerPath = "ledger/submissions.parquet"
)

function Get-RscriptPath {
    $cmd = Get-Command Rscript.exe -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    $candidates = Get-ChildItem "C:\Program Files\R\*\bin\Rscript.exe" -ErrorAction SilentlyContinue | Sort-Object FullName -Descending
    if ($candidates) { return $candidates[0].FullName }
    throw "Rscript.exe tidak ditemukan di PATH maupun C:\Program Files\R\"
}

Write-Host "==============================================================" -ForegroundColor Cyan
Write-Host "  AUDIT SUBMISSION PARQUET & KEBIJAKAN ZERO-CSV (GATEKEEPER)" -ForegroundColor Cyan
Write-Host "==============================================================" -ForegroundColor Cyan

# 1. Verifikasi Kebijakan Zero-CSV di Seluruh Repositori
$csvFiles = @(Get-ChildItem -Recurse -File -Filter "*.csv" -ErrorAction SilentlyContinue | Where-Object { $_.FullName -notmatch '\\\.git\\' })
if ($csvFiles.Count -gt 0) {
    Write-Host "[FAIL] Ditemukan $($csvFiles.Count) file .csv ilegal di repositori:" -ForegroundColor Red
    $csvFiles | ForEach-Object { Write-Host "  - $($_.FullName)" -ForegroundColor Red }
    exit 1
} else {
    Write-Host "[PASS] Kebijakan Repo 100% Parquet: 0 file .csv ditemukan." -ForegroundColor Green
}

# 2. Kumpulkan file submission*.parquet (kecuali *summary*)
$files = @()
if (Test-Path "data/submission.parquet") {
    $files += Get-Item "data/submission.parquet"
}
if (Test-Path $TargetDir) {
    $files += Get-ChildItem -Path $TargetDir -Filter "submission*.parquet" -ErrorAction SilentlyContinue | Where-Object { $_.Name -notlike "*summary*" }
}

if ($files.Count -eq 0) {
    Write-Host "Tidak ada file submission*.parquet ditemukan di data/ maupun $TargetDir" -ForegroundColor Yellow
    exit 0
}

$Rscript = Get-RscriptPath
$joinedPaths = ($files | ForEach-Object { $_.FullName }) -join "|"
$rOutput = & $Rscript --vanilla "R/submission_helper.R" "audit" $joinedPaths

$results = @()
foreach ($line in $rOutput) {
    if ([string]::IsNullOrWhiteSpace($line)) { continue }
    $parts = $line -split "`t"
    if ($parts.Count -lt 6) { continue }
    $fullPath = $parts[0]
    $hash = (Get-FileHash -Path $fullPath -Algorithm SHA256).Hash
    $relLabel = if ($fullPath -match "data[\\/]submission\.parquet$") { "data/submission.parquet" } else { [System.IO.Path]::GetFileName($fullPath) }
    $results += [PSCustomObject]@{
        FileName = $relLabel
        Status   = $parts[1]
        Rows     = [int]$parts[2]
        PosCount = [int]$parts[3]
        PosPct   = $parts[4]
        Format   = "PARQUET"
        SHA256   = $hash.Substring(0, 16) + "..."
        Issues   = $parts[5]
    }
}

$results | Format-Table -AutoSize

$failCount = @($results | Where-Object { $_.Status -ne "PASS" }).Count
Write-Host "==============================================================" -ForegroundColor Cyan
if ($failCount -gt 0) {
    Write-Host "PERINGATAN: Terdapat $failCount file berstatus FAIL!" -ForegroundColor Red
    exit 1
} else {
    Write-Host "Catatan: Seluruh $($results.Count) artefak Parquet berstatus PASS dan siap pakai." -ForegroundColor Green
}
Write-Host "==============================================================" -ForegroundColor Cyan
