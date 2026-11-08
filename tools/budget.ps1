# ==============================================================================
# tools/budget.ps1 — Submission Quota & Kaggle Reset Countdown (Parquet Ledger)
# Proyek : KAGGLE_AUTO_WIN (Praktikum Sains Data STA1581)
# ==============================================================================

param(
    [string]$LedgerPath = "ledger/submissions.parquet",
    [int]$DailyQuota = 5
)

function Get-RscriptPath {
    $cmd = Get-Command Rscript.exe -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    $candidates = Get-ChildItem "C:\Program Files\R\*\bin\Rscript.exe" -ErrorAction SilentlyContinue | Sort-Object FullName -Descending
    if ($candidates) { return $candidates[0].FullName }
    throw "Rscript.exe tidak ditemukan di PATH maupun C:\Program Files\R\"
}

$now = Get-Date

# Kaggle reset setiap 00:00 UTC = 07:00 WIB
$utcNow = $now.ToUniversalTime()
$nextResetUtc = $utcNow.Date.AddDays(1)
$secondsUntilReset = [math]::Round(($nextResetUtc - $utcNow).TotalSeconds)
$hoursLeft = [math]::Floor($secondsUntilReset / 3600)
$minsLeft  = [math]::Floor(($secondsUntilReset % 3600) / 60)

Write-Host "==============================================================" -ForegroundColor Cyan
Write-Host "  KAGGLE SUBMISSION BUDGET & COUNTDOWN (STA1581 - PARQUET)" -ForegroundColor Cyan
Write-Host "==============================================================" -ForegroundColor Cyan
Write-Host "  Waktu Sekarang (Lokal)  : $($now.ToString('yyyy-MM-dd HH:mm:ss'))"
Write-Host "  Waktu Sekarang (UTC)    : $($utcNow.ToString('yyyy-MM-dd HH:mm:ss'))"
Write-Host "  Kaggle Reset Harian     : 00:00 UTC (07:00 WIB)"
Write-Host "  Waktu Tersisa ke Reset  : $hoursLeft Jam $minsLeft Menit" -ForegroundColor Yellow

$todaySubmissions = 0
if (Test-Path $LedgerPath) {
    $Rscript = Get-RscriptPath
    $cntStr  = & $Rscript --vanilla "R/submission_helper.R" "budget" $LedgerPath ($now.ToString("yyyy-MM-dd"))
    if ($cntStr -match "^\s*(\d+)\s*$") {
        $todaySubmissions = [int]$matches[1]
    }
}

$remainingQuota = [math]::Max(0, $DailyQuota - $todaySubmissions)

Write-Host "--------------------------------------------------------------"
Write-Host "  Ledger Aktif            : $LedgerPath"
Write-Host "  Batas Kuota Harian      : $DailyQuota submit"
Write-Host "  Submit Terpakai Hari Ini: $todaySubmissions submit"
Write-Host "  Sisa Kuota Hari Ini     : $remainingQuota submit" -ForegroundColor $(if ($remainingQuota -gt 0) { "Green" } else { "Red" })
Write-Host "--------------------------------------------------------------"

$deadline = [datetime]"2026-11-28 23:59:00"
$daysToDeadline = [math]::Round(($deadline - $now).TotalDays, 1)
$totalBudgetLeft = [math]::Floor($daysToDeadline * $DailyQuota)

Write-Host "  Batas Akhir Kompetisi   : 28 November 2026 23:59 WIB"
Write-Host "  Hari Tersisa            : $daysToDeadline hari"
Write-Host "  Estimasi Kuota Tersisa  : ~$totalBudgetLeft submit total" -ForegroundColor Cyan
Write-Host "==============================================================" -ForegroundColor Cyan
