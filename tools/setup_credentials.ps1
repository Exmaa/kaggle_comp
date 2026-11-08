# ==============================================================================
# tools/setup_credentials.ps1 — Automated Setup for Antigravity <-> GitHub <-> Kaggle
# Referensi : workflow/PRD_SETUP_AUTOMATION.md (PRD-AUTO-01)
# Kebijakan : Zero-Leakage (kredensial tidak pernah dicetak ke layar/log atau git)
# ==============================================================================

param(
    [string]$KaggleUsername  = "",
    [string]$KaggleKey       = "",
    [string]$CompetitionSlug = "",
    [string]$GitHubRepo      = "",
    [string]$GitHubToken     = ""
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

Write-Host "==============================================================" -ForegroundColor Cyan
Write-Host "  SETUP OTOMASI ANTIGRAVITY <-> GITHUB <-> KAGGLE (PRD-AUTO-01)" -ForegroundColor Cyan
Write-Host "==============================================================" -ForegroundColor Cyan

# 1. Load existing .env if parameters are omitted
$EnvFile = Join-Path $RepoRoot ".env"
if (Test-Path $EnvFile) {
    Get-Content $EnvFile | ForEach-Object {
        if ($_ -match "^\s*([^#][^=]+)=(.*)$") {
            $k = $matches[1].Trim()
            $v = $matches[2].Trim()
            if ($k -eq "KAGGLE_USERNAME" -and [string]::IsNullOrWhiteSpace($KaggleUsername)) { $KaggleUsername = $v }
            if ($k -eq "KAGGLE_KEY" -and [string]::IsNullOrWhiteSpace($KaggleKey)) { $KaggleKey = $v }
            if ($k -eq "COMPETITION_SLUG" -and [string]::IsNullOrWhiteSpace($CompetitionSlug)) { $CompetitionSlug = $v }
            if ($k -eq "GITHUB_REPO" -and [string]::IsNullOrWhiteSpace($GitHubRepo)) { $GitHubRepo = $v }
            if ($k -eq "GITHUB_TOKEN" -and [string]::IsNullOrWhiteSpace($GitHubToken)) { $GitHubToken = $v }
        }
    }
}

# 2. Setup CRED-01: C:\Users\eman\.kaggle\kaggle.json
$KaggleDir = Join-Path $env:USERPROFILE ".kaggle"
$KaggleJsonPath = Join-Path $KaggleDir "kaggle.json"

if (-not [string]::IsNullOrWhiteSpace($KaggleUsername) -and -not [string]::IsNullOrWhiteSpace($KaggleKey)) {
    if (-not (Test-Path $KaggleDir)) {
        New-Item -ItemType Directory -Path $KaggleDir -Force | Out-Null
    }
    $jsonPayload = @{
        username = $KaggleUsername
        key      = $KaggleKey
    } | ConvertTo-Json -Compress

    Set-Content -Path $KaggleJsonPath -Value $jsonPayload -Encoding ASCII -NoNewline
    & icacls $KaggleJsonPath /inheritance:r /grant:r "$($env:USERNAME):(R,W)" | Out-Null
    Write-Host "[OK] CRED-01: File kaggle.json berhasil disimpan di $KaggleJsonPath (izin akses dikunci)." -ForegroundColor Green
} elseif (Test-Path $KaggleJsonPath) {
    $existingJson = Get-Content $KaggleJsonPath -Raw | ConvertFrom-Json
    if ([string]::IsNullOrWhiteSpace($KaggleUsername)) { $KaggleUsername = $existingJson.username }
    if ([string]::IsNullOrWhiteSpace($KaggleKey))      { $KaggleKey      = $existingJson.key }
    Write-Host "[OK] CRED-01: Membaca kaggle.json yang sudah ada di $KaggleJsonPath." -ForegroundColor Green
} else {
    Write-Host "[MENUNGGU] CRED-01: KaggleUsername & KaggleKey belum diisi." -ForegroundColor Yellow
}

# 3. Simpan konfigurasi lokal ke .env (terproteksi oleh .gitignore)
$envLines = @(
    "# File Konfigurasi Lokal (JANGAN DI-COMMIT KE GIT - PRD-AUTO-01)"
    "KAGGLE_USERNAME=$KaggleUsername"
    "KAGGLE_KEY=$KaggleKey"
    "COMPETITION_SLUG=$CompetitionSlug"
    "GITHUB_REPO=$GitHubRepo"
)
Set-Content -Path $EnvFile -Value ($envLines -join "`r`n") -Encoding UTF8
Write-Host "[OK] Konfigurasi lokal disimpan ke .env (dilindungi .gitignore)." -ForegroundColor Green

# 4. Verifikasi Koneksi Native REST API Kaggle (Tanpa ketergantungan .exe eksternal)
if (-not [string]::IsNullOrWhiteSpace($KaggleUsername) -and -not [string]::IsNullOrWhiteSpace($KaggleKey)) {
    try {
        $pair = "${KaggleUsername}:${KaggleKey}"
        $b64  = [Convert]::ToBase64String([Text.Encoding]::ASCII.GetBytes($pair))
        $headers = @{ Authorization = "Basic $b64" }

        if (-not [string]::IsNullOrWhiteSpace($CompetitionSlug)) {
            $url = "https://www.kaggle.com/api/v1/competitions/submissions/list/$CompetitionSlug"
            $resp = Invoke-RestMethod -Uri $url -Headers $headers -Method Get -ErrorAction Stop
            Write-Host "[OK] Koneksi Kaggle API & Kompetisi '$CompetitionSlug' TERVERIFIKASI! (Riwayat submit terdeteksi: $($resp.Count))" -ForegroundColor Green
        } else {
            $url = "https://www.kaggle.com/api/v1/competitions/list?search=data"
            $null = Invoke-RestMethod -Uri $url -Headers $headers -Method Get -ErrorAction Stop
            Write-Host "[OK] Koneksi & Autentikasi Kaggle REST API TERVERIFIKASI!" -ForegroundColor Green
        }
    } catch {
        Write-Host "[WARN] Verifikasi Kaggle API gagal: $($_.Exception.Message)" -ForegroundColor Yellow
    }
}

# 5. Setup CRED-02, CRED-03, CRED-04, CRED-05: GitHub CLI & GitHub Actions Secrets
$GhExe = "C:\Program Files\GitHub CLI\gh.exe"
if (Test-Path $GhExe) {
    if (-not [string]::IsNullOrWhiteSpace($GitHubToken)) {
        Write-Host "[INFO] Melakukan autentikasi GitHub CLI menggunakan token..." -ForegroundColor Cyan
        $GitHubToken | & $GhExe auth login --with-token
        & $GhExe auth setup-git
        Write-Host "[OK] CRED-02: GitHub CLI berhasil login." -ForegroundColor Green
    }

    $prevErr = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    $ghStatus = & $GhExe auth status 2>&1
    $ghExit = $LASTEXITCODE
    $ErrorActionPreference = $prevErr
    if ($ghExit -eq 0) {
        Write-Host "[OK] Status GitHub CLI: Terautentikasi." -ForegroundColor Green

        if (-not [string]::IsNullOrWhiteSpace($GitHubRepo)) {
            if (-not [string]::IsNullOrWhiteSpace($KaggleUsername)) {
                $KaggleUsername | & $GhExe secret set KAGGLE_USERNAME --repo $GitHubRepo
                Write-Host "[OK] CRED-03: GitHub Secret KAGGLE_USERNAME berhasil diatur pada $GitHubRepo." -ForegroundColor Green
            }
            if (-not [string]::IsNullOrWhiteSpace($KaggleKey)) {
                $KaggleKey | & $GhExe secret set KAGGLE_KEY --repo $GitHubRepo
                Write-Host "[OK] CRED-04: GitHub Secret KAGGLE_KEY berhasil diatur pada $GitHubRepo." -ForegroundColor Green
            }
            if (-not [string]::IsNullOrWhiteSpace($CompetitionSlug)) {
                $CompetitionSlug | & $GhExe secret set KAGGLE_COMPETITION --repo $GitHubRepo
                Write-Host "[OK] CRED-05: GitHub Secret KAGGLE_COMPETITION berhasil diatur pada $GitHubRepo." -ForegroundColor Green
            }
        }
    } else {
        Write-Host "[MENUNGGU] CRED-02: GitHub CLI belum login." -ForegroundColor Yellow
    }
} else {
    Write-Host "[WARN] GitHub CLI tidak ditemukan di $GhExe." -ForegroundColor Yellow
}

Write-Host "==============================================================" -ForegroundColor Cyan
Write-Host "  SETUP SELESAI" -ForegroundColor Cyan
Write-Host "==============================================================" -ForegroundColor Cyan
