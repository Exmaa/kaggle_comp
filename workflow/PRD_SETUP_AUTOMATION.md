# Product Requirements Document (PRD) — Setup Otomasi Antigravity <-> GitHub <-> Kaggle

**Kode Dokumen:** `PRD-AUTO-01`  
**Proyek:** `KAGGLE_AUTO_WIN` — Praktikum Sains Data (`STA1581`)  
**Status:** ACTIVE / Terhubung ke `https://github.com/Exmaa/kaggle_comp`  

---

## 1. Ringkasan Eksekutif & Tujuan

Tujuan dari sistem otomasi ini adalah menghubungkan 3 lingkungan kerja secara mulus, aman, dan tanpa friksi manual:
1. **Antigravity (Local Workstation):** Pusat kendali analisis, EDA, *feature engineering*, pengujian *Teacher-Simulation*, dan perancangan konfigurasi eksperimen (`configs/*.yaml`).
2. **GitHub (Version Control & CI/CD Orchestrator):** Penyimpan riwayat kode resmi, pengendali *branch* eksperimen, pengelola rahasia (*GitHub Secrets*), dan pemicu *GitHub Actions* otomatis.
3. **Kaggle (Compute Runner & Evaluation Engine):** Eksekutor komputasi *cloud* berspesifikasi tinggi (GPU/TPU/CPU paralel untuk tuning dan training model) serta penilai skor resmi *Leaderboard*.

---

## 2. Pemetaan Kredensial yang Dibutuhkan

> [!CAUTION]
> **Kebijakan Keamanan Zero-Leakage:** Kredensial API, token, dan kunci privat **DILARANG KERAS** di-*commit* atau diunggah ke dalam repositori git. Seluruh kunci disimpan di level sistem operasi pengguna atau dienkripsi pada *GitHub Secrets*.

### Tabel Kebutuhan Kredensial

| ID Kredensial | Nama Kredensial | Lokasi Penyimpanan | Bentuk / Format | Fungsi Utama |
| :--- | :--- | :--- | :--- | :--- |
| **CRED-01** | `kaggle.json` (Local) | `C:\Users\eman\.kaggle\kaggle.json` | JSON Object: `{"username":"...","key":"..."}` | Mengizinkan skrip lokal PowerShell & Python berinteraksi dengan API Kaggle. |
| **CRED-02** | `gh` Auth Token | Windows Credential Manager | OAuth Token via `gh auth login` | Mengizinkan Antigravity melakukan `git push/pull` dan membuat PR otomatis. |
| **CRED-03** | `KAGGLE_USERNAME` | GitHub Secrets Repo | String Text (Username Kaggle) | Variabel environment untuk GitHub Actions CI/CD runner. |
| **CRED-04** | `KAGGLE_KEY` | GitHub Secrets Repo | String Text (API Key 32 karakter) | Token autentikasi untuk GitHub Actions CI/CD runner. |
| **CRED-05** | `COMPETITION_SLUG` | GitHub Secrets / Config | String URL slug kompetisi | Target kompetisi submit otomatis. |

---

## 3. Spesifikasi Rinci Setup per Komponen

### 3.1 Setup Kredensial Kaggle Lokal (CRED-01)
- **Path Target:** `C:\Users\eman\.kaggle\kaggle.json`
- **Template Isi File:**
  ```json
  {
    "username": "MASUKKAN_USERNAME_KAGGLE_ANDA",
    "key": "MASUKKAN_32_KARAKTER_API_KEY_ANDA"
  }
  ```
- **Cara Mendapatkan:**
  1. Login ke [kaggle.com](https://www.kaggle.com).
  2. Buka menu profil kanan atas -> **Settings**.
  3. Gulir ke bagian **API** -> klik tombol **Create New Token**.
  4. Simpan file hasil unduhan ke path target di atas.

### 3.2 Setup GitHub CLI & Repository Lokal (CRED-02)
- **Status Alat:** `gh.exe` sudah terpasang di `C:\Program Files\GitHub CLI\gh.exe`.
- **Langkah Setup:**
  1. Jalankan di PowerShell:
     ```powershell
     & "C:\Program Files\GitHub CLI\gh.exe" auth login
     ```
  2. Pilih opsi: `GitHub.com` -> `HTTPS` -> `Yes (authenticate Git)` -> `Login with a web browser`.
  3. Masukkan kode 8 karakter di browser dan konfirmasi.
  4. Hubungkan remote repository repositori tim Anda:
     ```powershell
     git remote add origin https://github.com/<USERNAME_ATAU_ORG>/<NAMA_REPO>.git
     ```

### 3.3 Setup GitHub Actions Secrets (CRED-03, CRED-04, CRED-05)
Kredensial ini digunakan saat proses training atau evaluasi dijalankan secara *headless* di *cloud*:
1. Buka repositori GitHub Anda di browser: `https://github.com/<USER>/<REPO>/settings/secrets/actions`.
2. Klik tombol **New repository secret**.
3. Daftarkan 3 rahasia berikut:
   - Secret Name: `KAGGLE_USERNAME` -> Value: *username Kaggle Anda*.
   - Secret Name: `KAGGLE_KEY` -> Value: *API key dari `kaggle.json`*.
   - Secret Name: `KAGGLE_COMPETITION` -> Value: *slug kompetisi* (atau link invitation).

---

## 4. Keamanan & Proteksi `.gitignore`

Repositori lokal telah dikonfigurasi untuk secara eksplisit menolak file sensitif:

```gitignore
# Kredensial dan Token Sensitif
.kaggle/
kaggle.json
*.pem
*.key
.env
.env.*

# Data Besar & Binary Sementara
*.csv.zip
*.tar.gz
```

---

## 5. Alur Kerja Operasional Pasca-Setup

Setelah seluruh kredensial terpasang, siklus kerja menjadi semi-otomatis hingga otomatis penuh:

1. **Iterasi Cepat Lokal:**
   - Anda membuat fitur baru atau konfigurasi di `configs/exp_XXX.yaml`.
   - Jalankan pengujian cepat via `R/teacher_sim.R` untuk mengonfirmasi bahwa skor *worst-case* >= 0.74.
2. **Kaggle Kernel Push (Jika Perlu Cloud Compute):**
   - Jalankan `powershell tools/kaggle_run.ps1 -Config configs/exp_XXX.yaml`.
   - Kode otomatis diunggah ke Kaggle Notebook privat, dijalankan di server Kaggle, dan hasilnya ditarik kembali ke lokal.
3. **Submit Berpagar (Gatekeeper Submit):**
   - Jalankan `powershell tools/submit.ps1 -File modeling_script/output/submission_xxx.csv`.
   - Skrip memeriksa:
     - Apakah kuota hari ini masih tersisa? (Maksimal 5/hari, dicek via `tools/budget.ps1`).
     - Apakah file lolos 9 uji validasi integritas?
     - Jika lolos, kirim ke Kaggle via API -> tarik skor public -> catat otomatis ke `ledger/submissions.csv`.

---

## 6. Formulir Checklist Kesiapan (Ready-to-Fill)

- [x] **Kaggle Token:** File `kaggle.json` sudah terkonfigurasi di lokal.
- [x] **GitHub Login:** Perintah `gh auth login` sudah aktif (`Exmaa`).
- [x] **GitHub Repo Remote:** Repositori lokal sudah terhubung dengan `https://github.com/Exmaa/kaggle_comp.git`.
- [x] **GitHub Secrets:** `KAGGLE_USERNAME`, `KAGGLE_KEY`, `KAGGLE_COMPETITION`, dan `COMPETITION_SLUG` sudah ditambahkan pada GitHub Secrets.
- [x] **Slug Kompetisi:** Informasi nama/slug kompetisi Kaggle sudah terkonfigurasi.
