# KAGGLE_AUTO_WIN — Project Praktikum Sains Data (STA1581)

Repositori pengerjaan kompetisi Kaggle pada mata kuliah **Praktikum Sains Data (STA1581)** menggunakan bahasa pemrograman **R** (`pecutAI`).

- **Studi Kasus:** Memprediksi pelanggan pra-bayar telekomunikasi yang potensial (`Y = 1`) untuk diajak menggunakan kanal isi ulang (*recharge*) perbankan (ATM, internet/mobile banking).
- **Metrik Evaluasi Kompetisi:** **Balanced Accuracy** $=\frac{1}{2}(\text{Sensitivity} + \text{Specificity})$.

---

## Peraturan Wajib Repositori (Repository Governance Rules)

1. **Kebijakan Penyimpanan 100% Apache Parquet (`.parquet` — Zero CSV Policy):**
   - Seluruh dataset mentah (`data/`), tabel statistik EDA (`eda_script/output/`), fitur rekayasa (`feature_engineering_script/output/`), prediksi OOF & probabilitas uji, artefak *submission* (`modeling_script/output/` & `data/submission.parquet`), serta buku besar eksperimen & riwayat *submission* (`ledger/*.parquet`) **WAJIB** disimpan dalam format biner terkompresi **Apache Parquet (`.parquet`)** via `nanoparquet`.
   - **DILARANG KERAS** membuat, menyimpan, atau meng-*commit* file `.csv` maupun `.txt` tabular di mana pun di dalam repositori (dicekal otomatis oleh `.gitignore`, Gatekeeper `04_validate_submission.R`, `tools/audit_submissions.ps1`, dan GitHub Actions CI).
   - Apabila pengiriman HTTP ke endpoint API Kaggle membutuhkan aliran byte teks saat pengunggahan, serialisasi dilakukan murni di memori (*ephemeral in-memory*) melalui `tools/submit.ps1` tanpa pernah menulis file `.csv` ke disk.
2. **Bahasa Pemrograman Wajib:** Seluruh skrip analisis, rekayasa fitur, dan pemodelan wajib menggunakan **R** (`.R` / `.Rmd`).
3. **Pencegahan Data Leakage:** Kolom `ID` **wajib dibuang** dari prediktor model (karena pada data latih `ID` `1–36598` terurut kelas `0` dan `ID` `36599–45000` terurut kelas `1`), serta wajib menggunakan skema acak terstratifikasi `data/cv_folds.parquet` (`seed = 1581`).
4. **Keamanan Kredensial:** File `.env` dan `kaggle.json` tidak boleh dibaca ke log publik atau di-*commit* ke Git.
5. **Aturan Agen AI:** Lihat [`AGENTS.md`](AGENTS.md) dan [`AGENT.md`](AGENT.md) untuk protokol operasional lengkap.

---

## Peta Struktur Direktori & Pemetaan Terhadap Workflow

```text
KAGGLE_AUTO_WIN/
├── AGENTS.md                                                    # [ATURAN WORKSPACE AGEN AI — ALWAYS ON]
├── AGENT.md                                                     # [SOP LENGKAP RISET, MODELING & SUBMISSION]
├── configs/                                                     # [KONFIGURASI EKSPERIMEN YAML]
│   ├── exp_001_mvp_xgboost.yaml
│   ├── exp_002_teacher_sim.yaml
│   └── exp_003_fe_ensemble.yaml
│
├── data/                                                        # [INPUT & FINAL SUBMISSION PARQUET]
│   ├── training_data_tugas_praktikum_DataScience.parquet        # Data latih (45.000 baris, ID, X1–X11, Y)
│   ├── testing_data_tugas_praktikum_DataScience.parquet         # Data uji (5.000 baris, ID 45001–50000, X1–X11)
│   ├── cv_folds.parquet                                         # Indeks Stratified 5-Fold CV tetap (seed = 1581)
│   └── submission.parquet                                       # Artefak submission aktif (ID, Y)
│
├── eda_script/                                                  # [PREPRO -> EDA]
│   ├── 01_eda_mvp.R                                             # Skrip cepat EDA untuk pipeline MVP
│   ├── 01_eda_comprehensive.R                                   # Skrip EDA komprehensif & pengujian statistik
│   ├── 01_eda_report.Rmd                                        # Dokumen RMarkdown narasi & sintaks EDA
│   ├── 01_eda_report.html                                       # Laporan HTML standalone hasil render RMarkdown
│   ├── README.md                                                # Ringkasan eksekutif temuan EDA
│   ├── output/                                                  # Tabel Parquet ringkasan statistik, uji kelas, & anomali
│   └── plots/                                                   # Grafik PNG resolusi tinggi untuk laporan & presentasi
│
├── feature_engineering_script/                                  # [PREPRO -> FE]
│   ├── 02_fe_mvp.R                                              # Stratified 5-Fold split & preparasi fitur Vanilla MVP
│   ├── 02_fe_v2.R                                               # Rekayasa 22 fitur domain telekomunikasi (total 33 fitur)
│   └── output/                                                  # Dataset fitur siap pakai (.parquet & .rds)
│
├── modeling_script/                                             # [MODELING, ENSEMBLE & VALIDATION GATE]
│   ├── 03_modeling_xgboost_mvp.R                                # Baseline Vanilla XGBoost (5-Fold CV + Threshold Opt)
│   ├── 04_validate_submission.R                                 # Gatekeeper validasi Parquet & Zero-CSV
│   ├── 05_exp003_fe_ensemble.R                                  # Multi-Model Zoo (XGB+RF+ET+GLMNet) + Ensemble + Gate 0 Probes
│   └── output/                                                  # Artefak OOF, probabilitas uji, metrik & submission (.parquet)
│
├── ledger/                                                      # [BUKU BESAR EKSPERIMEN & SUBMISSION PARQUET]
│   ├── experiments.parquet                                      # Log metrik lintas eksperimen (EXP_001 s.d. EXP_003+)
│   └── submissions.parquet                                      # Registry SHA-256, kuota harian & skor publik Kaggle
│
├── R/                                                           # [MODUL FUNGSI INTI R]
│   ├── metrics.R                                                # Balanced Accuracy, Nested Threshold & Bootstrap CI
│   ├── teacher_sim.R                                            # Simulasi Ground-Truth Ensemble Penyelenggara
│   └── prediction_stability.R                                   # Audit konsistensi & korelasi lintas model (.parquet)
│
├── tools/                                                       # [OTOMASI POWERSHELL PARQUET]
│   ├── audit_submissions.ps1                                    # Audit SHA-256 & kepatuhan Zero-CSV seluruh submission Parquet
│   ├── budget.ps1                                               # Cek sisa kuota harian (5/hari) dari ledger/submissions.parquet
│   └── submit.ps1                                               # Gatekeeper pengiriman Parquet in-memory ke Kaggle REST API
│
├── workflow/                                                    # [ORCHESTRATION & ARCHITECTURE]
│   ├── workflow.png                                             # Diagram arsitektur Hybrid Local (Antigravity) - Kaggle
│   ├── PRD.md                                                   # Product Requirements Document (PRD) MVP
│   ├── PRD_v2.md                                                # PRD Strategi Pemenangan Private Leaderboard (EXP_003+)
│   ├── PRD_SETUP_AUTOMATION.md                                  # Dokumentasi integrasi GitHub & Kaggle Runner
│   ├── kaggle_runner.R                                          # Runner eksekusi untuk Kaggle Private Notebook
│   └── run_mvp_pipeline.R                                       # Master runner eksekusi end-to-end (Tahap 1 s.d. 4)
│
└── meta_data/                                                   # [PROJECT SPECIFICATION & TEAM METADATA]
    ├── Project Praktikum Sains Data.pptx                        # Panduan resmi proyek UAS, aturan Kaggle & kamus data
    └── Kelompok Project UAS.txt                                 # Daftar pembagian anggota kelompok UAS
```

---

## Kamus Data (`X1` – `X11` & Target `Y`)

| Fitur | Deskripsi |
| :--- | :--- |
| `X1` | Rata-rata durasi melakukan panggilan per bulan (menit) |
| `X2` | Rata-rata penghasilan dari transaksi panggilan per bulan (Rp) |
| `X3` | Rata-rata frekuensi melakukan panggilan per bulan |
| `X4` | Persentase panggilan di jam kerja (`0`–`1`) |
| `X5` | Persentase panggilan sesama operator (`0`–`1`) |
| `X6` | Rata-rata penghasilan dari transaksi SMS per bulan (Rp) |
| `X7` | Persentase transaksi SMS sesama operator (`0`–`1`) |
| `X8` | Rata-rata pemakaian data per bulan (MB) |
| `X9` | Rata-rata penghasilan dari transaksi pemakaian data per bulan (Rp) |
| `X10` | Rata-rata frekuensi melakukan recharge pulsa per bulan |
| `X11` | Rata-rata banyaknya wilayah dikunjungi per bulan |
| `Y` | Target biner penggunaan kanal perbankan (`0` = 81,33%, `1` = 18,67%) |

---

## Cara Menjalankan Pipeline & Audit Parquet

```powershell
# 1. Jalankan pipeline MVP end-to-end (100% Parquet)
& "C:\Program Files\R\R-4.6.1\bin\Rscript.exe" workflow/run_mvp_pipeline.R

# 2. Jalankan pipeline Multi-Model Ensemble EXP_003 (100% Parquet)
& "C:\Program Files\R\R-4.6.1\bin\Rscript.exe" modeling_script/05_exp003_fe_ensemble.R

# 3. Audit integritas seluruh file submission Parquet & kebijakan Zero-CSV
powershell -ExecutionPolicy Bypass -File tools/audit_submissions.ps1

# 4. Cek sisa kuota harian Kaggle dari ledger/submissions.parquet
powershell -ExecutionPolicy Bypass -File tools/budget.ps1
```
