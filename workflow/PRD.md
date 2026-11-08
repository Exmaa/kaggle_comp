# Product Requirements Document (PRD) & Spesifikasi Workflow — `KAGGLE_AUTO_WIN`

## 1. Gambaran Umum Proyek
- **Mata Kuliah:** Praktikum Sains Data (`STA1581`) — Proyek UAS
- **Studi Kasus Bisnis:** Operator telekomunikasi seluler menjajaki kerja sama dengan bank untuk meningkatkan angka isi ulang (*recharge*) pulsa pelanggan pra-bayar melalui kanal perbankan (ATM, internet/mobile banking, dsb.).
- **Tujuan Pemodelan:** Memprediksi pelanggan pra-bayar yang potensial (`Y = 1`) untuk diajak menggunakan kanal perbankan berdasarkan 11 fitur perilaku penggunaan bulanan (`X1`–`X11`) dari 45.000 pelanggan data latih dan 5.000 pelanggan data uji.
- **Metrik Kompetisi Kaggle:** **Balanced Accuracy**
  $$\text{Balanced Accuracy} = \frac{1}{2}\left(\text{Sensitivity} + \text{Specificity}\right) = \frac{1}{2}\left(\frac{\text{TP}}{\text{TP} + \text{FN}} + \frac{\text{TN}}{\text{TN} + \text{FP}}\right)$$
- **Bahasa Pemrograman Wajib:** **R** (`.R` / `.Rmd`).

---

## 2. Kamus Data (`X1` – `X11` & Target `Y`)

| Kolom | Deskripsi Fitur | Tipe & Karakteristik pada Data Latih ($N = 45.000$) |
| :--- | :--- | :--- |
| `ID` | Identitas unik pelanggan | Integer (`1`–`45000` pada Train, `45001`–`50000` pada Test) |
| `X1` | Rata-rata durasi melakukan panggilan per bulan (menit) | Numerik (`0`–`4130`, median `47`, mean `110,7`) |
| `X2` | Rata-rata penghasilan dari transaksi panggilan per bulan (Rp) | Numerik (`0`–`612.300`, median `22.050`, mean `27.880`) |
| `X3` | Rata-rata frekuensi melakukan panggilan per bulan | Numerik (`0`–`995`, median `24`, mean `40,11`) |
| `X4` | Persentase panggilan di jam kerja | Proporsi `[0, 1]` (median `0,75`, mean `0,7344`) |
| `X5` | Persentase panggilan sesama operator (*on-net*) | Proporsi `[0, 1]` (median `0,9706`, mean `0,8831`) |
| `X6` | Rata-rata penghasilan dari transaksi SMS per bulan (Rp) | Numerik (`0`–`244.400`, median `3.983`, mean `6.746`) |
| `X7` | Persentase transaksi SMS sesama operator (*on-net*) | Proporsi `[0, 1]` (median `0,8631`, mean `0,7723`) |
| `X8` | Rata-rata pemakaian data per bulan (MB) | Numerik (`0`–`59.610`, median `74`, mean `948,9`; **32,59% bernilai 0**) |
| `X9` | Rata-rata penghasilan dari transaksi pemakaian data per bulan (Rp) | Numerik (`0`–`504.100`, median `2.600`, mean `16.170`; **32,91% bernilai 0**) |
| `X10` | Rata-rata frekuensi melakukan recharge pulsa per bulan | Numerik (`0`–`30,7`, pecahan `.0/.3/.7` = rata-rata 3 bulan) |
| `X11` | Rata-rata banyaknya wilayah yang dikunjungi per bulan | Numerik (`0`–`35,7`, pecahan `.0/.3/.7` = rata-rata 3 bulan) |
| `Y` | Target: Penggunaan kanal perbankan (`0` = Tidak, `1` = Ya) | Biner timpang: `0` = `36.598` (**81,33%**), `1` = `8.402` (**18,67%**) |

---

## 3. Arsitektur End-to-End Workflow (MVP s.d. Production)

Alur kerja dibagi menjadi 4 tahapan modular yang dapat dijalankan secara lokal (`SCRIPT LOCAL - ANTIGRAVITY`) maupun diunggah ke `GITHUB` untuk dieksekusi di `KAGGLE REAL RUN`:

1. **Tahap 1 — Exploratory Data Analysis (`eda_script/01_eda_mvp.R`):**
   - Memeriksa integritas skema data latih (`45.000 x 13`) dan data uji (`5.000 x 12`).
   - Menghitung statistik deskriptif per kelas target (`Y = 0` vs `Y = 1`), rasio rata-rata antar-kelas, persentase nilai nol (*zero-inflation*), serta mendeteksi anomali pembulatan 3-bulanan.
   - Menghasilkan visualisasi distribusi kelas, perbandingan fitur utama, dan matriks korelasi pada `eda_script/output/`.

2. **Tahap 2 — Stratified Split & Preprocessing (`feature_engineering_script/02_fe_mvp.R`):**
   - Membangkitkan skema validasi **5-Fold Stratified Cross-Validation** (`set.seed(1581)`) yang konsisten dan menyimpannya ke `data/cv_folds.parquet`.
   - Menyiapkan dataset fitur Vanilla (`X1`–`X11`) untuk MVP (serta mendukung ekstensi fitur rasio/interaksi untuk iterasi berikutnya) ke `feature_engineering_script/output/` dalam format `.parquet` dan `.rds`.

3. **Tahap 3 — Modeling & Optimal Thresholding (`modeling_script/03_modeling_xgboost_mvp.R`):**
   - Melatih model **Vanilla XGBoost** menggunakan 5-Fold Stratified CV dengan *early stopping* berbasis `auc`.
   - Menyimpan prediksi probabilitas *Out-Of-Fold* (`oof_predictions_xgb_mvp.parquet`) dan probabilitas rata-rata 5 fold pada data uji (`test_probabilities_xgb_mvp.parquet`).
   - Melakukan **Nested / Out-Of-Fold Threshold Optimization** untuk mencari ambang batas probabilitas $t^*$ yang memaksimalkan **Balanced Accuracy** (mengatasi ketimpangan kelas 18,67% positif dimana threshold default `0.50` tidak optimal).
   - Mengekspor metrik lengkap ke `modeling_script/output/experiments_log.parquet` dan tingkat kepentingan fitur (*Gain*) ke `modeling_script/output/feature_importance_xgb_mvp.parquet`.

4. **Tahap 4 — Submission Gatekeeper (`modeling_script/04_validate_submission.R`):**
   - Memvalidasi file hasil prediksi Parquet sebelum diunggah ke Kaggle: tepat 5.000 baris, skema kolom `ID, Y`, rentang `ID` `45001`–`50000`, bebas `NA`/`NaN`, hanya bernilai `{0, 1}`, proporsi kelas positif berada dalam batas wajar, serta **0 file teks tabular terlarang di repositori**.
   - Menghasilkan file siap pakai pada `modeling_script/output/submission_xgb_mvp.parquet` dan memperbarui `data/submission.parquet`.

---

## 4. Dokumen Operasional & SOP Terkait

- **Aturan Workspace Agen AI:** [`AGENTS.md`](../AGENTS.md) — Peraturan wajib 100% Apache Parquet, pencegahan *data leakage*, dan proteksi kredensial.
- **SOP Agen Riset & Modeling:** [`AGENT.md`](../AGENT.md) — Panduan lengkap bagi Agen AI dalam mengeksekusi riset, validasi *Teacher-Simulation*, optimasi threshold, dan pencatatan eksperimen.
- **PRD Setup & Otomasi CI/CD:** [`PRD_SETUP_AUTOMATION.md`](PRD_SETUP_AUTOMATION.md) — Kebutuhan setup kredensial, integrasi GitHub Actions, dan Kaggle Runner API.
- **Buku Besar Riset (Parquet):** [`ledger/experiments.parquet`](../ledger/experiments.parquet) dan [`ledger/submissions.parquet`](../ledger/submissions.parquet).
