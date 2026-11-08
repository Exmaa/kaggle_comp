# Peraturan Wajib Repositori KAGGLE_AUTO_WIN (Workspace Rules)

Setiap agen AI yang beroperasi pada repositori ini WAJIB mematuhi peraturan mutlak berikut:

1. **Kebijakan 100% Parquet-Only (Dilarang CSV):**
   - Seluruh berkas data tabular (dataset `data/`, tabel ringkasan `eda_script/output/`, fitur `feature_engineering_script/output/`, probabilitas OOF/Test & file submission `modeling_script/output/`, serta buku besar `ledger/`) **HARUS disimpan dan dibaca secara eksklusif dalam format Apache Parquet (`.parquet`)** menggunakan `nanoparquet::read_parquet()` dan `nanoparquet::write_parquet()`.
   - **DILARANG KERAS** membuat, menulis, menyimpan, atau meng-commit file `.csv` atau `.txt` tabular di mana pun di dalam repositori.
   - Jika endpoint eksternal (seperti pengiriman HTTP ke Kaggle REST API pada `tools/submit.ps1`) membutuhkan aliran byte teks saat upload, konversi hanya boleh dilakukan secara *in-memory* tanpa pernah menulis file `.csv` ke disk repositori.

2. **Bahasa Pemrograman Wajib R:**
   - Seluruh skrip EDA, *feature engineering*, pelatihan model, validasi, dan evaluasi wajib menggunakan bahasa **R** (`C:\Program Files\R\R-4.6.1\bin\Rscript.exe`).

3. **Pencegahan Target Leakage & Disiplin Validasi:**
   - Kolom `ID` wajib selalu dihapus dari matriks fitur pelatihan (karena data latih terurut berdasarkan kelas `Y`).
   - Seluruh eksperimen wajib menggunakan pembagian fold tetap pada `data/cv_folds.parquet` (Stratified K-Fold dengan pengacakan) dan semua transformasi berbasis data/label wajib di-fit di dalam fold.
   - Optimasi ambang batas (*threshold*) untuk metrik **Balanced Accuracy** wajib dilakukan melalui *Nested Cross-Validation* atau prior populasi ($\pi \approx 0.1867$), tidak boleh di-tuning ke *Public Leaderboard*.

4. **Keamanan Kredensial (Zero-Leakage):**
   - Dilarang membaca untuk ditampilkan, mencetak, atau membocorkan isi `.env` maupun `kaggle.json`.
   - Ikuti SOP lengkap siklus 6-tahap pada `@[AGENT.md](AGENT.md)`.

5. **Wajib Persetujuan Eksplisit Pengguna Sebelum Setiap Submit (Zero-Autonomous Submission):**
   - **DILARANG KERAS** mengeksekusi submission ke Kaggle (baik via `tools/submit.ps1`, Kaggle CLI, REST API, GitHub Actions, maupun skrip apa pun) secara mandiri atau otomatis TANPA persetujuan eksplisit dari pengguna dalam sesi percakapan.
   - Agen AI WAJIB menyajikan proposal rencana submit (nama file `.parquet`, hash SHA-256, proporsi positif, estimasi risiko, dan alasan ilmiah) dan HANYA boleh mengeksekusi pengiriman setelah pengguna secara tegas menyatakan persetujuan (misalnya: "submit", "setuju", "kirim", "lanjutkan submit").
