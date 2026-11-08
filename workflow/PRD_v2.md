# PRD v2 — Decision-Driven Modeling Workflow (KAGGLE_AUTO_WIN)

| Item | Isi |
| :--- | :--- |
| Versi | 2.0 (menggantikan bagian strategi pada [PRD.md](PRD.md); kamus data tetap di sana) |
| Tanggal | 6 November 2026 |
| Tim | pecutAI — Praktikum Sains Data (STA1581) |
| Deadline Kaggle | 28 November 2026, 23:59 WIB |
| Bahasa wajib | R |
| Metrik | Balanced Accuracy (BA) = ½ (Sensitivity + Specificity) |

---

## 1. Latar belakang

- Operator telekomunikasi ingin memprediksi pelanggan pra-bayar yang potensial (`Y = 1`) untuk isi ulang lewat kanal bank. Ada 11 fitur perilaku (`X1`–`X11`), 45.000 data train (18,67% positif) dan 5.000 data test.
- Leaderboard dihitung dari 30% data test (public). Skor akhir memakai 70% sisanya (private).
- Kuota 5 submission/hari. Penyelenggara menyatakan *ground truth* adalah output model ensemble.

**Status saat ini:**

| Item | Nilai |
| :--- | :--- |
| Baseline MVP (Vanilla XGBoost, 11 fitur) | CV AUC 0,769, CV BA 0,703 (nested threshold, t* ≈ 0,194) |
| Public score sebelumnya | ~0,95 (tim kami dan tim lain) |
| Public score 6 Nov | **0,475–0,505 untuk kelima tim**; tim kami 0,48225 (terbaru 0,47729) |
| Adversarial validation train vs test | AUC 0,50 (tidak ada covariate shift) |

## 2. Rumusan masalah

1. **Kunci jawaban public berubah.** Semua tim kini di level acak, jadi public leaderboard tidak bisa dipakai untuk memilih model.
2. **Belum diketahui** apakah label evaluasi adalah data nyata, output teacher model, atau kunci yang rusak.
3. **Tanpa proses keputusan yang disiplin**, ~110 submission tersisa berisiko dipakai untuk mengejar noise, sehingga private score turun.

## 3. Tujuan

| ID | Tujuan | Ukuran |
| :--- | :--- | :--- |
| G1 | Memahami mekanisme evaluasi sebelum optimasi | Gate 0 selesai paling lambat 8 Nov |
| G2 | Meningkatkan kualitas model secara terukur | CV BA ≥ 0,72 (dari 0,703), holdout internal konsisten |
| G3 | Memaksimalkan skor private | 2 final dipilih berdasarkan CV dan stabilitas, bukan public |
| G4 | Memenuhi deliverable akademik | `.html` (R Markdown) + `.pdf` presentasi |

## 4. Non-goals

- Tidak memakai model non-R (mis. TabPFN, LLM sebagai prediktor) untuk submission.
- Tidak menyetel model atau threshold berdasarkan public leaderboard.
- Tidak membangun GNN pada tahap ini (tidak ada graph alami antar pelanggan).

## 5. Pengguna

| Pengguna | Kebutuhan |
| :--- | :--- |
| Anggota tim (4 orang) | Pipeline yang bisa dijalankan ulang dengan hasil identik, log eksperimen bersama |
| Dosen/penilai | Tahapan pemodelan yang jelas, visualisasi, interpretasi, feature engineering |
| Kaggle | File `ID,Y` valid, 5.000 baris, nilai {0, 1} |

---

## 6. Kebutuhan fungsional

### 6.0 Gate 0 — Audit evaluasi (prioritas tertinggi)

| ID | Kebutuhan |
| :--- | :--- |
| FR-0.1 | Membuat file probe: **P1** semua 0; **P2** prediksi MVP dibalik; **P3** `Y=1` untuk ~19% ID terakhir (ID ≥ 49067); **P4** ensemble threshold 0,5 (~6% positif); **P5** resubmit file lama yang byte-identik |
| FR-0.2 | Setiap probe dicatat di registry (hash SHA-256, timestamp, hipotesis, skor) |
| FR-0.3 | Aturan keputusan (lihat tabel di bawah) diterapkan dan hasilnya didokumentasikan |

| Hasil probe | Keputusan |
| :--- | :--- |
| P1 ≠ 0,50000 | Stop. Laporkan ke penyelenggara (metrik atau format bermasalah) |
| P2 ≈ 0,70 | Kunci terbalik → Jalur A dengan label dibalik |
| P3 ≫ 0,5 | Kunci berurutan per ID → laporkan, dokumentasikan |
| P4 ≫ 0,5 | Label = teacher model → **Jalur B** |
| Semua ≈ 0,5 | Kunci acak/rusak → **Jalur C** + tanyakan di Discussion |
| Tidak ada indikasi khusus, kunci diperbaiki | **Jalur A** |

**Jalur:**
- **A (label nyata):** optimasi CV BA, threshold dari nested CV.
- **B (label teacher):** tiru teacher (distillation), positive rate mengikuti hasil P4.
- **C (kunci acak):** abaikan LB, optimasi CV seperti Jalur A agar siap jika kunci diperbaiki.

### 6.1 S1 — Validasi

| ID | Kebutuhan | Kriteria terima |
| :--- | :--- | :--- |
| FR-1.1 | Repeated Stratified 5-Fold × 3, seed tetap, disimpan ke file | Proporsi positif tiap fold 18,67% ± 0,1% |
| FR-1.2 | Holdout internal 15%, tidak pernah dipakai tuning | Disimpan sebagai daftar ID terpisah |
| FR-1.3 | SE BA dari bootstrap OOF | Tercatat di log tiap eksperimen |
| FR-1.4 | Aturan perubahan nyata | **Δ CV > 2 × SE** |
| FR-1.5 | Kolom `ID` tidak boleh jadi fitur | Assertion di pipeline |

### 6.2 S2 — Preprocessing

| ID | Kebutuhan |
| :--- | :--- |
| FR-2.1 | Model pohon: fitur mentah tanpa scaling |
| FR-2.2 | Model non-pohon: `log1p` (uang/MB), rank/quantile transform, standardisasi, opsional winsorize 0,5–99,5 persentil |
| FR-2.3 | Semua transformasi yang belajar dari data di-fit **di dalam fold** |
| FR-2.4 | Resampling (SMOTE/undersampling) hanya dipakai jika CV naik > 2 SE |

### 6.3 S3 — Feature engineering

| ID | Kelompok | Fitur |
| :--- | :--- | :--- |
| FR-3.1 | F1 Hitungan asli | `round(3·X10)`, `round(3·X11)` |
| FR-3.2 | F2 Nominal isi ulang | `(X2+X6+X9)/(X10+0,1)` |
| FR-3.3 | F3 Komposisi belanja | `X9/tot`, `X2/tot`, `X6/tot`, `tot` |
| FR-3.4 | F4 Unit economics | `X2/X1`, `X2/X3`, `X1/X3`, `X9/X8` |
| FR-3.5 | F5 Off-net & jam kerja | `X1·(1−X5)`, `X1·X4`, `X6·(1−X7)` |
| FR-3.6 | F6 Flag | `X8==0`, `X3==0 & X2>0`, `X9==0 & X8>0`, `X1==0 & X3>0` |
| FR-3.7 | F7 Target encoding bin | Mean Y per bin kuantil, OOF di dalam fold |
| FR-3.8 | F8 Fitur kNN/graph | Mean Y k tetangga (OOF), jarak ke centroid kelas |

**Kriteria terima per kelompok:** dipertahankan jika Δ CV AUC > 2 SE (di GBDT dan/atau linear). Diuji satu kelompok per eksperimen.

### 6.4 S4 — Seleksi fitur

| ID | Kebutuhan |
| :--- | :--- |
| FR-4.1 | Permutation importance OOF, null importance (50× Y diacak), stability selection |
| FR-4.2 | Fitur dibuang jika gagal ≥ 2 dari 3 metode **dan** CV tanpa fitur itu tidak turun > 1 SE |

### 6.5 S5 — Model zoo

| ID | Model | Paket R |
| :--- | :--- | :--- |
| FR-5.1 | XGBoost, LightGBM, CatBoost | `xgboost`, `lightgbm`, `catboost` |
| FR-5.2 | Random Forest / ExtraTrees | `ranger` |
| FR-5.3 | Elastic net + spline, GAM | `glmnet`, `splines`, `mgcv` |
| FR-5.4 | kNN, SVM-RBF, MLP | `kknn`, `kernlab`, `nnet`/`torch` |
| FR-5.5 | Setiap model menyimpan `oof_<exp>.parquet` dan `test_<exp>.parquet` | |

**Kriteria masuk ensemble:** CV AUC ≥ AUC terbaik − 0,02, **atau** korelasi Spearman OOF dengan model terbaik < 0,90.

### 6.6 S6 — Tuning

| ID | Kebutuhan |
| :--- | :--- |
| FR-6.1 | Metrik tuning: AUC / log-loss OOF (bukan BA) |
| FR-6.2 | Random search 30–50 konfigurasi → Bayesian optimization 30–50 iterasi jika naik > 2 SE |
| FR-6.3 | Validasi di holdout internal; jika selisih dengan CV > 2 SE, pakai konfigurasi lebih sederhana |
| FR-6.4 | Model final = rata-rata 5–10 seed |

### 6.7 S7 — Ensemble

| Level | Metode | Naik level jika |
| :--- | :--- | :--- |
| E1 | Seed averaging | Selalu |
| E2 | Rank average | CV naik > 2 SE dari model tunggal terbaik |
| E3 | Weighted average (hill climbing Caruana, di dalam CV) | CV naik > 2 SE dari E2 |
| E4 | Stacking `glmnet` non-negatif di atas logit OOF | CV naik > 2 SE dari E3 dan holdout konsisten |
| E5 | Ensemble per segmen (`X8==0` vs `X8>0`) | Ada model yang jelas unggul di satu segmen |

### 6.8 S8 — Threshold

| ID | Kebutuhan |
| :--- | :--- |
| FR-8.1 | Jalur A/C: threshold dari nested CV, dinyatakan sebagai positive rate (persentil) |
| FR-8.2 | Jalur B: positive rate mengikuti hasil probe P4 |
| FR-8.3 | Threshold dikunci; tidak disetel ulang dari public LB |

### 6.9 S9 — Final

| ID | Kebutuhan |
| :--- | :--- |
| FR-9.1 | Final 1: ensemble CV BA tertinggi yang lolos holdout |
| FR-9.2 | Final 2: ensemble paling stabil (SD CV terendah) atau versi Jalur B |
| FR-9.3 | Kedua final **dipilih manual** di tab Submissions sebelum 28 Nov 23:59 |

### 6.10 Submission & registry

| ID | Kebutuhan |
| :--- | :--- |
| FR-10.1 | Seluruh artefak submission di repositori wajib disimpan dalam format biner **Apache Parquet (`.parquet`)** (kebijakan Zero-CSV, bebas kerusakan pemisah lokal Excel) |
| FR-10.2 | Gatekeeper ([04_validate_submission.R](../modeling_script/04_validate_submission.R)) dan [audit_submissions.ps1](../tools/audit_submissions.ps1) wajib lolos sebelum upload |
| FR-10.3 | Registry [`ledger/submissions.parquet`](../ledger/submissions.parquet): timestamp, file `.parquet`, SHA-256, eksperimen, threshold, positive rate, public score, catatan pengumuman |

---

## 7. Kebutuhan non-fungsional

| ID | Kebutuhan |
| :--- | :--- |
| NFR-1 | Reproducible: satu perintah `Rscript workflow/run_pipeline.R --exp <nama>`; seed, fold, versi paket tercatat (`renv`) |
| NFR-2 | Bebas leakage: semua langkah yang memakai label/statistik data di-fit di dalam fold |
| NFR-3 | Waktu: satu eksperimen model tunggal 5×3 fold < 15 menit di laptop lokal |
| NFR-4 | Artefak: OOF dan prediksi test tiap eksperimen disimpan (parquet) sehingga ensemble tidak perlu melatih ulang |
| NFR-5 | Kepatuhan: 100% R; Kaggle Notebook (jika dipakai) private; nama tim sesuai aturan |

---

## 8. Metrik keberhasilan

| Metrik | Target |
| :--- | :--- |
| CV BA (repeated, nested threshold) | ≥ 0,72 |
| Selisih CV vs holdout internal | < 2 SE |
| SD CV BA antar repeat | < 0,01 |
| Gate 0 terjawab | ≤ 8 Nov, ≤ 6 submission |
| Submission terpakai untuk milestone | Hanya yang lolos aturan Δ CV > 2 SE |

---

## 9. Timeline

| Tanggal | Milestone | Output |
| :--- | :--- | :--- |
| 6–8 Nov | Gate 0 + registry + fold repeated + holdout | Keputusan jalur A/B/C |
| 8–12 Nov | S2–S4: preprocessing, feature bank, seleksi | Set fitur v1 |
| 10–17 Nov | S5–S6: model zoo, tuning, seed averaging | OOF/test 6–10 model |
| 15–21 Nov | S7: ensemble E1–E5, fitur kNN | Ensemble v1–v3 |
| 21–25 Nov | S8–S9: threshold, uji robustness | 2 kandidat final |
| 25–27 Nov | Laporan `.Rmd → .html` + slide `.pdf` | Deliverable akademik |
| 28 Nov 23:59 | Pilih 2 final manual | — |

**Anggaran submission (~110):** 6–8 Nov probe (≤ 6); 8–20 Nov 1–2/hari untuk milestone; 20–27 Nov kandidat final; sisanya cadangan.

---

## 10. Risiko & mitigasi

| Risiko | Dampak | Mitigasi |
| :--- | :--- | :--- |
| Kunci jawaban berubah lagi | Skor LB tidak bermakna | Andalkan CV; registry + resubmit identik untuk deteksi perubahan |
| Overfit ke public LB | Private turun | Aturan 2 SE, threshold dikunci, final dipilih manual |
| Leakage dari target encoding / kNN feature | CV terlalu optimis | Fold-in-fold, cek di holdout internal |
| Overfit ke CV karena tuning berlebihan | Private turun | Holdout internal, konfigurasi sederhana jika tidak konsisten |
| File rusak karena Excel | Skor salah | Upload hanya file teks dari R, gatekeeper wajib |
| Paket R (lightgbm/catboost) sulit dipasang di Windows | Model zoo berkurang | Jalankan di Kaggle R Notebook (private) atau fokus xgboost + ranger + glmnet |

---

## 11. Pertanyaan terbuka

1. Apakah penyelenggara mengumumkan perubahan kunci jawaban atau aturan evaluasi (cek Discussion/Overview)?
2. Apakah tim lain bersedia berbagi hasil probe P1–P4 agar kuota tidak terbuang?
3. Apakah skor final dinilai dari private Kaggle saja, atau dosen juga menilai CV dan proses?
