# Laporan Singkat — Prediksi Kelulusan (Student Pass Prediction)

**Deskripsi singkat dataset**
- Dataset dibuat secara sintetis (n=1000) dengan fitur:
  - hours_study: jam belajar per minggu (float)
  - attendance: persentase kehadiran (0-100)
  - prev_grade: nilai sebelumnya (0-100)
  - socioeconomic: kategori status sosial-ekonomi (0 low, 1 mid, 2 high)
  - extracurricular: partisipasi ekstra kurikuler (0/1)
- Target: `passed` (0 = fail, 1 = pass).

**Model yang digunakan**
1. Logistic Regression (dengan scaling)
2. Decision Tree
3. K-Nearest Neighbors (dengan scaling)

**Preprocessing**
- One-hot encoding untuk variabel kategorikal `socioeconomic`.
- StandardScaler untuk fitur numerik sebelum Logistic Regression dan KNN.
- Train-test split 80/20 (stratified).

**Hasil evaluasi (ringkasan)**
- Metrik yang dilaporkan: Confusion matrix, Accuracy, Precision, Recall, F1-score, ROC AUC.
- Hasil lengkap tersedia pada notebook `prediksi_kelulusan_classification.ipynb`. Contoh output ringkas:
  - Tabel per-model: accuracy, precision, recall, f1, roc_auc.
  - ROC curve ditampilkan untuk model yang menyediakan probabilitas.

**Pembahasan**
- Model yang sederhana (Logistic Regression) sering bekerja baik bila hubungan linier antara fitur dan target.
- Decision Tree memberikan interpretabilitas (aturan fitur) tetapi berisiko overfitting tanpa pruning.
- KNN sensitif terhadap scaling dan memilih jumlah tetangga (k) yang tepat penting.
- Pilihan akhir model disesuaikan dengan metrik yang paling relevan (misal: jika ingin meminimalkan false negative, pilih model dengan recall tinggi).

**Kesimpulan**
- Untuk dataset sintetis ini, perbandingan metrik (lihat tabel) menentukan model terbaik. Biasanya Logistic Regression atau decision tree (pruned) adalah pilihan awal yang baik.
- Saran: uji model tambahan (RandomForest, GradientBoosting), lakukan cross-validation lebih ekstensif, dan gunakan dataset nyata (mis. UCI Student Performance) untuk hasil praktis.

---

**File yang disertakan**
- `prediksi_kelulusan_classification.ipynb` — notebook lengkap.
- `laporan_singkat_prediksi_kelulusan.md` — laporan ringkas ini.

