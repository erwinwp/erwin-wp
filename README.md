<div style="page-break-after: always;"></div>

# Laporan Ujian Tengah Semester

## Pembelajaran Mesin dengan Python
## Semester 1 2025-2026

**Dataset:** Global Coffee Health Dataset (Sintetis)
**Tugas:** Klasifikasi Multikelas Variabel `Health_Issues`

<br><br><br><br><br><br><br><br><br><br>

**Disusun oleh:**
*(Silakan isi nama dan NPM Anda di sini)*
**Nama:** ...
**NPM:** ...

<br><br><br><br>

**PROGRAM STUDI MATEMATIKA**
**FAKULTAS SAINS**
**UNIVERSITAS KATOLIK PARAHYANGAN**
**2025**

<div style="page-break-after: always;"></div>

## Daftar Isi

**1. Pendahuluan Masalah**
    * 1.1 Latar Belakang Masalah
    * 1.2 Variabel Respon dan Prediktor
    * 1.3 Pembagian Set Data dan Validasi

**2. Metode**
    * 2.1 Pra-pemrosesan Data
        * 2.1.1 Pemuatan dan Pembersihan Data Awal
        * 2.1.2 Penanganan Data Hilang (*Missing Values*)
        * 2.1.3 Encoding Variabel Target (Ordinal)
        * 2.1.4 Encoding Variabel Prediktor (Fitur)
        * 2.1.5 Standardisasi Fitur Numerik
        * 2.1.6 Konstruksi Pipeline Pra-pemrosesan
    * 2.2 Model dan Transformasi
        * 2.2.1 Model 1: Logistic Regression (ovr) + PCA
        * 2.2.2 Model 2: Random Forest Classifier
        * 2.2.3 Model 3: HistGradientBoostingClassifier
    * 2.3 Optimasi dan Validasi Model
        * 2.3.1 Optimasi Hyperparameter dengan Optuna
        * 2.3.2 Validasi Silang (Cross-Validation)
    * 2.4 Metrik Evaluasi

**3. Diskusi Hasil**
    * 3.1 Peringkat Kinerja Model
    * 3.2 Analisis Signifikansi Variabel (Logistic Regression + PCA)
    * 3.3 Komentar dan Analisis Hasil
        * 3.3.1 Perbandingan Kinerja Model
        * 3.3.2 Catatan Implementasi dan Penyesuaian

**4. Kesimpulan**

**5. Lampiran: Kode Program Lengkap**

<div style="page-break-after: always;"></div>

## 1. Pendahuluan Masalah

### 1.1 Latar Belakang Masalah

Ujian Tengah Semester (UTS) ini berfokus pada masalah klasifikasi menggunakan "Global Coffee Health Dataset". Berbeda dengan Tugas 1 yang merupakan masalah regresi, tugas ini meminta kita untuk memprediksi kategori diskrit. Tujuan utamanya adalah membangun, mengoptimalkan, dan mengevaluasi tiga model pembelajaran mesin yang berbeda untuk memprediksi kondisi kesehatan seseorang (`Health_Issues`) berdasarkan berbagai faktor gaya hidup, demografi, dan konsumsi kopi.

Masalah ini merupakan masalah **klasifikasi multikelas**, karena variabel respon yang kita prediksi memiliki lebih dari dua kategori yang saling eksklusif. Selain itu, kategori-kategori ini memiliki urutan alami (misalnya, dari 'Severe' hingga 'None'), menjadikannya masalah klasifikasi ordinal.

### 1.2 Variabel Respon dan Prediktor

Sesuai instruksi, variabel-variabel dalam analisis ini didefinisikan sebagai berikut:

* **Variabel Respon (y):**
    * `Health_Issues`: Variabel target yang akan diprediksi. Ini adalah variabel kategorikal ordinal dengan empat level: 'Severe', 'Moderate', 'Mild', dan 'None'.

* **Variabel Prediktor (X):**
    * Semua kolom lain dalam dataset, kecuali `ID`. Variabel-variabel ini mencakup campuran tipe data:
        * **Numerik Kontinu:** `Age`, `Coffee_Intake`, `Caffeine_mg`, `Sleep_Hours`, `BMI`, `Physical_Activity_Hours`.
        * **Numerik Diskrit (Biner):** `Smoking`, `Alcohol_Consumption`.
        * **Numerik Diskrit (Lainnya):** `Heart_Rate`.
        * **Kategorikal Nominal:** `Gender`, `Country`, `Occupation`.
        * **Kategorikal Ordinal:** `Sleep_Quality`, `Stress_Level`.

### 1.3 Pembagian Set Data dan Validasi

Mengikuti instruksi dari Tugas 1, metodologi pembagian data dan validasi yang digunakan adalah sebagai berikut:

* **Pembagian Data Latih dan Uji:**
    * Set data keseluruhan (setelah pembersihan data `nan` pada variabel respon) dibagi menjadi set data latih dan set data uji.
    * Proporsi set data uji adalah **20%** dari keseluruhan data.
    * Proporsi set data latih adalah **80%** dari keseluruhan data.
    * Parameter `stratify=y` digunakan selama pembagian untuk memastikan bahwa proporsi dari setiap kelas `Health_Issues` tetap sama di kedua set data (latih dan uji). Ini sangat penting untuk masalah klasifikasi, terutama jika ada ketidakseimbangan kelas.

* **Validasi Silang (Cross-Validation):**
    * Selama fase optimasi hyperparameter (menggunakan Optuna), kinerja model dievaluasi menggunakan **Validasi Silang K-Fold (K-Fold Cross-Validation)**.
    * Secara spesifik, `StratifiedKFold` dengan **5 fold (lipatan)** digunakan. Ini berarti set data latih (80% tadi) dibagi lagi menjadi 5 bagian yang proporsional secara kelas. Model dilatih 5 kali pada 4 bagian dan divalidasi pada 1 bagian yang tersisa, memastikan setiap data poin menjadi bagian dari set validasi tepat satu kali.

<div style="page-break-after: always;"></div>

## 2. Metode

Bagian ini merinci semua langkah teknis yang diambil, mulai dari persiapan data mentah hingga evaluasi model final, sesuai dengan instruksi UTS dan referensi Tugas 1.

### 2.1 Pra-pemrosesan Data

Pra-pemrosesan adalah langkah krusial untuk memastikan data siap dan dalam format yang tepat untuk diterima oleh model pembelajaran mesin.

#### 2.1.1 Pemuatan dan Pembersihan Data Awal

1.  Dataset `synthetic_coffee_health_10000.csv` dimuat ke dalam DataFrame `pandas`.
2.  Kolom `ID` dibuang karena tidak memiliki nilai prediktif.
3.  **Penanganan `nan` pada Variabel Target:** Dilakukan pengecekan nilai hilang (`nan`) pada kolom target `Health_Issues`. Ditemukan bahwa ada nilai `nan` pada kolom ini. Baris-baris yang mengandung `nan` pada `Health_Issues` **dibuang** (`dropna`). Langkah ini wajib dilakukan karena model tidak dapat dilatih pada data yang label targetnya tidak diketahui. `X` dan `y` kemudian dibuat dari DataFrame yang sudah bersih ini.

#### 2.1.2 Penanganan Data Hilang (*Missing Values*) pada Fitur

Setelah menangani `nan` pada target, masih ada potensi `nan` pada kolom-kolom fitur (X). Model seperti `StandardScaler` dan `LogisticRegression` tidak dapat menangani nilai `nan`. Oleh karena itu, *imputasi* diperlukan:

* `SimpleImputer(strategy='median')`: Diterapkan pada semua **fitur numerik**. Strategi median dipilih karena lebih robust (tahan) terhadap nilai pencilan (outlier) dibandingkan dengan strategi 'mean'.
* `SimpleImputer(strategy='most_frequent')`: Diterapkan pada semua **fitur kategorikal (ordinal dan nominal)**. Strategi ini mengisi nilai yang hilang dengan nilai yang paling sering muncul di kolom tersebut.

#### 2.1.3 Encoding Variabel Target (Ordinal)

Variabel respon `Health_Issues` diubah dari teks menjadi angka menggunakan `OrdinalEncoder`. Sesuai instruksi UTS, urutan yang didefinisikan secara eksplisit adalah:
`['Severe', 'Moderate', 'Mild', 'None']`
Ini menghasilkan pemetaan: `Severe=0`, `Moderate=1`, `Mild=2`, `None=3`.

#### 2.1.4 Encoding Variabel Prediktor (Fitur)

* **Fitur Ordinal (`Sleep_Quality`, `Stress_Level`):**
    * Sama seperti target, kedua fitur ini di-encode menggunakan `OrdinalEncoder` dengan urutan yang logis.
    * `Sleep_Quality`: `['Poor', 'Fair', 'Good', 'Excellent']` (Catatan: kategori 'Poor' ditemukan saat *runtime* dan ditambahkan untuk menghindari error).
    * `Stress_Level`: `['Low', 'Medium', 'High']`

* **Fitur Nominal (`Gender`, `Country`, `Occupation`):**
    * Fitur-fitur ini di-encode menggunakan `OneHotEncoder`. Metode ini mengubah setiap kategori menjadi kolom biner baru (0 atau 1). Ini penting untuk menghindari model mengasumsikan adanya urutan (misalnya, 'USA' > 'Canada') yang sebenarnya tidak ada.
    * `handle_unknown='ignore'` digunakan agar model tidak error jika menemukan kategori baru di set data uji yang tidak ada di set data latih.

#### 2.1.5 Standardisasi Fitur Numerik

Semua fitur numerik (`Age`, `Coffee_Intake`, dll.) disekalakan menggunakan `StandardScaler`. Ini adalah instruksi dari Tugas 1 dan merupakan persyaratan untuk `LogisticRegression` dan `PCA` agar berfungsi dengan baik. `StandardScaler` mengubah fitur-fitur sehingga memiliki rata-rata (mean) 0 dan standar deviasi 1.

#### 2.1.6 Konstruksi Pipeline Pra-pemrosesan

Semua langkah di atas (Imputasi, Encoding, Standardisasi) digabungkan menjadi satu objek `preprocessor` menggunakan `ColumnTransformer`. `ColumnTransformer` menerapkan transformasi yang berbeda ke subset kolom yang berbeda secara paralel. Pipeline ini memastikan bahwa semua langkah pra-pemrosesan diterapkan secara konsisten pada data latih dan data uji tanpa terjadi kebocoran data (*data leakage*).

<div style="page-break-after: always;"></div>

### 2.2 Model dan Transformasi

Tiga arsitektur model yang berbeda diimplementasikan dan dievaluasi sesuai instruksi UTS.

#### 2.2.1 Model 1: Logistic Regression (ovr) + PCA

* **Logistic Regression:** Ini adalah model linier fundamental untuk klasifikasi. Meskipun namanya "regresi", model ini memprediksi probabilitas keanggotaan kelas.
    * `multi_class='ovr'` (One-vs-Rest): Sesuai instruksi, strategi ini digunakan untuk menangani masalah multikelas. Model ini melatih *n* pengklasifikasi biner (dalam kasus ini, 4 pengklasifikasi), di mana setiap pengklasifikasi membedakan satu kelas melawan semua kelas lainnya.
* **PCA (Principal Component Analysis):**
    * PCA adalah teknik reduksi dimensi. Ini digunakan *sebelum* `LogisticRegression` (di dalam `Pipeline`). PCA mengubah fitur-fitur asli (yang mungkin berkorelasi) menjadi satu set fitur baru yang tidak berkorelasi yang disebut komponen utama, diurutkan berdasarkan jumlah varians yang mereka jelaskan.
    * Hyperparameter `n_components` (dijelaskan sebagai proporsi varians, misal 0.95) dioptimalkan oleh Optuna untuk menemukan jumlah komponen optimal yang menyeimbangkan antara retensi informasi dan reduksi dimensi.

#### 2.2.2 Model 2: Random Forest Classifier

* Ini adalah model pilihan dari dua opsi ensemble yang diberikan (Random Forest atau Extra Trees).
* **Random Forest:** Ini adalah model *ensemble* yang bekerja dengan membangun sejumlah besar (diatur oleh `n_estimators`) pohon keputusan (decision trees) pada berbagai sub-sampel data (teknik *bagging*).
* Untuk membuat prediksi, setiap pohon dalam "hutan" memberikan suaranya, dan kelas dengan suara terbanyak menjadi prediksi akhir model.
* Model ini sangat kuat, dapat menangkap hubungan non-linier yang kompleks, dan umumnya tahan terhadap *overfitting* jika dikonfigurasi dengan benar.

#### 2.2.3 Model 3: HistGradientBoostingClassifier

* **HistGradientBoostingClassifier:** Ini adalah implementasi `scikit-learn` yang modern dan sangat cepat dari *gradient boosting*, yang terinspirasi oleh LightGBM.
* Tidak seperti Random Forest yang membangun pohon secara paralel, model *boosting* membangun pohon secara berurutan (*sequential*).
* Setiap pohon baru dilatih untuk memperbaiki kesalahan (gradien dari *loss function*) yang dibuat oleh pohon-pohon sebelumnya.
* Model ini menggunakan *binning* (diatur oleh `max_bins`) pada fitur-fitur input, yang membuatnya sangat efisien secara komputasi dan memori.

<div style="page-break-after: always;"></div>

### 2.3 Optimasi dan Validasi

#### 2.3.1 Optimasi Hyperparameter (Optuna)

Untuk menemukan konfigurasi terbaik untuk setiap model, *library* `Optuna` digunakan untuk optimasi hyperparameter. Optuna secara otomatis mencari ruang hyperparameter yang didefinisikan untuk memaksimalkan metrik evaluasi.

* **Fungsi `objective`:** Sebuah fungsi `objective` didefinisikan untuk setiap model. Fungsi ini mengambil satu `trial` Optuna, menyarankan hyperparameter, membangun `Pipeline` model, dan mengevaluasinya menggunakan *cross-validation*.
* **Ruang Pencarian (Search Space):**
    * **LogReg+PCA:** `C` (kekuatan regularisasi), `penalty` (l1/l2), `n_components` (varians PCA).
    * **Random Forest:** `n_estimators` [50-500], `max_depth` [3-9], `max_features` [sqrt, log2, None], `max_leaf_nodes` [20-60].
    * **HistGradientBoosting:** `learning_rate` [0.01-0.2], `max_depth` [3-9], `max_leaf_nodes` [20-60], `max_bins` [128-255].
        * **Catatan Penting `max_bins`:** Instruksi UTS meminta rentang [128-512]. Namun, implementasi `sklearn` memiliki batas teknis `max_bins=255`. Rentang diubah menjadi `[128-255]` agar kode dapat berjalan. (Lihat Diskusi 3.3.2).
* **Jumlah Trial:** Setiap studi optimasi dijalankan untuk `N_TRIALS = 40`.

#### 2.3.2 Validasi Silang (Cross-Validation)

Di dalam setiap *trial* Optuna, kinerja hyperparameter dievaluasi menggunakan `cross_val_score` dengan `StratifiedKFold(n_splits=5)`. Metrik yang dioptimalkan adalah rata-rata `f1_score(average='weighted')` dari 5-fold tersebut. Penggunaan validasi silang memberikan estimasi kinerja yang lebih stabil dan dapat diandalkan daripada hanya menggunakan satu set validasi.

### 2.4 Metrik Evaluasi

Sesuai instruksi UTS, metrik evaluasi utama yang digunakan adalah **F1-Score (Weighted)**.

* **F1-Score:** Ini adalah rata-rata harmonik (harmonic mean) dari *Precision* dan *Recall*. F1-score memberikan keseimbangan yang baik antara kedua metrik tersebut, terutama berguna ketika berhadapan dengan kelas yang tidak seimbang.
    * *Precision*: Dari semua yang diprediksi sebagai kelas A, berapa persen yang benar-benar kelas A? (Mengukur kesalahan *false positive*).
    * *Recall*: Dari semua yang seharusnya kelas A, berapa persen yang berhasil diprediksi? (Mengukur kesalahan *false negative*).
* **`average='weighted'`:** Parameter ini menghitung F1-score untuk setiap kelas secara terpisah, lalu mengambil rata-ratanya, *dibobotkan* oleh jumlah sampel aktual di setiap kelas (*support*). Ini adalah pilihan yang sangat baik untuk data multikelas di mana satu kelas mungkin jauh lebih umum daripada yang lain.

<div style="page-break-after: always;"></div>

## 3. Diskusi Hasil

Bagian ini menyajikan hasil dari model-model yang telah dilatih dan dioptimalkan, serta analisis terhadap temuan tersebut. Model final dilatih pada *seluruh* set data latih (80%) menggunakan hyperparameter terbaik yang ditemukan oleh Optuna, dan kemudian dievaluasi *satu kali* pada set data uji (20%) yang tersembunyi.

### 3.1 Peringkat Kinerja Model

Kinerja model final pada set data uji diukur menggunakan metrik `f1_score (weighted)`. Hasilnya diurutkan dari yang terbaik hingga terburuk.

**Tabel 3.1: Peringkat Kinerja Model pada Set Data Uji**

| Peringkat | Model | F1-Score (Weighted) pada Set Data Uji |
| :--- | :--- | :--- |
| **1** | **Random Forest Classifier** | **0.9975** |
| **2** | **HistGradientBoostingClassifier** | **0.9958** |
| **3** | **Logistic Regression + PCA** | **0.8984** |

<br>

**Laporan Klasifikasi Rinci (Set Data Uji):**

---
**Model: Random Forest Classifier**
