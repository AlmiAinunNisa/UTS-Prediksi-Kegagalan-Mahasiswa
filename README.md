# Analisis dan Prediksi Kegagalan Mahasiswa

Nama    : Almi Ainun Nisa
Nim     : TI22250002
Kelas   : TI B
Semester : 3
Mata Kuliah : Desain dan Analisis Algoritma

## Gambaran Proyek

Proyek ini menganalisis karakteristik mahasiswa dan membangun model klasifikasi biner untuk memperkirakan status **lulus (0)** atau **gagal (1)** pada suatu modul perkuliahan. Analisis dilakukan dengan Python di Jupyter Notebook, dari eksplorasi data hingga pengujian prediksi terhadap lima data sintetis.

Dataset yang digunakan berisi **80 mahasiswa, 19 fitur, dan 1 kolom target**. Sebanyak 68 mahasiswa (85%) berlabel lulus dan 12 mahasiswa (15%) berlabel gagal. Ketidakseimbangan ini penting karena akurasi tinggi saja belum tentu berarti model mampu mendeteksi mahasiswa yang gagal.

## 1. Data Understanding

Kode `pd.read_csv('original - original.csv')` membaca data CSV ke dalam DataFrame `df`. `df.head()` memperlihatkan lima baris pertama, `df.shape` menampilkan jumlah baris dan kolom, sedangkan `df.info()` menunjukkan tipe data dan kelengkapan tiap kolom.

Distribusi target diperiksa melalui `value_counts()` dan `value_counts(normalize=True) * 100`. Hasilnya menunjukkan 68 data lulus dan 12 data gagal. Target asli bernama `label (fail=1, pass=0)`.

Fitur mencakup demografi (`Gender`, `Age`), indikator wilayah dan akses pendidikan (POLAR, Adult HE, TUNDRA, Gaps GCSE, Uni Connect), kondisi perjalanan/tinggal, aktivitas login dan jam belajar, serta kehadiran.

## 2. EDA dan Pemrosesan Data

Pemeriksaan `isna()` dan `duplicated()` menemukan **dua nilai hilang pada `# of Absence`**, tanpa baris atau kolom kosong seluruhnya dan tanpa baris duplikat.

Tiga kolom yang semestinya numerik terbaca sebagai teks: `distance to university (km)`, `% of Average Hours in Module Area`, dan `Percent Attended`. CSV memang memuat penulisan angka dengan banyak titik, misalnya `4.166.666.667`, sehingga pembacaan angka harus diperiksa dengan hati-hati.

Dalam notebook, dibuat salinan `df_clean`, dilakukan konversi sementara untuk jarak dan persentase kehadiran, dan dua nilai kosong pada `# of Absence` sempat diisi 0 berdasarkan asumsi. **Ini merupakan keterbatasan proses cleaning:** posisi desimal pada kolom jarak belum dipastikan sepenuhnya, dan pengisian absen 0 bukan nilai yang terverifikasi. Sebelum modeling, kolom rasio jam belajar dan jarak dikeluarkan; kolom absen juga tidak digunakan sebagai fitur.

### Visualisasi yang dibuat

1. **Histogram usia:** sebagian besar mahasiswa berada pada rentang usia muda; distribusi cenderung miring ke kanan.
2. **Boxplot usia:** menunjukkan beberapa usia ekstrem yang masih mungkin merupakan observasi valid.
3. **Boxplot kehadiran berdasarkan label:** kelompok lulus cenderung memiliki persentase kehadiran lebih tinggi daripada kelompok gagal.
4. **Heatmap korelasi:** memperlihatkan hubungan antarfitur terpilih, termasuk potensi redundansi antara ukuran aktivitas dan kehadiran.

Outlier dihitung dengan metode **IQR**: `Q1 = quantile(0.25)`, `Q3 = quantile(0.75)`, `IQR = Q3 - Q1`, lalu batas bawah dan atas ditetapkan sebagai `Q1 - 1.5*IQR` dan `Q3 + 1.5*IQR`. Hasil pemeriksaan menunjukkan **6 outlier usia, 2 outlier jumlah login, dan 3 outlier total jam belajar**. Observasi ini tidak otomatis dihapus karena mungkin menggambarkan kondisi mahasiswa sebenarnya.

## 3. Persiapan Data

`X` berisi fitur prediktor dan `y` berisi label kelulusan. Fitur `TUNDRA LSOA Quintile`, `# of presence`, dan `# of Absence` tidak dipakai untuk mengurangi informasi yang berulang. Kolom rasio jam belajar dan jarak juga tidak dipakai karena masalah validasi format.

Hasil akhir adalah **14 fitur** untuk 80 mahasiswa. `train_test_split(test_size=0.2, stratify=y, random_state=42)` membagi data menjadi **64 data latih** (54 lulus, 10 gagal) dan **16 data uji** (14 lulus, 2 gagal). `stratify=y` mempertahankan proporsi kelas.

`StandardScaler` dipasang dengan `fit_transform(X_train)` dan diterapkan ke data uji dengan `transform(X_test)`. Artinya, rata-rata dan standar deviasi dihitung dari data latih saja, sehingga data uji tidak ikut menentukan parameter scaling.

## 4. Modeling

Dua algoritma dibandingkan:

### Logistic Regression

`LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)` mempelajari hubungan fitur dengan peluang kelas gagal. `class_weight='balanced'` memberi bobot lebih besar pada kelas minoritas; `max_iter=1000` memberi kesempatan optimisasi untuk konvergen. Model dilatih pada fitur yang telah distandardisasi.

### Random Forest

`RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)` menggabungkan 100 pohon keputusan. Metode ini dapat mempelajari hubungan nonlinier dan tidak memerlukan scaling, sehingga dilatih menggunakan fitur asli yang sudah dipilih.

### Cross-validation

`StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` membagi data latih menjadi lima fold dengan proporsi kelas yang dijaga. `cross_val_score(..., scoring='f1')` menghitung F1-score untuk kelas gagal (1). Logistic Regression ditempatkan dalam `Pipeline` bersama `StandardScaler` agar scaling pada setiap fold hanya belajar dari bagian pelatihannya.

Hasil rata-rata yang diperoleh dalam notebook:

| Algoritma | F1-score cross-validation |
|---|---:|
| Logistic Regression | 0,46 |
| Random Forest | 0,44 |

Logistic Regression dipilih untuk evaluasi akhir karena memperoleh rata-rata F1-score sedikit lebih tinggi. Selisihnya kecil, sehingga tidak cukup untuk menyimpulkan keunggulan yang kuat.

## 5. Evaluasi Model

Model Logistic Regression digunakan untuk memprediksi 16 data uji yang disisihkan sejak awal. Fungsi `accuracy_score`, `precision_score`, `recall_score`, `f1_score`, dan `confusion_matrix` digunakan untuk mengukur hasil.

| Metrik | Hasil |
|---|---:|
| Accuracy | 68,75% |
| Precision kelas gagal | 20,00% |
| Recall kelas gagal | 50,00% |
| F1-score kelas gagal | 28,57% |

Confusion matrix, dengan urutan kelas **0 (lulus)** lalu **1 (gagal)**:

| Aktual / Prediksi | Lulus (0) | Gagal (1) |
|---|---:|---:|
| Lulus (0) | 10 | 4 |
| Gagal (1) | 1 | 1 |

Model benar pada 11 dari 16 kasus. Dari dua mahasiswa yang sebenarnya gagal, satu berhasil terdeteksi. Empat mahasiswa yang sebenarnya lulus salah diprediksi gagal. Karena kelas gagal hanya memiliki dua contoh pada data uji, metriknya sangat sensitif terhadap satu kesalahan.

### Interpretasi koefisien

Koefisien Logistic Regression diambil dari `model_lr.coef_[0]`, dipasangkan dengan nama fitur, lalu diurutkan menurut nilai absolutnya. Tiga fitur teratas dalam hasil notebook adalah:

- `Percent Attended`: sekitar **−1,099**
- `POLAR4 Quintile`: sekitar **+0,994**
- `attending from home?`: sekitar **+0,993**

Koefisien negatif pada kehadiran menunjukkan bahwa kenaikan fitur tersebut berkaitan dengan penurunan kecenderungan prediksi gagal, dengan fitur lain tetap. Koefisien positif menunjukkan arah sebaliknya. **Koefisien bukan bukti sebab-akibat**, dan peringkatnya dapat berubah jika data atau fitur berubah.

## 6. Prediksi Lima Data Baru

Lima baris sintetis dibuat dengan `numpy.random.default_rng(42)` dan `rng.choice()` yang mengambil nilai dari masing-masing kolom data latih. Ini menjaga setiap nilai individual berasal dari rentang data yang ada, tetapi **kombinasi antarkolom belum tentu realistis**.

Data baru disusun dalam urutan fitur yang sama dengan `X_train`, distandardisasi memakai `scaler.transform(data_baru)`, lalu diprediksi dengan `model_lr.predict()` dan `model_lr.predict_proba()[:, 1]`. Kolom probabilitas adalah perkiraan peluang kelas gagal menurut model, bukan kepastian hasil akademik.

| Mahasiswa sintetis | Prediksi | Probabilitas gagal |
|---|---|---:|
| 1 | Lulus | 6,85% |
| 2 | Lulus | 0,89% |
| 3 | Lulus | 32,71% |
| 4 | Gagal | 62,54% |
| 5 | Lulus | 17,45% |

Empat data diprediksi lulus dan satu diprediksi gagal. Probabilitas ini belum dikalibrasi dan harus ditafsirkan dengan hati-hati.

## Kesimpulan

Proyek berhasil menerapkan alur analisis data, visualisasi, pemilihan fitur, pembagian data, standardisasi, pelatihan dua algoritma, cross-validation, evaluasi, dan prediksi data sintetis. Logistic Regression memperoleh F1-score cross-validation sedikit lebih baik daripada Random Forest, tetapi pada data uji F1-score kelas gagal hanya **28,57%**.

Keterbatasan utama adalah **dataset kecil, kelas tidak seimbang, data numerik dengan format tidak konsisten, dan sedikitnya contoh mahasiswa gagal pada data uji**. Karena itu, model merupakan eksperimen pembelajaran dan belum layak digunakan sebagai satu-satunya dasar keputusan akademik.
