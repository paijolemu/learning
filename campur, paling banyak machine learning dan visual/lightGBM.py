# ==============================================================================
#           KUMPULAN KODE LENGKAP - LIGHTGBM
# ==============================================================================
# %%
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.datasets import load_breast_cancer, make_regression
import pandas as pd
import joblib

print(f"Versi LightGBM: {lgb.__version__}")

# ------------------------------------------------------------------------------
# 1. KLASIFIKASI DASAR
print("\n--- 1. Contoh Klasifikasi Biner ---")
X_c, y_c = load_breast_cancer(return_X_y=True)
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_c, y_c, test_size=0.2, random_state=42)

model_c = lgb.LGBMClassifier(random_state=42)
model_c.fit(X_train_c, y_train_c)
preds_c = model_c.predict(X_test_c)
print(f"Akurasi Klasifikasi: {accuracy_score(y_test_c, preds_c):.4f}")

# ------------------------------------------------------------------------------
# 2. REGRESI DASAR
print("\n--- 2. Contoh Regresi ---")
X_r, y_r = make_regression(n_samples=1000, n_features=10, noise=25, random_state=42)
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_r, y_r, test_size=0.2, random_state=42)

model_r = lgb.LGBMRegressor(random_state=42)
model_r.fit(X_train_r, y_train_r)
preds_r = model_r.predict(X_test_r)
print(f"MAE Regresi: {mean_absolute_error(y_test_r, preds_r):.4f}")

# ------------------------------------------------------------------------------
# 3. FITUR UNGGULAN: EARLY STOPPING & PENANGANAN KATEGORI OTOMATIS
print("\n--- 3. Contoh Lanjutan: Early Stopping & Kategori ---")

# Buat dataset contoh dengan fitur kategori
data = {'kota': ['A', 'B', 'A', 'C', 'B', 'A'] * 10,
        'suhu': [25, 30, 22, 28, 29, 24] * 10,
        'target': [1, 0, 1, 0, 0, 1] * 10}
df_kat = pd.DataFrame(data)

# PENTING: Ubah tipe data menjadi 'category'
df_kat['kota'] = df_kat['kota'].astype('category')

X_kat = df_kat.drop('target', axis=1)
y_kat = df_kat['target']
X_train_kat, X_test_kat, y_train_kat, y_test_kat = train_test_split(X_kat, y_kat, test_size=0.2, random_state=42)

model_kat = lgb.LGBMClassifier(n_estimators=500, random_state=42)
# Sintaks Early Stopping LightGBM menggunakan callbacks
model_kat.fit(X_train_kat, y_train_kat,
              eval_set=[(X_test_kat, y_test_kat)],
              eval_metric='logloss',
              callbacks=[lgb.early_stopping(10, verbose=False)])

print(f"Model seharusnya berjalan 500 kali, tapi berhenti di iterasi ke: {model_kat.best_iteration_}")
preds_kat = model_kat.predict(X_test_kat)
print(f"Akurasi pada data dengan kategori: {accuracy_score(y_test_kat, preds_kat):.4f}")


# ------------------------------------------------------------------------------
# 4. MENYIMPAN & MEMUAT MODEL
print("\n--- 4. Menyimpan dan Memuat Model ---")
filename = 'lgbm_model.pkl'
joblib.dump(model_kat, filename)
print(f"Model tersimpan sebagai {filename}")

# Memuat kembali model
loaded_lgbm_model = joblib.load(filename)
print("Model berhasil dimuat kembali.")
sample_pred = loaded_lgbm_model.predict(X_test_kat[:1])
print(f"Prediksi pada satu sampel data: {sample_pred[0]}")







# %%
# ==============================================================================
#           KODE LENGKAP - LIGHTGBM RANKER (LGBMRanker)
# ==============================================================================

import lightgbm as lgb
import pandas as pd
import numpy as np

# ------------------------------------------------------------------------------
# 1. Membuat Data Contoh (Skenario Mesin Pencari)
# ------------------------------------------------------------------------------
print("--- 1. Membuat Data Contoh ---")

# Bayangkan kita punya 2 query pencarian.
# Query 1 punya 4 dokumen hasil.
# Query 2 punya 3 dokumen hasil.
data = {
    'query_id':  [1, 1, 1, 1,   2, 2, 2],
    # Fitur dokumen: misal, apakah judul cocok, jumlah backlink, dll.
    'fitur_judul': [0.9, 0.2, 0.5, 0.1,   0.8, 0.3, 0.6],
    'fitur_konten': [0.8, 0.3, 0.6, 0.2,   0.9, 0.1, 0.5],
    # Target (y): Tingkat relevansi yang diberikan oleh manusia (0=jelek, 3=sempurna)
    'relevance':   [3,   0,   2,   1,     2,   0,   3] 
}
df = pd.DataFrame(data)

# PENTING: Data harus diurutkan berdasarkan query_id agar 'group' bisa dibuat dengan benar
df = df.sort_values(by='query_id')

print("Data contoh untuk ranking:")
print(df)


# ------------------------------------------------------------------------------
# 2. Mempersiapkan Data untuk LGBMRanker
# ------------------------------------------------------------------------------
print("\n--- 2. Mempersiapkan Data ---")

# Pisahkan fitur (X), target (y), dan query_id
X_features = df[['fitur_judul', 'fitur_konten']]
y_relevance = df['relevance']
query_ids = df['query_id']

# Membuat array 'group'
# Ini memberitahu model berapa banyak item di setiap grup.
# df.groupby('query_id')['query_id'].count() akan menghasilkan: query_id 1 -> 4, query_id 2 -> 3
group_counts = df.groupby('query_id')['query_id'].count().to_numpy()

print(f"Array 'group': {group_counts}")
print("Artinya: 4 baris pertama untuk grup 1, 3 baris berikutnya untuk grup 2.")

# Kita bisa membuat set validasi juga untuk early stopping
# Untuk contoh ini, kita gunakan data yang sama sebagai train dan validasi
X_train, y_train, group_train = X_features, y_relevance, group_counts
X_valid, y_valid, group_valid = X_features, y_relevance, group_counts


# ------------------------------------------------------------------------------
# 3. Melatih Model LGBMRanker
# ------------------------------------------------------------------------------
print("\n--- 3. Melatih LGBMRanker ---")

# Inisialisasi model
ranker = lgb.LGBMRanker(
    n_estimators=100,
    learning_rate=0.1,
    random_state=42
)

# Latih model dengan data group
# Perhatikan parameter 'group' dan 'eval_group' yang wajib ada
ranker.fit(
    X=X_train,
    y=y_train,
    group=group_train,
    eval_set=[(X_valid, y_valid)],
    eval_group=[group_valid],
    eval_at=[1, 3], # Evaluasi metrik ranking seperti NDCG@1 dan NDCG@3
    callbacks=[lgb.early_stopping(10, verbose=False)]
)

print("Model ranker berhasil dilatih.")


# ------------------------------------------------------------------------------
# 4. Menggunakan Model untuk Memprediksi Peringkat
# ------------------------------------------------------------------------------
print("\n--- 4. Memprediksi Peringkat ---")

# Buat data baru untuk diprediksi, misalnya untuk query_id = 1
# Kita acak urutannya untuk melihat apakah model bisa mengurutkannya dengan benar
data_untuk_prediksi = pd.DataFrame({
    'fitur_judul': [0.5, 0.9, 0.1, 0.2],
    'fitur_konten': [0.6, 0.8, 0.2, 0.3],
    'dokumen_asli': ['C', 'A', 'D', 'B'] # Label untuk kita lihat hasilnya
})

print("Data baru (tidak terurut) untuk Query 1:")
print(data_untuk_prediksi)

# Lakukan prediksi
# Outputnya adalah SKOR, bukan kelas atau nilai. Semakin tinggi skor, semakin baik.
pred_scores = ranker.predict(data_untuk_prediksi[['fitur_judul', 'fitur_konten']])
data_untuk_prediksi['pred_score'] = pred_scores

# Urutkan hasilnya berdasarkan skor prediksi
hasil_peringkat = data_untuk_prediksi.sort_values('pred_score', ascending=False)

print("\nHasil Peringkat dari Model:")
print(hasil_peringkat)
# Seharusnya model menempatkan Dokumen A di peringkat pertama.