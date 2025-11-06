# ==============================================================================
#           KUMPULAN KODE LENGKAP - CATBOOST
# ==============================================================================

# %%
import catboost as cb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.datasets import load_breast_cancer, make_regression
import pandas as pd

print(f"Versi CatBoost: {cb.__version__}")

# ------------------------------------------------------------------------------
# 1. KLASIFIKASI DASAR
print("\n--- 1. Contoh Klasifikasi Biner ---")
X_c, y_c = load_breast_cancer(return_X_y=True)
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_c, y_c, test_size=0.2, random_state=42)

# CatBoost sangat "berisik", verbose=0 akan membuatnya diam
model_c = cb.CatBoostClassifier(random_state=42, verbose=0)
model_c.fit(X_train_c, y_train_c)
preds_c = model_c.predict(X_test_c)
print(f"Akurasi Klasifikasi: {accuracy_score(y_test_c, preds_c):.4f}")

# ------------------------------------------------------------------------------
# 2. REGRESI DASAR
print("\n--- 2. Contoh Regresi ---")
X_r, y_r = make_regression(n_samples=1000, n_features=10, noise=25, random_state=42)
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_r, y_r, test_size=0.2, random_state=42)

model_r = cb.CatBoostRegressor(random_state=42, verbose=0)
model_r.fit(X_train_r, y_train_r)
preds_r = model_r.predict(X_test_r)
print(f"MAE Regresi: {mean_absolute_error(y_test_r, preds_r):.4f}")

# ------------------------------------------------------------------------------
# 3. FITUR UNGGULAN: EARLY STOPPING & PENANGANAN KATEGORI OTOMATIS
print("\n--- 3. Contoh Lanjutan: Early Stopping & Kategori ---")

# Buat dataset contoh dengan fitur kategori (sebagai string biasa)
data = {'kota': ['A', 'B', 'A', 'C', 'B', 'A'] * 10,
        'suhu': [25, 30, 22, 28, 29, 24] * 10,
        'target': [1, 0, 1, 0, 0, 1] * 10}
df_kat = pd.DataFrame(data)

X_kat = df_kat.drop('target', axis=1)
y_kat = df_kat['target']
X_train_kat, X_test_kat, y_train_kat, y_test_kat = train_test_split(X_kat, y_kat, test_size=0.2, random_state=42)

# PENTING: Beri tahu CatBoost mana kolom kategori, tidak perlu ubah tipe data!
categorical_features_indices = [0] # Indeks kolom 'kota'

model_kat = cb.CatBoostClassifier(iterations=500, # Sama seperti n_estimators
                                  random_state=42,
                                  cat_features=categorical_features_indices)

# CatBoost memiliki early stopping bawaan yang lebih terintegrasi
model_kat.fit(X_train_kat, y_train_kat,
              eval_set=(X_test_kat, y_test_kat),
              early_stopping_rounds=10,
              verbose=0) # Set ke 0 agar tidak print log, atau >0 untuk lihat

print(f"Model seharusnya berjalan 500 kali, tapi berhenti di iterasi ke: {model_kat.get_best_iteration()}")
preds_kat = model_kat.predict(X_test_kat)
print(f"Akurasi pada data dengan kategori: {accuracy_score(y_test_kat, preds_kat):.4f}")


# ------------------------------------------------------------------------------
# 4. MENYIMPAN & MEMUAT MODEL
print("\n--- 4. Menyimpan dan Memuat Model ---")
# CatBoost merekomendasikan metode simpan bawaannya
filename = 'catboost_model.cbm'
model_kat.save_model(filename)
print(f"Model tersimpan sebagai {filename}")

# Memuat kembali model
loaded_cb_model = cb.CatBoostClassifier()
loaded_cb_model.load_model(filename)
print("Model berhasil dimuat kembali.")
sample_pred = loaded_cb_model.predict(X_test_kat.iloc[:1])
print(f"Prediksi pada satu sampel data: {sample_pred[0]}")



# %%
import pandas as pd

# Data mentah kita
data = {
    'usia': [25, 30, 22, 45, 18],
    'pendapatan': [50, 80, 40, 150, 20],
    'platform': ['PC', 'PS5', 'PC', 'Switch', 'PS5'],
    'genre_favorit': ['RPG', 'FPS', 'RPG', 'Strategy', 'FPS'],
    'beli_game': [1, 1, 0, 1, 0] # Target: 1=beli, 0=tidak
}
df_mentah = pd.DataFrame(data)
# %%
# ini keunggulan cat yg dimana kita tidak perlu melakukan encoding manual
# --- LANGKAH PRE-PROCESSING MANUAL YANG DIPERLUKAN ---
print("--- Cara Sulit (XGBoost/Scikit-learn) ---")
print("Data Mentah:")
print(df_mentah)

# NOTE UNGGUL (CatBoost): Anda tidak perlu melakukan langkah di bawah ini dengan CatBoost.
# Kita harus melakukan One-Hot Encoding secara manual.
df_encoded = pd.get_dummies(df_mentah, columns=['platform', 'genre_favorit'])

print("\nData setelah One-Hot Encoding (siap untuk XGBoost):")
print(df_encoded)
# Perhatikan bagaimana kolom 'platform' dan 'genre_favorit' meledak menjadi banyak kolom baru.

# Memisahkan fitur dan target
X_encoded = df_encoded.drop('beli_game', axis=1)
y = df_encoded['beli_game']

# Di sini Anda baru bisa melatih model, misalnya XGBoost
# from xgboost import XGBClassifier
# model_xgb = XGBClassifier().fit(X_encoded, y)
print("\nModel baru bisa dilatih setelah data di-encode secara manual.")
# %%
from meong import CatBoostClassifier
from sklearn.metrics import accuracy_score

# Kita mulai dari data mentah yang SAMA persis
df_mentah = pd.DataFrame(data)

# Memisahkan fitur dan target dari data MENTAH
X_mentah = df_mentah.drop('beli_game', axis=1)
y = df_mentah['beli_game']

# --- Di sinilah Keajaiban CatBoost Terjadi ---
print("\n\n--- Cara Mudah & Unggul (CatBoost) ---")
print("Data Mentah (yang langsung dipakai CatBoost):")
print(X_mentah)

# NOTE UNGGUL 1: KEMUDAHAN PENGGUNAAN
# Tidak ada One-Hot Encoding. Cukup beritahu CatBoost mana kolom yang berisi kategori.
# Kode Anda jauh lebih bersih dan singkat.
categorical_features_names = ['platform', 'genre_favorit']

model_cat = CatBoostClassifier(iterations=50,
                               verbose=0, # verbose=0 agar tidak menampilkan log training
                               cat_features=categorical_features_names)

model_cat.fit(X_mentah, y)
print("\nModel berhasil dilatih langsung dari data mentah!")

# NOTE UNGGUL 2: PENANGANAN KATEGORI YANG SUPERIOR
# Di balik layar, CatBoost tidak hanya mengubahnya menjadi angka. Ia menggunakan
# teknik canggih (berbasis target encoding dengan regularisasi) yang seringkali
# menghasilkan akurasi lebih tinggi, terutama untuk fitur dengan banyak nilai unik.
# Teknik ini juga lebih tahan terhadap data leakage.

# Mari kita buktikan dengan prediksi
preds = model_cat.predict(X_mentah)
print(f"\nAkurasi di data training: {accuracy_score(y, preds):.2f}")
# %%
# --- Skenario Data Baru ---
data_baru = pd.DataFrame([{
    'usia': 28,
    'pendapatan': 75,
    'platform': 'Xbox', # <-- Nilai ini BELUM PERNAH ada di data training!
    'genre_favorit': 'RPG'
}])

print("\n\n--- UJIAN AKHIR: Prediksi pada Data Baru dengan Kategori Asing ---")
print("Data baru:")
print(data_baru)

# --- Prediksi dengan Cara Sulit (XGBoost/Scikit-learn) ---
try:
    # Jika kita coba prediksi dengan model yang dilatih cara lama, akan error
    # karena kolom 'platform_Xbox' tidak ada saat training.
    # Ini adalah sumber frustrasi yang sangat umum di dunia nyata.
    # model_xgb.predict(data_baru_encoded) # --> PASTI ERROR
    print("\nPrediksi dengan cara XGBoost/Scikit-learn: AKAN GAGAL karena kolom tidak cocok.")
except Exception as e:
    print(f"Error: {e}")

# --- Prediksi dengan Cara Mudah (CatBoost) ---
# NOTE UNGGUL 3: KETAHANAN (ROBUSTNESS)
# CatBoost secara otomatis menangani nilai kategori yang tidak dikenal.
# Tidak ada error. Model tetap bisa memberikan prediksi yang masuk akal.
prediksi_baru = model_cat.predict(data_baru)
hasil = 'Beli' if prediksi_baru[0] == 1 else 'Tidak Beli'
print(f"\nPrediksi dengan cara CatBoost: BERHASIL!")
print(f"Hasil prediksi untuk gamer baru: {hasil}")