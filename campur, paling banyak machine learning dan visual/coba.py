# %%
from warnings import filterwarnings
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
filterwarnings('ignore')
# 1. KUMPULKAN DATA HISTORIS
# Biasanya data ini dari file CSV atau database. Untuk contoh, kita buat sendiri.
# Fitur: Luas Rumah (meter persegi)
# Target: Harga Rumah (dalam Juta Rupiah)
data = {
    'luas_rumah': [50, 60, 70, 80, 100, 110, 120, 130, 150, 180],
    'harga_juta': [300, 350, 420, 480, 550, 580, 640, 690, 800, 950]
}
df = pd.DataFrame(data)

print("--- Data Historis Kita ---")
print(df)
print("\n")

# Pisahkan antara Fitur (X) dan Target (y)
X = df[['luas_rumah']]  # Fitur harus dalam bentuk 2D array/DataFrame
y = df['harga_juta']    # Target

# 2. SIAPKAN DATA
# Bagi data menjadi data latih (untuk belajar) dan data uji (untuk tes)
# 80% untuk latih, 20% untuk uji
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. PILIH & LATIH MODEL
# Kita gunakan model yang paling sederhana: Regresi Linear (mencari garis lurus yang paling pas)
model = LinearRegression()

# Latih model dengan data latih. Di sinilah proses "belajar" terjadi!
model.fit(X_train, y_train) 
print("--- Model selesai dilatih! ---\n")

# 4. EVALUASI MODEL (Langkah Opsional untuk contoh ini, tapi sangat penting di dunia nyata)
# Kita bisa lihat seberapa akurat modelnya di data uji
akurasi = model.score(X_test, y_test)
print(f"Akurasi model: {akurasi*100:.2f}%") # Semakin dekat ke 100%, semakin baik
print("\n")

# 5. GUNAKAN MODEL UNTUK PREDIKSI MASA DEPAN
# Sekarang, mari kita prediksi harga untuk rumah baru yang belum ada di data kita.
# Misalnya, berapa harga rumah seluas 160 m²?

luas_rumah_baru = [[160]] # Data baru harus dalam format yang sama dengan data latih (2D array)

# Lakukan prediksi!
prediksi_harga = model.predict(luas_rumah_baru)

# Tampilkan hasilnya
print("--- HASIL PREDIKSI ---")
print(f"Prediksi harga untuk rumah seluas {luas_rumah_baru[0][0]} m² adalah: Rp {prediksi_harga[0]:.2f} Juta")

# Contoh lain: rumah 90 m²
prediksi_lain = model.predict([[90]])
print(f"Prediksi harga untuk rumah seluas 90 m² adalah: Rp {prediksi_lain[0]:.2f} Juta")
# %%
