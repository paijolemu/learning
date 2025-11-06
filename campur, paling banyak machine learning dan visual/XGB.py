# XGB = extreme gradient boosting
# keunggulan dari mode; ini adalah algoritma boosting yang sangat efisien dan efektif untuk tugas-tugas klasifikasi dan regresi. Beberapa keunggulan utama dari XGBoost meliputi:
# jadi sering menjadi pemenang di kompetisi machine learning.
# - Performa Tinggi: XGBoost dirancang untuk memberikan performa yang sangat baik

# dibawah ini adalah kode XGB:
# sebagai catatan memang hanya ada 3 model: regression, classification, ranking
from xgboost import XGBRegressor
from xgboost import XGBClassifier
from xgboost import XGBRanker
# ini adalah semua model di xgb emang sangat sedikit sekali dibanding sklearn, karna ini 
# adalah library khusus untuk boosting, dan sebagai model saja.

# %%
# data
import pandas as pd

data = pd.read_csv(r'C:\Users\62812\Downloads\archive (11)\melb_data.csv')
data.head()
# %%
# info
print(f'ini adalah hasil dari: INFO:\n{data.info()}')
print(f'ini adalah hasil dari: DESCRIBE:\n{data.describe()}')
print(f'ini adalah hasil dari: ISNULL:\n{data.isnull().sum()}')
print(f'ini adalah hasil dari: COLOMNS:\n{data.columns}')
print(f'ini adalah hasil dari: SHAPE:\n{data.shape}')
print(f'ini adalah hasil dari: DTYPES:\n{data.dtypes}')
print(f'ini adalah hasil dari: NUNIQUE:\n{data.nunique()}')
print(f'ini adalah hasil dari: CORR:\n{data.corr(numeric_only=True)}')

# %%
print(f'ini adalah hasil dari: INFO:\n{data.info()}')
# %%
print(f'ini adalah hasil dari: DESCRIBE:\n{data.describe()}')
# %%
print(f'ini adalah hasil dari: ISNULL:\n{data.isnull().sum()}')
# %%
print(f'ini adalah hasil dari: COLOMNS:\n{data.columns}')
# %%
print(f'ini adalah hasil dari: SHAPE:\n{data.shape}')
# %%
print(f'ini adalah hasil dari: DTYPES:\n{data.dtypes}')
# %%
print(f'ini adalah hasil dari: NUNIQUE:\n{data.nunique()}')
# %%
print(f'ini adalah hasil dari: CORR:\n{data.corr(numeric_only=True)}')
# %%
cols_to_use = ['Rooms', 'Distance', 'Landsize', 'BuildingArea', 'YearBuilt']
X = data[cols_to_use]
y = data['Price']
# %%
# split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)
X_train.shape, X_test.shape, y_train.shape, y_test.shape
# %%
from xgboost import XGBRegressor
model = XGBRegressor() # pakai xgb
model.fit(X_train, y_train)

# %%
# cari mae / mean absolute error
from sklearn.metrics import mean_absolute_error
y_pred = model.predict(X_test)

print(f'hasil MAE: {mean_absolute_error(y_pred, y_test)}') 
print(f'hasil MAE: {mean_absolute_error(y_test, y_pred)}') # di tuker pun sama aja
# %%
model = XGBRegressor(n_estimators = 200, learning_rate = 0.2) # cuma mau copy disini aja
model.fit(X_train, y_train) # 0.2 itu cukup besar dan cendrung overfitting
# %%
model = XGBRegressor(n_estimators = 4000, learning_rate = 0.05,early_stopping_rounds = 5, n_jobs=-1) # early ini untuk menghentikan training kalo udah sama terus
model.fit(X_train, y_train,
          eval_set=[(X_test, y_test)],
          verbose=False)
# %%
# saya mencoba menbandingkan dengan sklearn
from sklearn.ensemble import RandomForestRegressor
rf_model = RandomForestRegressor(n_estimators=1000, random_state=42, n_jobs=-1,)
rf_model.fit(X_train, y_train)
# %%
from sklearn.metrics import mean_absolute_error
y_pred = rf_model.predict(X_test)

print(f'hasil MAE: {mean_absolute_error(y_pred, y_test)}') 
print(f'hasil MAE: {mean_absolute_error(y_test, y_pred)}')



# %%
##### coba bandingkan dengan sklearn vs xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import pandas as pd

# --- Asumsi ---
# Anda sudah memiliki variabel-variabel ini dari sel sebelumnya:
# X_train, X_test, y_train, y_test
# ----------------

# 1. Latih dan Evaluasi Model XGBoost (Kita ulangi untuk memastikan)
# Anda bisa menggunakan model yang sudah dilatih atau melatih ulang di sini.
# Pastikan parameternya sama dengan model terbaik Anda sejauh ini.
xgb_model = XGBRegressor(n_estimators=500, learning_rate=0.05, early_stopping_rounds=5, n_jobs=-1, random_state=42)
xgb_model.fit(X_train, y_train,  
              eval_set=[(X_test, y_test)], 
              verbose=False)

xgb_preds = xgb_model.predict(X_test)
xgb_mae = mean_absolute_error(y_test, xgb_preds)


# 2. Latih dan Evaluasi Model Scikit-learn: RandomForestRegressor
print("Melatih model RandomForestRegressor dari Scikit-learn...")

# Inisialisasi model
# n_jobs=-1 akan menggunakan semua core CPU untuk mempercepat training
# random_state=42 untuk memastikan hasilnya bisa direproduksi
rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)

# Latih model
rf_model.fit(X_train, y_train)

# Lakukan prediksi dan hitung MAE
rf_preds = rf_model.predict(X_test)
rf_mae = mean_absolute_error(y_test, rf_preds)
print("Training RandomForestRegressor selesai.")


# 3. Buat Tabel Perbandingan dan Tampilkan Hasilnya
print("\n===== HASIL PERBANDINGAN MODEL =====")

# Buat DataFrame untuk menampilkan hasil dengan rapi
comparison_df = pd.DataFrame({
    'Model': ['XGBoost Regressor', 'Scikit-learn Random Forest'],
    'Mean Absolute Error (MAE)': [xgb_mae, rf_mae]
})

# Format angka agar mudah dibaca
comparison_df['Mean Absolute Error (MAE)'] = comparison_df['Mean Absolute Error (MAE)'].map('${:,.2f}'.format)

print(comparison_df)

# 4. Berikan Kesimpulan Otomatis
print("\n--- Kesimpulan ---")
if xgb_mae < rf_mae:
    improvement = ((rf_mae - xgb_mae) / rf_mae) * 100
    print(f"XGBoost adalah pemenangnya dengan MAE yang lebih rendah.")
    print(f"Model XGBoost sekitar {improvement:.2f}% lebih akurat daripada Random Forest.")
else:
    improvement = ((xgb_mae - rf_mae) / xgb_mae) * 100
    print(f"Random Forest dari Scikit-learn adalah pemenangnya dengan MAE yang lebih rendah.")
    print(f"Model Random Forest sekitar {improvement:.2f}% lebih akurat daripada XGBoost.")
# %%
