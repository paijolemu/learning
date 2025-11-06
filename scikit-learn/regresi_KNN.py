# %%
import pandas as pd

sensus = {'tinggi': [158, 170, 183, 191, 155, 163, 180, 158, 170],
          'berat': [64, 86, 84, 80, 49, 59, 67, 54, 67],
          'jk': ['pria', 'pria', 'pria', 'pria', 'wanita', 'wanita', 'wanita', 'wanita', 'wanita']}
sensus_df = pd.DataFrame(sensus)
sensus_df
# %%
############# REGRESSION WITH KNN ###########
# melakukan features dan target
import numpy as np
X_train = np.array(sensus_df[['tinggi', 'jk']])
y_train = np.array(sensus_df['berat'])

print(f'bentuk X: {X_train}')
print(f'bentuk y: {y_train}')
# %%
########### preproses dataset | mengubah label/string menjadi binarizer number
X_train_transposed = np.transpose(X_train)

print(f'hasil normal X_train: \n{X_train}')
print(f'hasil transpose: \n{X_train_transposed}')
# %%
from sklearn.preprocessing import LabelBinarizer

lb = LabelBinarizer()
jk_binarised = lb.fit_transform(X_train_transposed[1])

print(f'hasil awal transpose:\n{X_train_transposed[1]}')
print(f'hasil LabelBinarizer:\n{jk_binarised}')

# %%
jk_binarised = jk_binarised.flatten() # agar menjadi 1d atau colom mayoritas
jk_binarised
# %%
X_train_transposed[1] = jk_binarised # ini mengubah pria, wanita menjadi biner
X_train = X_train_transposed.transpose() # mengubah yg awalnya flatten menjadi 2d dengan transposed

print(f'bentuk X_train_transposed:\n{X_train_transposed}')
print(f'bentuk dari X_train terbaru:\n{X_train}')
# %%
################# training KKN regression ################
from sklearn.neighbors import KNeighborsRegressor

K = 3
model = KNeighborsRegressor(n_neighbors = K)
model.fit(X_train, y_train)
# %%
################# memprediksi berat badan #################
X_new = np.array([[150,1]])
X_new
# %%
###### predict
y_new = model.predict(X_new)
y_new


# %%
############### EVALUASI MODEL KKN REGRESI
X_test = np.array([[168,0], [180,0],[160,1],[169,1]])
y_test = np.array([65, 96, 52, 67])

print(f'bentuk X_test:\n{X_test}')
print(f'bentuk y_test:\n{y_test}')
# %%
y_pred = model.predict(X_test) # hasilnya adalah sekitar 70,79,59,70 artinya model memprediksi dengan hasil ini.
y_pred # apakah hasilnya benar/mendekati? mungkin saja tidak. karna model memprediksi berat badan 70. yg harusnya 65
    # lalu 79 yg harusnya 96, lalu 59 yg harusnya 52, lalu terakhir 70 yg harusnya 67. jadi model memprediksi y. tapi kurang berhasil
# %%
############ R2 / R-squared coefficient of dettermination
from sklearn.metrics import r2_score
r_squared = r2_score(y_test, y_pred)
r_squared
# %% [markdown]
############ MAE / mean absolute error | MAD / mean absolute Deviation
# ![rumus](https://iili.io/KK4YlKN.jpg)
# catatan... hasilnya pasti positif karna rumUsnya mutlak | |
from sklearn.metrics import mean_absolute_error

MAE = mean_absolute_error(y_test, y_pred)
print(f'hasil MAE adalah:\n{MAE}')

# %% [markdown]
############# MSE / mean squared error | MSD / mean squared deviation
# ![rumus](https://iili.io/KK48Jus.jpg)
# catatan.. ini mirip-mirip sama MAE
from sklearn.metrics import mean_squared_error

MSE = mean_squared_error(y_test, y_pred)
print(f'hasil MSE adalah:\n{MSE}')



# %%
############## PERMASALAHAN SCALLING PADA FEATURES| INI PENTING!!!
# catatan... semakin banyak data/angkanya maka akan semakin besar jarak accuracynya atau kemampuan model untuk
    # mencari kebenaran berkurang(kadang bisa jauh sekali perbedaaannya)
        # jadi ini adalah cara agar mengatasi perbedaaan data yg banyak
from scipy.spatial.distance import euclidean

# tinggi dalam milimeter(mm)

X_train = np.array([[1700,0], [1600,1]])
X_new = np.array([[1640, 0]])

[euclidean(X_new[0],d) for d in X_train]
    # hasil euclidean jaraknya adalah 60:40
        # kita coba kalo datanya/angka kecil
# %%
# tinggi dalam meter(m)

X_train = np.array([[1.7, 0], [1.6, 1]])
X_new = np.array([[1.64, 0]])

[euclidean(X_new[0], d) for d in X_train]
    # hasil euclidean jaraknya adalah 0.06:1
        # artinya data mentah mempengaruhi hasil dari machine learning. masalahnya ada di scalling


# %% [markdown]
# ini adalah cara mengatasi ketimpangan data agar data dapat optimal saat di gunakan. scalling
# ![rumus]()
from sklearn.preprocessing import StandardScaler

ss = StandardScaler()
# catatan... yg di ss.fit_transform adalah data training. yg hanya ss.transform adalah data test
# tinggi dalam milimeter(mm)
X_train = np.array([[1700,0], [1600,1]])
X_train_scaled = ss.fit_transform(X_train)
print(f'hasil yg sudah di scale: {X_train_scaled}')

X_new = np.array([[1640, 0]])
X_new_scaled = ss.transform(X_new)
print(f'hasil yg sudah di scaled:{X_new_scaled}')

jarak = [euclidean(X_new_scaled[0], d) for d in X_train_scaled]
print(f'jaraknya adalah: {jarak}')
# %%
# tinggi dalam meter(m)
X_train = np.array([[1.7, 0], [1.6, 1]])
X_train_scaled = ss.fit_transform(X_train)
print(f'hasil yg sudah di scaled: {X_train_scaled}')

X_new = np.array([[1.64, 0]])
X_new_scaled = ss.transform(X_new)
print(f'hasil yg sudah di scaled: {X_new_scaled}')

jarak = [euclidean(X_new_scaled[0], d) for d in X_train_scaled]
print(f'jarak: {jarak}')


# %%
############# menerapkan features scalling pada KNN
###### dataset
# training set
X_train = np.array([[158, 0], [170, 0], [183, 0], [191, 0], [155,1],[163, 1], [180, 1], [158, 1], [170, 1]])
y_train = np.array([64, 86, 84, 80, 49, 59, 67, 64, 67])
# test set
X_test = np.array([[168, 0], [180, 0], [160, 1], [169, 1]])
y_test = np.array([65, 96, 52, 67])
# %%
# features scalling (standard scaler / ss)
X_train_scaled = ss.fit_transform(X_train)
X_test_scaled = ss.transform(X_test)

print(f'X_train_scaled: \n{X_train_scaled}')
print(f'X_test_scaled: \n{X_test_scaled}')


# %%
############## TRAINING DAN EVALUASI MODEL
model.fit(X_train_scaled, y_train) # ini artinya modelnya di train. jadi yg dimasukan adalah X dan y train.
y_pred = model.predict(X_test_scaled)

MAE = mean_absolute_error(y_test, y_pred)
MSE = mean_squared_error(y_test, y_pred)

print(f'hasil dari MAE DAN MSE ADALAH:\n MAE: {MAE}\n MSE: {MSE}')

