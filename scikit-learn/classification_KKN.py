# %% [markdown]
# KNN / K-NEAREST NEIGHBOURS
# KNN adalah model machine learning yg digunakan untuk melakukan prediksi berdasarkan kedekatan karakteristik dengan sejumlah tetangga terdekat
# KKN dapat di terapkan pada classification maupun ragression task
# %%
import pandas as pd

sensus = {'tinggi': [158, 170, 183, 191, 155, 163, 180, 158, 178],
          'berat': [64, 86, 84, 80, 49, 59, 67, 54, 67],
          'jenis_kelamin': ['laki-laki','laki-laki','laki-laki','laki-laki','perempuan','perempuan','perempuan','perempuan','perempuan']}

sensus_df = pd.DataFrame(sensus)
sensus_df

# %%
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
for jenis_kelamin, d in sensus_df.groupby('jenis_kelamin'):
    ax.scatter(d['tinggi'], d['berat'], label=jenis_kelamin)

plt.legend(loc= 'upper left')
plt.title('sebaran data sensus menurut tinggi dan berat badan')
plt.xlabel('tinggi (cm)')
plt.ylabel('berat (kg)')
# plt.ylim(0, 100)
plt.grid()
plt.show()
# %% [markdown]
########################### classification dengan KNN ##########################
import numpy as np

X_train = np.array(sensus_df[['tinggi', 'berat']]) # X = feature. inget ya penulisan kalo double bracket [[]]
y_train = np.array(sensus_df['jenis_kelamin']) # y = target

X_train
y_train # karna ini classification, maka y yg masih..huruf/string harus di ubah menjadi angka... caranya adalah dengan biner.. di bawah 
# %%
########## mengubah string menjadi angka biner ########
from sklearn.preprocessing import LabelBinarizer

lb = LabelBinarizer() # memanggil label binarizer, tapi ini ga saya pake wkwkwk, cuma ikut tutorial aja.
y_train = lb.fit_transform(y)
print(f'hasil yg sudah di ubah adalah: {y_train}') # hasilnya akan menjadi angka, 0 = laki-laki, 1 = perempuan
    # karna hasil sudah 2dimensi, jadi harus di ubah jadi 1 dimensi lagi karna ini adalah y/target
# %%
y_train = y_train.flatten() # membuat agar menjadi 1 dimensi
y_train

# %%
############### training model KKN Classification ###########
from sklearn.neighbors import KNeighborsClassifier

k = 3 # k = jumlah tetangga terdekat
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)
# %%
################################ PREDISKSI JENIS KELAMIN ######################
warga = {'tinggi': 155,
         'berat': 70}
X_new = np.array([warga['tinggi'], warga['berat']]).reshape(1, -1) # di sini agar sedikit susah, harus teliti karna errorn keyerror
X_new
# %%
y_new = model.predict(X_new)
y_new
# %%
lb.inverse_transform(y_new) # mengubah angka biner menjadi string/ huruf lagi
# %% [markdown]
# VIASUALISASI KNN / K-NEAREST NIGHBOURS
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
for jenis_kelamin, d in sensus_df.groupby('jenis_kelamin'):
    ax.scatter(d['tinggi'], d['berat'], label = jenis_kelamin)

plt.scatter(warga['tinggi'],
            warga['berat'],
            marker= 's',
            color = 'red',
            label = 'misterius')
plt.legend(loc= 'upper left')
plt.title('sebaran data menurut tinggi dan berat')
plt.xlabel('tinggi')
plt.ylabel('berat')
plt.grid()
plt.show()
    # akan mengahsilkan scatter plot dengan titik merah (misterius) yang menunjukkan posisi individu baru dalam konteks data pelatihan.
# %% [markdown]
# ![rumus](https://iili.io/KKzkZiu.jpg)
misterius = np.array([warga['tinggi'], warga['berat']])
misterius
# %%
X_train
# %%
from scipy.spatial.distance import euclidean

data_jarak = [euclidean(misterius, d) for d in X_train]
data_jarak
# %%
sensus_df['jarak'] = data_jarak
sensus_df.sort_values(['jarak'])
sensus_df.sort_values('tinggi', ascending=False)
# %%
################ TESTING SET ################
X_test = np.array([[168, 65], [180, 96], [160, 52], [169, 67]])
y_test = lb.transform(np.array(['laki-laki', 'laki-laki', 'perempuan', 'perempuan'])).flatten()

print(f'{X_test}\n')
print(y_test)
# %%
######## prediksi ##########
y_pred = model.predict(X_test)
y_pred
# %% [markdown]
############## AKURASI / ACCURACY
# ![rumus](https://iili.io/KKIWW4p.jpg)
from sklearn.metrics import accuracy_score

acc = accuracy_score(y_test, y_pred)
print(f'akurasi model adalah: {acc * 100}%') # hasilnya 0,75% karna dari empat yg bener 3
# %% [markdown]
############# PRESISI / PRECISSION
# ![rumus](https://iili.io/KKIDIte.jpg)
from sklearn.metrics import precision_score

pre = precision_score(y_test, y_pred)
print(f'presisi skornya adalah: {pre * 100}%') # hasilnya 0,66 karna rumusnya tp/tp+fp | 2/3 hasilnya 0,66
# %% [markdown]
############# RECALL
# ![rumus](https://iili.io/KKImst9.jpg)
from sklearn.metrics import recall_score

rec = recall_score(y_test, y_pred)
print(f'hasil recall skor adalah: {rec * 100}%')
# %% [markdown]
############# F1 SCORE
# ![rumus](https://iili.io/KKTKZmJ.jpg)
from sklearn.metrics import f1_score

f = f1_score(y_test, y_pred)
print(f'hasil dari f1 adalah: {f}')
# %%
# ini penting
################### CLASSIFICATION_REPORT INI ADALAH GABUNGAN HITUNGAN DARI SEMUANYA. JADI INI LUMAYAN OVERPOWER
from sklearn.metrics import classification_report

cls_report = classification_report(y_test, y_pred)
print(f'hasil dari classifikasi adalah: \n{cls_report}')
    # sebagai catatan. di hasil classification ini, ada yg support | support adalah berapa jenis nilainya ini ada 2 yaitu laki, perempuan
        # yg dibawah hasil accuracy, recall, f1 adalah  hasil average/rata-rata dari nilai diatasnya.
# %% [markdown]
############# MCC / MATTHEWS CORRELATION COEFFICIENT
# ![rumus](https://iili.io/KKT4gEB.jpg)
from sklearn.metrics import matthews_corrcoef

mcc = matthews_corrcoef(y_test, y_pred)
print(f'hasil dari penghitungan MCC adalah: {mcc * 100}%')