# %% [markdown]
from IPython.display import display, Markdown
display(Markdown(
f'''
- pengertian dari SVM(Support Vector Machine) adalah algoritma supervised learning untuk klasifikasi dan regresi.
- Intinya SVM mencoba menemukan garis (atau hiperplane) yang memisahkan kelas-kelas sehingga margin antara kelas maksimum.
- Sederhananya: cari garis terbaik yang memisahkan dua kelas dengan jarak terjauh ke titik terdekat dari tiap kelas.
- memaksimalkan margin (jarak terdekat di antar sisi kelas)

svm bisa di lakukan untuk klasifikasi dan regresi. tapi lebih sering di gunakan untuk klasifikasi.

'''))

# %% [markdown]
# referensi : https://www.svm-tutorial.com

# KONSEP DASAR
# Decision Boundary (Hyperplane)
# ![gambar hyperplane](https://i.ibb.co.com/X6PxGMC/5ee9e189-8a2e-4984-9092-20f316edfdda.jpg)
# sedikit penjelasan untuk gambar, ada 3 garis h1, h2, h3 setiap h menunjukan garis pemotong/pemisah antar class. dari ketiga garis ini mana pemisah terbaik?
    # jawabnnya adalah h3, karna itu adalah maksimum margin, mari kita bahas setiap garis pemisahnya
        #- H1 = untuk h1 bahkan tidak semua data di pisahkan dengan benar, ada beberapa titik yg salah klasifikasi.
        #- H2 = untuk h2 memang benar(semua data di pisahkan dengan tepat), tapi tidak mencapai margin maksimum.
        #- H3 = untuk h3 adalah garis pemisah terbaik, karna semua data di pisahkan dengan tepat dan mencapai margin **maksimum**.
# %% [markdown]
# ![maximum margin](https://i.ibb.co.com/B5Gx97SH/a9a4fdb9-7ad8-41d0-b0e7-09656f9cc369.jpg)
# ini adalah visual margin maxiumum, jadi mencari garis terdekat antar class.

# %% [markdown]
# ![ilustrasi](https://i.ibb.co.com/k6Hc8mP6/950dffbd-b753-4956-a965-1f81ebd45b65.jpg)
# Linearly Insperable & Kernel Tricks  / linear tak terpisahkan & kernel trik
# Linearly Inseparable

# Linearly separable artinya data bisa dipisahkan dengan garis lurus (kalau 2D) atau bidang (kalau 3D).
# Contoh: titik merah di kiri, titik biru di kanan → cukup tarik garis lurus di tengah untuk memisahkan.

# Linearly inseparable artinya data tidak bisa dipisahkan hanya dengan garis lurus.
# Lihat gambar kiri: ada titik hitam di luar membentuk lingkaran, dan titik biru di tengah.
# 👉 Kalau kita coba pisahkan dengan garis lurus, tidak mungkin karena bentuknya "lingkaran".
# gampangnya jika kondisi data tidak bisa di tarik garis pemisah, maka di namakan linearly insperable.
# caranya adalah svm, svm membuat agar menjadi 3 dimensi. masalahnya ada di komputasi yg berat karna menjadikannya 3 dimensi.
# cara mengatasi komputasi it adalah dengan kernel tricks. ada beberapa kernel tricks. cari sendiri.

# %%
import ssl

# Baris ini menonaktifkan verifikasi sertifikat SSL secara global
ssl._create_default_https_context = ssl._create_unverified_context

# %%
# Load Dataset
from sklearn.datasets import fetch_openml

X, y= fetch_openml('mnist_784', data_home= './dataset/mnist', return_X_y = True)
X.shape 
# catatan. sangat susah sekali untuk meload dataset ini. alasannya masih kurang tau. tapi kita harus
    # mengdownload datasets nya dengan manual juga di kaggle agar berhasil. 
        # harus mengnonaktifkan ssl juga. jadi cukup ribet banget.
            # dan internet harus lancar banget. kalo ngeleg dikit langsung error
# %%
import numpy as np
# --- BAGIAN INI YANG DIPERBAIKI ---
# 1. Ubah DataFrame menjadi NumPy array. .to_numpy() adalah cara modern.
X = X.to_numpy()
# 2. Sekarang X adalah NumPy array, Anda bisa mengubah tipe datanya.
X = X.astype(np.float32)
print(X.dtype)
# data X di ubah menjadi numpy array karna memang sepertinya komputer mengambil data yg saya download.
    # jadi tidak mengload data di cell atas. jadi harus saya ubah menjadi ndarray agar bisa di reshape. 
        # karna jika data frame itu tidak akan bisa.
# %%
# melakukan visualisasi
import matplotlib.pyplot as plt
import matplotlib.cm as cm # cm = color map

pos = 1
for data in X[:8]:
    plt.subplot(1, 8, pos)
    plt.imshow(data.reshape((28, 28)),
                cmap=cm.Greys_r)
    plt.axis('off')
    pos += 1
plt.show()
# cell ini untuk menunjukan visualisasi data. memang agak report. karna load datanya susah. dan harus mengubahnya menjadi 
# data numpy array kenapa 
print(type(X))
# %%
y = y.to_numpy().astype(np.int64) # X sudah saya ubah di aas. y saya ubah di sini. kenapa di pisah? 
# sebenarnya di gabungpun tidak maslaah, akan tetapi saya lupa.
# lalu kenapa np,int64 ini berbeda dengan np.float32 di X? alasannya adalah karna
# memang harus seperti itu. itu aturan umum, karna kumputer memang memiliki kemampuan membedakan klasifikasi.
# jika np.float32 / np.int64 di taruh sembarangan antar X dan y akan komputer akan error/ gagal total. bisa memprediksi 
# tapi jelek sekali. itupun kalo tidak error.
# %%
y[:8]
# %%
X_train = X[:6000]
y_train = y[:6000]
X_test = X[6000:]
y_test = y[6000:]
# %% [markdown]
# Classification dengan SVC (supoort Vector Classifier)
# training
from sklearn.svm import SVC

model = SVC(random_state = 0)
model.fit(X_train, y_train)
# %%
# test
from sklearn.metrics import classification_report

y_pred = model.predict(X_test)
hasil = classification_report(y_test, y_pred)
print(hasil)
# %% [markdown]
# PENTING !
# HyperParameter Tuning / hypertuning dengan GridSearchCV
from sklearn.model_selection import GridSearchCV

parameters = {
    'kernel': ['rbf', 'poly', 'sigmoid'], # rbf (Radial Basis Function): Pilihan default yang sangat kuat, bisa membuat batas keputusan yang melingkar/kompleks.
    'C': [0.5, 1, 10, 100],
    'gamma': ['scale', 1, 0.1, 0.01, 0.001]
} # ini ada 60 kombinasi parameter. jadi akan sangat lama.
grid_search = GridSearchCV(estimator= SVC(random_state= 0), # ini adalah modelnya
                           param_grid= parameters, # ini variable/ cara cross valdasinya
                           n_jobs= 6, # ini ibarat teknisinya ada 6. jika ingin semua teknisi (-1)
                           verbose= 1, # ini adalah seberapa cerewet. jika verbose = 3 maka akan menjelaskan dengan detail.
                           scoring = 'accuracy') # ini adalah metricnya. bisa di ganti dengan f1, recall, precision, dll.

grid_search.fit(X_train, y_train)
# %%
print(f'best score: {grid_search.best_score_}')
best_params = grid_search.best_estimator_.get_params()
print(best_params)
for param in parameters:
    print(f'\t{param} : {best_params[param]}')


# %%
# predict and evaluate
y_pred = grid_search.predict(X_test)
hasil = classification_report(y_test, y_pred)
print(hasil)
# %%
