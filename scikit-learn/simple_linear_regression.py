# %%
import pandas as pd

pizza = {'diameter': [6, 8, 10, 14, 18],
         'price': [7, 9, 13, 17.5, 18]}

df = pd.DataFrame(pizza)
df
# %%
import matplotlib.pyplot as plt

df.plot(kind='scatter', x='diameter', y='price')

plt.title('compared diameter and price of pizza')
plt.xlabel('diameter(inch)')
plt.ylabel('price(dollar)')
plt.xlim(0, 25)
plt.ylim(0, 25)
plt.grid(True)
plt.show()
    # hasilnya akan menampilkan 5 titik, karna memang data hanya ada lima baris yg bisa dijadikan data.
# %%
###################### SIMPLE LINEAR REGRESSION ###################
# penyesuaian dataset
import numpy as np

X = np.array(df['diameter']) # ini adalah feature/ dibaca vicer
y = np.array(df['price']) # ini adalah target

print(f'nilai X adalah: {X}\n') # nilainya masih dalam bentuk 1 dimensi. tidak sesuai dengan prosedur sklearn machine learning
print(f'nilai y adalah: {y}')
# %%
# di shape/bentuk
X = X.reshape(-1, 1) # -1 ini agak unik, "Saya tidak tahu/tidak mau menghitung berapa baris yang dibutuhkan.
    # Tolong hitung sendiri jumlah baris yang sesuai agar semua data asli bisa muat, dengan syarat jumlah kolomnya harus 1."
X.shape # hasilnya (5, 1) artinya 5 baris dan 1 kolom
# %%
X
# %%
# Training the model / Training simple Linear Regression
from sklearn.linear_model import LinearRegression

model = LinearRegression() # memanggil linear
model.fit(X, y)

# %%
# Visualisasi Simple Linear Regression
# ini model prediksi untuk garis regresinya...
X_vis = np.array([0, 25]).reshape(-1, 1)
y_vis = model.predict(X_vis)

plt.scatter(X, y) # memanggil gambar di atas/ yg tadi
plt.plot(X_vis, y_vis,'-r')

plt.title('compared diameter and price of pizza with regression')
plt.xlabel('inch')
plt.ylabel('dollar')
plt.xlim(0, 25)
plt.ylim(0, 25)
plt.grid()
plt.show()

# %%
########## catatan matematika ########
'''
FORMULA LINEAR REGRESSION
y = a + Bx

y = response variable
a = intercept --> bahasa indonesia: titik potong/ mencegat
B = slope --> bahasa indonesia: lereng/cekungan/kemiringan
x = explanatory variable
'''
print(f'hasil intercept(a) adalahL {model.intercept_}') # ini adalah hasil perhitungan
print(f'hasil slope(B) adalah: {model.coef_}') # ini adalah hasil perhitungan
# %% [markdown]
########### mencari nilai slope, ini rumusnya ########## / slope adalah kemiringan
# ![rumus](https://i.ibb.co.com/F4yXYWwS/3dfaf3ff-2bb8-4dad-936e-982f79dcf03a.jpg)
# ini benar gaes. cara memasukan gambar adalah dengan seperti ini.
# %%
print(f'nilai X:\n{X}\n') # nilai aslinya masih berbentuk 2d jadi harus di ubah menjadi 1d jika ingin menghitung nilainya.
print(f'nilai X: {X.flatten()}\n') # ini adalah cara mengubah menjadi 1d, dijadikan flat | kuncinya adalah .flaten()
print(f'nilai y: {y}\n')
# %%
######## variance ######## / perbedaaan
variance_x = np.var(X.flatten(), ddof = 1) #  ddof = 1 ini adalah untuk sample bukan population
print(f' variance_x adalah: {variance_x}') # hasilnya 23.2
# %%
######### covariance ######## /kovariansi
np.cov(X.flatten(), y) # hasilnya adalah array 2x2, jadi kita ambil nilai covariance saja yg 22.65 karna ada 2
# %%
covariance_Xy = np.cov(X.transpose(), y)[0][1] # [0][1] ini adalah untuk mengambil nilai covariance saja yg 22.65
print(f'covariance_xy adalah: {covariance_Xy}') # ini mencetak yg di ambil di atas


# %% [markdown]
############# mencari nilai intercept #############
# ! [rumus](https://i.ibb.co.com/NnKzYcnY/Whats-App-Image-2025-08-29-at-23-06-35.jpg)
# %%
intercept = np.mean(y) - slope * np.mean(X) # ini gagal karna slope belum di definisikan, tapi di yt bisa.

# %%
print(f'intercept adalah: {intercept}') # ini gagal masalahnya karna diatas ini slope ga bisa. jadi kita percaya saja hasilnya sama dengan 
    # intercept dari model di atas. | model.intercept_


# %%
################# prediksi harga pizza ################
diameter_pizza = np.array([12, 20, 23, 50, 111]).reshape(-1, 1) 
diameter_pizza
# %%
prediksi_harga = model.predict(diameter_pizza)
prediksi_harga
# %%
### looping agar terlihat rapih. ### ini adalah cara menampilkan hasil prediksi dengan rapi. absolut peak
for dmtr, hrg in zip(diameter_pizza, prediksi_harga): # looping ini bagus. harus segera saya ingat.
    print(f'diameter pizza: {dmtr} inch, prediksi harga: {hrg} dollar')




# %%
######### evaluasi simple linear regression
# training dan testing dataset
X_train = np.array([6, 8, 10, 14, 18]).reshape(-1, 1)
y_train = np.array([7, 9, 13, 17.5, 18])

X_test = np.array([8, 9, 11, 16, 12]).reshape(-1, 1)
y_test = np.array([11, 8.5, 15, 18, 11])

# %%
model = LinearRegression()
model.fit(X_train, y_train)
# %%
####### ini adalah scikit-learn untuk mencari nilai R²/R-squared ##########
from sklearn.metrics import r2_score
y_pred = model.predict(X_test)
r_squared = r2_score(y_test, y_pred)
print(f'r_squared adalah: {r_squared}')
# %% [markdown]
############## mencari nilai R²/R-squared ################ --> menghitung manual melalui rumus
# ![rumus](https://iili.io/KFGpdva.jpg)
# %%
# SS_res | res itu residual atau sisa. SS_res artinya total residu. Total kesalahan jika kita hanya menebak rata-rata.
ss_res = sum([(y_i - model.predict(x_i.reshape(-1, 1))[0])**2
              for x_i, y_i in zip(X_test, y_test)])
print(f'ss_res adalah: {ss_res}')
# %%
# SS_tot | tot itu total. Total kesalahan yang dibuat oleh model ML kita.
mean_y = np.mean(y_test)
SS_tot = sum([(y_i - mean_y)**2 
              for y_i in y_test])
print(f'SS_tot/total adalah: {SS_tot}')
# %%
# R²/R-squared | hasilnya sama dengan di atas yg memanggil library sklearn.metrics import r2_score
r_squared = 1 - (ss_res / SS_tot)
print(f'r_squared adalah: {r_squared}')