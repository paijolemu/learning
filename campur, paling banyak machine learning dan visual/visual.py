# %% dengan ini artinya mengubah terminal menjadi jupyter
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
sns.set()

##########################
# %%
df = pd.read_csv('hypertension_dataset.csv')
df.head()



# %%
df.hist() # akan membuat histogram semua colom
df.Age.hist() # akan membuat histogram Age
# %%
df.Age.plot.hist(bins= 30) # bins = artinya jumlah batang.. semakin banyak bins= semakin detail. bukan berarti bagus ya.
# %%
df.Age.plot.hist(bins= 30, alpha = 0.3) # alpha artinya adalah transparan | alpha = 1 artinya hilang / 100 transparan

# %% [markdown]
#**ini adlaah contoh yg benar**


# %%
np.random.seed(100)
angka = []
for i in range(100):
    angka.append(i)
# %%
angka
# %%
buat = pd.DataFrame(np.random.rand(100,4), angka, ['A', 'B', 'C', 'D']) # membuat data frame. dengan baris 100. dan kolom 5
buat.head() # membuat data frame berdasarkan data buat
# %%
buat.plot.area()
# %%
buat.plot.area(alpha = 0.4)
# %%
buat.plot.bar() # hasilnya akan terlihat error. karna banyak sekali data disitu. makanya saya buat data baru di bawah
# %%
data_baru = pd.DataFrame(np.random.rand(5,4), [1,2,3,4,5], ['A', 'B', 'C', 'D'])
print(data_baru)
data_baru.plot.bar(stacked = True) # membuat data seperti bar. | stacked adalah menumpuk data sebenarnya... jadi datanya akan terlihat banyak/lebih besar
data_baru.C.plot.bar()
# %%
buat.plot.scatter(x = 'A', y = 'B', c = 'C', cmap = 'inferno') # artinya membuat membuat titik titik scatter/menyebarkan
                    # yg mana titik titik itu ada warna inferno. dan di visualkan oleh c = B artinya c = color, B untuk visualkan
# %%
buat.plot.scatter(x = 'A', y = 'B', c = 'C') # c = ini adalha color
# %%
buat.plot.box()
# %%
buat.plot.line(y ='B', lw = 0.2, figsize = (15,5)) # artinya membuat visual line. dari variable buat. dengan B yg kita ambil
            # lw = adalah line width/ lebar garis, dan figsize adalah lebar visual yg di tambilkan. 15 kesamping dan 5 ke bawah. | [persegi]
# %%
buat.plot.hexbin(x = 'A', y = 'B', gridsize= 15) # hexbin adalah hexagonal / 6 sudut.  gridsize adalah ukuran setiap hex
                            # semakin besar gridsize akan semakin kecil justru. apabila gridsize 5 artinya itu besar jika di bandingkan 15.


















# %%
satu = pd.DataFrame({
                        'A':['a0', 'a1', 'a2', 'a3', 'a4', 'a5'],
                        'B':['b0', 'b1', 'b2', 'b3', 'b4', 'b5'],
                        'C':['c0', 'c1', 'c2', 'c3', 'c4', 'c5'],
                        'D':['d0', 'd1', 'd2', 'd3', 'd4', 'd5']},
                        index = [0,1,2,3,4,5])
satu
# %%
dua = pd.DataFrame({
                        'A':['a6', 'a7', 'a8', 'a9', 'a10', 'a11'],
                        'B':['b6', 'b7', 'b8', 'b9', 'b10', 'b11'],
                        'C':['c6', 'c7', 'c8', 'c9', 'c10', 'c11'],
                        'D':['d6', 'd7', 'd8', 'd9', 'd10', 'd11']},
                        index = [6,7,8,9,10,11])
dua
# %%
tiga = pd.DataFrame({
                        'A':['a12', 'a13', 'a14', 'a15', 'a16', 'a17'],
                        'B':['b12', 'b13', 'b14', 'b15', 'b16', 'b17'],
                        'C':['c12', 'c13', 'c14', 'c15', 'c16', 'c17'],
                        'D':['d12', 'd13', 'd14', 'd15', 'd16', 'd17']},
                        index = [12,13,14,15,16,17])
tiga
# %%
# concatination
pd.concat([satu,dua,tiga]) # concat/concatenation / penggabungan --> akan menggabung semua data | kuncinya concat()
# %%
# mengecek cara berpikir concat()
pd.concat([satu,dua,tiga], axis = 1) #--> bisa dilihat dari gambar bahwa. cara berfikir concat adalah menyambung indeks .
                                    # karna tidak ada yg cocok maka akan NaN di nilainya. untuk indeks akan terus menumpuk.



###### membuat bisnis wkwkwk #########



# %%
# ini datanya.
bisnis = pd.DataFrame({
                'perusahaan': ['telkom','kaggle', 'microsoft', 'google', 'telkom', 'microsoft', 'meachine l'],
                'karyawan' : ['adnan', 'paijo', 'wawan', 'gallen', 'tulung', 'prayt', 'mosst'], 
                'usia': [22, 33, 44, 78, 43, 11, 21]})
bisnis
# %%
bisnis.groupby('perusahaan').describe() # dengan memakai describe. itu akan memberi sedikit info tentang datanya. 
                # groupby() dalam pandas. Proses ini dikenal sebagai "Split-Apply-Combine".| artinya banyak rumus yg ada di dalamnya

# %%
bisnis.groupby('perusahaan').describe().transpose() # transpose() adalah untuk menukar baris menjadi kolom dan kolom menjadi baris.
                     # biasanya hasilnya akan ada banyak 0 di belakangnya

# %%
# menghitung secara sub kelompok
bisnis.groupby('perusahaan').count() # dengan data yg saya miliki ini akan menghitung jumlah karyawan.
# %%
# mengurutkan TERbesar dan TERkecil.
bisnis.groupby('perusahaan').max() # ini Data Frame ya...
bisnis.groupby('perusahaan').min()
# %%
bisnis.groupby('perusahaan').usia.min() # tampilan akan jelek. data series
# %%
satu = pd.DataFrame({
    'key': ['k0', 'k1', 'k2', 'k3', 'k4'],
    'one': [1,2,3,4,5],
    'two': [6,7,8,9,10]
})
satu
# %%
dua = pd.DataFrame({
    'key': ['k0', 'k1', 'k2', 'k3', 'k4'],
    'one': [1,2,3,4,5],
    'two': [6,7,8,9,10]
})
dua
# %%
banyak = pd.merge(satu,dua, how='inner', on='key' ) # pd.merge() digunakan untuk menggabungkan dua DataFrame (tabel) berdasarkan nilai-nilai di kolom yang sama
                                # 'key' sebagai kunci atau acuan untuk menggabungkan
                                # inner artinya menggunakan teknik inner. konsepnya sama seperti sql. walaupun saya belum belajar sql wkwkwk
                                #  tetapi hanya menyertakan baris-baris yang nilai pada kolom 'key'-nya cocok dan ada di kedua tabel tersebut.

banyak
# %% [markdown]
# **masih ada banyak lagi seperti join. --> menggabungkan. akan tetapi harus ada kecocokan di index nya. jika indeks cocok akan tergabung.
# **sebagai catatan... perbedaan merge dengan concat adalah merge akan menggabungkan berdasarkan kecocokan ( sistem tebang pilih )
# ** jika concat itu akan menggabungkan tanpa peduli tentang kecocokan.. intinya seperti menumpuk kertas.
# %%
