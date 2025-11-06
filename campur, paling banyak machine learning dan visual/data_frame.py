import pandas as pd
import numpy as np
import os

########### DATA_FRAME DI PANDAS ########
# data dasar
nama = ['paijo', 'wawan', 'gallen']
umur = [20, 18, 17]
dictionary = {
    'dio': 17,
    'dia': 15,
    'dani': 22
    }
# cek sebentar....
os.system('cls')
np_umur = np.array(umur)
print(np_umur)
np_nama = np.array(nama)
print(np_nama)

# pd
pd.Series(data = umur) # jika sperti ini akan menampilkan umur dan index... kebawah seperti looping
print(pd.Series(data = umur)) # ini umur --> int
pd.Series(data= nama) # saya coba di nama.. ada indexnya juga..
print(pd.Series(data= nama)) # ini nama --> object atau biasanya string/str


kamus = pd.Series(data=nama, index=umur) # indeks/ nomor urut di ubah jadi umur
print(kamus)
print(kamus[20]) # saya mengambil nilai yg ada di indeks 20. yaitu paijo
                   # disini error terus waktu latihan.. jadi yg bener adalah seperti ini... yeayy
print(np_umur[0]) # ini saja mengambil indeks pertama dari np_umur

#
pd.Series(dictionary) # mencoba mencetak dictionary.. hasilnya adalah nama dengan umurnya.. tidak ada index
print(pd.Series(dictionary))

print(pd.Series(np_umur)) # jika seperti ini bukan dictionary maka hasilnya umur dengan indeks
print(pd.Series(umur)) # hasilnya sama saja dengan numpy. typenya pun sama int64

# pandas
kamus.to_frame()
print(kamus.to_frame) # ini membuat data yg kita punya menjadi lebih bagus di lihat seperti sebuah table

########## BUAT DATA BARU YG LEBIH BAGUS ############
no = np.arange(3)
huruf = ['kendi','wiwit','sukidi']
df = pd.Series(data = huruf, index = no)
print(df)

nomor = np.arange(5)
abjad = ['pemuda', 'pancasila', 'psht', 'pasukan', 'banser']
ds = pd.Series(data = abjad, index = nomor)
print(ds)
hasil = df + ds
print(hasil)
    # maksud dari ini adlah.. jika pd itu bisa menambahkan data type apapun
    # tidak equal/ ada lawannya maka akan menjadi NaN

#################### DATA FRAME ###################
# peraturan         : 1. jika kita membuat 2 baris 3 colom.. lalu kita hanya menuliskan nama masing masing 2 nama untuk baris dan 2 nama untuk colom.. maka akan error. karna
#                     2. perintah menganggap bahwa ada 2 baris dan 3 colom tapi kenapa kita hanya memberi nama untuk 2 baris dan 2 colom nama saja.
#                     3. Cara mengambil data menggunakan kurung siku [] pada DataFrame secara default adalah untuk mengambil KOLOM, bukan BARIS. | jadi A tidak bisa di ambil
#                     4. jika ingin agar bisa mengambil A harus pakai loc
#                     5. axis=0 berarti operasi dilakukan pada baris (rows). Ini adalah nilai default untuk banyak fungsi.
#                     6. axis=1 berarti operasi dilakukan pada kolom (columns). <-- ini penting sekalii!!!!!
os.system('cls')

np.random.seed(100) # ini guna untuk machine learning... berfungsi untuk memastikan agar angka random yg kita buat tetep sama.

soal = pd.DataFrame(np.random.randn(3,4), ['A','B','C'], ['Satu', 'Dua', 'Tiga', 'Empat']) # np.random.randn(3,4) --> Ini membuat data acak dengan bentuk 3 baris dan 4 kolom.
                                                                # barisnya adalah A B C dan colomnya adalah SATU, DUA, TIGA, EMPAT
print(soal)

# mengambil
print(soal['Satu']) # mengambil colom 1 dengan isinya. | peraturan jika seperti ini bisa(mengambil nilai dari colom)
print(soal.loc['A']) # mengambil nilai dari baris A | tidak bisa kalau tidak ada .loc

# membuat dataframe baru dari menambahkan colom
soal['Lima'] = soal['Satu'] + soal['Tiga'] # ini menambahkan data frame dengan menjumlahkan colom satu + colom 3 
print(soal) # hasilnya akan ada colom baru bernama Lima

# me remove / hapus sebagian
soal = soal.drop('Tiga', axis=1) # ini adalah cara untuk remove colom...| kata kunci .drop axis =1 artinya bagian pada colom, jika axis = 0, artinya baris
print(soal) # variable harus di tulis ulang sebagai variable juga... kalau tidak maka tidak akan tersave oleh sistem | soal = soal.drop <-- harus seperti ini.

soal.drop('C', axis= 0, inplace= True) # jika ada inplace True maka tidak perlu variable berulang untuk save sistem
print(soal) # hasilnya baris C akan remove.

# shape / condition
soal.shape # artinya menunjukan kondisi dari variable soal. hasilnya adlah (2, 4) --> 2 baris: A,B dan 4 colol: Satu,Dua,Empat,Lima
print(soal.shape) # inget ya tidak perlu pakai tuples

# bisa juga bollean
print(soal<0) # jika hasil kurang dari nol (0) maka akan True jika lebih akan false.
soal_bol = soal<0 # artinya hanya menampilkan nilai jika hasilnya kurang dari 0, jika lebih dari 0 maka akan menjadi NaN
print(soal[soal_bol]) # mencetak yg kurang dari 0


################## PANDAS DI CSV ###############
os.system('cls')
buku = pd.read_csv('hypertension_dataset.csv')
print(buku)

# mengambil colom Age
print(buku['Age']) # artinya mengambil colom Age
age = buku['Age'] # membuat variable baru yg namanya age
print(age.mean()) # ini adalah rumus menghitung rata-rata Age/umur di dalam data csv
print(age.median()) # ini adalah rumus untuk mengambil nilai tengah.. jika semua dari di urutkan dari awal-terakhir maka nilai tengah adalah ini.
print(age.sum()) # semua umur jika di jumlahkan hasilnya seperti ini..
print(age.std()) # Standar Deviasi (.std()): Memberi tahu seberapa LEBAR sebaran data Anda di sekitar pusat itu. Ini adalah PENGGARIS Anda.

# menggunakan dropna()
hilang_buku= buku.dropna() # artinya menghilangkan data jika ada NaN.. tentu jika ada nilainya ya tidak hilang
print(hilang_buku) # intinya ini jika baris/colomnya ada yg nan maka akan hilang nilainya
# bisa manipulasi dropna() dengan axis
hilang_buku = buku.dropna(axis = 1) # jika NaN ada di colom maka akan hilang.| karna axis = 1--> adalah colom
hilang_buku = buku.dropna(axis = 1, thresh = 1) # artinya jika di colom ada NaN dengan jumlah 1 maka tidak apa-apa.. jika lebih dari 1 maka akan hilang
                            # thresh adalah threshold/ambang batas, yg mana ini jumlah maximum untuk menghilangkan NaN.
                            # ini bertujuan agar menghilangkan colom yg banyak NaN karna untuk mengfilter.
print(hilang_buku)

# memberi value/nilai NaN
buku.Age.fillna(value = buku.Age.mean()) # ini adalah rumus untuk mengisi NaN. khususnya umur.. tapi karna umur di sini sudah terisi jadi tidak ada yg berubah
print(buku.Age.fillna(value = buku.Age.mean())) # karna semua umur sudah terisi jadi tidak ada yg diisi oleh perintah ini. | kuncinya .fillna--> untuk mengisi NaN(tidak ada)









