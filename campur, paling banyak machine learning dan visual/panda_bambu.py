import pandas as pd
import os


############## BELAJAR PANDAS ############
# PERATURAN     : 1. format type data ada pandas series dan dan pandas dataframe | perbedaanya data frame lebih rapih
#                 2. jika series cara braketnya adalah ()/[] | kalo data frame pakai double list [[]]
#                 3. sorry to say.. tapi untuk pandas di VSC ini bener bener terlihat jelek.. berbeda dengan spyder atau jupyter

os.system('cls')

data = pd.read_csv('hypertension_dataset.csv') # ini coding untuk membaca akan tetapi tidak ada perintah menampilkan
print(data.head()) # akan memunculkan 5 karna ini adalah head
print(data.tail()) # akan memunculkan 5 terbawah


print(pd.options.display.max_rows) # ini adalah menjunjukan jumlah rows yg di tampilkan tapii... max | total 60
print(pd.options.display.min_rows) # 10. artinya adalah jumlah data yg di tampilkan adalah 10 rows

# mengganti jumlah total yg di tampilkan
pd.options.display.max_rows = 100
pd.options.display.min_rows = 100

print(data)
data.info() # ini akan menampikan info detail tentang data dan teknik pemrogramannya
print('\n\n')
data.describe()
print(data.describe) # entah ini kok ga bisa.. tapi ini intinya adalah untuk menjelaskan/mendeskripsi bagaimana data total di baca
                     # ini lebih ramah pengguna yg tidak paham coding. | orang bilang statistik sederhana.

data.describe(include = 'O') # untuk ini malah berhasil. jadii.. ini menjelaskan dengan lebih rinci dengan menghilangkan angka 0 
print(data.describe(include = 'O')) # count = total, unique = unik/data yg berbeda, top = yg paling banyak, 
                                    # freq = jumlah yg paling banyak atau menunjukan data int top. kalo top dan berisi str, freq berisi int

# mengambil data sebagian saja. dan mencoba mengubah index langsung ke no 1
data_ambil = data[['Age', 'Stress_Score']].reset_index(drop = True) # artinya mengambil data age, dan stress_score saja. dan membuat index mulai dari 1
data_ambil.index += 1 # membuat indeks mulai dari 1 
print(type(data)) # type data adalah dataframe. | pandas.core.frame.DataFrame
print(data_ambil)

# bisa juga ambil pakai dot(.)
data.Age # ini juga akan menampilkan sama.

# mengecek apakah sama
data.Age.equals(data['Age']) # kita cek apakah sama???? hasilnya adalah......
print(data.Age.equals(data['Age'])) # TRUE!!!!!! YEAHH!!!

# mengambil 1 baris saja.. sebelumnya kan mengambil colom nah ini 1 baris
data.iloc[0] # ini akan mengambil 1 baris saja.
print(data.iloc[0:8]) # yes sirr... ini artinya saya mengambil baris ke-0 sampe ke-7 total ada 8 baris
# kebawah/ colom
data.iloc[:,1] # ambil data indeks ke-1. age.
print(data.iloc[:,1]) # hasilnya yesss keambil
data.iloc[[0,4,7]] # seperti ini juga bisa

# mengambil yg agak advance | baris dan colom
print('\n\n\n')
data.iloc[[1,3,6,8],[1,2,3]] # artinya 4 baris indeks ke-1,3,6,8 dan 3 colom ke-1,2,3
print(data.iloc[[1,3,6,8], [1,2,3]]) # kita print saja. gaes dan berhasilll












