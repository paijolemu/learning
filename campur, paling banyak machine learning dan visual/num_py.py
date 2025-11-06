############# NUMPY ################
# peraturan:         1. untuk operasi kalkulator(tambah,kurang,bagi,kali) tidak akan bisa dilakukan jika. angka dalam matrix tidak sama/sejajar jumlahnya.
#                    2. oyaa... dan tuples nya jangan lupa ada 2 ya (( ))
#                    3. golden rules adalah matrix itu harus sejajar jika kita ingin mengoperasikan baris atau kolomnya.
# [1,2,3]
# [4,5,6] --> inget ya... cara menghitungnya adalah 1. 2. 3. 4. | yg salah 1. 4. 5. | yg salah 1. 2. 3. 6. 5.
# [7,8,9] --> biasanya untuk sort()

import numpy as np

data = np.array([2,4,6,8,12])
nomor = [1,3,5,7,9]

print( data)
print(nomor)

a = data + 3 # artinya akan menambahkan setiap int yg ada dilist +3
b = nomor + [2] # artinya append 2

print(a)
print(b)


print('ini adalah hasil dari numpy')
print('\n============ NUMPY ARRAY ==========\n')
####### NUMPY AS NP ARRAY #######

# membuat vektor
c = np.array([1.5, 12.5]) # bisa di buat pakai titik (.) juga ternyata
print(c) # ini juga bisa ambil 1 nilainya saja lhoo

# membuat vektor dengan arange() | tidak pakai list [] ya! X
d = np.arange(1,12,2) # artinya adalah 1 sampai 12 increment 2 | hasilnya akan ganjil kalau mulai dari 1 | kacakunci: arange()
print(d)

# membuat linspace()
e = np.linspace(1,12,3) # artinya adalah 1 sampai 12 buat 3 angka saja yg ada di dalam space.| kuncinya adalah linspace
print(e)

# membuat matrix / multidimensi
f = np.array(([1,3,5,7] , [2,4,6,8])) # artinya membuat matrix atau meng enter list seperti looping... inget ya ini namnaya adalah matrix
print(f)

# membuat zeros / membuat agar hasilnya 0.
g = np.zeros([3,3,3,]) # membuat 3 kolom 3 baris 3 biji table. nilainya semua 0.
print(g)

# membuat ones / membuat agar hasilnya 1. value nya 1 ya...
h = np.ones([4,4,4]) # membuat 4 kolom 4 baris 4 biji table. nilai semua yg ada di dalam list 1.
print(h)
print('\n')

# matrix indentitas
j1= np.identity(5)
j2 = np.eye(5)
print(j1)
print(j2)
print('\n')

# coba sedikit lebih susah
L1= np.identity(5) # ini adalah paten/ tidak bisa di ubah... berbeda dengan eye yg fleksibel
L2 = np.eye(5, k=1) # eyes lebih fleksibel.. bisa di ubah k,m,n nya...| k artinya memajukan 1 kedepan alhasil 1 lainnya ikut kegeser
L2 = np.eye(5, M=1) # artinya mengambil 1 kolom/M
L3 = np.eye(N=1 ,M=5 ) # artinya mengambil 1 Baris/N
L3 = np.eye(1 ,5 ) # artinya mengambil 1 Baris/N
print(L1)
print('\n\n')
print(L3)

###### operasi aritmatika #####
print('\n ============= operasi aritmatika ================\n')

# list python
abc = [1,2,3,4,5]
defg = [6,7,8,9,10]

# list numpy
ghi = np.array([1,2,3,4,5])
jkl = np.array([6,7,8,9,10])


## ELEMENT WISE OPERATION
# penjumlahan / tambahan
hasil = abc + defg #--> ini kalo pertambahan python
hasil2 = ghi + jkl # --> ini pertambahan dengan numpy
print(hasil) # hasilnya akan menambahkan list ke list . | seperti append
print(hasil2) # hasilnya akan menambahkan list 1 ke list 2 dengan cara langsung ditambahkan secara eksplisit


# pengurangan / minus
'hasil = abc-defg --> hasilnya akan error... alasannya karna ini adalah python sederhana'
hasil2 = ghi - jkl
print(hasil2) # hasilnya akan mengurangi list 1 kepada list 2 secara eksplisit | -5 -5 -5 -5 -5


# perkalian 
'''hasil = abc * defg --> hasilnya akan error... alasanya karna ini adalah python'''
hasil2 = ghi * jkl # perkalian list pertama ke list ke 2
print(hasil2) # hasilya tidak erro karna numpy


# pembagian
'''hasil = abc / defg --> hasilnya akan error... alasannya seperti di atas'''
hasil2 = ghi / jkl
print(hasil2) # hasilnya akan membagi


print('\nmatrix\n')
## kita coba di multidimensi array numpy / matrix
yxz = np.array(([2,4,6] , [1,3,6]))
mno = np.array(([1,2,3] , [4,5,6]))
#
hasil = yxz + mno
hasil2 = yxz - mno
hasil3 = yxz / mno
hasil4 = yxz * mno
print(hasil)
print(hasil2)
print(hasil3)
print(hasil4)


print('\n============== indexing,slice,iterasi/looping =============')
############ index slice iterasi/loop ##########


contoh = np.arange(10)**3
print(contoh)

# indexing
print(f'element ke 1 dari contoh adalah: {contoh[0]}')
print(f'element ke 1 sampai 10 adalah: {contoh}')
print(f'element terakhir dari contoh adalah: {contoh[-1]}')

# slicing
print(f'element dari 1 ke 7 adalah: {contoh[0:7]}')
print(f'element dari 4 sampai akhir/selesai adalah: {contoh[4:]}')
print(f'element dari pertama sampai ke 5 adalah: {contoh[:5]}')

# iterasi / looping for
for i in contoh:
    print(f'hasil perulangan adalah: {i}')


############# PERKALIAN MATRIX/ MULTIDIMENSI ############
# peraturan:    1. np.ones tidak bisa berjalan/error saat dikali (8) jikalau jumlah baris dan kolomnya berbeda. jadi kolom dan barisnya sama.
#               2. perkalian matrix sebetulnya bisa fleksibel dengan @ --> ini sebenrnya seperti numpy dot()
#               3. inget ya... jangan sampe lupa perbedaan * dengan @.
#               4. * untuk element wise = pengalikan per element
#               5. @ ini sama saja dengan dot. atau perkalian matrix
# gold.rules->  6. Aturan Emas Perkalian Matriks (untuk @ dan np.dot) | Jumlah KOLOM matriks pertama HARUS SAMA DENGAN jumlah BARIS matriks kedua.
# 

print('\n=================== PERKALIAN MATRIX ======================\n'.center(45))
# data
num = np.array(([2,3,4] , 
                [4,5,6]))
nom1 = np.ones((3,2)) # 3 adalah kolom/ kebawah | 2 adalah baris atau kesamping | semua nilai 1. ya karna ini np.ones
#[1. 1. 
# 1. 1.
# 1. 1.]
print('contoh datanya adalah: ')
print(num)
print(nom1)

# perkalian matrix numpy biasa
'''jumlah = num * nom1
print('perkalian biasa')
print(jumlah)'''

# perkalian matrix menggunakan  numpy dot()
p = np.dot(num,nom1)
print('ini adlah perkalian menggunakan dot:')
print(p)
# akan mencoba seperti oop. hasil tetap sama ya!!
V = num.dot (nom1)
print('ini adlah perkalian menggunakan dot:')
print(p)

################## MANIPULASI MATRIX ##############
# peraturan:        1. reshape dan transpose itu berbeda!
#                   2. reshape lebih seperti urut dari kiri ke kanan | [1,2] [3,4] [5,6]
#                   3. transpose lebih seperti urut dari atas ke bawah baru di isi kanan | [1,4] [2,5] [3,6]
#                   4. flatten dan ravel itu berbeda!!! | walaupun sama sama membuat agar 1 dimensi
#                   5. flatten mengandung fitur copy() sedangkan ravel tidak.

print('\n=== manipulasi matrix===')
##
# data
example = np.array(( # bisa juga seperti ini
                    [1,2,3] , 
                    [4,5,6]
                    )) # seperti c++ ya hehe...
# mari kita memanipulasiii...

# shape
print('ini adalah contoh dari shape/bentuk matrix: ')
print(example.shape) # jadi ini memberitahu bahwa matrixnya (2,3) 2 kolom | 3 baris
# reshape
print('ini adalah contoh dari reshape/membentuk ulang matrix tanpa mengubah nilainya...')
print(example.reshape(3,2)) # hasilnya akan (3,2) 3 kolom | 2 baris. akan tetapi tidak mengubah nilai sebenarnya(awal)
# transpose matrix
print('ini adalah contoh dari transpose/mengubah kolom menjadi baris dan sebaliknya...')
print(example.T) # banyak caranya.. bisa seperti ini
print(np.transpose(example)) # ini cara lain juga
print(example.transpose()) # ini akan mengubah kolom menjadi baris dan baris menjadi kolom akan tetapi ini berbeda dengan reshape...
                           # bagian berbedanya ada di baris yg di ubah... coba lihat nilainya pasti berbeda
# flatten array, vektor baris
print('\nini adalah contoh dari flatten array/menjadikan matrix baris')
print(example.flatten()) # flatten itu ada fitur copy() | ini akan menjadikan matrix yg tadinya seperti kotak persegi menjadi datarrr....
print(example.ravel()) # perbedaanya flatten dengan ravel adalah flatten memiliki fitur copy() sedangkan ravel tidak
print(np.ravel(example)) # bisa juga seperti ini... yeay!!
# resize / mengubah nilainya terkhusus matrix
print('\nini adalah resize/mengubah nilai matrix: ')
print('\n==== resize ====')
example.resize(3, 2) # artinya mengubah menjadi (3, 2) 3 kolom 2 baris
print(example.shape) # ini untuk mengecek bagaimana bentuk matrixnya
print(example)
print('\n==== di resize lagi ====')
example.resize(6, 1) # artinya mengubah menjadi (6, 1) 6 kolom sebaris
print(example.shape) # kuncinya adalah .shape
print(example)

############### stacking matrix ###############
print('\n=== ini adalah stcking ===')

''' hasilnya akan error jika numpy atau python ini di jumlahkan akan tetapi saat penjumlahan tidak sepadan.
    [1,2,3] + [8,9] --> ini akan error karna barisnya tidak sepadan | yg pertama ada 3 baris yg kedua ada 2 baris.. ini ga sepadan'''

# ini data list ya
angka = np.array(([3,4,5]))
berapa = np.array(([7,8,9]))
#
angka_matrix = np.ones((2,3))
berapa_matrix = np.zeros((2,3)) # inget ya kalo ga gini.. bisa error karna baris yg ingin di operasikan harus sama.
#
# stacking
m = np.hstack((angka,berapa)) # ini artinya adalah horizontal stack/baris dengan variable angka di depan dan variable berapa di belakang.
n = np.vstack((angka,berapa)) # ini artinya adalah vertikal stack/kolom dengan variable angka di depan dan variable berapa di belakang.| inget tuples 2 ya
print('ini adalah hstack dan vstack: ')
print(m)
print(n)
# kita coba kalau matrix
M = np.hstack((angka_matrix, berapa_matrix)) # ingt tuples 2 ya.. saya waktu latihan banyak salahnya disini.
N = np.vstack((angka_matrix, berapa_matrix))
print('ini hasil hstack dan vstack jika di matrix: ')
print(M)
print(N)


############# CARA ADVANCE MEMBUA ARRAY ###############
print('\n\n==ini cara memuat array advance==')
print('=== INI LUAMAYAN SUSAH GAES ===')
##
# data sederhana seperti biasa
meja = np.array(([1,2,3],
                 [5,6,7]), dtype = int)
print(f'ini data: {meja}') # data basic

# pakai def
def kuadrat(kolom, baris): # isi fungsi ada 2 kolom dan baris.
    return kolom**2

def pangkat(vertikal, horizontal): # ini sengaja saya ubah agar fun
    return vertikal + horizontal
# array yg lebih advance
Q = np.fromfunction(kuadrat, (3,3), dtype = int) # ini adalah array numpy yg lebih advance dengan def.| ini arange. dari 0-2 akan kuadrat(dari def)
W = np.fromfunction(pangkat, (2,4), dtype = float) # artinya memanggil fungsi def dengan nilai matrix (2,4) dengan pertambahan
# kita print ya... inget ini berdasarkan funci def juga.
print(Q)
print(W)
print('\n\n')

# iterable atau looping
ulang = [i*2 for i in range(5)] # i* artinya nilai yg ada di variable i di kali 2
print(ulang)
R = np.fromiter(ulang, dtype = float) # ini untuk print advance dari looping'an | kuncinya ada di fromiter
print(R)

# multitype array

dtipe = [('nama','S10'), ('berat badan', int)] # ini ada type datanya.. kita buat sendiri... akan tetapi
    # ini kuncinya ada di S. yg dimana artinya adalah S untuk string khusus stype bytes dan 10 untuk memorynya... sebenarnya ini bisa di ganti menjadi U
kolom = [
            ('paijo', 90),
            ('gallen', 100),
            ('wawan', 91)
]

E = np.array(kolom, dtype = dtipe)
print(E)


############# SORT ARRAY NUMPY ##########
print('\n\n========= sort array numpy ==========')
##
# data unik | pakai random,, btw saya udah bingung namain variablenya
lembar = np.floor(np.random.randn(1,5)*2) # ini sudah random ya...| kuncinya ada di np.random.randn
    # artinya adalah floor = tampilkan 1 angka saja di depan titik (.) | randn = untuk membuat range matrix menjadi random
print(lembar)

# ingett ya ini belajar sort
print('nilai max adalah: ',lembar.max()) # artinya mengambil angka terbesar(max) | kuncinya max()
print('posisi max ada di indeks ke-',lembar.argmax()) # artinya nilai terbesar yg di ambil max itu indeks ke berapa | kuncinya argmax()
print('nilai min adalah: ',lembar.min()) # artinya mengambil angka terkecil(min) | kuncinya min()
print('posisi min ada di indeks ke-', lembar.argmin()) # artinya nilai yg terkecil itu indeks ke berapa | kuncinya argmin()
print('\n=== ini sort basic ===') # enter
# sort
print('urutkan dari yg terkecil: ')
print(np.sort(lembar)) # mengurutkan nilai terkecil
print(np.argsort(lembar)) # artinya nilai terkecil itu indeks ke berapa... inget ini tidak urut karna memang mengambil indeks bukan nilainya.
    # saya sempet salah penulisan... yg bener seperti ini ya!

############ ini saya mencoba yg 2 dimensi atau matrix ###########
# [1,2,3]
# [4,5,6] --> inget ya... cara menghitungnya adalah 1. 2. 3. 4. | yg salah 1. 4. 5. | yg salah 1. 2. 3. 6. 5.
# [7,8,9]

print('\n\n========= sort array numpy salinan uji coba matrix/2 dimensi ==========')
##
# data unik | pakai random,, btw saya udah bingung namain variablenya
lembar = np.floor(np.random.randn(3,3)*2) # ini sudah random ya...| kuncinya ada di np.random.randn
    # artinya adalah floor = tampilkan 1 angka saja di depan titik (.) | randn = untuk membuat range matrix menjadi random
print(lembar)

# ingett ya ini belajar sort
print('nilai max adalah: ',lembar.max()) # artinya mengambil angka terbesar(max) | kuncinya max()
print('posisi max ada di indeks ke-',lembar.argmax()) # artinya nilai terbesar yg di ambil max itu indeks ke berapa | kuncinya argmax()
print('nilai min adalah: ',lembar.min()) # artinya mengambil angka terkecil(min) | kuncinya min()
print('posisi min ada di indeks ke-', lembar.argmin()) # artinya nilai yg terkecil itu indeks ke berapa | kuncinya argmin()
print('\n=== ini sort basic ===') # enter
# sort
print('urutkan dari yg terkecil: ')
print(np.sort(lembar)) # mengurutkan nilai terkecil
print(np.argsort(lembar)) # artinya nilai terkecil itu indeks ke berapa... inget ini tidak urut karna memang mengambil indeks bukan nilainya.
    # saya sempet salah penulisan... yg bener seperti ini ya!

####### sorting khusus ######
print('\n\n')

# hasil salinan di atas...wkwkwk
dtipe = [('nama','S10'), ('berat badan', int)] # ini ada type datanya.. kita buat sendiri... akan tetapi
    # ini kuncinya ada di S. yg dimana artinya adalah S untuk string khusus stype bytes dan 10 untuk memorynya... sebenarnya ini bisa di ganti menjadi U
kolom = [
            ('paijo', 90),
            ('gallen', 100),
            ('wawan', 91)
]

E = np.array(kolom, dtype = dtipe)
print(E)
# jika kita ingin mengambil namanya saja... atau berat badannya saja...
print(np.sort(E, order='nama')) 
print(np.sort(E, order='berat badan'))


























