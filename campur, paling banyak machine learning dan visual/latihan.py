# catatan. latihan ini sangat matematika karna urusan dengan aljabar linear
import numpy as np

# data biasa 2 dimensi
data = np.array([1,3]) # X = 1 Y = 3
data2 = np.array([2,1]) # X 2 Y = 1
''' cara menghitung ini adalah seperti ini.| 1 * 3 = 3
                                           | 2 * 1 = 2
                                           | 3 + 2 = 5 '''
# perkalian dot. dalam matrix
A = np.dot(data,data2) # artinya dot arah data sumbu x(1) y(3) ke arah data2 sumbu x(2) y(1)
print(A) # hasilnya 5
# data baru 3 dimensi
angka = np.array([1,3,0]) # X/i = 1     Y/j = 3     Z/k = 0
angka2 = np.array([2,1,0]) # X/i = 2     Y/j = 1     Z/k = 0
'''cara menghitung matrix 3 dimensi..| (3 * 0) - (1 * 0) = 0 --> menghitung X/i
life hack!! jika ingin menghitung... | (1 * 0) - (2 * 0) = 0 --> untuk Y agak spesial harus di * -1     <-- ini penting!!
dibawah....                          | (1 * 1) - (2 * 3) = -5 --> menghitung Z/k
jika ingin menghitung X tutup secara mental(jangan di lihat) sumbu X. jadi akan seperti ini [3 * 0] - [1 * 0] = 0
jika ingin menghitung Y tutup secara mental(jangan di lihat) sumbu Y. jadi akan seperti ini [1 * 0] - [2 * 0] = 0 | case Y ini spesial hasil Y akan di *-1
jika ingin menghitung Z tutup secara mental dan kali kan secara silang... namanya determinan [1 * 1] - [2 * 3] = -5 

ini adalah rumusnya.
'''
# perkalian cross. dalam matrix
B = np.cross(angka,angka2)
print(B) # hasilnya [0  0 -5]
# catatan... inget ya ini rumus seperti menghitung sumbu x dan y.| bisa juga untuk menghitung derajat dengan sin, cos, tan (costanta)



####### invers dan determinan/penentu ######
print('\n====== invers and determinant ======')

D = np.array([(2,2), (-2,2)])
print(D)

# invers matrix --> linear algebra/aljabar linear
E = np.linalg.inv(D) # --> artinya matrix yg bisa undo transformasi dari variable D
'''
Pojok kiri atas: 2 * (1/8) = 2/8 = 1/4 = 0.25
Pojok kanan atas: -2 * (1/8) = -2/8 = -1/4 = -0.25
Pojok kiri bawah: 2 * (1/8) = 2/8 = 1/4 = 0.25
Pojok kanan bawah: 2 * (1/8) = 2/8 = 1/4 = 0.25
'''
print(E) # ini penjelasan kenapa hasil invers seperti itu....

# determinan matrix --> linear algebra/aljabar linear
F = np.linalg.det(D) # saya ubah agar mudah di pahami --> [a,b]-[c,d] rumusnya adalah: [a * d] - [b * c] = hasil
G = np.linalg.det(E) # hasilnya 0.125 kenapa??... karna 1/8 adalah 0.1250000
print(F)
print(G)
print('\n=== persamaan linear ===')

######## persamaan linear untuk menghitung X dan Y pada aljabar. #########
# penjelasan singkat... kita mencari x dan y  dengan data yg ada adalah soal dan hasilnya.. akan etapi albajarnya atau x dan y nya belum ketemu...
#                       jadi ini adalah rumus agar aljabarnya ketemu.. | katakunci invers(), dot(@), solve(), linalg


a = np.array(([2, 3], 
              [2, 1]))
b = np.array(([23,14]))
print(a)
print(b)

a_invers = np.linalg.inv(a)
print('\n', a_invers)
# dot
X1 = np.dot(a_invers, b)
print(X1)
# cara yg lebih mudah dan advance.
X2 = np.linalg.solve(a,b)
print(X2)
''' Penjelasan:
                Secara matematika, jika kita punya A * X = Y, maka solusinya adalah X = A_invers * Y. Kode ini melakukan hal tersebut.
                A_inv = np.linalg.inv(A): Ini adalah perintah untuk mencari "kebalikan" (invers) dari matriks A.
                X1 = np.dot(A_inv, Y): Ini adalah perintah untuk mengalikan matriks invers A dengan matriks Y. Hasilnya adalah solusi yang kita cari,
                yang disimpan dalam variabel X1.'''


########### KOLABORASI ANTARA NUMPY DENGAN MATPLOTLIB ##########
import matplotlib.pyplot as plt
print(f'\n\n========= ini adalah kolaborasi antara numpy dengan matplotlib ========'.center(55))

# membuat garis lurus dan lingkaran
daftar = np.arange(10) # ini sama seperti range python
hitam = daftar*2 + 3
print(daftar)
print(hitam)
# garis lurus
plt.plot(daftar,hitam) # ini adalah function dari python.. biasanya untuk dataset atau menampilkan gambar
'''plt.show() # ini adalah untuk menampilkan gambar.. biasanya ada di akhir dari program.'''

# lingkaran
jari = 5

sudut = np.linspace(0,np.pi*2,100)
i = jari * np.cos(sudut)
j = jari * np.sin(sudut)

plt.figure(2)
plt.plot(i,j)
plt.show() # ini adalah untuk menampilkan gambar.. biasanya ada di akhir dari program.
# akan ada 2 hasil.. pertama adalah garis lurus diagonal keatas kanan. kedua adalah lingkaran