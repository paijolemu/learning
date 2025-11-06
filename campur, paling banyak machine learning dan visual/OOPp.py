
'''class kulkas:

    penjual = 'paijo'

    def __init__(self, merk, harga):
        self.merk = merk
        self.harga = harga
        self.pembeli = 'wawan'
        self.tenaga = 'listrik'
    
    def masuk(self, unit= 1):
        print('saya {} membeli kulkas di {} bermerk {} dengan harga {} ini pakai {} ya. saya membeli {} unit'.format(self.pembeli, self.penjual, self.merk, self.harga, self.tenaga, unit))

barang = kulkas(merk= 'ODDO', harga= 1200)
barang.masuk(unit= 2)'''

################

class lingkaran:
    phi = 3.14 # ini adalah rumus phi 

    def __init__(self, radius): # dalam matematika radius adalah nama satuan di lingkaran.
        self.radius = radius

    def keliling(self):
        return 2*self.phi*self.radius # ini adalah rumus lingkaran
    
    def luas(self):
        return print(2*self.phi*self.radius*self.radius)
    
ling1 = lingkaran(10)
print(ling1.keliling()) # inget harus di ikuti oleh def akhir ya...
    # saya sempet stress dengan kode ini. karna tidak mau print. berbeda dengan atas yg bisa ngeprint sendiri. ternyata di atas ada printah print di dalam def.
        # jadi jika di panggil dia bisa print otomatis.

ling1.luas()




########### inheritance / warisan #########

# ini data induk
class Mahasiswa: # biasakan agar huruf depan kapital. jika ada 2 kata. maka jangan pake _. akan tetapi gabung saja keduanya dengan setiap kata huruf pertama kapital. contoh NamaDepan


    def __init__(self, nama, kelas):
        self.nama = nama
        self.kelas = kelas

    def keterangan(self):
        return print('{} adalah seorang mahasiswa kelas {}'.format(self.nama, self.kelas)) 
        # jika ingin melihat kode apa yg bisa di masukan adalah ctrl+shift+bar space
        

SmaNegeri = Mahasiswa('paijo',12 )
SmaNegeri.keterangan()

# ini data cabang
class Nilai(Mahasiswa):


    def __init__(self, nama, kelas):
        super().__init__(nama, kelas) # untuk baris ini tidak perlu self. karna sudah ada fungsi super() | super() guna untuk melihat def yg sama di atas. dia akan mencari. | biasanya berguna untuk saat data banyak.
        self.nilai_update = []

    def input_nilai(self, tambah):
        return self.nilai_update.append(tambah)
    

paijo = Nilai('paijo', 21)
paijo.keterangan

paijo.input_nilai(20)
print(paijo.nilai_update) # ini mengupdate nilainya

paijo.input_nilai(20)
print(paijo.nilai_update) # hasinya akan di tambahkan di belakang listnya. | append


#########################
# ini namanya adalah
class Kucing: # oop 1 
    

    def __init__(self, name):
        self.name = name

    def respons(self):
        return self.name + ' meow-meow'
    

class Anjing: # oop 2


    def __init__(self, name): # saya sampe depresi salahnya dimana. __init__ harus huruf kecil semua. kalo tidak akan error
        self.name = name

    def respons(self):
        return self.name + ' guk-guk'
    

cat = Kucing('cat')
print(cat.respons()) # inget ini !!!!!! penting banget!!!! jangan sampai lupa!!!! | cat.repost --> variable baru dengan def, ini benar ya!!!!
                     # saya salah terus karna variable baru dengan class --> cat.Kucing --> ini salah besar!!!!!
dog = Anjing('dog')
print(dog.respons())




for i in [cat,dog]:
    print(type(i)) # classnya adalah __main__.kucing dan __main__.anjing
    print(i.respons())

def X(coba): # mencoba dari variable aneh/ngasal
    print(coba.respons()) # argument dengan def ya... bukan malah X.respons --> INI SALAH!!! 
                        # depresi juga disini... salah karna kurang (). yg bener adalah coba.respons( ) --> jika tidak seperti ini hanya menampilkan memory saja.

X(dog)

##################################### DUNDER / MAGIC METHOD

class Coba:


    def __init__(self, nama):
        self.nama = nama

    def Cetak(self):
        return self.nama
    
tes = Coba('testing')
print(tes.Cetak())

#
print(dir(Coba)) # ini akan mencetak perintah apa saja. yg boleh/bisa kita lakukan
    # ada __str__ , ada __len__ dan masih banyak sekali. sesuai kegunaan.







