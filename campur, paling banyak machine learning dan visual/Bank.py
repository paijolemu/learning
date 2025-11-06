# buatlah sebuah class dengan nama RekeningBank.
# kemudian class ini memiliki beberapa medthod seperti mencetak saldo, menabung, dan menarik uang.
# jika rekening kurang, maka akan menunjukan keterangan bahwa saldo tidak bisa ditarik karna tidak mencukupi.


class RekeningBank:


    def __init__(self, tabungan):
        self.tabungan = tabungan

    def cek_saldo(self):
        print('jumlah saldo anda saat ini adalah {}'.format(self.tabungan))

    def menabung(self):
        tambah = int(input('masukan jumlah yg ingin anda tabung: '))
        self.tabungan += tambah

    def menarik(self):
        tarik = int(input('masukan jumlah yg ingin anda tarik: '))
        self.tabungan -= tarik
        if self.tabungan < 0 :
            print('maaf saldo anda tidak mencukupi ')
        else:
            print('sisa saldo anda setelah penarikan adalah: {}'.format(self.tabungan))
        

p = RekeningBank(10000)
p.cek_saldo()
p.menabung()
p.cek_saldo()


p.menarik()

