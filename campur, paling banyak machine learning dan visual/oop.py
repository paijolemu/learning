##### OOP AWAL #####


'''class Hero:
    def __init__(self, name, health, power, armor):
        self.name = name
        self.health = health
        self.power = power
        self.armor = armor

hero_pertama = Hero('paijo',1200,50,11)
hero_kedua = Hero('Dio',400,21,2)
hero_ketiga = Hero('galen',450,24,3)

print(hero_pertama) # hasilnya akan adress.

print(hero_pertama.name) # jika seperti ini akan menampilkan nama dari hero.
print(hero_pertama.health) # akan menampikan health dari hero paijo
print(hero_pertama.power) # akan menampilkan jumlah power / 50
print(hero_pertama.armor) # akan menampilkan jumlah armor dari hero paijo.

print(hero_pertama.__dict__) # ini akan mencetak semua stats tapi dalam string beripe dictionary { }
    # hasil: {'name': 'paijo', 'health': 1200, 'power': 50, 'armor': 11}

print(hero_ketiga.__dict__)'''

### INSTANCE VARIABLE ### lanjutan dari yg atas.

'''class Hero:

    jumlah = 0

    def __init__(self, name, health, power, armor):
        self.name = name
        self.health = health
        self.power = power
        self.armor = armor
        Hero.jumlah += 1
        print(f'hero yg pertama kali di buat adalah: {name} {health} {power} {armor}')

hero_pertama = Hero('paijo',1200,50,11)
print(Hero.jumlah)
hero_kedua = Hero('Dio',400,21,2)
print(Hero.jumlah)
hero_ketiga = Hero('galen',450,24,3)
print(Hero.jumlah)


print(hero_pertama) # hasilnya akan adress.

print(hero_pertama.name) # jika seperti ini akan menampilkan nama dari hero.
print(hero_pertama.health) # akan menampikan health dari hero paijo
print(hero_pertama.power) # akan menampilkan jumlah power / 50
print(hero_pertama.armor) # akan menampilkan jumlah armor dari hero paijo.

print(hero_pertama.__dict__) # ini akan mencetak semua stats tapi dalam string beripe dictionary { }
    # hasil: {'name': 'paijo', 'health': 1200, 'power': 50, 'armor': 11}

print(hero_ketiga.__dict__)'''


####### METHODS ######


'''class Hero:

    jumlah = 0

    def __init__(self, name, health, power, armor):
        self.name = name
        self.health = health
        self.power = power
        self.armor = armor
        Hero.jumlah += 1

    # void function
    def nama(self):
        print(f'hero yg dipakai adalah: ' , {self.name})

    # menambah darah
    def health_up(self,up):
        self.health += up

    # mengambil darah
    def gethealth(self):
        return self.health
    
# ini hero
hero1 = Hero('paijo lemu',1200,12,11)
hero2 = Hero('wawan',1400,1,1)
hero3 = Hero('gallen',600,1,2)

# ini akan sedikit rumit....
hero1.nama() # artinya ini hanya untuk memanggil saja dan langsung terprint karna memang di def sudah ada perintah untuk print
hero1.health_up(300) # artinya ini tanpa return dan tanpa perintah print... ini dipanggil adalah untuk menambahkan health. bukan untuk mencetak atau yg lain
hero1.gethealth() # artinya ini adalah menunjukan jumlah health yg telah di tambah itu berapa... dan ini bertugas untuk mencetak health...

print(hero1.gethealth()) # ini adalah untuk mengambil nilai healt yg telah di tambah/health_up '''

######## pelatihan serangan oop #######

class Hero:

    def __init__(self, nama, health, power, armor):
        self.nama = nama
        self.health = health
        self.power = power
        self.armor = armor

    def serang(self, lawan):
        print(self.nama + ' menyerang ' + lawan.nama)
        lawan.diserang(self, self.power)

    def diserang(self, lawan, power):
        print(self.nama + ' diserang ' + lawan.nama)
        serangan_diterima = power- self.armor
        print('serangan di terima: ' + str(serangan_diterima))
        self.health -= serangan_diterima
        print('darah ' + self.nama + ' tersisa ' + str(self.health))


paijo = Hero('paijo', 1200, 12, 40)
gallen = Hero('gallen', 121, 100, 3)

paijo.serang(gallen)
print('\n')
gallen.serang(paijo)
print('\n')
paijo.serang(gallen)
print('\n')
gallen.serang(paijo)
print('\n')
paijo.serang(gallen)
print('\n')
gallen.serang(paijo)






















