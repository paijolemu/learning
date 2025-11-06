import numpy as np
import matplotlib.pyplot as plt

############## pendahuluan matplotlib ##################
### peraturan:      1. membuat data dulu
#                   2. membuat plot
#                   3. menampilkan show()

'''# membuat data
jarak = np.arange(10)
pangkat_2 = jarak**2
pangkat_3 = pangkat_2 * 2
print(jarak)
print(pangkat_2)
print(pangkat_3)

# membuat plot
plt.plot(jarak,pangkat_2, 'r',pangkat_3, 'y') # artinya menggambar garis dengan layer 1 berwarna 'r'/red layer2 berwarna 'y'/yellow

# menampilkan / show()
plt.show()'''


'''###### set warna dan marker pada sinus generator #####
# membuat data (sin(2wt + theta))
# camel case
def sinusGenerator(amplitudo, frekuensi, tAkhir, theta):
    t = np.arange(0, tAkhir, 0.1) # Pabrik ini butuh sumbu waktu (sumbu X). Saya buat rentang waktu dari 0 sampai tAkhir dengan langkah yang sangat 
                                  # kecil (0.1) agar gelombangnya terlihat mulus.
    y = amplitudo * np.sin((frekuensi* 2 * t) + np.deg2rad(theta)) # y = amplitudo * np.sin(...): "Inilah inti dari mesinnya, rumus untuk membuat gelombang (sumbu Y)."
                                                                # np.sin(...): Ini fungsi sinus dari NumPy.
                                                                # 2*frekuensi*t: Ini mengatur kerapatan gelombang.
                                                                # + np.deg2rad(theta): Ini bagian penting. Komputer biasanya menghitung sinus dalam satuan radian, bukan derajat. 
                                                                # Jadi, "bahan" theta yang kita masukkan dalam derajat harus diubah dulu ke radian. np.deg2rad() melakukan ini.
                                                                # amplitudo * ...: Hasil dari sinus (yang nilainya antara -1 dan 1) dikalikan dengan amplitudo untuk mengatur tingginya.
    return t,y

# membuat plot
t1,y1 = sinusGenerator(1,1,4,0)
plt.plot(t1,y1, 'k')

t2,y2 = sinusGenerator(1,1,4,30)
plt.plot(t2, y2, 'r--')

t3,y3 = sinusGenerator(1,1,4,60)
plt.plot(t3,y3, 'y-o')


# menampilkan show
plt.show()
'''

'''######### mmebuat lebih rapih dengan set properti #######

def sinusGenerator(amplitudo, frekuensi, tAkhir, theta):
    t = np.arange(0, tAkhir, 0.1)
    y = amplitudo * np.sin(frekuensi*2*t + np.deg2rad(theta))
    return t,y

# membuat plot data
t1,y1 = sinusGenerator(1,1,4,0)
t2,y2 = sinusGenerator(1,1,4,90)
t3,y3 = sinusGenerator(1,1,4,180)

# ini plot sebenarnya
alur1 = plt.plot(t1,y1)
alur2 = plt.plot(t2,y2)
alur3 = plt.plot(t3,y3)

# setting properti
plt.setp(alur1, color = 'r', linestyle = '-', linewidth = '0.5')
plt.setp(alur2, color = 'k', linestyle = '-.', linewidth = '2')
plt.setp(alur3, color = 'b', linestyle = '--', linewidth = '5')

## axis / sumbu x dan y
# plt.axis([xmin, xmax, ymin, ymax]) <-- ini rumusnya. pakai list ya dek ya...
plt.axis([0, 4, -1, 1]) # hasilnya akan mepet... intinya axis ini bisa mengatur lebar dari tampilan sumbu.

# menampilkan show()
plt.show()
'''



'''
############### MEMBUAT SET LABEL/SUMBU X,Y | TITLE ##################

# membuat data
def sinusGenerator(amplitudo, frekuensi, tAkhir, theta):
    t = np.arange(0, tAkhir, 0.1)
    y = amplitudo * np.sin(frekuensi*2*t + np.deg2rad(theta))
    return t,y



amplitudo = 1
frekuensi = 1
tAkhir = 4
theta = 0

t1,y1 = sinusGenerator(amplitudo, frekuensi, tAkhir, theta)
t2,y2 = sinusGenerator(amplitudo, frekuensi, tAkhir, 90)
t3,y3 = sinusGenerator(amplitudo, frekuensi, tAkhir, 180)

# data plot
judul = 'GRAFIK SINUSOIDAL\n'
rumus = r'$ \mathcal{Y} = A.sin(2  \omega t + \theta) $'+ '\n'
parameter1 = r'A =' + str(amplitudo)+  'cm, '
parameter2 = r'$ \omega = $' + str(frekuensi) + r'$ \mathit{Hz} $' + ', '
parameter3 = r'$ \theta = $' + str(theta) + r'$ ^{0} $'

title_lengkap = f'{judul} {rumus} {parameter1} {parameter2} {parameter3}'

# plot sumbu x,y dan title
plt.plot(t1,y1, label = 'sin(0)')
plt.plot(t2,y2, label = 'sin(90)')
plt.plot(t3,y3, label = 'sin(180)')
plt.title(title_lengkap)

plt.xlabel('X = waktu/detik')
plt.ylabel('Y = magnituda(cm)')

######## MEMBUAT LEGEND ##########
# plt.legend()
# plt.legend(loc = 'upper center') --> membuat agar legend nya berada di atas tengah
# plt.legend(loc = 'upper center', bbox_to_anchor = (0.5, -0.5))
    # akan sedikit sulit ya...
box = plt.subplot(111).get_position()
plt.subplot(111).set_position([box.x0, box.y0, box.width*0.7, box.height])
plt.legend(loc = 'upper center', bbox_to_anchor = (1.2, 1))


######### MEMBUAT TEKS ########
plt.text(1, -0.25, 'inget ya jumlah X cuma sampe 4 dan jumlah Y cuma sampe 1')

# menampilkan show()
plt.show()
'''

####### set Ticks ######
# projek membuat sudut
# data
sudut = np.arange(360)
y = np.sin(np.deg2rad(sudut))

# plot

plt.plot(sudut, y)
plt.title('grafik sinusoidal')

# set xticks dan yticks
plt.xticks([0, 90, 180, 270, 360], [r'$0^o$', r'$90^o$', r'$180^o$', r'$270^o$', r'$360^o$'])
plt.yticks([-1, -0.5, 0, 0.5, 1])

# set posisi dan warna spines
ax = plt.gca()
ax.spines['left'].set_position(('data', 180))
ax.spines['bottom'].set_position(('data', 0))
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')

# set teks
plt.text(360, 0.1, 'sudut')
plt.text(190, 1, 'magnituda')

# tampilkan
plt.show()





