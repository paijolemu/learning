

# %%
import matplotlib.pyplot as plt

x = [1, 2, 3, 4, 5]
y1 = [1, 4, 9, 16, 25]
y2 = [1, 2, 3, 4, 5]

plt.plot(x, y1, label='x^2', color='blue', marker='o')
plt.plot(x, y2, label='x', color='red', linestyle='--')

plt.title('Perbandingan Pertumbuhan')
plt.xlabel('Nilai X')
plt.ylabel('Nilai Y')
plt.legend() # Menampilkan label
plt.grid(True, linestyle=':', alpha=0.6)
plt.show()


# %%
import matplotlib.pyplot as plt
import numpy as np

# Data acak terdistribusi normal
data = np.random.normal(loc=170, scale=10, size=1000)

plt.hist(data, bins=30, color='skyblue', edgecolor='black')
plt.title('Distribusi Tinggi Badan')
plt.xlabel('Tinggi (cm)')
plt.ylabel('Frekuensi')
plt.show()
# %%
import matplotlib.pyplot as plt
import numpy as np

# Data nilai dari 3 kelas berbeda
kelas_a = np.random.normal(80, 10, 50)
kelas_b = np.random.normal(75, 15, 50)
kelas_c = np.random.normal(85, 5, 50)
data_nilai = [kelas_a, kelas_b, kelas_c]

plt.boxplot(data_nilai, labels=['Kelas A', 'Kelas B', 'Kelas C'])
plt.title('Sebaran Nilai Ujian Antar Kelas')
plt.ylabel('Nilai')
plt.show()
# %%
import matplotlib.pyplot as plt

labels = ['Chrome', 'Safari', 'Edge', 'Firefox']
sizes = [64.8, 19.1, 5.4, 3.2]
colors = ['#ff9999','#66b3ff','#99ff99','#ffcc99']
explode = (0.05, 0, 0, 0) # menonjolkan slice pertama

plt.pie(sizes, explode=explode, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=90)
plt.title('Pangsa Pasar Browser Desktop')
plt.axis('equal') # Pastikan pie chart bulat
plt.show()
# %%
import matplotlib.pyplot as plt
import numpy as np

# Matriks data acak 5x5
data = np.random.rand(5, 5) 
labels = ['A', 'B', 'C', 'D', 'E']

fig, ax = plt.subplots()
im = ax.imshow(data, cmap='viridis')

# Menambahkan colorbar
cbar = ax.figure.colorbar(im, ax=ax)

# Mengatur label
ax.set_xticks(np.arange(len(labels)), labels=labels)
ax.set_yticks(np.arange(len(labels)), labels=labels)
plt.title('Heatmap Korelasi Antar Variabel')
plt.show()
# %%
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 2 * np.pi, 400)
y_sin = np.sin(x ** 2)
y_cos = np.cos(x ** 2)

# Membuat figure dengan 2 baris, 1 kolom
fig, axs = plt.subplots(2)
fig.suptitle('Plot Sinus dan Cosinus')

axs[0].plot(x, y_sin, color='navy')
axs[0].set_title('sin(x^2)')

axs[1].plot(x, y_cos, color='orange')
axs[1].set_title('cos(x^2)')
plt.tight_layout() # Merapikan layout
plt.show()
# %%
