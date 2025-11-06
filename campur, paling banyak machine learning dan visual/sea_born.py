# alamat website seaborn adalah: seaborn.pydata org

# %%
from warnings import filterwarnings
import scipy.stats as stat
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
filterwarnings('ignore') # ini adalah untuk menghilangkan warnings.
# %%
file_csv = 'hypertension_dataset.csv'

df_file = pd.read_csv(file_csv)
df_file.head()
# %%
# membuat distribution plot
# sns.distplot(df_file) --> ini akan gagal karna distplot hanya menerima numberik saja.
sns.distplot(df_file.Age); # perintah ini menunjukan berapa umur yg ada di data. secara komperhensif
        # garis warna biru namanya KDE (kernel Density Estimation)/kira kira kepadatan
# %%
sns.distplot(df_file.Age, kde = False, bins= 30) # membuat kde tidak ada. dan menambah bins/batang
sns.distplot(df_file.Age, kde = True, bins= 50) 
# %%
sns.distplot(df_file.Age, kde = True, bins= 50) # setelah trial and error. kde bisa muncul kalo dia sendirian.

# %%
df_file.head(221)

# %%
sns.jointplot(x='Age', y='Stress_Score', data = df_file)
# %%
# ini adalah fitur yg akan sering di pakai untuk membandingkan.| harus di inget ini overpower.
sns.jointplot(x = 'Age', y = 'Stress_Score', data = df_file, marginal_kws=dict(kde=True)); # ini perintah dari sns/seaborn joinplot
        # dengan sumbu x = Age dan sumbu y = stress_score, dan... datanya adalah df_file(diatas)| ;(agar rapih)
# %%
import scipy.stats as stat
# %%
sns.kdeplot(x='Age', y='Stress_Score', data = df_file) # ini membuat konture gunung.
# %%
sns.jointplot(data=df_file, x='Age', y='Stress_Score', kind='reg'); # kind='reg' adalah singkatan dari 'regression' (regresi).
            # Menggambar garis regresi linear (juga dikenal sebagai best-fit line atau garis tren) di atas titik-titik tersebut.
# %%







######## out layer ######### | yg kira kira mengganggu data akan di singkirkan.
# versi manual
df_file.Stress_Score[df_file.Stress_Score > 9] # perintah untuk mencari orang yg stress scorenya lebih dari 9. di file ini ada 159.
# %%
df_new = df_file[df_file.Stress_Score <= 9] # membuat file baru yg score stres kurang dari/sama dengan 9 kebawah.

# %%
sns.jointplot(data = df_new, x='Age', y = 'Stress_Score', kind='reg')
######################################################################################### versi manual ######################
# %%
sns.jointplot(data = df_new, x='Age', y = 'Stress_Score', kind='hex')
sns.jointplot(data = df_new, x='Age', y = 'Stress_Score', kind='kde')


# %%
sns.pairplot(data = df_new) # ini adalah mahakuasa. perintah ini akan menampilkan semua dataset yg kita miliki dengan syarat numerik.
# %%
df_new.info()
# %%
sns.pairplot(data = df_new, hue = 'Age') # memberi warna pada Age
# %%
