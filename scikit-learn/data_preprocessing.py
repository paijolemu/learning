# %%
import numpy as np
from sklearn import preprocessing

sample_data = np.array([[2.1, -1.9, 5.5],
                        [-1.5, 2.5, 3.5],
                        [0.5, -7.9, 5.6],
                        [5.9, 2.3, -5.8]])
sample_data
# %%
sample_data.shape # (4, 3) artinya 4 baris dan 3 kolom
# %%
sample_data
# %%
######## binarization #########
preprocessor = preprocessing.Binarizer(threshold= 0.5) # artinya membuat matrix biner([0.0   1.0])<-- seperti ini.
binarized_data = preprocessor.transform(sample_data) # jika data lebih dari 0.5 maka akan menjadi 1.0, jika kurang dari 0.5 maka akan menjadi 0.0
binarized_data
# %%
############# scaling / penskalaan ###########
preprocessor = preprocessing.MinMaxScaler(feature_range=(0, 1)) # artinya membuat skala data dari 0 sampai 1
preprocessor.fit(sample_data) # fit itu untuk menyesuaikan data
scaled_data = preprocessor.transform(sample_data) # transform itu untuk mengubah data sesuai dengan fit yang sudah di sesuaikan
scaled_data # ini cara yg lain kecuali biner
# %%
'''########## normalisation: L1 least absolute deviations ##########'''
l1 = preprocessing.normalize(sample_data, norm='l1') # artinya membuat normalisasi data dengan metode L1
l1 # untuk selengkapnya cari di dokumentasi sklearn
# %%
''' normalisation: L2 least square'''
l2 = preprocessing.normalize(sample_data, norm= 'l2') # artinya membuat normalisasi data dengan metode L2
l2