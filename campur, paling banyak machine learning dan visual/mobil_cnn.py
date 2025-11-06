# mobile net
# ini adalah alat yg jauh lebih ringan jika di bandingkan dengan vgg16
# hebatnya dari model ini adalah bisa memprediksi hewan tanpa kita melatihnya kembali. memakai transfer learning
# transfer leaning di sebut sebagai fine-tunning
# %%
import numpy as np
import itertools
import random
import matplotlib.pyplot as plt
import os
import shutil
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.applications import imagenet_utils, mobilenet_v3
from sklearn.metrics import confusion_matrix
from IPython.display import Image
%matplotlib inline

# %%
mobile = tf.keras.applications.mobilenet.MobileNet()
# %%
# 1. Dapatkan path direktori kerja saat ini (folder 'pelatihan')
samples_dir = r"C:\Users\62812\Documents\pelatihan\MobileNet-samples"


def prepare_image(file_name):
    """
    Fungsi untuk memuat, mengubah ukuran, dan memproses gambar
    agar siap dimasukkan ke model MobileNet.
    """
    # Buat path absolut ke file gambar spesifik
    img_path = os.path.join(samples_dir, file_name)
    
    # Cek apakah file ada sebelum diproses
    if not os.path.exists(img_path):
        print(f"ERROR: File tidak ditemukan di '{img_path}'. Pastikan nama file dan path folder sudah benar.")
        return None

    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array_expanded_dims = np.expand_dims(img_array, axis=0)
    return tf.keras.applications.mobilenet.preprocess_input(img_array_expanded_dims)
# %%
image_to_display = '1.jpg'
Image(filename=os.path.join(samples_dir, image_to_display), width=300, height=200)
# %%
preprocessed_image = prepare_image('1.jpg')
predictions = mobile.predict(preprocessed_image)
result = imagenet_utils.decode_predictions(predictions)
result
# %%
assert result[0][0][1] == 'crane' # di sini model memprediksi crane/ bangau dan benar bahwa ini adalah bangau
# %%
preprocessed_image = prepare_image('2.jpg')
predictions = mobile.predict(preprocessed_image)
result = imagenet_utils.decode_predictions(predictions)
result
# %%
assert result[0][0][1] == 'crane'
# %%
preprocessed_image = prepare_image('3.jpg')
predictions = mobile.predict(preprocessed_image)
result = imagenet_utils.decode_predictions(predictions)
result
# %%
assert result[0][0][1] == 'crane'
# %%
preprocessed_image = prepare_image('6.jpg') # ini elang / bald eagle
predictions = mobile.predict(preprocessed_image)
result = imagenet_utils.decode_predictions(predictions)
result
# %%
assert result[0][0][1] == 'bald_eagle'
# %%
preprocessed_image = prepare_image('420.jpg') # ini salah, malah memprediksi tulang
predictions = mobile.predict(preprocessed_image)
result = imagenet_utils.decode_predictions(predictions)
result
# %%
preprocessed_image = prepare_image('325.jpg') # ini salah, memprediksi kucing saja salah.
predictions = mobile.predict(preprocessed_image)
result = imagenet_utils.decode_predictions(predictions)
result



# %%
# modify / moditikasi model
mobile = tf.keras.applications.mobilenet.MobileNet()
# %%
mobile.summary()
# %%
params = count_params(mobile)
assert params['trainable_params'] == 4231976
assert params['non_trainable_params'] == 21888
# %%
x = mobile.layers[-6].output
output = Dense(units = 10, activation = 'softmax')(x)
# %%
model = Model(inputs=mobile.input, outputs = output)
# %%
for layer in model.layers[:-23]:
    layer.trainable = False
# %%
model.summary()
# %%
params = count_params(model)
assert params['trainable_params'] == 2136074
assert params['non_trainable_params'] == 1103040
########### SELESAI!!!!! ##############






# %%
# COBA-COBA.
model = tf.keras.applications.MobileNetV3Large()

# Opsi B: Versi kecil, lebih cepat dan ringan
# model = mobilenet_v3.MobileNetV3Small()
# -----------------------------

# Tampilkan ringkasan untuk melihat arsitekturnya
model.summary()
# %%
# --- MEMPERSIAPKAN GAMBAR (PROSESNYA SAMA) ---
def prepare_image(file_path):
    img = image.load_img(file_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array_expanded_dims = np.expand_dims(img_array, axis=0)
    # PENTING! Gunakan preprocess_input dari mobilenet_v3
    return mobilenet_v3.preprocess_input(img_array_expanded_dims)
# %%
image_to_display = '325.jpg'
Image(filename=os.path.join(samples_dir, image_to_display), width=300, height=200)

# %%
samples_dir = r"C:\Users\62812\Documents\pelatihan\MobileNet-samples"
nama_file = '325.jpg'

# Buat alamat lengkap (path absolut) ke file tersebut
path_lengkap = os.path.join(samples_dir, nama_file)

# Sekarang, berikan alamat LENGKAP ini ke fungsi prepare_image
preprocessed_image = prepare_image(path_lengkap)

# Jalankan prediksi (hanya jika gambar berhasil ditemukan dan diproses)
if preprocessed_image is not None:
    predictions = model.predict(preprocessed_image)
    result = imagenet_utils.decode_predictions(predictions)
    
    # Cetak hasil prediksi
    print("Hasil Prediksi:")
    print(result)
# %%
assert result[0][0][1] == 'tiger_cat' 
if result[0][0][1] == 'tiger_cat':
    print('ini benar!')