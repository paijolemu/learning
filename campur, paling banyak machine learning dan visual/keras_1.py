# %%
# Data Preprocessing
import numpy as np
from random import randint
from sklearn.utils import shuffle # ini untuk mengocok/acak data
from sklearn.preprocessing import MinMaxScaler # scala data 0 - 1

# %%
# membuat data train
train_label = [] # untuk mengisi nilai yg di loop / wadahnya. hasilnya [0,1] | target
train_samples = [] # ini akan berisi features

# %%
for i in range(50):
    random_younger = randint(13, 64) # muda ~5% yg mengalami efek samping
    train_samples.append(random_younger)
    train_label.append(1)

    random_older = randint(65, 100) # tua ~95% yg mengalami efek samping
    train_samples.append(random_older)
    train_label.append(0)

for i in range(1000):
    random_younger = randint(13, 64) # muda ~95% yg tidak mengalami efek samping
    train_samples.append(random_younger)
    train_label.append(0)

    random_older = randint(65, 100) # tua ~95% yg tidak mengalami efek samping
    train_samples.append(random_older)
    train_label.append(1)
# %%
for i in train_samples: # ini untuk mengecek sementara
    print(i)
# %%
for i in train_label:
    print(i)
# %%
train_label = np.array(train_label) # membuat data menjadi numpy
train_samples = np.array(train_samples)
train_label, train_samples = shuffle(train_label, train_samples) # mengocok data
# %%
scaler = MinMaxScaler(feature_range = (0, 1)) # memanggil scaler
scaler_train_samples = scaler.fit_transform(train_samples.reshape(-1, 1)) # ini mengubah banyak hal,
# 1. reshape(-1, 1) -> mengubah data menjadi 2 dimensi
# 2. fit -> mempelajari data yg akan di scalar
# 3. transform -> mengubah data yg awalnya numberic menjadi antara 0 - 1
# 4. fit_transform -> mengaplikasikan scaler ke data
# %%
# cek
for i in scaler_train_samples: # ini adalah hasil dari features yg sudah kita scalar
    print(i)


# %%
# membuat simple seqeuential model
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential # ini adalah model yg kita pakai
from tensorflow.keras.layers import Activation, Dense # ini adalah layer/lapisan yg kita pakai
from tensorflow.keras.optimizers import Adam # ini adalah optimizer yg kita pakai
from tensorflow.keras.metrics import categorical_crossentropy # ini adalah metrik yg kita pakai

# %%
# # cek apakah GPU terdeteksi, kalau tidak ada akan error
# physical_devices = tf.config.experimental.list_physical_devices('GPU')
# print("Num GPUs Available: ", len(physical_devices))
# tf.config.experimental.set_memory_growth(physical_devices[0], True)
# %%
model = Sequential([ # Sequential artinya lapisan disusun secara berurutan
    Dense(units = 16, input_shape = (1,), activation = 'relu'), # ini lapisan pertama, input shape artinya data masukannya 1 dimensi. ini tersembunyi
    Dense(units = 32, activation = 'relu'), # memiliki 32 neuron, lapisan tersembunyi kedua
    Dense(units = 2, activation = 'softmax') # ini lapisan untuk output/hasil prediksi hasilnya
])  # akan 0 untuk tidak ada efek samping, 1 untuk ada efek samping
    # ReLU adalah singkatan dari Rectified Linear Unit. ini sering di pakai. 
# %%
model.summary() # untuk melihat ringkasan model, ini cukup penting karna memang kalo kita load. melihat summary model
    # bahkan ada parameter weights juga
# %%
# compile model / mengsiapkan model sebelum bertempur dengan data. agar hasil maksimal
model.compile(optimizer = Adam(learning_rate = 0.0001), loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
# adam adalah tools optimasi, sparse_categorical_crossentropy adalah mengukur seberapa salah model, dan juga tools yg di gunakan
#  untuk mengukur/menghukum model agar semakin baik lalu metrics nya adalah akurasi
# %%
model.fit(x = scaler_train_samples, y = train_label, validation_split = 0.1 , batch_size = 10, epochs = 30, shuffle = True, verbose = 1)
# training model dengan data yg sudah kita siapkan dengan x dan y lalu validasinya 10%, batch size 10 artinya mempelajari langsung 10 data.
# soalnya jika data 1 atau semua model akan belajar sedikit sedikit dan tidak perlu dan sebaliknya jika semua maka akan banyak sekali sekaligus.
# epoch berapa kali melatih datanya, shuffle mengocok data , verbose adalah banyaknya pemeritahuan yg di tampilkan semakin besar angkanya semakin banyak 
 

# %%
# membuat data test
test_samples = []
test_label = []

for i in range(10):
    random_younger = randint(13, 64) # muda ~5% yg mengalami efek samping
    test_samples.append(random_younger)
    test_label.append(1)

    random_older = randint(65, 100) # tua ~95% yg mengalami efek samping
    test_samples.append(random_older)
    test_label.append(0)

for i in range(200):
    random_younger = randint(13, 64) # muda ~95% yg tidak mengalami efek samping
    test_samples.append(random_younger)
    test_label.append(0)

    random_older = randint(65, 100) # tua ~95% yg tidak mengalami efek samping
    test_samples.append(random_older)
    test_label.append(1)
# %%
test_samples = np.array(test_samples) # mengubah menjadi numpy
test_label = np.array(test_label)
test_label, test_samples = shuffle(test_label, test_samples)
# %%
scaled_test_samples = scaler.transform(test_samples.reshape(-1, 1)) # menscalar data test
# %%
# prediksi / predict
predictions = model.predict(x = scaled_test_samples, batch_size = 10, verbose = 1) # memprediksi pakai model
for i in predictions: # ini untuk melihat hasil prediksi (real)
    print(i)
# %%
rounded_predictions = np.argmax(predictions, axis = -1) # ini untuk mengambil nilai terbesar dari hasil prediksi
for i in rounded_predictions: # atau untuk menetapkan prediksi mana yg di pilih
    print(i)




# %%
# confusion matrix
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt
%matplotlib inline
# %%
cm = confusion_matrix(y_true=test_label, y_pred = rounded_predictions)
# %%
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
# %%
cm_plot_labels = ['No Side Effects', 'Had Side Effects']
plot_confusion_matrix(cm = cm, classes = cm_plot_labels, title = 'Confusion matrix', cmap=plt.cm.Greens)

# %%
# 1. Model Save
import os.path # panggil modul path dari sistem operasi
if os.path.isfile('model.h5') is False: # .h5 / .keras | 'coba cari nama model.h5' di sistem apakah ada? kalo tidak simpan
    model.save('model.h5') # menyimpan model
# %%
from tensorflow.keras.models import load_model
new_model = load_model ('model.h5') # load model
# %%
new_model.summary() # melihat ringkasan model yg di load
# %%
new_model.get_weights() # melihat weights model yg di load
# %%
new_model.optimizer # melihat optimizer model yg di load
# optimiernya adalah adam kita bisa lihat


# %%
# model.to_json()
# save as JSON
json_file = model.to_json()
# %%
json_file
# %%
from tensorflow.keras.models import model_from_json
model_arsitektur = model_from_json(json_file)
# %%
model_arsitektur.summary()


# %%
# model.save_weights()
import os.path
if os.path.isfile('model.h5') is False:
    model.save_weights('model.h5') # menyimpan weights model
# %%
model2 = Sequential([ # model baru. kebetulan sama
    Dense(units = 16, input_shape = (1,), activation = 'relu'),
    Dense(units = 32, activation = 'relu'),
    Dense(units = 2, activation = 'softmax')
])
# %%
model2.load_weights('model.h5') # load memory/pikiran baru
# %%
model2.get_weights() # melihat weights model yg di load
# PENTING!!!! SAYA GA BISA PAKE CODE INI. PUSYINGG
# PENTING!!!! sekarang sudah bisa, ternyata file harus sama dengan model yg dulu

# %%
