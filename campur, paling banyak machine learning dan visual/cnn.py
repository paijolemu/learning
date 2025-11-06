# CONVOLUTIONAL NEURAL NETWORK ( CNN )

# %%
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Flatten, BatchNormalization, Conv2D, MaxPool2D # ini adalah layer/jaringan saraf otak
# COnv2D = inti core untuk gambar/cnn, maxpool2d = mengurangi dimensi gambar, flatten agar data bisa di baca. batchnormalize = stabilizer 
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator # ini membuat dan augmentasi data gambar
from sklearn.metrics import confusion_matrix
import itertools
import os # operasi sistem
import shutil # manipulasi file tingkat tinggi, salin, pindah, hapus file
import random
import glob # menemukan file dan direktori berdasarkan pola pencocokan, filter. misal *.jpg saja.
import matplotlib.pyplot as plt
import seaborn as sns
from warnings import filterwarnings
filterwarnings('ignore')
sns.set()
%matplotlib inline

# %%
base_dir = r'C:\Users\62812\Documents\pelatihan'
source_data_dir = os.path.join(base_dir, 'dogs-vs-cats')
organized_data_dir = os.path.join(base_dir, 'dogs-vs-cats-organized')

if not os.path.exists(organized_data_dir):
    print(f"Membuat folder baru di: {organized_data_dir}")
    
    train_dir = os.path.join(organized_data_dir, 'train')
    valid_dir = os.path.join(organized_data_dir, 'valid')
    test_dir = os.path.join(organized_data_dir, 'test')

    train_cat_dir = os.path.join(train_dir, 'cat')
    train_dog_dir = os.path.join(train_dir, 'dog')
    valid_cat_dir = os.path.join(valid_dir, 'cat')
    valid_dog_dir = os.path.join(valid_dir, 'dog')
    test_cat_dir = os.path.join(test_dir, 'cat')
    test_dog_dir = os.path.join(test_dir, 'dog')

    os.makedirs(train_cat_dir, exist_ok=True)
    os.makedirs(train_dog_dir, exist_ok=True)
    os.makedirs(valid_cat_dir, exist_ok=True)
    os.makedirs(valid_dog_dir, exist_ok=True)
    os.makedirs(test_cat_dir, exist_ok=True)
    os.makedirs(test_dog_dir, exist_ok=True)

    # Ambil daftar semua file dari subfolder 'Cat' dan 'Dog'
    all_cats = glob.glob(os.path.join(source_data_dir, 'Cat', '*'))
    all_dogs = glob.glob(os.path.join(source_data_dir, 'Dog', '*'))
    
    random.shuffle(all_cats)
    random.shuffle(all_dogs)

    def copy_files(file_list, dest_folder):
        for file_path in file_list:
            shutil.copy(file_path, dest_folder)

    # --- PERUBAHAN UTAMA DI SINI ---
    # Kita akan gunakan jumlah absolut sesuai tutorial
    print("Menyalin file...")
    copy_files(all_cats[:500], train_cat_dir)
    copy_files(all_dogs[:500], train_dog_dir)
    
    copy_files(all_cats[500:600], valid_cat_dir)
    copy_files(all_dogs[500:600], valid_dog_dir)

    copy_files(all_cats[600:650], test_cat_dir)
    copy_files(all_dogs[600:650], test_dog_dir)
    
    print("Proses persiapan data selesai.")
else:
    print(f"Folder '{organized_data_dir}' sudah ada. Melewati proses persiapan data.")
# %%
base_organized_dir = r"C:\Users\62812\Documents\pelatihan\dogs-vs-cats-organized"

# Buat path untuk train, valid, dan test (mirip tutorial)
train_path = os.path.join(base_organized_dir, 'train')
valid_path = os.path.join(base_organized_dir, 'valid')
test_path = os.path.join(base_organized_dir, 'test')

# Cetak untuk verifikasi
print(f"Training path: {train_path}")
print(f"Validation path: {valid_path}")
print(f"Test path: {test_path}")
# %%
train_batches = ImageDataGenerator(preprocessing_function = tf.keras.applications.vgg16.preprocess_input).flow_from_directory(directory = train_path, target_size = (224,224), classes = ['cat', 'dog'], batch_size = 10)
valid_batches = ImageDataGenerator(preprocessing_function = tf.keras.applications.vgg16.preprocess_input).flow_from_directory(directory = valid_path, target_size = (224,224), classes = ['cat', 'dog'], batch_size = 10)
test_batches = ImageDataGenerator(preprocessing_function = tf.keras.applications.vgg16.preprocess_input).flow_from_directory(directory = test_path, target_size = (224,224), classes = ['cat', 'dog'], batch_size = 10, shuffle = False)
# %%
# CATATAN!!! SAYA PUSING SEKALI INI... BANYAK SEKALI YG SALAH! WAKTU SAYA BELAJAR INI KAYA TAIK!
# %%
assert train_batches.n == 1000
assert valid_batches.n == 200
assert test_batches.n == 100

# %%
imgs, label = next(train_batches)
# %%
# membuat gambar dari prediksinya dan hasilnya menggunakan label, 0 untuk kucing dan 1 untuk anjing
def plotImages(images_arr):
    fig, axes = plt.subplots(1, 10, figsize=(20, 25))
    axes = axes.flatten()
    for img, ax, in zip(images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()

plotImages(imgs)
print(label)


# %%
model = Sequential([
    Conv2D(filters = 32, kernel_size = (3, 3), activation = 'relu', padding = 'same', input_shape = (224, 224, 3)),
    MaxPool2D(pool_size = (2, 2), strides = 2),
    Conv2D(filters = 32, kernel_size = (3, 3), activation = 'relu', padding = 'same'),
    MaxPool2D(pool_size = (2, 2), strides = 2),
    Flatten(),
    Dense(units = 2, activation = 'softmax'),
])
# %%
model.summary()
# %%
model.compile(optimizer =Adam(learning_rate = 0.0001), loss = 'categorical_crossentropy', metrics = ['accuracy'])
# %%
model.fit(x = train_batches, validation_data = valid_batches, epochs = 10, verbose = 2)



# %%
# predict
test_imgs, test_label = next(test_batches)
plotImages(test_imgs)
print(test_label)
# %%
test_batches.classes
# %%
test_batches.reset()
# %%
predictions = model.predict(x = test_batches, verbose = 0)
# %%
np.round(predictions)
# %%
cm = confusion_matrix(y_true=test_batches.classes, y_pred=np.argmax(predictions, axis =1))
# %%
def plot_confusion_matrix(cm,
                      classes,
                      normalize= False,
                      title = 'confusion matrix',
                      cmap = plt.cm.Blues):
    # membuat plot
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation = 45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print('normalize confusion matrix')
    else:
        print('confusion matrix, without normalize')

    print(cm)

    thresh = cm.max() / 2
    for j, i in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment='center',
                 color = 'white'if cm[i, j] > thresh else 'black')

    plt.tight_layout()
    plt.xlabel('predict')
    plt.ylabel('true')
    plt.show()
# %%
test_batches.class_indices # untuk memeriksa label
# %%
cm_plot_label = ['cat', 'dog']
plot_confusion_matrix(cm =cm, classes = cm_plot_label, title = 'Confusion Matrix')



# %%
# BUILD fine-tunned vgg16 model / membangun model yg disetel dengan baik

# download model, perlu internet 
vgg16_model = tf.keras.applications.vgg16.VGG16()
# %%
vgg16_model.summary()
# %%
def count_params(model):
    non_trainable_params = np.sum([np.prod(v.shape) for v in model.non_trainable_weights])
    trainable_params = np.sum([np.prod(v.shape) for v in model.trainable_weights])
    return {'non_trainable_params': non_trainable_params, 'trainable_params': trainable_params}
# %%
params = count_params(vgg16_model)
assert params['non_trainable_params'] == 0
assert params['trainable_params'] == 138357544
# %%
model = Sequential()
for layer in vgg16_model.layers[:-1]:
    model.add(layer)
# %%
model.summary()
# %%
params = count_params(vgg16_model)
assert params['non_trainable_params'] == 0
assert params['trainable_params'] == 138357544
# %%
for layer in model.layers:
    layer.trainable = False
# %%
model.add(Dense(units = 2, activation='softmax'))
# %%
model.summary()
# %%
params = count_params(model)
assert params['non_trainable_params'] == 134260544
assert params['trainable_params'] == 8194


# %%
# TRAIN the fine-tuned vgg16 model / model latih dengan setel yg baik pakai vgg16
model.compile(optimizer= Adam(learning_rate = 0.0001), loss = 'categorical_crossentropy', metrics = ['accuracy'])
# %%
model.fit(x = train_batches, validation_data = valid_batches, epochs = 5, verbose = 1)
# %%
assert model.history.history.get('accuracy')[-1] > 0.95



# %%
# PREDICT using fine-tuned vgg16 model / memprediksi menggunakan tuning
predictions = model.predict(x = test_batches, verbose=0)
# %%
test_batches.classes
# %%
cm = confusion_matrix(y_true=test_batches.classes, y_pred=np.argmax(predictions, axis=-1))
# %%
test_batches.class_indices
# %%
cm_plot_label = ['cat', 'dog']
plot_confusion_matrix(cm=cm, classes=cm_plot_label, title='Confusion Matrix')
# %%
