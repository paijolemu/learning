# ===================================================================
# KODE LENGKAP UNTUK VISUALISASI AUGMENTASI (PERBAIKAN IndexError)
# ===================================================================

import monai
import os
import tempfile
import torch
import matplotlib.pyplot as plt
from monai.transforms import LoadImaged, RandRotate90d, RandZoomd, ScaleIntensityd

# --- BAGIAN 1: DOWNLOAD DATASET (Tetap sama) ---
print("Mempersiapkan dataset...")
directory = os.environ.get("MONAI_DATA_DIRECTORY")
root_dir = tempfile.mkdtemp() if directory is None else directory
data_dir = os.path.join(root_dir, "MedNIST")
resource = "https://github.com/Project-MONAI/MONAI-extra-test-data/releases/download/0.8.1/MedNIST.tar.gz"
compressed_file = os.path.join(root_dir, "MedNIST.tar.gz")

if not os.path.exists(data_dir):
    print("Dataset tidak ditemukan. Memulai download...")
    monai.apps.download_and_extract(resource, compressed_file, root_dir)
    print("Download selesai.")
else:
    print("Dataset sudah ada.")

# --- BAGIAN 2: VISUALISASI AUGMENTASI (SUDAH DIPERBAIKI) ---
# Ambil beberapa contoh gambar
class_names = sorted(x for x in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, x)))
image_files = [os.path.join(data_dir, class_names[i], os.listdir(os.path.join(data_dir, class_names[i]))[0]) for i in range(len(class_names))]

# Definisikan transformasi augmentasi
transforms = monai.transforms.Compose([
    LoadImaged(keys=["img"]),
    ScaleIntensityd(keys=["img"]),
    RandRotate90d(keys=["img"], prob=1.0),
    RandZoomd(keys=["img"], prob=1.0, min_zoom=1.1, max_zoom=1.5)
])

# Terapkan transformasi pada satu gambar
img_dict = {"img": image_files[0]} 
transformed_data = transforms(img_dict)
transformed_img = transformed_data["img"]

# === PERBAIKAN UTAMA ADA DI SINI ===
# Muat gambar asli secara terpisah untuk ditampilkan
loader_for_original = LoadImaged(keys=["img"])
original_img_data = loader_for_original(img_dict)
original_img = original_img_data["img"]

# Visualisasi
print("\nMembuat plot perbandingan...")
plt.figure("check", (8, 4))
plt.subplot(1, 2, 1)
plt.title("Gambar Asli")
plt.imshow(original_img.squeeze(), cmap="gray") # Gunakan variabel yang sudah dimuat
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Setelah Augmentasi")
plt.imshow(transformed_img.squeeze(), cmap="gray")
plt.axis('off')

plt.tight_layout()
plt.show()
# Simpan gambar ini sebagai 'monai_augmentation.png'