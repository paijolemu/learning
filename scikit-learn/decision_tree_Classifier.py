# %% [markdown]
# Konsep Dasar Decision Tree Classifier.
# ![visual](https://i.ibb.co.com/WWMgx8F5/e00b7cbc-9998-4370-8221-adf0c3f78a4e.jpg)
# ada 3 cabang.      1. root node(utama/paling atas), 2. internal node(ditengah), 3. leaf node(daun/paling bawah)

# %% [markdown]
# Gini Impurity
# kenapa awalan pakai gini impurity? karna otak dari decision tree adalah gini impurity.
# ![visual](https://i.ibb.co.com/TxdtFDtw/0df94185-1d5f-48ba-921d-cc1832aac830.jpg)
# ![rumus ruas kiri & ruas kanan](https://i.ibb.co.com/q3p7h9mb/c0b7250e-31af-4bb2-ad98-4212e2bd7d8b.jpg)
# ![avg gini impurity](https://i.ibb.co.com/KcpjQczW/5e4f0cf9-3f29-456c-86e0-c6c1de957e81.jpg)
# ini bertujuan untuk membedakan / mengklasifikasi data. data mana yg masuk ke kiri atau ke kanan.
# ![information gain](https://i.ibb.co.com/kgX6vDz6/f5106417-4d2e-4433-a323-ba09f2285202.jpg)

# %% [markdown]
# konsep workflow
# ![membangun decision Tree](https://i.ibb.co.com/v6ysZXYz/1f7b2e2b-d178-4af5-9105-cef5be02498e.jpg) 


# %%
# dataset
from sklearn.datasets import load_iris

X, y = load_iris(return_X_y = True)
print(f'X_shape: {X.shape}') # 150, 4 artinya 150 baris, 4 kolom
print(f'class / set: {set(y)}') # hasilnya masih ada np.int64

# %%
y = y.tolist() # ini mengubah agar tidak ada np.int64
print(f'class: {set(y)}')
# %%
# split data
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size = 0.3,
                                                    random_state= 0)

# %%
# Classification with Decision Tree
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(max_depth= 4) # maksudnya membuat 4 layers
model.fit(X_train, y_train)

# %%
# Visualisasi Model
import matplotlib.pyplot as plt
from sklearn import tree

# plt.rcParams['figure.dpi'] = 85
# plt.subplot(figsize = (10, 10))
# tree.plot_tree(model, fontsize=10)
# plt.show()
# ini error, yg bener yg bawah

# %%
plt.rcParams['figure.dpi'] = 85
fig, ax = plt.subplots(figsize=(10, 10)) # harus di berikan variable fig, ax. fig untuk tempat atau canvas kosongnya.
tree.plot_tree(model, ax=ax, fontsize=10) #  ax adalah tempat spesifik di dalam area gambar.
plt.show()
# %%
# Evaluasi Model
from sklearn.metrics import classification_report

y_pred = model.predict(X_test)
hasil = classification_report(y_test, y_pred)
print(hasil)
# %%
