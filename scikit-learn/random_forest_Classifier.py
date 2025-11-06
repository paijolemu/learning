# %% [markdown]
# ![general model](https://i.ibb.co.com/R4gmxzCV/3ebe7768-704b-4fa5-8124-c06fc222bc6f.jpg)
# ini adalah workflow bagaimana machine learning bekerja.
# %% [markdown]
# ![ensemble learning](https://i.ibb.co.com/KxFXFncW/9f8cc983-f7a2-4acb-9b85-0804d59f4e1c.jpg)
# saya sudah belajar bagaimana ml bekerja. akan tetapi itu sendiri sendiri. jadi ensemble ini adalah gabungan/banyak model langsung.
# menggabungkan beberapa model untuk mendapatkan hasil prediksi yang lebih baik dan lebih akurat daripada jika kita hanya menggunakan satu model saja.
# heterogeneous = gabungan dari model yg berbeda. contoh logistic, svm, decision tree.
# homogeneous = gabungan dari model yg sama. contoh bagging(random forest)

# %% [markdown]
# BAGGING: BOOTSTRAP AGGREGATING
# ![Bagging](https://i.ibb.co.com/jPY3zsQC/605b8b81-c8db-4a92-a627-29ee4a5e7855.jpg)
# ini adalah bagaimana flow kerja bagging. termasuk homogeneous.

# %% [markdown]
# ![random forest](https://i.ibb.co.com/ymvrPw7H/44986293-8e3f-4020-9517-1c0bfdc77b0e.jpg)
# random forest adalah bagging + random features(X)

# %%
# Dataset
from sklearn.datasets import load_iris

X, y = load_iris(return_X_y= True)

print(f'X_shape: {X.shape}')
y = y.tolist() # ini mengubah agar tidak ada np.int64
print(f'class: {set(y)}')
# %%
# split data
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size= 0.3,
                                                    random_state= 0)
print(f'X_train_shape: {X_train.shape}')
# %%
# Classification with RandomForestclassifier
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators = 200,
                               random_state = 0)
model.fit(X_train, y_train)
# %%
# classifier report
from sklearn.metrics import classification_report

y_pred = model.predict(X_test)
hasil = classification_report(y_test, y_pred)
print(hasil)
# %%
