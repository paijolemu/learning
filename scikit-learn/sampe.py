# %%
from sklearn.datasets import load_iris
iris = load_iris()
iris
# %%
iris.keys()

# %%
print(iris.DESCR)
# %%
X = iris.data
# %%
y = iris.target
y
    # baris pertama di array itu milik angka pertama di target jadi, 5.1, 3.5, 1.4, 0.2 itu milik angka 0
# %%
X.shape

# %%
y.shape
# %%
iris.feature_names
    # artinya ini berkorelasi dengan iris data/X, 4 kolom dari data X menandakan 4 nama: sepal length (cm), sepal width (cm), petal length (cm), petal width (cm)
# %%
iris.target_names
    # artinya berkorelasi dengan target y, 3 nama: setosa(0), versicolor(1), virginica(2)
# %%
import matplotlib.pyplot as plt
import seaborn as sns


X = X[:, :2] # kita ambil 2 kolom pertama dari data X. kenapa hanya kolom 0 dan 1? karna :2 menujukan sebelum 2 bukan sampe 2.

x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    # baris ini untuk mengatur batas sumbu x dan y pada grafik

plt.scatter(X[:, 0], X[:, 1], c=y) # ini untuk menampilkan grafiknya. c=y artinya colornya mengikuti jumlah nilai y.
plt.xlabel('sepal length/panjang kelopak')
plt.ylabel('sepal width/lebar kelopak')

plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.show()
# %%
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size = 0.2,
                                                    random_state = 1)
print(f'hasil x_train adalah: {x_train.shape}')
print(f'hasil x_test adalah: {x_test.shape}')
print(f'hasil y_train adalah: {y_train.shape}')
print(f'hasil y_test adalah: {y_test.shape}')

# %%
iris = load_iris(as_frame=True)
iris.frame.head()
    # pakai frame untuk membuat agar kita bisa menampilkan datanya agar terlihat bagus/frame. dan harus seeperti ini.
        # bisa juga pakai iris.data.head() 
# %%
import pandas as pd
df = pd.DataFrame(data = iris['feature_names'])
df