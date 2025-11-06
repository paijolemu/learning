# Label encoding dan hot-one encoding
# Label Encoding = memberikan label numberic kepada categorical/string. jadi intinya adalah mengubah string menjadi numberic, pemberian numberic sesuai urutan.
    #  contoh [merah,kuning,hijau] menjadi --> [0  1  2]
        # biasanya di gunakan saat data terstruktur dan banyak data
# One-Hot Encoding = memberikan number byte / biner kepada categorical/string. jadi intinya adalah mengubah string menjadi struktur byte/biner.
    # pemberian biner sesuai dengan jumlah nilai categorical dan seperti biner. contoh [SMA -> s1 -> s2] menjadi --> [1  0  0] | SMA
    #                                                                                                          [0  1  0] | s1
    #                                                                                                          [0  0  1] | s2
        # salah satu kelebihan onehot adalah biner nya akan mengurutkan berdasarkan alphabet/ascending
            # biasanya digunakan saat data tidak memiliki struktur dan data tidak banyak


# %%
# dataset
import pandas as pd

df = pd.DataFrame({'country': ['India', 'US', 'Japan', 'US', 'Japan'],
                   'Age': [44, 34, 46, 35, 23],
                   'Salary': [72000, 65000, 98000, 45000, 34000]})
df

# %%
# Label Encoding
from sklearn.preprocessing import LabelEncoder

labelencoder = LabelEncoder()
df['country'] = labelencoder.fit_transform(df['country']) # ini mengubah country menjadi numbering. dan hanya country saja
df
# %%
labelencoder.classes_ # ini menunjukan jumlah yg di beri Label Encoding


# %%
# dataset
import pandas as pd

df = pd.DataFrame({'country': ['India', 'US', 'Japan', 'US', 'Japan'],
                   'Age': [44, 34, 46, 35, 23],
                   'Salary': [72000, 65000, 98000, 45000, 34000]})
df
# %%
# ini tahap 
df_country = df['country'].values.reshape(-1, 1) # jika di one-hot maka harus di values reshape() agar komputer paham 
df_country
# %%
# One-Hot Encoding
from sklearn.preprocessing import OneHotEncoder

onehot = OneHotEncoder()
X = onehot.fit_transform(df_country).toarray() # harus pakai toarray() jika tidak maka tidak berhasil, walaupun memang tidak error
X
# %%
onehot.categories_ # ini tidak classes_ karna memang berbeda tugasnya dengan Label encoder diatas.



# %%
# membuat one-hot / biner / byte menjadi DataFrame
df_onehot = pd.DataFrame(X, columns = [str(i) for i in range(X.shape[1])])
df_onehot
# %%
# menggabungkan
df = pd.concat([df_onehot, df], axis= 1) # concat() artinya concatenate/menggabungkan df_onehot dengan df sebagai axis =1 (sebagai Colom)
df
# %%
df = df.drop(['country'], axis = 1)
df
# %%
