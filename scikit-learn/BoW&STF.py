# %%
# DATASET
import numpy as np
corpus = ['Linux has been around since the mid-1990s.',
          'Linux disribution include the linux kernel', 
          'Linux is the one of the most prominent open-source software']

corpus


# %%
# BoW / BAG OF WORDS 
# menggunakan Bow dengan CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus).toarray()
X
# %%
vectorizer.get_feature_names_out().reshape(-1, 1) # ini untuk mengatahui ada saja yg di jadikan biner.


# %%
# EUCLIDEAN Distance untuk mengukur kedekatan/jarak antar dokumen(vector)
from sklearn.metrics.pairwise import euclidean_distances

for i in range(len(X)):
    for j in range(i, len(X)):
        a = X[i].reshape(1, -1)
        b = X[j].reshape(1, -1)
        jarak = euclidean_distances(a, b)[0, 0]
        print(f'jarak dokumen {i+1} dan {j+1}: {jarak}')
        # catatan taikkk ini sudah banget codenya astagaaa......


# %%
# STOP WORD FILTERING Pada Text
# berbeda dengan BoW adalah ini lebih efisien karna mengabaikan kata tambahan seperti if, how, the, a , an , do, will
# DATASET
corpus
# %%
# SWF / Stop Word Filtering dengan CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(stop_words='english')
X = vectorizer.fit_transform(corpus).todense() # dense itu mengubah menjadi tipe matrix
X

# %%
vectorizer.get_feature_names_out()
# %% [markdown]
# sudah cuma seperti ini, artinya ini belajar bagaimana mengubah code categorial menjadi number dan number biner.
    # kenapa di ubah? agar mesin komputer machine learning dapat memproses bagaimana ini didapatkan. 
        # karna memang laptop/komputer hanya bisa untuk numbering bytes
