# %% [markdown]
# bayes theorem, sering dikenal sebagai conditional probability
    # di gunakan untuk menghitung probabilitas suatu kejadian berdasarkan informasi yang sudah diketahui sebelumnya.
        # theori ini di temukan oleh thomas bayes
# dari AI | Disebut "Naive Bayes" karena dua alasan utama yang berasal dari namanya:
    # Bagian "Bayes": Karena algoritma ini didasarkan pada Teorema Bayes.
        # Bagian "Naive": Karena algoritma ini membuat sebuah asumsi yang sangat kuat 
            # (dan seringkali tidak realistis) yang dianggap "naif" atau polos.
# ![rumus](https://i.ibb.co.com/Qj6tRxqg/458c7ce8-497c-4d24-b003-10b703172f3c.jpg)
# %% [markdown]
# pengenalan naive bayes
# ![rumus](https://i.ibb.co.com/Pvcwv0dQ/878899df-a904-4711-8863-c3e5d712626b.jpg)
from IPython.display import display, Markdown

display(Markdown(''' studi kasus 1:
Asep
- siomay 0.1
- bakso 0.8
- lumpia 0.1

Budi
- siomay 0.5
- bakso 0.2
- lumpia 0.3
'''))


# %% [markdown]
display(Markdown(f'''
misi : lakukan prediski, siapa pelanggan yg melakukan pemesanan dengan diketahui pesanannya adalah Lumpia dan Bakso\n

Prior Probability: **P(y)**
        - **P**(Asep) = 0.5
        - **P**(Budi) = 0.5
                 
Likelihood: **P**(X|y)
        - **P**(Lumpia, Bakso|Asep) = (0.1 * 0.8)
                                    = 0.08
        - **P**(Lumpia, Bakso|Budi) = (0.3 * 0.2)
                                    = 0.06

[rumus](https://i.ibb.co.com/LzC9BdwC/de4f277c-1baf-4e45-82ad-8335dbe4aa08.jpg)                             
Evidence atau Normalizer: **P**(X)
        - Evidence = $ \sum $ (Likelihood * Prior)
        - **P**(Lumpia, Bakso) = (0.008 * 0.5) + (0.06 * 0.5)
                               = 0.07  
                   
[rumus](https://i.ibb.co.com/cXM6gvTh/3422a1dd-8c16-4e38-b6dd-fa605ce719c6.jpg)
Posterior Probability: **P**(y|X)
        - Posterior = Likelihood * prior / Evidence --> formula
        - **P**(Asep|Lumpia, Bakso) = (0.08 * 0.5) / 0.07
                                    = 0.57
        - **P**(Budi|Lumpia, Bakso) = (0.06 * 0.5) / 0.07
                                    = 0.43
                  
'''))

# %% [markdown]
# studi kasus 2
# ! [rumus](https://i.ibb.co.com/9mzxJsjL/0bda5ae5-c893-4e6a-a863-6c7fcf0803ad.jpg)
display(Markdown(
f'''
Asep:
- siomay 0.1
- bakso 0.8
- lumpia 0.1 
                 
Joko:                 
- siomay 0.5
- bakso 0.2
- lumpia 0.3

misi: lakukan prediski, siapa pelanggan yg melakukan pemesanan dengan diketahui pesanannya adalah **Siomay dan Bakso**\n
                 '''))
# %% [markdown]
display(Markdown(
f'''
misi: siomay dan bakso
Posterior Probability: **P**(y|X) **(kasus 2)**
Evidence: **P(X)**
        - **P**(Siomay, Bakso) = (0.08 * 0.5) + (0.1 * 0.5)
                               = 0.09

- **P**(Asep|Siomay, Bakso) = (0.1 * 0.8) * 0.05 / 0.09
                            = 0.444
- **P**(Joko|Siomay, Bakso) = (0.5 * 0.2) * 0.05 / 0.09
                            = 0.555

'''))

# %% [markdown]
# Dataset: Breast Cancer Wisconsin (Diagnostic)
# 

# load dataset 
from sklearn.datasets import load_breast_cancer

print(load_breast_cancer())
# %%
load_breast_cancer?
# %%
X, y = load_breast_cancer(return_X_y = True)
X.shape
# %%
y.shape # coba-coba
# %%
y # coba-coba


# %%
# Training dan testing set
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size = 0.2,
                                                    random_state= 0)

print(f'X_train and X_test: {X_train.shape}\t|\n{X_test.shape}')
print(f'y_train and y_test: {y_train.shape}\t|\n{y_test.shape}')
# catatan, beda yg train dengan yg test adalah hanya jumlahnya yg train adalah 80% sedangkan test 20%
# %%
# NAIVE BAYES dengan Scikit-Learn
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

model = GaussianNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy_score(y_test, y_pred)
# %%
model.score(X_test, y_test)