# %% [markdown]
# logistic regression pada binary classification task
# ![rumus linear regression untuk numeric](https://i.ibb.co.com/7dTVDzhw/35c71316-7159-4319-a6f0-d8800add03da.jpg)
# ![rumus logistic regression untuk categorial](https://i.ibb.co.com/7tjzBwJc/34169ea8-b0c6-4250-bd7c-09942c2dc168.jpg)

# Dataset : SMS Spam Collection
# link : (https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection)
# %%
# import datasets
import pandas as pd

df = pd.read_csv(r'C:\Users\62812\Documents\pelatihan\scikit-learn\SMSSpamCollection',
                  sep = '\t',
                  header = None,
                  names = ['label', 'sms'])
df.head()
# %%
df.info() # memberi informasi tentang datanya
df.describe() # memberi describsi tentang datanya berupa dataframe
# %%
df['label'].value_counts() # value_counts ini menujukan jumlah nilai yg ada di filenya. contoh [M, p, M, M, p]
    # hasil value_counts adalah     M   3
        #                           p   2
# %%
# Training dan Testing Datasets
from sklearn.preprocessing import LabelBinarizer

X = df['sms'].values
y = df['label'].values
print(X)
print(y)
lb = LabelBinarizer() # labelBinarizer untuk membuat nilainya menjadi angka, [pria,wanita] --> [0] pria
                                                                                #              [1] wanita
y = lb.fit_transform(y).ravel() # ravel() untuk membuat flat, sama seperti flatten
y

# %%
lb.classes_ # kode ini. classes_ mengambil kode unik. (ada apa saja di binernya)
    # [0] ham
    # [1] spam
# %%
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size= 0.25,
                                                    random_state= 0)
# hasilnya bisa seperti ini karna library ini cukup canggih, bisa mencocokan data mana milik X dan data mana milik y
print(f'X_train: \n{X_train}\n')
print(f'y_train: \n{y_train}\n')
print(f'X_test: \n{X_test}\n')
print(f'y_test: \n{y_test}')

# %%
# Feature Extraction dengan TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(stop_words='english')

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print(f'X_train_tfidf: \n{X_train_tfidf}\n')
print(f'X_test_tfidf: \n{X_test_tfidf}')
# %%
# Binary Classification dengan Logistic Regression
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train_tfidf, y_train)
y_pred = model.predict(X_test_tfidf)

for pred, sms in zip(y_pred[:5], X_test[:5]):
    print(f'PRED: {pred} -- SMS: {sms}')
# artinya pred 0 adalah ini memprediksi bahwa ham/pesan sah (kemungkinanbesar penting. bukan spam)
    # lalu sms: adalah menujukan isi pesannya adalah ini. | yg banyak sekali. 
        # kesimpullanya sms nya di prediksi spam atau bukan.... pred untuk prediksi 0,1 dan sms untuk isi pesannya
# %%
y_pred
# %%
# Evaluation Metrics pada Binary Classification
    # Confusion Matrix, Accuracy, Precission & Recall(sensitivity), F1 score, ROC

# CONFUSION MATRIX
    # confusion matrix biasa di sebut sebagai error matrix.
from sklearn.metrics import confusion_matrix

matrix = confusion_matrix(y_pred, y_test) # confusion matrix adalah evaluasi pada biner classifikasi
matrix
# %%
tn, fp, fn, tp = matrix.ravel() # ravel() ini untuk menjadikan agar menjadi flat/ mendatar dan agar kode tidak error, karna
    # sistem komputer expect 4 tapi yg diberikan malah 2.

print(f'TN/True Negatif adalah: {tn}')
print(f'FP/False Positif adalah: {fp}')
print(f'FN/False Negatif adalah: {fn}')
print(f'TP/True Positif adalah: {tp}')

# %%
# Visualisasi
import matplotlib.pyplot as plt

# plt.matshow(matrix)
im = plt.imshow(matrix, interpolation='nearest', cmap=plt.cm.plasma)
plt.colorbar(im)

plt.title('Confusion Matrix')
plt.xlabel('Predict Label')
plt.ylabel('True Label')
plt.show()


# %%
# ACCURACY
    # ini sudah cukup sering dipelajari
from sklearn.metrics import accuracy_score

akurasi = accuracy_score(y_test, y_pred)
akurasi # bisa juga tidak usah variable, hasilnya 0.96 artinya sangat baik nyaris sempurna
    # untuk rumus dari akurasi silahkan lihat di google saja



# %%
# PRECISSION AND RECALL
    # Precission / PPV      | RECALL(Sensitivitas) / TPR           | lalu ada juga Spesifitas / TNR
        # ini cukup di pelajari juga, dan untuk r2_score sering keluar jika kita belajar dan mencoba tentang machine learning
from sklearn.metrics import precision_score, recall_score

presisi = precision_score(y_test, y_pred) # nilai prediksinya bisa berubah kalo nilainya y nya berubah/diubah tempatnya.
    # rumusnya adalah (y_true, y_pred) harus seperti ini.
print(presisi)


sensitifitas = recall_score(y_test, y_pred)
print(sensitifitas) # hasilnya tidak sebaik presisi, artinya untuk case ini presisi lebih di unggulkan, jadi pake presisi aja.



# %%
# f1_Score / f-measure
    # ini sering di temui di machine learning
from sklearn.metrics import f1_score

f1 = f1_score(y_test, y_pred)
f1
# %%
# ini namanya adalah ROC ,roc adalah grafik yg didalamnya ada auc
from sklearn.metrics import roc_curve, auc

prob_estimates = model.predict_proba(X_test_tfidf)
prob_estimates

fpr, tpr, threshhold = roc_curve(y_test, prob_estimates[:, 1]) # [:, 1] artinya mengambil semua baris dari indeks ke 2
nilai_auc = auc(fpr, tpr) # artinya fungsi auc dari sklearn menghitung AUC. lalu ia mengambil fpr sebagai sumbu (X) dan tpr sebagai sumbu (y)
# fpr = false positive rate
# tpr = true positif rate
# threshhold = batas
# ROC = Receive Operating Characteristic
# AUC = Area Under The Curve

plt.plot(fpr, tpr, 'b', label = f'AUC: {nilai_auc}')
plt.plot([0, 1], [0, 1], '--r', label = 'classifier')

plt.title('ROC: Receiver Operating Characteristic')
plt.xlabel('Fallout or False Positive Rate(FPR)')
plt.ylabel('recall or sensitivitas or True Positive Rate(TPR)')
plt.legend()
plt.show()
# %%
