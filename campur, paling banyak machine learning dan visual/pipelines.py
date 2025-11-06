# pipelines di butuhkan untuk menjaga agar agar data tetap bersih dan tersruktur.
# 1 . agar bersih dan jelas (clean and clear)
# 2 . agar mengetahui bug dimana (debugging)
# 3 . mempermudahkan untuk mengerjakan karna ada alur yg jelas
# 4 . dapat memikirkan model apa saja yg cocok

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# %%
data = pd.read_csv(r"C:\Users\62812\Downloads\archive (10)\Melbourne_housing.csv")
data.head()
# %%
print(type(data))
# %%
data.info()
# %%
data.describe()
# %%
data.isnull().sum()

# %%
# saya ingin mambuat dan mencari nilai y
y = data['Price']
X = data.drop('Price', axis = 1)
# %%
# saya ingin memecah data menjadi train dan test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# %%
X_train.shape, X_test.shape, y_train.shape, y_test.shape

# %%
# memisahkan data/cleaning data dengan memsihkan mana data numerik dan mana data kategorikal dan nilai 
# unik kurang dari < 11

# categorical / string
categorical_kolom = [cname for cname in X_train
                      if X_train[cname].dtype == "object" and X_train[cname].nunique() < 11]
categorical_kolom
# %%
# numerical / integer / float
numberical_kolom = [cname for cname in X_train
                    if X_train[cname].dtype in ['int64', 'float64']]
numberical_kolom
# %%
# menyimpan kolom yg di pilih
my_cols = categorical_kolom + numberical_kolom
X_train = X_train[my_cols].copy()
X_test = X_test[my_cols].copy() # dengan copy() membuat salinan data baru
# %%
X_train.shape, X_test.shape

# %%
###############3333 memasuki tahap pipelines #####################
# preprocessing
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
# %%
# preprocessing untuk data numberical
numberical_transformer = SimpleImputer(strategy = 'constant') # artinya mengisi nilai kosong dengan nilai konstan (misalnya 0 atau -1)
# %%
# preprocessing untuk data categorical_kolom
categorical_transformer = Pipeline(steps = [
    ('imputer', SimpleImputer(strategy = 'most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown = 'ignore'))
])
# %%
# bundle / bundel preprocessing data numberical dan categorical
preprocessor = ColumnTransformer(
    transformers = [
        ('num', numberical_transformer, numberical_kolom),
        ('cat', categorical_transformer, categorical_kolom)
    ])
preprocessor


# %%
# MENENTUKAN MODELNYA
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators=100, random_state=42)

# %%
# MEMBUAT DAN MENGEVALUASI PIPELINE
from sklearn.metrics import mean_absolute_error

my_pipleline = Pipeline(steps = [
    ('preprocessor', preprocessor),
    ('model', model)
])
# %%
my_pipleline.fit(X_train, y_train)
y_pred = my_pipleline.predict(X_test)


# %%
score = mean_absolute_error(y_test, y_pred)
print(score)

# %%
# imports yang diperlukan
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor

# contoh: pastikan ini berisi list nama kolom (string)
# numerical_kolom = ['Price', 'Bedroom2', ...]  # sesuaikan dengan datasetmu
# categorical_kolom = ['Suburb', 'Type', ...]   # sesuaikan dengan datasetmu

# transformer numerik: imputer + scaler (opsional)
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# transformer kategorikal: imputer (most_frequent) + one-hot
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))  # pastikan tidak ada spasi di 'ignore'
])

# gabungkan di ColumnTransformer — pastikan variable kolom adalah list
preprocessor = ColumnTransformer(transformers=[
    ('num', numerical_transformer, numberical_kolom),
    ('cat', categorical_transformer, categorical_kolom)
])

# model & pipeline akhir
model = RandomForestRegressor(n_estimators=100, random_state=42)
my_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', model)
])

# cek cepat sebelum fit
print("preprocessor type:", type(preprocessor))
print("categorical_transformer type:", type(categorical_transformer))
print("categorical_transformer steps:", getattr(categorical_transformer, 'steps', None))
print("numerical_transformer steps:", getattr(numerical_transformer, 'steps', None))
print("num cols:", numberical_kolom)
print("cat cols:", categorical_kolom)

# lalu fit
my_pipeline.fit(X_train, y_train)
y_pred = my_pipeline.predict(X_test)

score = mean_absolute_error(y_test, y_pred)
print("MAE:", score)



##################### ini akan error karna data tidak ada yg berupa float atau int
############### jadi akan error, tapi untuk kode ini benar. sekian terimakasih
# %%
