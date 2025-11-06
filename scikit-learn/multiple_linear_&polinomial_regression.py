# %%
########### dataset
import pandas as pd
# training dataset
pizza = {'diameter': [6, 8, 10, 14, 18],
         'n_topping': [2, 1, 0, 2, 0],
         'price': [7, 9, 13, 17.5, 18]}
pizza_df_train = pd.DataFrame(pizza)
pizza_df_train
# %%
# testing dataset
pizza = {'diameter': [8, 9, 11, 16, 12],
         'n_topping': [2, 0, 2, 2, 0],
         'price': [11, 8.5, 15, 18, 11]}
pizza_df_test = pd.DataFrame(pizza)
pizza_df_test
# %%
import numpy as np
# prepocessing
X_train = np.array(pizza_df_train[['diameter','n_topping']])
y_train = np.array(pizza_df_train['price'])

print(f'X_train:\n{X_train}')
print(f'y_train:\n{y_train}')
# %%
X_test = np.array(pizza_df_test[['diameter', 'n_topping']])
y_test = np.array(pizza_df_test['price'])

print(f'X_test:\n{X_test}')
print(f'y_test:\n{y_test}')


# %% [markdown]
################ MULTIPLE LINEAR REGRESSION
# ![rumus](https://iili.io/KfuLI5u.jpg)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
r_squared = r2_score(y_test, y_pred)
print(f'r2_score: {r_squared}')


# %%
################### POLINOMIAL REGRESSION
# Preprocessing dataset
X_train = np.array(pizza_df_train['diameter']).reshape(-1, 1)
y_train = np.array(pizza_df_train['price'])

print(f'X_train:\n{X_train}')
print(f'y_train:\n{y_train}')
# %% [markdown]\
# POLINOMIAL REGRESSION : QUADRATIC
# ![rumus](https://iili.io/KfaoqLg.jpg)
from sklearn.preprocessing import PolynomialFeatures

quadratic_pol = PolynomialFeatures(degree=2)
X_quadratic = quadratic_pol.fit_transform(X_train)
print(f'poli quadratic:\n {X_quadratic}')
    # hasilya ada 3. kita ambil pertama saja | 1. 6. 36. maksud dari hasil ini adalah karna x = 6, lalu sesuai rumus maka.. 
        # 6 ** 0 = 1, 6 ** 1 = 6, lalu 6 ** 2 = 36. jadi seperti itu cara menghitungnya
# %%
model.fit(X_quadratic, y_train) # training model


# %%
##### VIASUALISASI MODEL
import matplotlib.pyplot as plt

X_vis = np.linspace(0, 25, 100).reshape(-1, 1)
X_vis_quadratic = quadratic_pol.transform(X_vis)
y_vis_quadratic = model.predict(X_vis_quadratic)

plt.scatter(X_train, y_train)
plt.plot(X_vis, y_vis_quadratic, '-r')

plt.title('Perbandingan Diameter Dan Harga Pizza')
plt.ylabel('Price($)')
plt.xlabel('Diameter(cm)')
plt.xlim(0, 25)
plt.ylim(0, 25)
plt.grid()
plt.show()


# %% [markdown]
# FINISH!!
# PERBANDINGAN POLINOMIAL REGRESSION : QUADRATIC VS CUBIC
# training set
plt.scatter(X_train, y_train)

# linear
model = LinearRegression()
model.fit(X_train, y_train)
X_vis = np.linspace(0, 25, 100).reshape(-1, 1)
y_vis = model.predict(X_vis)
plt.plot(X_vis, y_vis, '--r', label = 'linear')

# QUADRATIC
quadratic_features = PolynomialFeatures(degree= 2)
X_quadratic = quadratic_features.fit_transform(X_train)
model = LinearRegression()
model.fit(X_quadratic, y_train)
X_vis_quadratic = quadratic_features.transform(X_vis)
y_vis = model.predict(X_vis_quadratic)
plt.plot(X_vis, y_vis, '--g', label = 'quadratic')

# CUBIC
cubic_features = PolynomialFeatures(degree= 3)
X_cubic = cubic_features.fit_transform(X_train)
model = LinearRegression()
model.fit(X_cubic, y_train)
X_vis_cubic = cubic_features.transform(X_vis)
y_vis = model.predict(X_vis_cubic)
plt.plot(X_vis_cubic, y_vis, '--y', label = 'cubic')

# visual
plt.title('perbandingan diameter dan harga pizza')
plt.xlabel('harga')
plt.ylabel('diameter')
plt.xlim(0, 30)
plt.ylim(0, 30)
plt.grid()
plt.show()
# %%
