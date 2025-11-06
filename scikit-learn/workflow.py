# %%
from sklearn.datasets import load_iris

iris = load_iris()
X = iris.data
y = iris.target
iris
# %%
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.3,
                                                    random_state= 1)

# %%
from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier(n_neighbors= 3)
model.fit(X_train, y_train)
# %%
from sklearn.metrics import accuracy_score

y_prediksi = model.predict(X_test)
akurasi = accuracy_score(y_test, y_prediksi)
print(f'akurasi model adalah: {akurasi}')
# %%
#################### melakukan train ##########
data_baru = [[5,5,3,2],
             [2,4,3,5]]
prediksi = model.predict(data_baru)
prediksi

pred_species = [iris.target_names[i] for i in prediksi]
print(f'hasil prediksi adalah: {pred_species}') # hasilnya ada 2 yaitu, versicolor, virginica
# %%
import joblib

joblib.dump(model, 'iris_classifier_knn.joblib')
# %%
production_model = joblib.load('iris_classifier_knn.joblib')
# %%
