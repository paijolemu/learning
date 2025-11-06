# cross validation di gunakan untuk mengevaluasi model dan melihat yg mana yg terbaik di antara semua urutan validasi
# jadi mengurutan data training menjadi beberapa bagian (fold) / lipatan biasa 5 lipatan dan diantara semua ipatan itu ana yg terbaik
# konsepnya seperti ini:
# 10000   01000  00100  00010  00001 --> ini namanya cross validation 5 folds
############### mencoba cross validation di sklearn ##################
# %%
data = pd.read_csv(r"C:\Users\62812\Downloads\archive (11)\melb_data.csv")
data.head()
# %%
# memilih data untuk x dan y
X = data[['Rooms', 'Distance', 'Landsize', 'BuildingArea', 'YearBuilt']]
y = data.Price

# %%
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline  import Pipeline

my_Pipeline = Pipeline(steps = [
    ('imputer', SimpleImputer()), # mengisi nilai kosong dengan nilai rata-rata
    ('model', RandomForestRegressor(n_estimators = 100, random_state = 0))
])

# %%
from sklearn.model_selection import cross_val_score
scores = -1 * cross_val_score(my_Pipeline, X, y,
                         cv = 5, # ini adalah kuncinya.. menggunakan 5 folds / lipatan
                         scoring = 'neg_mean_absolute_error')
# %%
print('MAE:', scores ) # fold 4 best, nilai paling kecil yg terbaik
print(f'\nMSE mean: {scores.mean()}\n') # rata-rata dari semua fold