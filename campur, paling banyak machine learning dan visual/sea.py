# %%
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
sns.set() # ini jalankan agar yg muncul adalah format sns bukan format matplotlib

# %%
dataku = pd.read_csv('hypertension_dataset.csv')
dataku.head()
# %%
dataku.describe(include='O')
# %%
# Barplot (histogram)
sns.barplot(x= 'Medication', y='Age', data= dataku, color = 'r' ) #  Tinggi Batang Merah: Rata-rata (Mean)
# Ini adalah bagian yang paling penting. Ketinggian setiap batang BUKAN menunjukkan jumlah orang, melainkan menunjukkan nilai RATA-RATA (MEAN) dari Age untuk setiap kelompok di sumbu X.
# %%
sns.barplot(x='Age' , y='Medication', data= dataku, color = 'r' )
# %%
sns.barplot(x= 'Medication', y='Age', data= dataku, color = 'r', estimator=sum )
# %%
sns.barplot(x= 'Medication', y='Age', data= dataku, color = 'r', estimator=np.std )
# %%
# countplot /jumlah
sns.countplot(x='Medication',data=dataku ) # ini akan menjumlahkan pergolongan.
# %%
sns.countplot(x='BP_History', hue = 'Medication', data=dataku)



# %%
######### boxplot(box % whisker plot)
sns.boxplot(x='Medication', y='Age', data=dataku)

# %%
sns.boxplot(x='Medication', y='Age', hue= 'BP_History', data=dataku) # cara bacanya. Q1 yg paling bawah | Q2 yg berada di tengah.
            # Q3 yg berada di atas. | garis diatas dan dibawah box namanya adalah outlier. | outlier atas = Q3 +1.5*(Q3-Q1)
            # kenapa bisa ada outlier? alasannya karna ada yg menyebar terlalu jauh        | outlier bawah = Q1 - 1.5*(Q3-Q1)


# %%
# violin plot
sns.violinplot(x='Medication', y='Age', data=dataku) # akan berbentuk seperti violin.
# %%
sns.violinplot(x='Medication', y='Age', hue= 'BP_History', data=dataku)


# %%
####### strip plot
sns.stripplot(x='Medication', y='Age',hue = 'Medication', data=dataku)


# %%
# swarmplot
sns.swarmplot(x='Medication', y='Age', hue='Medication', data=dataku)



################## belajar matrix ##################
# membuat heatmap
# catatan... heatmap hanya bisa di lakukan didata numeric/angka
# %%
korelasi = dataku.corr(numeric_only=True) # hanya mengambil data numeric
korelasi
# %%
sns.heatmap(korelasi)
        # 1 artinya sempurna. 0 artinya tidak ada hubungan -1 artinya negatif sempurna.
# %%
sns.heatmap(korelasi, annot=True) # annot untuk mendetailkan / memberi nilai pasti di heatmapnya.
# %%
sns.heatmap(korelasi, annot=True, cmap= 'Reds') # dengan Reds akan menjadi lebih nyaman.
# %%
dataku.head()
# %%
########### menghilangkan index #########
data2 = pd.read_csv('hypertension_dataset.csv', index_col = 'Family_History')
data2.head()
# %%
sns.heatmap(data2.iloc[:,[0,1,3,4,5,6]].head()) # ini adalah
# %%
############### heatmap advance #############
terbang = sns.load_dataset('flights') # memakai dataset seaborn
terbang.head()
# %%
terbang 
# %%
naik = terbang.pivot_table(index = 'month', columns = 'year', values = 'passengers')
naik
# %%
sns.heatmap(naik, cmap = 'Reds') # ini artinya semakin gelap selakin banyak orangnya/passengers
# %%
sns.heatmap(naik, cmap = 'Reds', annot= True) # ini tidak bisa karna.. terlalu banyak angka. ribet
# %%
sns.heatmap(naik, cmap = 'Reds', linecolor = 'white', linewidths = 1) # ini akan lebih rapih dan terlihat bagus
# %% 
sns.heatmap(naik, cmap = 'Reds', linecolor = 'black', linewidths = 1) # linenya akan berwarna hitam
# %%
############################ machine learning #####################
# cluster map
sns.clustermap(naik); 
# %%
sns.clustermap(naik, cmap = 'coolwarm'); # membuat lebih enak dibaca | 'coolwarm'
# %%
sns.clustermap(naik, cmap = 'coolwarm', standard_scale = 1); # dengan ini akan membuat lebih mudah di baca karna lebih dominan
# %%
sns.get_dataset_names
# %%
berlian = sns.load_dataset('diamonds')
berlian
# %%
berlian.info() # dataset ini lebih banyak di banding flights
# %%
berlian.cut.unique()
# %%
berlian.color.unique()
# %%
sns.pairplot(berlian)
# %%
i =sns.PairGrid(berlian)
i
# %%
i.map(plt.scatter)
plt.show()
# %%
i =sns.PairGrid(berlian) # ini pairGrid dengan ini kita bisa mmebuat macam macam data dengan lebih leluasan. 
i.map_diag(sns.distplot) # kita pasang data di diagonal pakai distplot
i.map_upper(plt.scatter) # kita pasang yg atas pakai scatter plot
i.map_lower(sns.kdeplot) # kita pasang yg bawah pakai kde plot
    # hasil: yg paling atas ada scatter plot
    # diagonal adalah distibution plot
    # lalu yg bawah adalah kde plot
# %%
tabrak = sns.load_dataset('car_crashes')
tabrak
x =sns.PairGrid(tabrak) # ini pair yg kosong yg akan kita gantikan dengan macam macam plot
x.map_diag(sns.distplot) # pasang dist plot di diagonal
x.map_upper(plt.scatter) # pasang scatter di atas
x.map_lower(sns.kdeplot) # pasang kde plot di bawah
# %%
##################### Facet Grid ##################
tips = sns.load_dataset('tips')
tips
# %%
p = sns.FacetGrid(tips, row ='smoker', col = 'day') # artinya row ada 2 jenis object
        # col ada 4 jenis object
p.map(sns.distplot, 'total_bill'); 
        # ini sebenarnya adlah kombinasi dari 3 paramater sebetulnya
plt.show()
# %%
p = sns.FacetGrid(tips, row ='smoker', col = 'day')
p.map(sns.distplot, 'total_bill', kde = False); 
        # dengan mematikan kde akan lebih mudah di baca untuk sebagian orang
plt.show()
# %%
p = sns.FacetGrid(tips, row ='smoker', col = 'day', hue= 'sex')
p.map(plt.scatter, 'total_bill','tip' ); # inget ya. plt tidak punya kde 
# %%
