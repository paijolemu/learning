# %%
corpus = ['he house had a tiny little house',
          'the cat saw the mouse',
          'the mouse run away from the house',
          'the cat finally ate the mouse',
          'the end of he mouse story']
corpus
import pandas as pd

df = pd.DataFrame(data=corpus)
df
# %%
# TF-IDF WEIGHT With TfidfVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(stop_words='english')
response = vectorizer.fit_transform(corpus)
print(response)

# %%
vectorizer.get_feature_names_out().reshape(-1, 1)
# %%
response.todense()
response.todense().reshape(-1, 1)
# %%
df = pd.DataFrame(response.todense().T,
                  index = vectorizer.get_feature_names_out(),
                  columns= [f'D{i+1}' for i in range(len(corpus))])
df
# %%
