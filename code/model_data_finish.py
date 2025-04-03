#%%
import pandas as pd
#%%
# import 
file_path = f'../data/scrape_mediestream_1850_2010.csv'
df = pd.read_csv(file_path)
df['year'] = df['year'].astype(int)
df['hits'] = pd.to_numeric(df['hits'], errors='coerce')
display(df)

#%%
# Filter
df = df.loc[df.year <= 2007]
df['artpub'] = df.hits / df.hits.max()
# %%
# export 
df.to_csv('../data/model_artpub.csv', index=False)
# %%
