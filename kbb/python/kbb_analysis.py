#!/usr/bin/env python
# coding: utf-8

# In[188]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from string import ascii_letters as letters
from string import digits
import itertools
import seaborn as sns

plt.style.use(['dark_background'])
# plt.rcParams['figure.dpi'] = 300
get_ipython().run_line_magic('matplotlib', 'inline')


# In[189]:


df = pd.read_csv('csv/kbb_scraper-2000.csv', header=None)
df = df.drop_duplicates()
df.head()


# In[190]:


df.columns = ['title', 'price','owner', 'accident', 'price_rating', 'mileage', 'drive_type', 'engine', 'transmission', 'fuel_type', 'mpg', 'exterior', 'interior']
df.head()


# In[191]:


df.shape
# df.to_csv('kbb_data_dupdropped2.csv')


# In[192]:


df.isna().sum()


# In[193]:


df.info()


# ## ===== (1) Cleaning Data =====

# In[194]:


# Seeing where in the dataframe it contains 'MSRP'
mask = np.column_stack([df[col].str.contains('MSRP', na=False) for col in df])
df.loc[mask.any(axis=1)]


# In[195]:


df['mileage'] = df['mileage'].replace('\,', '', regex=True).replace(letters, '', regex=True)
df['mileage'] = pd.to_numeric(df['mileage'])

df['price'] = df['price'].replace('\,', '', regex=True).replace(r'MSRP', '', regex=True)
df['price'] = pd.to_numeric(df['price'])

df.info()


# ### ===== mpg =====

# In[196]:


# df['MPG (City)'] = df['']
for item in df['mpg'][:10]:
    print(item.split('/'))


# In[197]:


df['mpg_city'] = [item.split('/')[0] for item in df['mpg']]
df['mpg_highway'] = [item.split('/')[1] if len(item.split('/')) == 2 else item.split('/')[0] for item in df['mpg']]

df['mpg_city'] = df['mpg_city'].str.replace('City', '').replace("'", '').map(lambda x: x.strip())
df['mpg_highway'] = df['mpg_highway'].str.replace('Highway', '').replace("'", '').map(lambda x: x.strip())


# In[198]:


df['mpg_city'].unique()
# df['MPG (City)'].value_counts()


# In[199]:


df['mpg_highway'].unique()
# df['MPG (Highway)'].value_counts()


# In[200]:


df[df['mpg_city'].str.startswith('5')][:5]


# In[201]:


len(df.iloc[16]['mpg'])


# In[202]:


df.loc[df['mpg_city'].str.len() > 3, 'mpg_city'] = df['mpg_city'].str[0]
df.loc[df['mpg_highway'].str.len() > 3, 'mpg_highway'] = df['mpg_highway'].str[-1]

df['mpg_city'].replace(r'[a-zA-Z]', '0', regex=True, inplace=True)
df['mpg_highway'].replace(r'[a-zA-Z]', '0', regex=True, inplace=True)

df['mpg_city'] = df['mpg_city'].astype(str).replace(r'[^0-9]', '0', regex=True)
df['mpg_highway'] = df['mpg_highway'].astype(str).replace(r'[^0-9]', '0', regex=True)


# In[203]:


df['mpg_city'] = pd.to_numeric(df['mpg_city'])
df['mpg_highway'] = pd.to_numeric(df['mpg_highway'])


# In[204]:


df = df.drop('mpg', axis=1)


# In[205]:


df.reset_index(drop=True, inplace=True)


# ### ===== fuel_type =====

# In[206]:


df['fuel_type'].unique()


# In[207]:


df[df['fuel_type'].str.contains('Miles')][:5]


# In[208]:


df['fuel_type'] = df['fuel_type'].replace(r'[0-9]+\s(m|M)iles', 'N/A', regex=True)
df['fuel_type'].unique()


# In[209]:


# pd.Categorical(df['Fuel Type'])
df['fuel_type'] = pd.factorize(df['fuel_type'])[0]
df['fuel_type'].unique()


# ### ===== drive_type =====

# In[210]:


df['drive_type'].value_counts()


# In[211]:


df['drive_type'] = df['drive_type'].replace('All wheel drive', '4 wheel drive').replace('4 wheel drive - rear wheel default', '4 wheel drive')
df['drive_type'].value_counts()


# In[212]:


df[df['drive_type'] == 'Information Unavailable']


# In[213]:


df['drive_type'] = df['drive_type'].replace('Information Unavailable', '2 wheel drive - rear')
df['drive_type'].value_counts()


# In[214]:


df['drive_type'] = pd.factorize(df['drive_type'])[0]
df['drive_type'].value_counts()


# In[215]:


df['engine'].value_counts()


# In[216]:


df['transmission'].value_counts()


# ### ===== condition =====

# In[217]:


df['combined_mpg'] = df['mpg_city'] + df['mpg_highway']
df['condition'] = [item.split()[0] for item in df['title']]
df.head()


# In[218]:


df.info()


# In[219]:


df['condition'].value_counts()


# In[220]:


condition_dict = {'Used': 1, 'Certified': 2, 'New': 3}
df['condition'] = df['condition'].map(condition_dict)


# In[221]:


df.head()


# ### ===== year, make_model =====

# In[222]:


year = [item.split()[1] for item in df['title']]
isnum = [s for s in year if s.isdigit()]

len(year), len(isnum)


# In[223]:


df['year'] = [item.split()[1] for item in df['title']]
df['year'] = pd.to_numeric(df['year'])
df.head()[:2]


# In[224]:


df['brand'] = [item.split()[2] for item in df['title']]
df['brand'].value_counts()


# In[225]:


df['brand'] = pd.factorize(df['brand'])[0]


# In[226]:


df['make_model'] = [' '.join(item.split()[2:]) for item in df['title']]
df = df.drop('title', axis=1)
df.info()


# In[227]:


df.isna().sum()


# ### ===== owner =====

# In[228]:


df['owner'].value_counts()


# In[229]:


df[df['owner'] == 'No Accident / Damage Reported']


# In[230]:


df.loc[558, 'accident'] = 'No Accident / Damage Reported'
df.loc[558, 'owner'] = np.nan


# In[231]:


df['owner'].value_counts()


# In[232]:


df.drop(df[(df['owner'].isna()) & (df['combined_mpg'] < 11)].index, inplace=True)


# In[233]:


df.loc[(df['owner'].isna()) & (df['mileage'] < 500), 'owner'] = 'One Owner'
df.loc[(df['owner'].isna()) & (df['mileage'] > 500), 'owner'] = 'Multiple Owners'
df.loc[(df['owner'].isna()) & (df['condition'] == 3), 'owner'] = 'One Owner'


# In[234]:


owner_dict = {'Multiple Owners': 0, 'One Owner': 1}
df['one_owner'] = df['owner'].map(owner_dict)


# ### ===== price =====

# In[235]:


df[df['price'].isna()]


# In[236]:


# Got from KBB
df.loc[113, 'price'] = 20923
df.loc[604, 'price'] = 14936


# In[237]:


df['price'].describe()


# In[238]:


df['price'].plot.kde(color='red');


# In[239]:


df['price'].kurtosis()


# In[240]:


df[df['price'] > 30000].shape


# In[241]:


bins = [0, 7000, 10000, 13000, 16000, 18000, 20000, 22000, 24000, 26000, 28000, 30000, 35000, 40000, 50000, 95000]
labels = np.arange(0, 15)
df['price_bin'] = pd.cut(df['price'], bins=bins, labels=labels)
df.head()


# ### ===== accident =====

# In[242]:


df.loc[(df['accident'].isna()) & (df['mileage'] < 500), 'accident'] = 'No Accident / Damage Reported'
df.loc[(df['accident'].isna()) & (df['condition'] == 3), 'accident'] = 'No Accident / Damage Reported'
df.loc[(df['accident'].isna()) & (df['condition'] == 2), 'accident'] = 'No Accident / Damage Reported'
df.loc[(df['accident'].isna()) & (df['price'] > 10000), 'accident'] = 'No Accident / Damage Reported'
df.loc[(df['accident'].isna()) & (df['price'] < 10000), 'accident'] = 'Accident / Damage Reported'

df = df.drop('owner', axis=1)
df.head()


# In[243]:


df['accident'].value_counts()


# In[244]:


accident_dict = {'No Accident / Damage Reported': 1, 'Accident / Damage Reported': 0}

df['accident'] = df['accident'].map(accident_dict)


# ### ===== price_rating =====

# In[245]:


df[df['price_rating'].isna()][:5]


# In[246]:


df['price_rating'].fillna('N/A', inplace=True)


# In[247]:


price_rating_dict = {'N/A': 0, 'GOOD': 1, 'GREAT': 2}
df['price_rating'] = df['price_rating'].map(price_rating_dict)
df.head()


# ### ===== mileage ===== 

# In[248]:


df['mileage'].describe()


# In[249]:


pd.qcut(df['mileage'], q=15).value_counts()


# In[250]:


df['mileage_15'] = pd.qcut(df['mileage'], q=15, labels=np.arange(0, 15))


# ### ===== engine =====

# In[251]:


df['engine'].value_counts()


# In[252]:


df['engine_#'] = [item.split('-')[0] for item in df['engine']]
df.head()


# In[253]:


df['engine_#'].value_counts()


# In[254]:


df['electric'] = [1 if item == 'Electric' else 0 for item in df['engine_#']]


# In[255]:


df['turbo'] = [item.split()[-1] for item in df['engine']]


# In[256]:


df.loc[(df['turbo'] == 'Turbo') | (df['turbo'] == 'Supercharged'), 'turbo'] 


# In[257]:


df['turbo'] = [1 if item == 'Turbo' or item == 'Supercharged' else 0 for item in df['turbo']]


# ### ===== transmission =====

# In[258]:


df['transmission'].value_counts()


# In[259]:


df['trans_#'] = df['transmission'].str.extract('(\d+)')
df.loc[df['transmission'] == 'Single-Speed', 'trans_#'] = 1

# NOT SURE ON THIS
df.loc[df['trans_#'].isna(), 'trans_#'] = 0


# In[260]:


df['automatic'] = df['transmission'].str.contains('Automatic').astype(int)
df.head()


# ### ===== exterior =====

# In[261]:


df['exterior'].value_counts()


# In[262]:


df.loc[df['exterior'].str.contains('(\d+)'), 'exterior'] = 'N/A'


# In[263]:


df['color'] = df['exterior'].str.extract('(White|Black|Red|Orange|Yellow|Blue|Purple|Green|Silver|Burgundy|Gray|Grey|Brown|Pearl|Gold|Mocha|Nickel|Beige|Ingot|Tan|Tungsten|Granite|Metallic|Ebony)')
df['color'].value_counts()


# In[264]:


df['color'].fillna('N/A', inplace=True)


# ### ===== interior =====

# In[265]:


df['interior'].fillna('N/A', inplace=True)
df.loc[df['interior'].str.contains('(\d+)', regex=True), 'interior'] = 'N/A'


# In[266]:


# [item.split() for item in df['interior']]


# In[267]:


df['interior'] = df['interior'].str.extract('(White|Black|Red|Orange|Yellow|Blue|Purple|Green|Silver|Burgundy|Gray|Grey|Brown|Pearl|Gold|Mocha|Nickel|Beige|Ingot|Tan|Tungsten|Granite|Metallic|Sandstone|Ebony|Adobe)')
df['interior'].value_counts()


# In[268]:


df['interior'].fillna('N/A', inplace=True)


# ## ===== Exploratory Data Analysis =====

# In[269]:


df.columns.values


# In[270]:


plt.scatter(df['price'], df['mileage'])


# In[271]:


cols = ['year', 'brand', 'make_model', 'price', 'price_bin', 'price_rating', 'mileage', 'mileage_15', 'mpg_city', 'mpg_highway', 'combined_mpg', 'condition', 'accident', 'one_owner', 'drive_type', 'fuel_type', 'engine', 'engine_#', 'automatic', 'turbo', 'electric', 'transmission', 'trans_#', 'interior', 'exterior', 'color']
len(col1)


# In[272]:


df = df[cols]
df.head()


# In[ ]:




