#!/usr/bin/env python
# coding: utf-8

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

df = pd.read_csv('csv/kbb_scraper-2000.csv', header=None)
df = df.drop_duplicates()
df.head()

df.columns = ['title', 'price','owner', 'accident', 'price_rating', 'mileage', 'drive_type', 'engine', 'transmission', 'fuel_type', 'mpg', 'exterior', 'interior']
df.head()

df.shape
# df.to_csv('kbb_data_dupdropped2.csv')

df.isna().sum()

df.info()

# ## ===== (1) Cleaning Data =====

# Seeing where in the dataframe it contains 'MSRP'
mask = np.column_stack([df[col].str.contains('MSRP', na=False) for col in df])
df.loc[mask.any(axis=1)]

df['mileage'] = df['mileage'].replace('\,', '', regex=True).replace(letters, '', regex=True)
df['mileage'] = pd.to_numeric(df['mileage'])

df['price'] = df['price'].replace('\,', '', regex=True).replace(r'MSRP', '', regex=True)
df['price'] = pd.to_numeric(df['price'])

df.info()

# ### ===== mpg =====

# df['MPG (City)'] = df['']
for item in df['mpg'][:10]:
    print(item.split('/'))

df['mpg_city'] = [item.split('/')[0] for item in df['mpg']]
df['mpg_highway'] = [item.split('/')[1] if len(item.split('/')) == 2 else item.split('/')[0] for item in df['mpg']]

df['mpg_city'] = df['mpg_city'].str.replace('City', '').replace("'", '').map(lambda x: x.strip())
df['mpg_highway'] = df['mpg_highway'].str.replace('Highway', '').replace("'", '').map(lambda x: x.strip())

df['mpg_city'].unique()
# df['MPG (City)'].value_counts()

df['mpg_highway'].unique()
# df['MPG (Highway)'].value_counts()

df[df['mpg_city'].str.startswith('5')][:5]

len(df.iloc[16]['mpg'])

df.loc[df['mpg_city'].str.len() > 3, 'mpg_city'] = df['mpg_city'].str[0]
df.loc[df['mpg_highway'].str.len() > 3, 'mpg_highway'] = df['mpg_highway'].str[-1]

df['mpg_city'].replace(r'[a-zA-Z]', '0', regex=True, inplace=True)
df['mpg_highway'].replace(r'[a-zA-Z]', '0', regex=True, inplace=True)

df['mpg_city'] = df['mpg_city'].astype(str).replace(r'[^0-9]', '0', regex=True)
df['mpg_highway'] = df['mpg_highway'].astype(str).replace(r'[^0-9]', '0', regex=True)

df['mpg_city'] = pd.to_numeric(df['mpg_city'])
df['mpg_highway'] = pd.to_numeric(df['mpg_highway'])

df = df.drop('mpg', axis=1)

df.reset_index(drop=True, inplace=True)

# ### ===== fuel_type =====

df['fuel_type'].unique()

df[df['fuel_type'].str.contains('Miles')][:5]

df['fuel_type'] = df['fuel_type'].replace(r'[0-9]+\s(m|M)iles', 'N/A', regex=True)
df['fuel_type'].unique()

# pd.Categorical(df['Fuel Type'])
df['fuel_type'] = pd.factorize(df['fuel_type'])[0]
df['fuel_type'].unique()

# ### ===== drive_type =====

df['drive_type'].value_counts()

df['drive_type'] = df['drive_type'].replace('All wheel drive', '4 wheel drive').replace('4 wheel drive - rear wheel default', '4 wheel drive')
df['drive_type'].value_counts()

df[df['drive_type'] == 'Information Unavailable']

df['drive_type'] = df['drive_type'].replace('Information Unavailable', '2 wheel drive - rear')
df['drive_type'].value_counts()

df['drive_type'] = pd.factorize(df['drive_type'])[0]
df['drive_type'].value_counts()

df['engine'].value_counts()

df['transmission'].value_counts()

# ### ===== condition =====

df['combined_mpg'] = df['mpg_city'] + df['mpg_highway']
df['condition'] = [item.split()[0] for item in df['title']]
df.head()

df.info()

df['condition'].value_counts()

condition_dict = {'Used': 1, 'Certified': 2, 'New': 3}
df['condition'] = df['condition'].map(condition_dict)

df.head()

# ### ===== year, make_model =====

year = [item.split()[1] for item in df['title']]
isnum = [s for s in year if s.isdigit()]

len(year), len(isnum)

df['year'] = [item.split()[1] for item in df['title']]
df['year'] = pd.to_numeric(df['year'])
df.head()[:2]

df['brand'] = [item.split()[2] for item in df['title']]
df['brand'].value_counts()

df['brand'] = pd.factorize(df['brand'])[0]

df['make_model'] = [' '.join(item.split()[2:]) for item in df['title']]
df = df.drop('title', axis=1)
df.info()

df.isna().sum()

# ### ===== owner =====

df['owner'].value_counts()

df[df['owner'] == 'No Accident / Damage Reported']

df.loc[558, 'accident'] = 'No Accident / Damage Reported'
df.loc[558, 'owner'] = np.nan

df['owner'].value_counts()

df.drop(df[(df['owner'].isna()) & (df['combined_mpg'] < 11)].index, inplace=True)

df.loc[(df['owner'].isna()) & (df['mileage'] < 500), 'owner'] = 'One Owner'
df.loc[(df['owner'].isna()) & (df['mileage'] > 500), 'owner'] = 'Multiple Owners'
df.loc[(df['owner'].isna()) & (df['condition'] == 3), 'owner'] = 'One Owner'

owner_dict = {'Multiple Owners': 0, 'One Owner': 1}
df['one_owner'] = df['owner'].map(owner_dict)

# ### ===== price =====

df[df['price'].isna()]

# Got from KBB
df.loc[113, 'price'] = 20923
df.loc[604, 'price'] = 14936

df['price'].describe()

df['price'].plot.kde(color='red');

df['price'].kurtosis()

df[df['price'] > 30000].shape

bins = [0, 7000, 10000, 13000, 16000, 18000, 20000, 22000, 24000, 26000, 28000, 30000, 35000, 40000, 50000, 95000]
labels = np.arange(0, 15)
df['price_bin'] = pd.cut(df['price'], bins=bins, labels=labels)
df.head()

# ### ===== accident =====

df.loc[(df['accident'].isna()) & (df['mileage'] < 500), 'accident'] = 'No Accident / Damage Reported'
df.loc[(df['accident'].isna()) & (df['condition'] == 3), 'accident'] = 'No Accident / Damage Reported'
df.loc[(df['accident'].isna()) & (df['condition'] == 2), 'accident'] = 'No Accident / Damage Reported'
df.loc[(df['accident'].isna()) & (df['price'] > 10000), 'accident'] = 'No Accident / Damage Reported'
df.loc[(df['accident'].isna()) & (df['price'] < 10000), 'accident'] = 'Accident / Damage Reported'

df = df.drop('owner', axis=1)
df.head()

df['accident'].value_counts()

accident_dict = {'No Accident / Damage Reported': 1, 'Accident / Damage Reported': 0}

df['accident'] = df['accident'].map(accident_dict)

# ### ===== price_rating =====

df[df['price_rating'].isna()][:5]

df['price_rating'].fillna('N/A', inplace=True)

price_rating_dict = {'N/A': 0, 'GOOD': 1, 'GREAT': 2}
df['price_rating'] = df['price_rating'].map(price_rating_dict)
df.head()

# ### ===== mileage ===== 

df['mileage'].describe()

pd.qcut(df['mileage'], q=15).value_counts()

df['mileage_15'] = pd.qcut(df['mileage'], q=15, labels=np.arange(0, 15))

# ### ===== engine =====

df['engine'].value_counts()

df['engine_#'] = [item.split('-')[0] for item in df['engine']]
df.head()

df['engine_#'].value_counts()

df['electric'] = [1 if item == 'Electric' else 0 for item in df['engine_#']]

df['turbo'] = [item.split()[-1] for item in df['engine']]

df.loc[(df['turbo'] == 'Turbo') | (df['turbo'] == 'Supercharged'), 'turbo'] 

df['turbo'] = [1 if item == 'Turbo' or item == 'Supercharged' else 0 for item in df['turbo']]

# ### ===== transmission =====

df['transmission'].value_counts()

df['trans_#'] = df['transmission'].str.extract('(\d+)')
df.loc[df['transmission'] == 'Single-Speed', 'trans_#'] = 1

# NOT SURE ON THIS
df.loc[df['trans_#'].isna(), 'trans_#'] = 0

df['automatic'] = df['transmission'].str.contains('Automatic').astype(int)
df.head()

# ### ===== exterior =====

df['exterior'].value_counts()

df.loc[df['exterior'].str.contains('(\d+)'), 'exterior'] = 'N/A'

df['color'] = df['exterior'].str.extract('(White|Black|Red|Orange|Yellow|Blue|Purple|Green|Silver|Burgundy|Gray|Grey|Brown|Pearl|Gold|Mocha|Nickel|Beige|Ingot|Tan|Tungsten|Granite|Metallic|Ebony)')
df['color'].value_counts()

df['color'].fillna('N/A', inplace=True)

# ### ===== interior =====

df['interior'].fillna('N/A', inplace=True)
df.loc[df['interior'].str.contains('(\d+)', regex=True), 'interior'] = 'N/A'

# [item.split() for item in df['interior']]

df['interior'] = df['interior'].str.extract('(White|Black|Red|Orange|Yellow|Blue|Purple|Green|Silver|Burgundy|Gray|Grey|Brown|Pearl|Gold|Mocha|Nickel|Beige|Ingot|Tan|Tungsten|Granite|Metallic|Sandstone|Ebony|Adobe)')
df['interior'].value_counts()

df['interior'].fillna('N/A', inplace=True)

# ## ===== Exploratory Data Analysis =====

df.columns.values

plt.scatter(df['price'], df['mileage'])

cols = ['year', 'brand', 'make_model', 'price', 'price_bin', 'price_rating', 'mileage', 'mileage_15', 'mpg_city', 'mpg_highway', 'combined_mpg', 'condition', 'accident', 'one_owner', 'drive_type', 'fuel_type', 'engine', 'engine_#', 'automatic', 'turbo', 'electric', 'transmission', 'trans_#', 'interior', 'exterior', 'color']
len(col1)

df = df[cols]
df.head()
