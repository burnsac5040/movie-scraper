# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.9.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import numpy as np 
import pandas as pd
import time
import re

# ### Importing the Data

# +
df_main = pd.read_csv('df/df_main.csv', index_col=0)
df_ind = pd.read_csv('df/df_ind.csv', index_col=0)

df = pd.merge(df_main, df_ind, left_index=True, right_index=True)
df.to_csv('df/df_tot.csv', index=True, columns=df.columns.values)
# -

df.sample(5)

# ----
# ### Region
# The ones that I entered in and specifically scraped for:
# - Saint Louis
# - Kansas City
# - Lees Summit
# - Springfield
# - Columbia
#
# Below is what I am reclassifying cities to be in line with the 5 cities above.
#
# I looked each one up on Google.
#
# ----

# ```
# Raytown                74 = Kansas City
# Parkville              28 = Kansas City
# Florissant             20 = Saint Louis
# Gladstone              13 = Kansas City
# Independence           12 = Kansas City
# Ashland                10 = Columbia
# Riverside              10 = Kansas City
# Nixa                   7 = Springfield
# Liberty                7 = Kansas City
# Greenwood              6 = Kansas City
# Centralia              5 = Columbia
# Lake Waukomis          5 = Kansas City
# Weatherby Lake         4 = Kansas City
# Kirkwood               3 = Saint Louis
# Fulton                 3 = Columbia
# Republic               3 = Springfield
# Strafford              3 = Springfield
# Kearney                3 = Lees Summit
# Hallsville             3 = Columbia
# Platte Woods           2 = Kansas City
# Town And Country       2 = Saint Louis
# Black Jack             2 = Saint Louis
# Brookline              2 = Springfield
# Grandview              2 = Kansas City
# New Bloomfield         2 = Columbia
# Platte City            2 = Kansas City
# California             1 = Columbia
# Brighton               1 = Springfield
# Fair Grove             1 = Springfield
# Hartsburg              1 = Columbia
# Berkeley               1 = Saint Louis
# Hazelwood              1 = Saint Louis
# Ozark                  1 = Springfield
# Willard                1 = Springfield
# Auxvasse               1 = Columbia
# Webster Groves         1 = Saint Louis
# Harrisburg             1 = Columbia
# Rocheport              1 = Columbia
# Kingdom City           1 = Columbia
# Blue Springs           1 = Kansas City
# ```

# +
df['region'] = df['region'].replace(', MO', '', regex=True)

# Other
df.loc[df['region'].str.contains('Independence'), 'region'] = 'Independence'
df.loc[df['region'].str.contains('Raytown'), 'region'] = 'Raytown'
# Saint Louis
df.loc[df['region'].str.contains('Saint Louis|St Louis'), 'region'] = 'Saint Louis'
df.loc[df['region'].str.contains('Florissant|Kirkwood|Town And Country|Black Jack|Berkeley|Hazelwood|Webster Groves'), 'region'] = 'Saint Louis'
# Kansas City
df.loc[df['region'].str.contains('Kansas City'), 'region'] = 'Kansas City'
df.loc[df['region'].str.contains('Raytown|Parkville|Independence|Gladstone|Riverside|Liberty|Greenwood|Lake Waukomis|Weatherby Lake|Platte Woods|Grandview|Platte City|Blue Springs'), 'region'] = 'Kansas City'
# Lees Summit
df.loc[df['region'].str.contains("Lees Summit|Lee's Summit"), 'region'] = 'Lees Summit'
df.loc[df['region'].str.contains('Kearney'), 'region'] = 'Lees Summit'
# Springfiled
df.loc[df['region'].str.contains('Springfield'), 'region'] = 'Springfield'
df.loc[df['region'].str.contains('Nixa|Republic|Strafford|Brookline|Brighton|Fair Grove|Ozark|Willard'), 'region'] = 'Springfield'
# Columbia
df.loc[df['region'].str.contains('Columbia'), 'region'] = 'Columbia'
df.loc[df['region'].str.contains('Ashland|Centralia|Fulton|Hallsville|New Bloomfield|California|Hartsburg|Auxvasse|Harrisburg|Rocheport|Kingdom City'), 'region'] = 'Columbia'

# df['region'] = df['region'].astype('category').cat.codes
# -

df['region'].value_counts()

# ----
# ### New, Price
# ----

# +
df.loc[df['new'].str.contains('NEW|OWNER|COMING', na=False), 'new'] = 1
df.loc[df['new'].str.contains('BANK|OPEN|AUCTION|OLD', na=False), 'new'] = 0
df['new'] = pd.to_numeric(df['new'])

df['price'] = df['price'].replace(r'\$|,', '', regex=True)
df['price'] = pd.to_numeric(df['price'].replace(r'\W', np.nan, regex=True), errors='coerce')
df = df[df['price'].notnull()]
# -

# ----
# ### Bedrooms, Bathrooms, Square Footage
# ----

df['bedrm'] = pd.to_numeric(df['bedrm'].replace(r'bd', '', regex=True))
df['bth'] = pd.to_numeric(df['bth'].replace(r'ba|(\.\d)?', '', regex=True))
df['sqft'] = pd.to_numeric(df['sqft'].replace(r'sqft|,', '', regex=True))

# ----
# ### Crime
# ----

# +
df['crime'] = df['crime'].replace(r'Crime', '', regex=True) 
# Some do not have a rating and instead have 'Learn about crime in this area'
print(df.loc[df['crime'].str.contains('Learn'), 'crime'][0])

df['crime'] = [x.split()[0] for x in df['crime']]
# So I will replace 'Learn about crime...' with 'Low'
df.loc[df['crime'].str.contains('Learn'), 'crime'] = 'Low'
# -

df['crime'].value_counts()

# +
crime_d = {'Lowest': 1, 'Low': 2, 'Moderate': 3, 'High': 4, 'Highest': 5}

df['crime'] = df['crime'].map(crime_d)
# -

# ----
# ### Schools
# ----

# +
from textwrap import wrap

df['schools'] = df['schools'].replace(r'Schools', ' ', regex=True)

df['schools'] = df['schools'].replace(r'Elementary (School)?', 'E', regex=True)
df['schools'] = df['schools'].replace(r'Middle (School)?', 'M', regex=True)
df['schools'] = df['schools'].replace(r'High (School)?', 'H', regex=True)

school = [wrap(''.join(x.split()).strip(), 2) for x in df['schools']]
school[0]

# + tags=[]
for el in school:
    try: 
        re.search('E', ''.join(el)).group()
    except:
        el.insert(0, '0E')

    try: 
        re.search('M', ''.join(el)).group()
    except:
        el.insert(1, '0M')

    try: 
        re.search('H', ''.join(el)).group()
    except:
        el.insert(2, '0H')


df['eschl'] = [x[0] for x in school]
df['eschl'] = pd.to_numeric(df['eschl'].replace(r'E', '', regex=True))

df['mschl'] = [x[1] for x in school]
df['mschl'] = pd.to_numeric(df['mschl'].replace(r'M', '', regex=True))

df['hschl'] = [x[2] for x in school]
df['hschl'] = pd.to_numeric(df['hschl'].replace(r'H', '', regex=True))

df = df.drop('schools', axis=1)
# -

# ----
# ### Details
#
# ```json
# {
#   "Basement": "boolean",
#   "Heating": ["Forced Air", "Gas", "Electric", "Solar", "Heat", "Pump", "Radiant", "Oil"],
#   "Days on Market": "47 Days on Trulia",
#   "Year Built": "1951",
#   "Year Updated": "2008",
#   "Stories": "2",
#   "Property Type": "Single Family Home",
#   "Number of Rooms": "6",
#   "Cooling System": "Central",
#   "Assigned Parking Space": "Garage",
#   "Architecture": "Ranch / Rambler",
#   "Price Per Sqft": "$135",
#   "MLS/Source ID": "396348",
#   "Roof": "Composition",
#   "Floors": "Carpet",
#   "Exterior": "Stucco",
#   "Foundation Type": "Stone",
#   "Lot Size": "0.77 acres",
#   "Parking Spaces": "2",
#   "Parking": "Garage Garage Attached",
#   "HOA Fee": "$400/monthly",
#   "Security System": "bool"
# }
#
#   ```
#
#   ----

# +
import ast
from operator import itemgetter

# Create a column that is made of lists
df['details_l'] = [ast.literal_eval(x) for x in df['details']]
df['details'] = df['details'].replace(r"'|\[|\]", '', regex=True)
df['details'] = df['details'].replace(r'\,', '', regex=True)

# Create an actual list of lists
details = [x for x in df['details_l']]
len_ = [(len(x), idx) for idx, x in enumerate(details)]
# Finding the largest list to possibly find more features
max(len_, key=itemgetter(0))

# +
### Basement ### 
df['bsmnt'] = np.where(df['details'].str.contains('Basement', case=False), 1, 0)

### Heating ###
df['gas'] = np.where(df['details'].str.contains('Gas', case=False), 1, 0)
df['forsdair'] = np.where(df['details'].str.contains('Forced Air', case=False), 1, 0)
df['elctric'] = np.where(df['details'].str.contains('Electric', case=False), 1, 0)
df['solar'] = np.where(df['details'].str.contains('Solar', case=False), 1, 0)

### Days on Market ###
df['dayonmark'] = pd.to_numeric(df['details'].str.extract('(Days on Market:)\s(\d*)')[1])
df['dayonmark'] = df['dayonmark'].fillna(1)

### Year Built ###
df['yrbuilt'] = df['details'].str.extract('(Year Built:)\s(\d*)')[1]
df.loc[(df['yrbuilt'].isna()) & (df['new'] == 1), 'yrbuilt'] = round(df.loc[df['new'] == 1, 'yrbuilt'].astype(float).median(), 0)
df.loc[(df['yrbuilt'].isna()) & (df['price'].gt(400000)), 'yrbuilt'] = round(df.loc[df['price'].gt(400000), 'yrbuilt'].astype(float).median(), 0)
df.loc[(df['yrbuilt'].isna()) & (df['price'].between(300000, 400000)), 'yrbuilt'] = round(df.loc[df['price'].between(300000, 400000), 'yrbuilt'].astype(float).median(), 0)
df['yrbuilt'] = df['yrbuilt'].fillna(round(df['yrbuilt'].astype(float).mean(), 0))
df['yrbuilt'] = pd.to_numeric(df['yrbuilt'])

### Year Updated ###
df['update'] = np.where(df['details'].str.contains('Year Updated', case=False), 1, 0)

### Stories ###
df['stories'] = df['details'].str.extract('(Stories:)\s(\d*)')[1]
# Average price of a Missouri home from Zillow: $176,609
df.loc[(df['price'].lt(176609)) & (df['stories'].isna()) & (df['bsmnt'] == 0), 'stories'] = 1
df.loc[(df['price'].gt(176609)) & (df['stories'].isna()) & (df['bsmnt'] == 1), 'stories'] = 2
# Average square footage of a one story home from finanicalsamurai.com: 2422 sq. ft.
df.loc[(df['stories'].isna()) & (df['sqft'].lt(1816)), 'stories'] = 1 # 2422 - (.25 * 2422)
df.loc[(df['stories'].isna()) & (df['sqft'].between(1816, 3027)), 'stories'] = 2 # 2422 + (.25 * 2422)
df.loc[(df['sqft'].ge(3027)) & (df['stories'].isna()), 'stories'] = 3
df['stories'] = pd.to_numeric(df['stories'])

### Property Type ###
# df['details'].str.extract('(Property Type:)\s(\w*)(\s?\w*)')[1].unique() # All are the same

### Number of Rooms ###
df['n_rooms'] = df['details'].str.extract('(Number of Rooms:)\s(\d*)')[1]
df['n_rooms'] = df['n_rooms'].fillna(df[['bedrm', 'bth']].sum(axis=1))
df['n_rooms'] = pd.to_numeric(df['n_rooms'])

### Cooling System ###
# df['details'].str.extract('(Cooling System:)\s(\w*)')[1].value_counts() # Not important

### Garage ###
df['garage'] = np.where(df['details'].str.contains('Garage', case=False), 1, 0)

### Architecture ###
# df['details'].str.extract('(Architecture:)\s(\w*)')[1].isna().sum() # Too many nan's & too many uniques

### Price Per Square Foot
df['pp_sqft'] = pd.to_numeric(df['details'].str.extract('(Sqft:)\s(\$\d*)')[1].replace(r'\$', '', regex=True))
df['pp_sqft'] = df['pp_sqft'].fillna(df['pp_sqft'].median())

### Roof ###
# df['details'].str.extract('(Roof:)\s(\w*)')[1].value_counts()
df['roof'] = df['details'].str.extract('(Roof:)\s(\w*)')[1].fillna('Composition').astype('category').cat.codes

### Floors ###
# df['details'].str.extract('(Floors:)\s(\w*)')[1].value_counts()
df['floors'] = df['details'].str.extract('(Floors:)\s(\w*)')[1]
df['floors'] = df['floors'].apply(lambda x: np.random.choice(df['floors'].dropna().values) if pd.isna(x) else x)
df['floors'] = df['floors'].astype('category').cat.codes

### Exterior ###
# df['details'].str.extract('(Exterior:)\s(\w*)')[1].replace(r'Composition', 'Wood', regex=True).value_counts()
df['exterior'] = df['details'].str.extract('(Exterior:)\s(\w*)')[1].replace(r'Composition', 'Wood', regex=True)
df['exterior'] = df['exterior'].apply(lambda x: np.random.choice(df['exterior'].dropna().values) if pd.isna(x) else x).astype('category').cat.codes

### Foundation Type ###
# df['details'].str.extract('(Foundation Type:)\s(\w*)')[1].value_counts() 

### Lot Size ###
df['lot_sz'] = pd.to_numeric(df['details'].str.extract('(Lot Size:)\s(\w*)(\s?\w*)')[1]).fillna(df['sqft'])
# df['lot_sz'] = np.where(df['lot_sz'] == 0, df['sqft'], df['lot_sz'])
df['lot_sz'][df['lot_sz'] == 0] = df['sqft']

### Parking Spaces ###
df['prk_spc'] = df['details'].str.extract('(Parking Spaces:)\s(\w*)')[1].astype(float)
df['prk_spc'] = pd.to_numeric(df['prk_spc'].fillna(round(df['prk_spc'].mean(), 2)))

### Parking Area ###
# df[(df['parking'].isna()) & (df['garage'] == 1)] # If you run this before filling na, it = None
df['parking'] = df['details'].str.extract('(Parking:)\s(\w*)')[1].replace(r'None', 'Off', regex=True).replace(r'Built|On', 'Garage', regex=True).fillna('Off')
park_dct = {'Off': 0, 'Underground': 1, 'Carport': 2, 'Garage': 3}
df['parking'] = df['parking'].map(park_dct)

### HOA Fee ###
df['hoa_fee'] = pd.to_numeric(df['details'].str.extract('(HOA Fee:)\s\$(\d*)')[1].fillna(0))

### Security System ###
df['sec_sys'] = (df['details'].str.contains('Security System', case=False)).astype(int)

### Pool ### 
df['pool'] = (df['details'].str.contains('Pool', case=False)).astype(int)

df = df.drop(['details', 'details_l'], axis=1)
# -

# ----
# ### Listing History
# ----

# +
# df['list_hist'] = df['list_hist'].replace(r"[\[\]\'\,a-zA-JLN-Z]", '', regex=True).replace(r'(\d+\/\d+\/\d+)', '', regex=True)
# df['list_hist'] = df['list_hist'].fillna(1).replace(r'\/', '', regex=True)
# df['list_cnt'] = [str(l).strip().split() for l in df['list_hist']]

# Another (much much simpler) way that I don't know why I didn't think of first
df['list_hist'] = df['list_hist'].fillna(1)
df['list_cnt'] = [str(x).count('$') for x in df['list_hist']]

df = df.drop('list_hist', axis=1)
# -

# ----
# ### Tax Assessment
# - 1 if assessment is greater than the price listing
#
# ----

# +
df['tax'] = pd.to_numeric(df['tax'].replace(r"\[|\]|\'|,", '', regex=True).str.extract(r'(Assessment\$(\d*))')[1])
df['tax'] = df['tax'].fillna(df['price'])
df['assess'] = (df['tax'] > df['price']).astype(int)

df = df.drop('tax', axis=1)
# -

# ----
# ### Typical Home Value of Similar Houses
# ----

df['typ_val'] = pd.to_numeric(df['typ_val'].replace(r'\D', '', regex=True))
df['typ_val'] = df['typ_val'].fillna(df['price'])

df['typ_val']

# #### How Price of House relates to other Houses (above or below)
#
# ################################################################################################################
# - EITHER DROP THIS OR TYP_VAL
#
# ################################################################################################################

# + tags=[]
df['val_pct'] = [f'-{el}' if 'below' in el else el for el in df['val_pct']]
df['val_pct'] = df['val_pct'].replace(r'above|below|%|,', '', regex=True).replace(r'[a-zA-Z]', '', regex=True)
df['val_pct'] = [''.join(x.replace(' ', '')) for x in df['val_pct']]

df['val_pct'] = pd.to_numeric(df['val_pct'])

df.loc[(df['val_pct'].isna()) & (df['price'].lt(100000)), 'val_pct'] = df.loc[(df['price'].lt(100000)), 'val_pct'].mean()
df.loc[(df['val_pct'].isna()) & (df['price'].between(100000, 200000)), 'val_pct'] = df.loc[(df['price'].between(100000, 200000)), 'val_pct'].mean()
df.loc[(df['val_pct'].isna()) & (df['price'].between(200000, 300000)), 'val_pct'] = df.loc[(df['price'].between(200000, 300000)), 'val_pct'].mean()
df.loc[(df['val_pct'].isna()) & (df['price'].between(300000, 400000)), 'val_pct'] = df.loc[(df['price'].between(300000, 400000)), 'val_pct'].mean()
df.loc[(df['val_pct'].isna()) & (df['price'].gt(400000)), 'val_pct'] = df.loc[(df['price'].gt(400000)), 'val_pct'].mean()
# -

# ----
# ### Typical Square Footage Price of Similar Houses
# ----

# +
df['typ_sqft'] = pd.to_numeric(df['typ_sqft'].replace(r'\$|\D', '', regex=True))

df.loc[(df['typ_sqft'].isna()) & (df['sqft'].lt(1000)), 'typ_sqft'] =  df.loc[df['sqft'].lt(1000), 'sqft'].mean()
df.loc[(df['typ_sqft'].isna()) & (df['sqft'].between(1000, 2000)), 'typ_sqft'] =  df.loc[df['sqft'].between(1000, 2000), 'sqft'].mean()
df.loc[(df['typ_sqft'].isna()) & (df['sqft'].between(2000, 3000)), 'typ_sqft'] =  df.loc[df['sqft'].between(2000, 3000), 'sqft'].mean()
# -

# #### How Price of a Square Foot relates to other Houses (above or below)
#
# ################################################################################################################
# - EITHER DROP THIS OR TYP_SQFT
#
# ################################################################################################################

# +
df['sqft_pct'] = [f'-{el}' if 'below' in el else el for el in df['sqft_pct']]
df['sqft_pct'] = df['sqft_pct'].replace(r'above|below|%|,', '', regex=True).replace(r'[a-zA-Z]', '', regex=True)
df['sqft_pct'] = [''.join(x.replace(' ', '')) for x in df['sqft_pct']]

df['sqft_pct'] = pd.to_numeric(df['sqft_pct'])

df.loc[(df['sqft_pct'].isna()) & (df['price'].lt(100000)), 'sqft_pct'] = df.loc[df['sqft_pct'].lt(100000), 'sqft_pct'].mean()
df.loc[(df['sqft_pct'].isna()) & (df['price'].between(100000, 300000)), 'sqft_pct'] = df.loc[df['sqft_pct'].between(100000, 300000), 'sqft_pct'].mean()
# -

df.info()

df = df.drop_duplicates()
df.to_csv('df/df_full.csv', index=True, columns=df.columns.values)
