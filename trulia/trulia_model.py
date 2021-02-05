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

# +
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns

plt.rcParams['figure.dpi'] = 200
plt.style.use(['dark_background'])

# +
from sklearn.model_selection import train_test_split

df = pd.read_csv('df/df_full.csv', index_col=0)
# Drop these because I think they give too much data
df = df.drop(['typ_val', 'typ_sqft', 'pp_sqft'], axis=1)
df['region'] = df['region'].astype('category').cat.codes

# df = df.drop(['val_pct', 'sqft_pct', 'eschl', 'mschl', 'hschl', 'gas', 'forsdair', 'elctric', 'solar', 'roof', 'floors', 'exterior'], axis=1)

# X = df[['new', 'bedrm', 'bth', 'sqft']]
X = df.drop('price', axis=1)
y = df['price']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# +
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler

# ss = StandardScaler()
# ss.fit(X_train)
# X_train = ss.transform(X_train)
# X_test = ss.transform(X_test)

regr = LinearRegression()
regr.fit(X_train, y_train)

y_pred = regr.predict(X_test)
# -

regr.coef_

print(f'MSE: {mean_squared_error(y_test, y_pred)}')
print(f'R2: {r2_score(y_test, y_pred)}')

df.columns.values
