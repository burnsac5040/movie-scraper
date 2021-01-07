#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
from bs4 import BeautifulSoup as bs
import re
import requests
from unicodedata import normalize
import torch
import torch.nn as nn

# %matplotlib inline # VSCode vs Jupyter Notebook
plt.style.use(['dark_background'])

# ## Scraping one movie

dfs = pd.read_html('https://en.wikipedia.org/wiki/Escape_Room_(film)', encoding='utf-8')
#dfs = pd.read_html('https://en.wikipedia.org/wiki/Godzilla:_King_of_the_Monsters_(2019_film)', encoding='utf-8')
df = dfs[0]
df

# ## Getting rid of special characters (e.g. '\xa0')

def clean_normalize_whitespace(x):
    if isinstance(x, str):
        return normalize('NFKC', x).strip()
    else:
        return x

df = df.applymap(clean_normalize_whitespace)

# ### Replacement and setting header

df.iloc[:, 1].replace(r'(\[\w\])', '', regex=True, inplace=True)
df.iloc[:, 1].replace(r'\$', '', regex=True, inplace=True)
header = df.T.iloc[0]
df = df.T.iloc[1:]
df.columns = header 

df.index = df.index.str.replace(r'\.1', '', regex=True)
df = df.reset_index().rename(columns={'index':'title'})
df.columns = df.columns.str.lower().str.replace(' ', '_')
df = df.drop('theatrical_release_poster', axis=1, errors='ignore')
df.drop(list(df.filter(regex='poster')), axis=1, inplace=True, errors='ignore')
df.columns.name = None

df.head()

# ### Mapping words to numbers

repl_dict = {'(T|t)housand': '*1e3', '(M|m)illion': '*1e6', '(B|b)illion': '*1e9'}
df['box_office'] = df['box_office'].replace(repl_dict, regex=True).replace(r' ', '', regex=True).map(pd.eval)

# ### Minimalist testing function

def clean_minimal(df):
    df = df.applymap(clean_normalize_whitespace)
    df.iloc[:, 1].replace(r'(\[\w\])', '', regex=True, inplace=True)
    df.iloc[:, 1].replace(r'\$', '', regex=True, inplace=True)
    header = df.T.iloc[0]
    df = df.T.iloc[1:]
    df.columns = header    

    df.index = df.index.str.replace(r'\.1', '', regex=True)
    df = df.reset_index().rename(columns={'index':'title'})
    df.columns = df.columns.str.lower().str.replace(' ', '_')
    df = df.drop('theatrical_release_poster', axis=1, errors='ignore')
    df.drop(list(df.filter(regex='poster')), axis=1, inplace=True, errors='ignore')
    df.columns.name = None

    return df

# ## Scraping Data from all Movies

url = 'https://en.wikipedia.org/wiki/List_of_American_films_of_2019'
r = requests.get(url)
soup = bs(r.text, 'html.parser')

def get_table(url):
    r = requests.get(url)
    soup = bs(r.content, 'html.parser')
    
    dfs = pd.read_html(url)
    df = clean_minimal(dfs[0])

    return df

get_table('https://en.wikipedia.org/wiki/Escape_Room_(film)')

hrefs = soup.select('tr td i a')
links = [href['href'] for href in hrefs]
base_path = 'https://en.wikipedia.org/'

link_test = links[:3]

movie_data_list = []
for idx, link in enumerate(links):
    if (idx+1) % 10 == 0:
        print(idx+1)
    try:
        full_path = base_path + link        
        movie_data_list.append(get_table(full_path))
        
    except Exception as e:
        print(str(e))

big_df = pd.concat(movie_data_list)
# Remove last 3 irrelevant columns
big_df = big_df.drop(big_df.columns[-3:], axis=1)
# Save to csv
big_df.to_csv('240-minus3.csv', index=False, header=big_df.columns.values)

# ### Read back in the saved csv

df = pd.read_csv('240-minus3.csv')
df.head()

# ### Function to clean whole datafrme

repl_dict = {'(T|t)housand': '*1e3', '(M|m)illion': '*1e6', '(B|b)illion': '*1e9'}

def clean_table(df):
    # Release date
    df['release_date'] = [item.split(')')[-1].strip() for item in df['release_date']]
    df.loc[df['release_date'].str.len() > 19, 'release_date'] = [re.split('\d{4}', item, 1)[-1] for item in df['release_date'] if len(item) > 19]
    df.loc[df['release_date'].str.len() > 19, 'release_date'] = [re.split('\d{4}', item, 1)[-1] for item in df['release_date'] if len(item) > 19]
    df['release_date'] = pd.to_datetime(df['release_date'])

    # Running time
    df = df[df['running_time'].notna()]
    df['running_time'] = [pd.to_numeric(item.split(' ')[0]) for item in df['running_time']]

    # Budget
    df = df[df['budget'].notna()]
    df['budget'].replace(r'€', '', regex=True, inplace=True)
    df['budget'] = [item.split('–')[-1] for item in df['budget']]
    df['budget'] = [item.split('-')[-1] for item in df['budget']]
    df['budget'].replace(r'000', 'thousand', regex=True, inplace=True)
    df['budget'] = df['budget'].replace(r'\([^)]*\)', '', regex=True)
    df['budget'] = df['budget'].replace(r'\[[^\]]*\]', '', regex=True)
    df['budget'] = df['budget'].replace(r'more than', '', regex=True)
    df['budget'] = [item.replace(',', ' ') for item in df['budget']]
    df['budget'] = df['budget'].replace(repl_dict, regex=True).replace(r' ', '', regex=True).map(pd.eval)

    # Box Office
    df = df[df['box_office'].notna()]
    df['box_office'] = [item.split('–')[-1] for item in df['box_office']]
    df['box_office'] = [item.split('-')[-1] for item in df['box_office']]
    df['box_office'] = df['box_office'].replace(r'\([^)]*\)', '', regex=True)
    df['box_office'] = df['box_office'].replace(r'\[[^\]]*\]', '', regex=True)
    df['box_office'] = df['box_office'].replace(r'at least', '', regex=True)
    df['box_office'] = [item.replace('+', '') for item in df['box_office']]
    df['box_office'] = [item.replace(',', '') for item in df['box_office']]
    df['box_office'] = df['box_office'].replace(repl_dict, regex=True).replace(r' ', '', regex=True).map(pd.eval)

    return df

df = clean_table(df)
df.head()

# ### Attempt to clean up `country`

# df = pd.read_csv('240-minus3.csv')
df['country'] = df['country'].fillna('NA')
# Split based on capital letters
st = [', '.join(re.findall('[a-zA-Z][^A-Z]*', str(item))) for item in df['country']]

us_list = []
uk_list = []
cr_list = []
nz_list = []

for item in st:
    if item.find('United') != -1 and item.find('States') != -1:
        us = ', '.join(item.replace('United', '').replace('States', 'USA').replace(',', '').split())
        us_list.append(us)
    if item.find('United') != -1 and item.find('Kingdom') != -1:
        uk = ', '.join(item.replace('United', '').replace('Kingdom', 'UK').replace(',', '').split())
        uk_list.append(uk)
    if item.find('Czech') != -1 and item.find('Republic') != -1:
        cr = ', '.join(item.replace('Czech', '').replace('Republic', 'Czech-Republic').replace(',', '').split())
        cr_list.append(cr)
    if item.find('New') != -1 and item.find('Zealand') != -1:
        nz = ', '.join(item.replace('New', '').replace('Zealand', 'New-Zealand').replace(',', '').split())
        nz_list.append(nz)

df.loc[df['country'].str.contains('States'), 'country'] = us_list
df.loc[df['country'].str.contains('Kingdom'), 'country'] = uk_list
df.loc[df['country'].str.contains('Czech'), 'country'] = cr_list
df.loc[df['country'].str.contains('Zealand'), 'country'] = nz_list

df['country']

# ## Exploratory Data Analysis

# df.to_csv('157-cleaned.csv', index=False, header=df.columns.values)

df.info()

# ### Budget vs Box office

plt.figure(figsize=(10,6), dpi=300)
plt.ticklabel_format(style='plain', axis='y')
plt.ticklabel_format(style='plain', axis='x')
sns.scatterplot(x='budget', y='box_office', data=df, color='#d43d17', alpha=0.86, edgecolor='#547BCA')
plt.xticks(rotation=-30)
plt.title('Budget vs Box Office');

plt.figure(figsize=(10,6), dpi=300)
plt.ticklabel_format(style='plain', axis='y')
plt.ticklabel_format(style='plain', axis='x')
sns.scatterplot(x='budget', y='box_office', data=df, color='#8878C3', linewidth=0.7, alpha=0.86, edgecolor='#547BCA')
plt.xticks(rotation=-30)
plt.title('Budget vs Box Office (Zoomed in)')
plt.xlim(0, 125000000)
plt.ylim(0, 500000000);

# ### Running time vs Box office

plt.figure(figsize=(10,6), dpi=300)
plt.ticklabel_format(style='plain', axis='y')
plt.ticklabel_format(style='plain', axis='x')
sns.scatterplot(x='running_time', y='box_office', data=df, color='#5FB97B', alpha=0.86, edgecolor='#72C9A6')
plt.xticks(rotation=-30)
plt.title('Running time vs Box Office');

plt.figure(figsize=(10,6), dpi=300)
plt.ticklabel_format(style='plain', axis='y')
plt.ticklabel_format(style='plain', axis='x')
sns.scatterplot(x='running_time', y='box_office', data=df, color='#FD7878', alpha=0.86, edgecolor='#F2848F')
plt.xticks(rotation=-30)
plt.title('Running time vs Box Office (Zoomed in)')
plt.xlim(80, 180)
plt.ylim(0, 500000000);

# ## Running time vs Budget

plt.figure(figsize=(10,6), dpi=300)
plt.ticklabel_format(style='plain', axis='y')
plt.ticklabel_format(style='plain', axis='x')
sns.scatterplot(x='running_time', y='budget', data=df, color='#D4CF72', alpha=0.86, edgecolor='#D4A472')
plt.xticks(rotation=-30)
plt.title('Running time vs Budget');

plt.figure(figsize=(10,6), dpi=300)
plt.ticklabel_format(style='plain', axis='y')
plt.ticklabel_format(style='plain', axis='x')
sns.scatterplot(x='running_time', y='budget', data=df, color='#C993BF', alpha=0.86, edgecolor='#C993BF')
plt.xticks(rotation=-30)
plt.title('Running time vs Budget (Zommed in)')
plt.xlim(80, 180)
plt.ylim(0, 210000000);

# ### Distributions

plt.figure(figsize=(10,6), dpi=300)
plt.ticklabel_format(style='plain', axis='x')
sns.distplot(df['budget'], hist=False, kde_kws={'shade': True}, color='#5689F7', rug=True, label='budget')
sns.distplot(df['box_office'], hist=False, kde_kws={'shade': True}, color='#FD7878', rug=True, label='box_office')
plt.xlim(0, 600000000)
plt.xlabel('Dollar Amount (USD)')
plt.title('Budget and Box Office Distribution')
plt.legend();

plt.figure(figsize=(10,6), dpi=300)
plt.ticklabel_format(style='plain', axis='x')
sns.histplot(df[['budget', 'box_office']], multiple='stack', bins=20, kde=False, palette='Dark2', legend=True, element='bars', stat='count')
plt.xlabel('Dollar Amount (USD)')
plt.title('Budget and Box Office Distribution');

plt.figure(figsize=(10,6), dpi=300)
plt.ticklabel_format(style='plain', axis='x')
sns.histplot(df[['budget', 'box_office']], multiple='stack', bins=50, kde=False, palette='Dark2', legend=True, element='poly', stat='count')
plt.xlim(20000000, 600000000)
plt.xlabel('Dollar Amount (USD)')
plt.title('Budget and Box Office Distribution');

# ## Machine Learning

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn

class LinearRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)

df['budget'] = df['budget'].fillna(lambda x: x.median())
df['box_office'] = df['box_office'].fillna(lambda x: x.median())

X = np.array(df[['budget']])
y = np.array(df['box_office'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=14)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

X_train = torch.from_numpy(X_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.float32))

y_train = y_train.view(y_train.shape[0], 1)
y_test = y_test.view(y_test.shape[0], 1)

n_samples, n_features = X.shape

input_dim = n_features
output_dim = 1

model = LinearRegression(input_dim, output_dim)
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

num_epochs = 100

for epoch in range(num_epochs):
    y_pred = model(X_train)
    loss = criterion(y_pred, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch: {epoch+1} // Loss: {loss}')

print(f'Mean Squared Error: {mean_squared_error(y_test, predicted)}')
print(f'R2 Score: {r2_score(y_test, predicted)*100:.3f}%')

predicted = model(X_test).detach().numpy()
plt.figure(figsize=(10, 6), dpi=300)
plt.ticklabel_format(style='plain', axis='y')
plt.plot(X_test, y_test, 'o', color='#5DB26D')
plt.plot(X_test, predicted, color='#FD7878');

# ## Turning into a Classification

df = df[['title', 'release_date', 'running_time', 'budget', 'box_office']].copy()
df.head()

df['profit'] = 0
df.loc[df['budget'] < df['box_office'], 'profit'] = 1
df['profit'].value_counts()

# ### Logistic Regression

from sklearn.metrics import classification_report

X, y = np.array(df[['running_time', 'budget']]), np.array(df['profit'])

n_samples, n_features = X.shape

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

X_train = torch.from_numpy(X_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.float32))

y_train = y_train.view(y_train.shape[0], 1)
y_test = y_test.view(y_test.shape[0], 1)

class LogisticRegression(nn.Module):
    def __init__(self, n_input_features):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(n_input_features, 1)

    def forward(self, x):
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred

model = LogisticRegression(n_features)

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.2)

num_epochs = 100
for epoch in range(num_epochs):
    # forward pass and loss
    y_pred = model(X_train)
    loss = criterion(y_pred, y_train)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    if (epoch + 1) % 10 == 0:
        print(f'epoch: {epoch+1}, loss = {loss.item():.4f}')

with torch.no_grad():
    y_pred = model(X_test)
    y_pred_class = y_pred.round()
    acc = y_pred_class.eq(y_test).sum() / float(y_test.shape[0])
    print(f'accuracy = {acc*100:.4f}%')
    print(f'\n Classification Report: {classification_report(y_test, y_pred_class)}')
