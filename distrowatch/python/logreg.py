#!/usr/bin/env python
# coding: utf-8

# ----
# ----
# # Minor Exploratory Data Analysis
# ----
# ----

# In[1]:


import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns

# %matplotlib inline # (VSCode doesn't need this)
plt.style.use(['dark_background'])
plt.rcParams['figure.dpi'] = 300


# In[2]:


df = pd.read_csv('data/df-ohe.csv', index_col=0).sample(frac=1)
df


# In[3]:


fig, axes = plt.subplots(2, 2, sharex=True, sharey=True)
plt.yticks(np.arange(0, 4001, 1000))

axes = axes.ravel()
cols = ['1_month', '3_months', '6_months', '12_months']
colors = iter(['rosybrown', 'darkorange', 'mediumvioletred', 'blueviolet'])

for i, col in enumerate(cols):
    axes[i].scatter(x=df['ranks'], y=df[col], c=next(colors), ec='white', lw=0.5, alpha=0.7)
    axes[i].set(title=col)

plt.tight_layout()


# ### Distribution of Ratings

# In[4]:


df['rating_bin'] = pd.cut(df['rating'], bins=np.arange(0, 11), right=True)

sns.countplot(df['rating_bin']);


# In[5]:


fig, axes = plt.subplots(2, 2, sharex=True)

axes = axes.ravel()
cols = ['1_month', '3_months', '6_months', '12_months']
colors = iter(['PuBu', 'RdPu', 'PuRd', 'BuPu'])

for i, col in enumerate(cols):
    sns.barplot(ax=axes[i], x=df['rating_bin'], y=df[col], 
                                    palette=next(colors)).set(ylabel=None, title=col)
    axes[i].tick_params('x', labelrotation=45)

fig.suptitle('Average Page Views per Bin vs Rating')
plt.tight_layout()


# In[6]:


df.loc[df['rating_bin'].apply(lambda x: x.overlaps(pd.Interval(8, 9))), '1_month']


# In[7]:


from collections import defaultdict

inter = np.arange(0, 11)
cols = ['1_month', '3_months', '6_months', '12_months']
max = {}
min = {}

for col in cols:
    for i in np.arange(0, 10):
        max[f'{col}: ({i}, {i+1})'] = df.loc[df['rating_bin'].apply(lambda x: x.overlaps(pd.Interval(i, i+1))), col].max()
        min[f'{col}: ({i}, {i+1})'] = df.loc[df['rating_bin'].apply(lambda x: x.overlaps(pd.Interval(i, i+1))), col].min()

max_d = defaultdict(dict)
for mb, v in max.items():
    m, b = mb.split(':')
    max_d[m][b.strip()] = v
    
min_d = defaultdict(dict)
for mb, v in min.items():
    m, b = mb.split(':')
    min_d[m][b.strip()] = v

max_d['1_month']['(0, 1)']


# In[8]:


max_d


# In[9]:


# sns.barplot(x=list(max_d['1_month'].keys()), y=list(max_d['1_month'].values()))

fig, axes = plt.subplots(2, 2, sharex=True)

axes = axes.ravel()
cols = ['1_month', '3_months', '6_months', '12_months']
colors = iter(['twilight', 'twilight_shifted', 'tab20', 'gnuplot'])

for i, col in enumerate(max_d.keys()):
    sns.barplot(ax=axes[i], x=list(max_d[col].keys()), y=list(max_d[col].values()), palette=next(colors)).set(title=col)
    axes[i].tick_params('x', labelrotation=45)
    axes[i].set_yticks(np.arange(0, 4001, 1000))
    axes[i].grid(b=True, which='both', axis='y', c='#ebdbb2', alpha=0.35, ls=':')

fig.suptitle('Max Page Views per Bin vs Rating')
plt.tight_layout()


# ## Machine Learning

# In[10]:


df.head()


# In[11]:


# df['fav'] = np.where((df['rating'] > 7) & (df['ranks'] < 85), 1, 0)

df['fav'] = np.where(df['rating'] > 8.5, 1, 0)

df = df.drop(['rating_bin', 'rating'], axis=1)

indices = list(range(len(df.index)))
np.random.shuffle(indices)

x_train_indices, x_test_indices = np.split(indices, [int(0.8*len(indices))])

df.head()
X_train = df.iloc[x_train_indices, 1:-1]
X_test = df.iloc[x_test_indices, 1:-1]
y_train = df.iloc[x_train_indices, -1]
y_test = df.iloc[x_test_indices, -1]


# In[12]:


print('X_train: %s, X_test: %s \n y_train: %s, y_test: %s' % (X_train.shape, X_test.shape, y_train.shape, y_test.shape))


# In[13]:


from sklearn.preprocessing import StandardScaler

ss = StandardScaler()
ss.fit(X_train)
X_train = ss.transform(X_train)
X_test = ss.transform(X_test)


# In[14]:


# Will attempt to do this manually at some point
def col_mean(df):
    rows, cols = df.shape
    means = np.zeros(cols).tolist()
    for i in range(cols):
        means[i] = sum(df.iloc[:, i].values) / rows
    return means

def col_std(df, col_means):
    rows, cols = df.shape
    stds = np.zeros(cols).tolist()
    for i in range(cols):
        stds[i] = ((df.iloc[:, i].values - means[i])**2).sum()
    return [np.sqrt(el/rows-1) for el in stds]


# In[15]:


from timeit import default_timer as t

class LogisticRegression:
    def __init__(self, lr=0.001, n_iters=1000, bias=True):
        self.lr = lr
        self.n_iters = n_iters
        self.w = None
        self.b = bias

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
        
    def fit(self, X, y):
        if isinstance(self.b, bool):
            X = np.hstack([np.ones([X.shape[0], 1]), X])

        n_samp, n_feat = X.shape
        self.w = np.random.rand(n_feat)
        
        start = t()
        for _ in range(self.n_iters):
            z = np.dot(X, self.w)
            y_pred = self._sigmoid(z)
            dw = (1 / n_samp) * (X.T @ (y_pred - y))
            self.w -= self.lr * dw
            
        end = t()
        print(f'Elapsed time: {end - start:.4f}')
            
    def predict_proba(self, X, threshold=0.5):
        if isinstance(self.b, bool):
            X = np.hstack([np.ones([X.shape[0], 1]), X])

        probs = self._sigmoid(X @ self.w)
        clss = probs >= threshold
        return probs, clss


# In[16]:


np.random.seed(5040)

lr = LogisticRegression(n_iters=10000, bias=True)
lr.fit(X_train, y_train)
probs, pred = lr.predict_proba(X_test)


# In[17]:


def accuracy(y_true, y_pred):
    acc = np.sum(y_true == y_pred) / len(y_true)
    return round(acc, 3)

print(f'Accuracy: {accuracy(y_test, pred)}\nClass 1 count: {y_test.sum()} Class 0 count: {y_test.size - y_test.sum()}')


# ----
# ### Dataframe of Probabilities and Thresholds
# ----

# In[25]:


thresh_df = pd.DataFrame(probs, columns=['probs'])
thresh_df['true'] = y_test.tolist()

for i in np.arange(.10, 1, .05):
    thresh_df[f'thresh_{i:.2f}'] = np.where(thresh_df['probs']>=i, 1, 0)

thresh_df.head()


# In[19]:


for col in thresh_df.iloc[:, 2:]:
    print(f'{col}: {thresh_df[thresh_df[col] == 1].shape[0]}/{y_test.sum()}')

