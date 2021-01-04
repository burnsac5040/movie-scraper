#!/usr/bin/env python
# coding: utf-8

# In[8]:


import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import requests
from bs4 import BeautifulSoup as bs  
plt.style.use(['dark_background'])


# In[7]:


df = pd.read_csv('data/df-ohe.csv', index_col=0)
df.head()


# In[20]:


import pickle

with open('data/href2distro.pickle', 'rb') as f:
    href2distro = pickle.load(f)

distro2href = {b: a for a, b in href2distro.items()}


# In[25]:


dfs = pd.read_html('https://en.wikipedia.org/wiki/Comparison_of_Linux_distributions')

general = dfs[3]
technical = dfs[4]
architecture = dfs[5]
pack_manage = dfs[6]
live_media = dfs[7]


# In[ ]:




