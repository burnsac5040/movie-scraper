#!/usr/bin/env python
# coding: utf-8

# In[1]:


from bs4 import BeautifulSoup as bs
import requests

import numpy as np 
import pandas as pd 
import time
import ssl
import re
import csv
import datetime


# In[33]:


base_url = 'https://www.trulia.com'
url = 'https://www.trulia.com/for_sale/Columbia,MO/1p_beds/SINGLE-FAMILY_HOME_type/'
headers = {
    'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
    'accept-encoding': 'gzip, deflate, br',
    'accept-language': 'en-US,en;q=0.9,es;q=0.8',
    #'cache-control': 'max-age=0',
    'dnt': '1',
    'sec-fetch-dest': 'document',
    'sec-fetch-mode': 'navigate',
    'sec-fetch-site': 'cross-site',
    'sec-fetch-user': '?1',
    'sec-gpc': '1',
    'upgrade-insecure-requests': '1',
    'user-agent':'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.101 Safari/537.36'
    }

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

r = requests.get(url, headers=headers, verify=False)
soup = bs(r.text, 'html.parser')


# ## Single Search Page

# In[3]:


def get_page(url):
    """Returns a beautiful soup object."""
    r = requests.get(url, headers=headers, verify=False)
    
    if not r.ok:
        print('Server Responded: ', r.status_code)
    else:
        soup = bs(r.text, 'html.parser')
    return soup


def get_main_attrs(soup):
    """Gets the attributes of each listing from a search page and returns
    a dictionary where the address is the key and the attributes are the values."""
    addr_ = soup.find_all('div',attrs={'data-testid': 'property-street'})
    addr = [x.get_text().strip() for x in addr_]
    region_ = soup.find_all('div',attrs={'data-testid': 'property-region'})
    region = [x.get_text().strip() for x in region_]
    new_ = soup.find_all('div', attrs={'data-testid': 'property-tags'})
    new = [x.get_text().strip() if x.get_text() is not '' else 'OLD' for x in new_]
    prices_ = soup.find_all('div', attrs={'data-testid': 'property-price'})
    prices = [x.get_text().strip() for x in prices_]
    bedrooms_ = soup.find_all('div', attrs={'data-testid': 'property-beds'})
    bedrooms = [x.get_text().strip() for x in bedrooms_]
    baths_ = soup.find_all('div', attrs={'data-testid': 'property-baths'})
    baths = [x.get_text().strip() for x in baths_]
    sqft_ = soup.find_all('div', attrs={'data-testid': 'property-floorSpace'})
    sqft = [x.get_text().strip() for x in sqft_]

    c = zip(addr, region, new, prices, bedrooms, baths, sqft)
    d = {}

    for a, *x in c:
        d.setdefault(a, []).append(x)

    for k, v in d.items():
        d[k] = v[0]

    df = pd.DataFrame(d).T.rename(columns={0:'region', 1:'new', 2:'price', 3:'bedrm', 4:'bth', 5:'sqft'})
    return df


def get_urls(soup):
    """Gets the external URLs on the page."""
    hrefs = []

    for listing in soup.find_all('div', attrs={'data-testid': 'home-card-sale'}):
        for link in listing.find_all('a', attrs={'href': re.compile('^/')}):
            hrefs.append(link)
        
    return [base_url + x['href'] for x in hrefs]


# ## Single Listing

# In[4]:


s_url = 'https://www.trulia.com/p/mo/columbia/2815-wild-plum-ct-columbia-mo-65201--2060813753'

page = get_page(s_url)


# In[5]:


def get_page_attrs(soup):
    """Gets a single page's attributes and returns a dataframe."""
    # Getting address again to map it with the address collected on main page
    try:
        addr_p = soup.find('span', attrs={'data-testid': 'home-details-summary-headline'}).get_text()
    except:
        addr_p = ''

    # Crime
    try:
        crime = soup.find('div',attrs={'aria-label': 'Crime'}).get_text()
    except:
        crime = ''
    # Schools in the area
    try:
        schools = soup.find('div', attrs={'aria-label': 'Schools'}).get_text()
    except:
        schools = ''

    # Home Details -- (heating, roof, etc.)
    try:
        details_ = soup.find('ul', attrs={'data-testid': 'home-features'})
        details = [x.get_text() for x in details_.find_all('li')]
    except:
        details = ''

    # History of listings
    try:
        list_hist_ = soup.find('div', attrs={'data-testid': 'price-history-container'}).find('table').find_all('tr')
        list_hist = [x.get_text().strip() for x in list_hist_]
    except:
        list_hist = ''

    # Taxes on the house
    try:
        tax_table_ = soup.find('div', attrs={'data-testid': 'property-tax-container'}).find('table').find_all('tr')
        tax_table = [x.get_text().strip() for x in tax_table_]
    except:
        tax_table = ''

    try:
        price_trends_ = soup.find_all('div', attrs={'class': 'Text__TextBase-sc-1i9uasc-0-div Text__TextContainerBase-sc-1i9uasc-1 epkfvN'})
        # Typical home value compared to others like it
        typ_home_val = [x.get_text() for x in price_trends_][-2] 
        # Typical price per sqft of houses similar to it
        typ_sqft_pri = [x.get_text() for x in price_trends_][-1]
    except:
        typ_home_val = ''
        typ_sqft_pri = ''

    try:
        pcts_ = soup.find_all('span', attrs={'class': 'Text__TextBase-sc-1i9uasc-0 fUDZSu'})
        # How much this house varies from others (based on typ_home_val)
        val_pct = [x.get_text() for x in pcts_][-2]
        # How much this house varies from others (based on typ_sqft_pri)
        sqft_pct = [x.get_text() for x in pcts_][-1]
    except:
        val_pct = ''
        sqft_pct = ''

    l = [addr_p, crime, schools, details, list_hist, tax_table, typ_home_val, val_pct, typ_sqft_pri, sqft_pct]

    names = ['addr', 'crime', 'schools', 'details', 'list_hist', 'tax', 'typ_val', 'val_pct', 'typ_sqft', 'sqft_pct']

    df = pd.DataFrame(l).T

    return df.rename(columns=dict(zip(df.columns.values, names)))


# ## Using a Single Search page to Scrape 30 Listings

# In[6]:


spage = 'https://www.trulia.com/for_sale/Columbia,MO/1p_beds/SINGLE-FAMILY_HOME_type/2_p/'

l_urls = get_urls(get_page(spage))


# In[39]:


dfs = []

for i, url in enumerate(l_urls):
    df = get_page_attrs(get_page(url))
    dfs.append(df)
    print(f'Listing {i:-<5} completed')


# In[97]:


df_main = get_main_attrs(get_page(spage)).rename_axis('addr').dropna(how='all')
df_ind = pd.concat(dfs).set_index('addr').dropna(how='all')

# df_main.join(df_ind, how='outer').dropna(how='all')
pd.merge(df_main, df_ind, left_index=True, right_index=True).head(1)


# ## Getting Main Attributes for Multiple Pages
# - I got the ranges to use on the website by viewing the number of pages available

# In[7]:


from itertools import chain

como = [f'https://www.trulia.com/for_sale/Columbia,MO/1p_beds/SINGLE-FAMILY_HOME_type/{x}_p/' for x in range(1, 8)]
kc = [f'https://www.trulia.com/for_sale/Kansas_City,MO/1p_beds/SINGLE-FAMILY_HOME_type/{x}_p/' for x in range(1, 29)]
stl = [f'https://www.trulia.com/for_sale/Saint_Louis,MO/1p_beds/SINGLE-FAMILY_HOME_type/{x}_p/' for x in range(1, 38)]
spring = [f'https://www.trulia.com/for_sale/Springfield,MO/1p_beds/SINGLE-FAMILY_HOME_type/{x}_p/' for x in range(1, 15)]
lees = [f'https://www.trulia.com/for_sale/Lees_Summit,MO/1p_beds/SINGLE-FAMILY_HOME_type/{x}_p/' for x in range(1, 16)]

list_all_urls = [como, kc, stl, spring, lees]
all_urls = list(chain.from_iterable(list_all_urls))

len(all_urls)


# In[137]:


all_main = []

for i, url in enumerate(all_urls):
    df = get_main_attrs(get_page(url))
    all_main.append(df)
    if (i+1) % 5 == 0:
        print(f'URL {i+1:-<5} completed')


# In[144]:


df_main = pd.concat(all_main)
df_main.sample(5)
# df_main.to_csv('df_main.csv', index=True, columns=df_main.columns.values)


# In[9]:


df_main = pd.read_csv('df/df_main.csv', index_col=0)


# ## Getting all of the individual listing's URLs
# - I had to separate the two for loops (main page attributes/individual page attributes) because I kept getting blocked/captcha.

# In[20]:


ind_urls = []

for idx, url in enumerate(all_urls):
    urls = get_urls(get_page(url))
    ind_urls.append(urls)
    time.sleep(3)
    if (idx+1) % 20 == 0:
        print(f'URL {idx+1:-<5} completed {urls[0][12:40]:-<5}')


# In[23]:


import pickle

page_urls = list(chain.from_iterable(ind_urls))

with open('pickle/page_urls.pickle', 'wb') as f:
    pickle.dump(page_urls, f, protocol=pickle.HIGHEST_PROTOCOL)

len(page_urls)


# In[81]:


ind_dfs = []

for idx, url in enumerate(page_urls):
    df = get_page_attrs(get_page(url))
    ind_dfs.append(df)
    time.sleep(2)

    if (idx+1) % 50 == 0:
        print(f'URL {idx+1:-<5}completed{url[28:40]:->15}')


# In[126]:


# Saving the list of dataframes
with open('pickle/df_ind.pickle', 'wb') as f:
    pickle.dump(ind_dfs, f, protocol=pickle.HIGHEST_PROTOCOL)

with open('pickle/df_ind.pickle', 'rb') as f:
    ind_list = pickle.load(f)

# Saving the concatenated dataframes
df_ind = pd.concat(ind_dfs).replace('', np.nan).dropna(how='all').set_index('addr')
df_ind.to_csv('df/df_ind.csv', index=True, columns=df_ind.columns.values)


# In[71]:


print(f'df_main: {df_main.shape}, df_ind: {df_ind.shape}')

