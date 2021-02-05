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
import requests
from bs4 import BeautifulSoup as bs 

import numpy as np 
import pandas as pd 
import time, re, json, pickle

# +
url = 'https://distrowatch.com/search.php?status=All'
base_url = 'https://distrowatch.com/'

r = requests.get(url)
soup = bs(r.text, 'html.parser')
# -

b = soup.find_all('b')
b[21]

# +
ranking = soup.find_all('b')[21:]

distros = [a.get_text() for a in ranking if a.find('a') is not None]
hrefs = [a.find('a')['href'] for a in ranking if a.find('a') is not None]

ranked_distros = distros[:275]
ranked_hrefs = hrefs[:275]
# -

# ## Analyzing Single Page

r = requests.get(base_url + hrefs[1])
soup = bs(r.content, 'html.parser')

# +
# Distro Name
name = soup.find('h1').text

# OS Information
out = soup.select('ul')[1]

outline_ = out.find_all('b')
outline = [b.get_text() for b in outline_]

os_info_ = out.find_all('a')
os_info = [spec.get_text(strip=True) for spec in os_info_]

# +
print(outline)
print()
print(os_info)

t = out.text
print()
print(t)


# -

def get_os_info(text):
    t = text

    split_index = [i for i, e in enumerate(t) if e.islower() and t[i+1].isupper()
                                        or e.isnumeric() and t[i+1].isupper()]

    for i, j in enumerate(split_index, start=1):
        t = t[:j+i] + ' ' + t[j+i:]

    t = re.sub(r'\n', ' ', t)

    # a = re.search(r'\b(OS Type):', t).span()[0]
    b = re.search(r'\b(Based on):', t).span()[0]
    c = re.search(r'\b(Origin):', t).span()[0]
    d = re.search(r'\b(Architecture):', t).span()[0]
    e = re.search(r'\b(Desktop):', t).span()[0]
    f = re.search(r'\b(Category):', t).span()[0]
    g = re.search(r'\b(Status):', t).span()[0]
    h = re.search(r'\b(Popularity):', t).span()[0]

    idx = [b, c, d, e, f, g, h]

    split_list = [t[i:j].strip() for i, j in zip(idx, idx[1:]+[None])]

    return split_list


# +
text = get_os_info(t)

# Regular dictionary
di = {}

for a in text:
    di[a.split(':')[0]] = a.split(':')[1].strip()


# Defaultdict with lists
from collections import defaultdict
text = get_os_info(t)
d = defaultdict(list)

for a in text:
    d[a.split(':')[0]].append(a.split(':')[1].strip())

di['Origin']


# +
def os_to_dict(text):
    d = {}
    for val in text:
        d[val.split(':')[0]] = val.split(':')[1].strip()

    return d

test = os_to_dict(text)

test
# -

# ### Releases / Versions Dataframe

# +
dfs = pd.read_html(base_url + ranked_hrefs[1])

release = dfs[15].set_index(0)
release_df = release.iloc[:15].copy()
release_df.head()
# -

# ### Rating

# +
rating = soup.find('div', attrs={'style': 'font-size: 64px; text-align: left'}).text

rating
# -

# ### Overall Ranking df

# +
rank_dfs = pd.read_html('https://distrowatch.com/dwres.php?resource=popularity')

mo_12 = rank_dfs[8].drop('Last 12 months', axis=1).rename(columns={
            'Last 12 months.1': 'Distro',
            'Last 12 months.2': '12 months'
})

mo_6 = rank_dfs[9].drop('Last 6 months', axis=1).rename(columns={
            'Last 6 months.1': 'Distro',
            'Last 6 months.2': '6 months'
})

mo_3 = rank_dfs[10].drop('Last 3 months', axis=1).rename(columns={
            'Last 3 months.1': 'Distro',
            'Last 3 months.2': '3 months'
})

mo_1 = rank_dfs[11].drop('Last 1 month', axis=1).rename(columns={
            'Last 1 month.1': 'Distro',
            'Last 1 month.2': '1 month'
})

# +
from functools import reduce

rank_df = reduce(lambda x,y: pd.merge(x,y, on='Distro', how='outer'), [mo_12, mo_6, mo_3, mo_1])

# rank_df.to_csv('df-ranks.csv', index=False, header=rank_df.columns.values)

# The numbers represent the clicks per day
rank_df.head()


# -

# ## All distros

def get_os_info_full(text):
    t = text

    split_index = [i for i, e in enumerate(t) if e.islower() and t[i+1].isupper()
                                        or e.isnumeric() and t[i+1].isupper()]

    for i, j in enumerate(split_index, start=1):
        t = t[:j+i] + ' ' + t[j+i:]

    t = re.sub(r'\n', ' ', t)

    # a = re.search(r'\b(OS Type):', t).span()[0]
    try:
        b = re.search(r'\b(Based on):', t).span()[0]
    except:
        b = None
    
    try:
        c = re.search(r'\b(Origin):', t).span()[0]
    except:
        c = None

    try: 
        d = re.search(r'\b(Architecture):', t).span()[0]
    except:
        d = None

    try:
        e = re.search(r'\b(Desktop):', t).span()[0]
    except:
        e = None
    
    try:
        f = re.search(r'\b(Category):', t).span()[0]
    except:
        f = None

    try:
        g = re.search(r'\b(Status):', t).span()[0]
    except:
        g = None

    try:
        h = re.search(r'\b(Popularity):', t).span()[0]
    except:
        h = None

    idx = [b, c, d, e, f, g, h]

    split_list = [t[i:j].strip() for i, j in zip(idx, idx[1:]+[None])]

    return split_list


# #### I was going to use this because it looks much more elegant, however it returns multiple entries in the list and isn't split up right

# +
def try_except(string):
    try:
        return re.search(rf'\b({string}):', out).span()[0]
    except:
        return None


def get_os_info_full_2(text):
    t = text

    split_index = [i for i, e in enumerate(t) if e.islower() and t[i+1].isupper()
                                        or e.isnumeric() and t[i+1].isupper()]

    for i, j in enumerate(split_index, start=1):
        t = t[:j+i] + ' ' + t[j+i:]

    t = re.sub(r'\n', ' ', t)

    b = try_except('Based on')
    c = try_except('Origin')
    d = try_except('Architecture')
    e = try_except('Desktop')
    f = try_except('Category')
    g = try_except('Status')
    h = try_except('Status')

    idx = [a, b, c, d, e, f, g, h]

    split_list = [t[i:j].strip() for i, j in zip(idx, idx[1:]+[None])]

    return split_list

# get_os_info_full_2(out)


# +
def get_page(url):
    response = requests.get(url)

    if not response.ok:
        print('Server responded: ', response.status_code)
    else:
        soup = bs(response.text, 'html.parser')

    return soup


def get_distro(soup):
    name = soup.find('h1').text
    out = soup.select('ul')[1].text

    os_info = get_os_info_full(out)
    os_dict = os_to_dict(os_info)

    full_d = {}
    temp_d = {name: {}}
    for k, v in os_dict.items():
        temp_d[name].setdefault(k, []).append(v)

    full_d.update(temp_d)
    return temp_d



# -

# ### Testing on small sample

# +
test_hrefs = np.random.choice(ranked_hrefs, size=5).tolist()
test_urls = [base_url + item for item in test_hrefs]

all_d = {}

for url in test_urls:
    t_dict = get_distro(get_page(url))
    all_d.update(t_dict)
# -

# ### Getting all of the ranked distros

len(ranked_hrefs)

# +
full_urls = [base_url + item for item in ranked_hrefs]

full_d = {}

for url in full_urls:
    t_dict = get_distro(get_page(url))
    full_d.update(t_dict)
# -

# ### Saving the full dictionary to json and pickle

# +
with open('distro_dict.json', 'w', encoding='utf-8') as f:
    json.dump(full_d, f, ensure_ascii=False, indent=4)

with open('distro_dict.json', 'r') as f:
    data = json.load(f)
# -

data['Manjaro Linux']['Based on']

# +
with open('data/distro_dict.pickle', 'wb') as f:
    pickle.dump(full_d, f, protocol=pickle.HIGHEST_PROTOCOL)

with open('data/distro_dict.pickle', 'rb') as f:
    data2 = pickle.load(f)

# +
os_info_df = pd.DataFrame(data)

# os_info_df.to_csv('df-os_info.csv', index=True, columns=os_info_df.columns.values)

os_info_df.head()


# -

# ## Getting all versions / releases of each distro

def get_versions(href):

    dfs = pd.read_html(base_url + href)

    ix = [11, 12, 13, 14, 15, 16]

    for i in ix:
        try:
            dfs[i] = dfs[i].set_index(0)
            if 'Feature' in dfs[i].index.values:
                release_df = dfs[i]
                return release_df.iloc[:15]
            else:
                pass
        except:
            pass


# ### Testing on sample first

# +
df_list = []

for val in test_hrefs:
    x = get_versions(val)
    df_list.append(x)

# +
sample_df = pd.concat([item.T for item in df_list], keys=[item for item in test_hrefs], axis=0).reset_index(level=1, drop=True)

sample_df.sample(n=5)
# -

# ### Getting all versions

# +
df_list = []

for idx, val in enumerate(ranked_hrefs):
    x = get_versions(val)
    df_list.append(x)

    if (idx+1) % 10 == 0:
        print(f'Finished {val}, Number: {idx+1}')

# +
with open('data/versions.pickle', 'wb') as f:
    pickle.dump(df_list, f, protocol=pickle.HIGHEST_PROTOCOL)

with open('data/versions.pickle', 'rb') as f:
    data2 = pickle.load(f)
# -

# #### The first go around I had some missing dataframes, so I added 11 to the ranged to check and no longer have any NoneTypes

# +
count = 0
idxs = []

for idx, item in enumerate(df_list):
    if isinstance(item, type(None)):
        count += 1
        idxs.append(idx)

print(count)
print(idxs)

# +
full_df = pd.concat([el.T for el in df_list], 
                    keys=[el for el in ranked_hrefs], axis=0).reset_index(level=1, drop=True)

full_df.index.name = 'distro'
# full_df.to_csv('df-all_versions.csv', index=True, columns=full_df.columns.values)

full_df.sample(n=5)
# -

# ## Loading it back in

# +
releases = pd.read_csv('data/df-all_versions.csv', index_col=0)
os_info = pd.read_csv('data/df-os_info.csv', index_col=0)
distro_rank = pd.read_csv('data/df-ranks.csv', index_col=0)

# df[df.index == 'arch']['Journaled File Systems'].value_counts()

# +
idxs = np.unique(releases.index.values, return_index=True)[1]
distro_hrefs = [releases.index.values[idx] for idx in sorted(idxs)]

print(len(os_info.columns.values), len(distro_hrefs), distro_rank.shape[0])
# -

href2distro = dict(zip(distro_hrefs, os_info.columns.values))

# +
distro_rank = distro_rank.rename(index={b: a for a, b in href2distro.items()})
distro_rank.index = distro_rank.index.str.lower()

distro_rank.head(5)

# +
os_info = os_info.rename(columns={b: a for a, b in href2distro.items()})

ranks = pd.DataFrame(np.arange(1, os_info.shape[1]+1)).rename(columns={0: 'ranks'})
ranks = ranks.T
ranks.columns = os_info.columns

os_info = pd.concat([ranks, os_info])
os_info = pd.concat([os_info, distro_rank.T])

cols = os_info.T.columns.tolist()
cols = [cols[0]] + cols[9:][::-1] + cols[1:9]

os_info = os_info.T[cols].T

# I guess the column lengths changed, so:
os_info.iloc[0] = np.where(os_info.iloc[0].isnull(), [x for x in np.arange(1, 290)], os_info.iloc[0])

os_info.head()
# -

# ### Getting all the ratings

# +
full_urls = [base_url + item for item in ranked_hrefs]


def get_rating(soup):
    name = soup.find('h1').text
    rating = soup.find('div', attrs={'style': 'font-size: 64px; text-align: left'}).text

    d = {name: rating}

    return d


# +
test_ratings = np.random.choice(full_urls, 5).tolist()

ratings_dict = {}

[ratings_dict.update(get_rating(get_page(el))) for el in test_ratings]

# +
full_ratings = {}

for i, el in enumerate(full_urls):
    full_ratings.update(get_rating(get_page(el)))

    if (i+1) % 10 == 0:
        print(f'Distro {el} completed, number {i+1}')
# -

with open('data/ratings.json', 'w', encoding='utf-8') as f:
    json.dump(full_ratings, f, ensure_ascii=False, indent=4)

with open('data/ratings.json', 'r') as f:
    ratings = json.load(f)

# ### Adding ratings to the dataframe

# +
distro2href = {b: a for a, b in href2distro.items()}

r_keys = list(ratings.keys())
d_keys = list(distro2href.keys())

ra_keys = distro_rank.T.columns.values

# Symmetric difference between the two
{*d_keys} ^ {*r_keys}

# +
print('Scientific Linux' in d_keys)
print('Baltix GNU/Linux' in r_keys)

# Scientific Linux = scientific
# Baltix GNU/Linux = baltix

print(len(r_keys), len(d_keys), len(ra_keys), os_info.shape[1])

print('scientific' in os_info.columns.values)
print('baltix' in os_info.columns.values)

# {*os_info.columns.values} ^ {*href2distro}

# +
ratings['Baltix GNU/Linux'] = np.nan
href2distro['scientific'] = 'Scientific Linux'

distro2href = {b: a for a, b in href2distro.items()}

# +
ratings_dict = {distro2href.get(k, v): v for k, v in ratings.items()}

# s_ratings = dict(sorted(ratings.items()))
# s_distro2href = dict(sorted(distro2href.items()))

# dict(zip(s_distro2href.values(), s_ratings.values()))

# +
from collections import Counter

print(Counter(list(ratings.values())).most_common(1))

na = [k for k, v in ratings.items() if v == 'N/A']

# +
os_turned = os_info.T.copy()
os_turned['rating'] = os_turned.index.map(ratings_dict)

# Popularity column is redundant since I have the rank and hits per day in separate columns
os_turned = os_turned.drop('Popularity', axis=1)
os_turned['rating'] = os_turned['rating'].replace('N/A', np.nan).fillna(np.nan).astype(np.float32)

os_turned[os_turned.index == 'mx']
# -

full_releases = releases.copy()

# +
idxs = np.unique(releases.index.values, return_index=True)[1]
distro_hrefs = [releases.index.values[idx] for idx in sorted(idxs)]

recent_release_idx = [(full_releases.index == i).argmax() for i in distro_hrefs]
# full_releases.groupby(full_releases.index).first()
recent_releases = full_releases.iloc[recent_release_idx]

info_releases = pd.concat([os_turned, recent_releases], join='outer', axis=1)
info_releases = info_releases.drop(info_releases.tail(14).index) # Last 14 are all nan
info_releases.head()
# -

# ----
# ----
# ## Cleaning the Data
# ----
# ### Desktops:
# Most common: KDE, Xfce, Budgie, Cinnamon, GNOME, MATE, LXDE, LXQt, Openbox, Fluxbox
#
# ----

# +
i_release = info_releases.copy()
i_release['ddesktop'] = i_release['Desktop'].astype(str) + ' ' + i_release['Default Desktop'].astype(str)
i_release = i_release.replace(r"\[|\]|\'|\,|\(|\)", '', regex=True)

i_release = i_release.rename(columns={'1 month': '1_month', '3 months': '3_months', '6 months': '6_months', '12 months': '12_months'}).drop(['Desktop', 'Default Desktop'], axis=1)

i_release.columns = i_release.columns.str.lower()


# -

def other(df:pd.core.frame.DataFrame, l:list, col, new_col:str):
    other = df[col].str.contains('|'.join(l), case=False, na=False).astype(int)
    df[new_col] = 0
    df.loc[other[other == 0].index, new_col] = 1

    df = df.drop(col, axis=1)

    return df


# +
# Counter(' '.join(i_release['ddesktop']).split()).most_common()

i_release['desk_kde'] = i_release['ddesktop'].str.contains('KDE', case=False, na=False).astype(int)
i_release['desk_xfce'] = i_release['ddesktop'].str.contains('Xfce', case=False, na=False).astype(int)
i_release['desk_budgie'] = i_release['ddesktop'].str.contains('Budgie', case=False, na=False).astype(int)
i_release['desk_cinnamon'] = i_release['ddesktop'].str.contains('Cinnamon', case=False, na=False).astype(int)
i_release['desk_gnome'] = i_release['ddesktop'].str.contains('GNOME', case=False, na=False).astype(int)
i_release['desk_mate'] = i_release['ddesktop'].str.contains('MATE', case=False, na=False).astype(int)
i_release['desk_lxde'] = i_release['ddesktop'].str.contains('LXDE', case=False, na=False).astype(int)
i_release['desk_lxqt'] = i_release['ddesktop'].str.contains('LXQt', case=False, na=False).astype(int)
i_release['desk_openbox'] = i_release['ddesktop'].str.contains('openbox', case=False, na=False).astype(int)
i_release['desk_fluxbox'] = i_release['ddesktop'].str.contains('fluxbox', case=False, na=False).astype(int)
i_release['desk_nodesk'] = i_release['ddesktop'].str.contains('no desktop', case=False, na=False).astype(int)
i_release['desk_web'] = i_release['ddesktop'].str.contains('web', case=False, na=False).astype(int)
i_release['desk_i3'] = i_release['ddesktop'].str.contains('i3', case=False, na=False).astype(int)

# Other desktops
desktops = ['KDE', 'Xfce', 'Budgie', 'Cinnamon', 'GNOME', 'MATE', 'LXDE', 'LXQt', 'openbox', 'fluxbox', 'no desktop', 'web', 'i3']

i_release = other(df=i_release, l=desktops, col='ddesktop', new_col='desk_other')

i_release.sample(2)

# +
# other = i_release['ddesktop'].str.contains('|'.join(desktops), case=False, na=False).astype(int)
# i_release['desk_other'] = 0
# i_release.loc[other[other == 0].index, 'desk_other'] = 1

# i_release
# -

# ----
# ### Based On:
# Most common: Fedora, Red Hat, Debian, Ubuntu, Independent, Arch (pacman), Gentoo, Slackware 
#
# ----

# +
# Counter(' '.join(i_release['Based on']).split()).most_common()

i_release['based_cent'] = i_release['based on'].str.contains('Cent|Clear|Scientific', case=False, na=False).astype(int)
# Fedora and Red Hat are different (red hat spnosors fedora) so I'm combining them
i_release['based_fedora'] = i_release['based on'].str.contains('Fedora|Red|Hat', case=False, na=False).astype(int)
# Ubuntu is Debian based, and there is a line of Unbuntu-based distros but i'm combining them
i_release['based_debian'] = i_release['based on'].str.contains('Debian|Ubuntu', case=False, na=False).astype(int)
i_release['based_indep'] = i_release['based on'].str.contains('Independent', case=False, na=False).astype(int)
i_release['based_pacman'] = i_release['based on'].str.contains('Arch', case=False, na=False).astype(int)
i_release['based_gentoo'] = i_release['based on'].str.contains('Gentoo', case=False, na=False).astype(int)
i_release['based_slack'] = i_release['based on'].str.contains('Slackware', case=False, na=False).astype(int)

# Other 'based_on'
based_on = ['Cent', 'Clear', 'Scientific', 'Fedora', 'Red', 'Hat', 'Debian', 'Ubuntu', 'Independent', 'Arch', 'Gentoo', 'Slackware']

i_release = other(df=i_release, l=based_on, col='based on', new_col='based_other')
i_release.sample(2)
# -

# ----
# ### OS Type
#
# Doesn't look useful, so I will drop it
#
# ----

# +
print(Counter(' '.join(i_release['os type'].astype(str)).split()).most_common())

i_release = i_release.drop('os type', axis=1)
# -

# ----
# ### Architecture
# ----

# +
i_release['processor'] = i_release['architecture'].astype(str) + ' ' + i_release['processor architecture'].astype(str)

i_release = i_release.drop(['architecture', 'processor architecture'], axis=1)

Counter(' '.join(i_release['processor'].astype(str)).split()).most_common(5)

# +
i_release['arc_x86'] = i_release['processor'].str.contains('x86_64', case=False, na=False).astype(int)
i_release['arc_i686'] = i_release['processor'].str.contains('i686', case=False, na=False).astype(int)
i_release['arc_i386'] = i_release['processor'].str.contains('i386', case=False, na=False).astype(int)
i_release['arc_arm'] = i_release['processor'].str.contains('armfh', case=False, na=False).astype(int)
i_release['arc_aarch'] = i_release['processor'].str.contains('aarch64', case=False, na=False).astype(int)

# Other architecture
arc = ['x86_64', 'i686', 'i386', 'armfh', 'aarch64']
i_release = other(df=i_release, l=arc, col='processor', new_col='arc_other')

i_release.sample(2)
# -

# ----
# ### Category
# ----

Counter(' '.join(i_release['category'].astype(str)).split()).most_common(5)

# +
i_release['cat_live'] = i_release['category'].str.contains('Live', case=False, na=False).astype(int)
i_release['cat_med'] = i_release['category'].str.contains('Medium', case=False, na=False).astype(int)
i_release['cat_desk'] = i_release['category'].str.contains('Desktop', case=False, na=False).astype(int)
i_release['cat_serv'] = i_release['category'].str.contains('Server', case=False, na=False).astype(int)

cat = ['Live', 'Medium', 'Desktop', 'Server']

i_release = other(df=i_release, l=cat, col='category', new_col='cat_other')

i_release.sample(2)
# -

# ----
# ### Status
# All are active, so I am going to drop this column
#
# ----
#

# +
print(Counter(' '.join(i_release['status'].astype(str)).split()).most_common(5))

i_release = i_release.drop('status', axis=1)
# -

# ----
# ### Origin
#
# Since I am from the United States and for simplicities sake, I am going to do either USA or not
#
# ----

# +
print(Counter(' '.join(i_release['origin'].astype(str)).split()).most_common(5))

i_release = other(df=i_release, l=['USA'], col='origin', new_col='org_usa')
# -

# ----
# ### Ranks and Clicks
# ----

# +
# Clicks
i_release.iloc[:, 1:5] = i_release.iloc[:, 1:5].fillna(0).astype(int)

# Ranks
na_df = i_release.loc[i_release['rating'].isna(), 'ranks']
na_ranks = list(zip(na_df.index, na_df))

fill_ranks = {}

for dist, rank in na_ranks:
    fill_val = round(i_release.loc[i_release['ranks'].between(rank-10, rank+10), 'rating'].mean(), 1)
    fill_ranks[dist] = fill_val

i_release['rating'] = i_release['rating'].fillna(fill_ranks)
# -

# ----
# ### Feature
#
#
# ----

# +
print(Counter(' '.join(i_release['feature'].astype(str)).split()).most_common(5))

i_release['feature'] = i_release['feature'].str.extract('(\d+.?\d+.?\d+)', expand=False).fillna(1.0)

# Show what the other versions look like since they don't follow the traditional incremental versions
start20_n = i_release[i_release['feature'].str.startswith('2020', na=False)].index.tolist()
start20 = [full_releases.groupby(full_releases.index).get_group(name=el)[['Feature', 'Release Date']] for el in start20_n]
# -

# #### I decided to create a column that will show the number of releases for the distribution in the past year instead of using the version

# +
rel2020 = full_releases['Release Date'].str.startswith('2020', na=False).groupby(full_releases.index).sum()
rel2020.name = 'rel_2020'

i_release = pd.concat([i_release, rel2020], join='outer', axis=1)
i_release = i_release.drop('feature', axis=1)
# -

# ----
# ### Release date
#
# I'm not sure if the month would have any correlation, but I'm going to add it
#
# #### Could create season
#
# ----

# +
i_release['release_date'] = pd.to_datetime(i_release['release date'])
i_release['release_month'] = i_release['release_date'].dt.month

i_release = i_release.drop('release date', axis=1)
i_release = i_release.drop('release_date', axis=1)
# -

# ----
# ### End of Life
#
# Doesn't seem useful so I'm going to drop it
#
# ----

i_release = i_release.drop('end of life', axis=1)

# ----
# ### Price
#
# 94% of them are free so I'm going to drop it as well
#
# ----

i_release = i_release.drop('price (us$)', axis=1)

# ----
# ### Image Size (MB)
#
# Since most are a range, I'm going to average it
#
# ----

i_release = i_release.rename(columns={'image size (mb)': 'image_size'})
i_release['image_size']

# +
fill_avg = i_release['image_size'].str.split('-', expand=True).astype(float).mean(axis=0)
fill_val = int((fill_avg[0] + fill_avg[1]) / 2)

i_release['image_size'] = (i_release['image_size'].str.split('-', expand=True).fillna(fill_val)
                                                         .astype(int).mean(axis=1).astype(int))
# -

# ----
# ### Free Download
# ----

# +
i_release['down_iso'] = i_release['free download'].str.contains('ISO', case=False, na=False).astype(int)
i_release['down_img'] = i_release['free download'].str.contains('IMG', case=False, na=False).astype(int)

i_release = other(df=i_release, l=['ISO', 'IMG'], col='free download', new_col='down_oth')
# -

# ----
# ### Installation
# ----

# +
i_release['inst_graph'] = i_release['installation'].str.contains('Graphic', case=False, na=False).astype(int)
i_release['inst_text'] = i_release['installation'].str.contains('Text', case=False, na=False).astype(int)

i_release = other(df=i_release, l=['Graphic', 'Text'], col='installation', new_col='inst_oth')
# -

# ----
# ### Package Management
# ----

Counter(' '.join(i_release['package management'].astype(str)).split()).most_common(5)

# +
i_release['pack_deb'] = i_release['package management'].str.contains('DEB', case=False, na=False).astype(int)
i_release['pack_rpm'] = i_release['package management'].str.contains('RPM', case=False, na=False).astype(int)
i_release['pack_apt'] = i_release['package management'].str.contains('Pacman', case=False, na=False).astype(int)
i_release['pack_pacman'] = i_release['package management'].str.contains('APT', case=False, na=False).astype(int)

i_release = other(df=i_release, l=['DEB', 'RPM', 'Pacman', 'APT'], col='package management', new_col='pack_other')
# -

# ----
# ### Release Model
# ----

Counter(' '.join(i_release['release model'].astype(str)).split()).most_common(15)

# +
i_release['rel_fix'] = i_release['release model'].str.contains('Fixed', case=False, na=False).astype(int)
i_release['rel_roll'] = i_release['release model'].str.contains('Rolling', case=False, na=False).astype(int)

i_release = other(df=i_release, l=['Fixed', 'Rolling'], col='release model', new_col='rel_other')
# -

# ----
# ### Office Suite
#
# Doesn't seem to be unique/useful
#
# ----

i_release = i_release.drop('office suite', axis=1)

# ----
# ### Init Software
# ----

Counter(' '.join(i_release['init software'].astype(str)).split()).most_common(5)

# +
i_release['init_sysd'] = i_release['init software'].str.contains('systemd', case=False, na=False).astype(int)
i_release['init_sysv'] = i_release['init software'].str.contains('sysv', case=False, na=False).astype(int)
i_release['init_oprc'] = i_release['init software'].str.contains('OpenRC', case=False, na=False).astype(int)

i_release = other(df=i_release, l=['systemd', 'sysv', 'openrc'], col='init software', new_col='init_other')
# -

# ----
# ### Journaled File System
# ----

Counter(' '.join(i_release['journaled file systems'].astype(str)).split()).most_common(5)

# +
i_release['jour_ext4'] = i_release['journaled file systems'].str.contains('ext4', case=False, na=False).astype(int)
i_release['jour_ext3'] = i_release['journaled file systems'].str.contains('ext3', case=False, na=False).astype(int)
i_release['jour_xfs'] = i_release['journaled file systems'].str.contains('XFS', case=False, na=False).astype(int)
i_release['jour_btrfs'] = i_release['journaled file systems'].str.contains('btrfs', case=False, na=False).astype(int)
i_release['jour_reiser'] = i_release['journaled file systems'].str.contains('Reiser', case=False, na=False).astype(int)

# Possibly add JFS / NFS

i_release = other(df=i_release, l=['ext4', 'ext3', 'xfs', 'btrfs', 'reiser'], col='journaled file systems', new_col='journ_oth')
# -

# ----
# ### Multilingual
# ----

# +
i_release['multiling'] = np.where(~i_release['multilingual'].str.contains('No|--', case=False, na=True), 1, 0)

i_release = i_release.drop('multilingual', axis=1)
# -

i_release.sample(2)

i_release.to_csv('data/df-ohe.csv', index=True, columns=i_release.columns.values)
