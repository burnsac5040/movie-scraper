# -*- coding: utf-8 -*-
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
#     display_name: Cat
#     language: python
#     name: cat
# ---

# +
# =======================================================
# NOTE: Piping in Python (similar to %>% in R, | in Unix)
from datetime import datetime as dt
from itertools import chain
import math
import re

from bs4 import BeautifulSoup as bs
import numpy as np
import pandas as pd
import requests
from sspipe import p, px                  # piping tool like bash '|' or R '%>%'

import tabloo                             # dataframe visualizer
# import dfply                            # like dplyr


# +
url = "https://slashdot.org/"

headers = {
    "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9",
    "accept-encoding": "gzip, deflate, br",
    "accept-language": "en-US,en;q=0.9,es;q=0.8",
    # 'cache-control': 'max-age=0',
    "dnt": "1",
    "sec-fetch-dest": "document",
    "sec-fetch-mode": "navigate",
    "sec-fetch-site": "cross-site",
    "sec-fetch-user": "?1",
    "sec-gpc": "1",
    "upgrade-insecure-requests": "1",
    "user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.101 Safari/537.36",
}

r = requests.get(url, headers=headers)
soup = bs(r.text, "html.parser")

#### ┌─────────────────────────────────────────────────┐
#### │ Scraping one page                               │
#### └─────────────────────────────────────────────────┘

# +
# title, link, rating
tit = [t.get_text() for t in soup.select("h2 span a")]

# title of post
title = [j for i in soup.find_all("span", class_="story-title") for j in i.find("a")]

# date/time of post
import dateutil.parser
# dateutil.parser.parse('string')
dated = [d.get_text() for d in soup.select("span.story-byline time")]
date = [re.sub("on|@", "", x).strip() for x in dated]
dt = [dt.strptime(d, "%A %B %d, %Y %I:%M%p") for d in date]

# external link to post
elink = [l.text.strip() for l in soup.select("h2 span span")]

# comments on post
comments = (
    [x.get_text() for x in soup.select("span.comment-bubble a")]
    | p(np.array)
    | px.astype("int")
)

# category of post
cat = [b.get("alt") for b in soup.find_all("img")] | p(list, p(filter, None, px))
# Using this sort of as a try except in case it were to not exist
category = [x.replace("Icon", "") for x in cat] | p(filter, None) | p(list)

# user who made the post
user = [
    u.get_text(" ", strip=True).replace("\n", "").replace("\t", "")
    for u in soup.select("span.story-byline")
]
user = [" ".join(a.split()) | p(re.findall, r"Postedby\s(\w+)", px) for a in user]

# popularity of post (ratings? red?)
pop = [
    re.findall("'([a-zA-Z0-9,\s]*)'", prop["onclick"]) | px[1]
    for prop in soup.find_all("span", attrs={"alt": "Popularity"})
]
# -

# ### ┌─────────────────────────────────────────────────┐
# ### │ Dataframe                                       │
# ### └─────────────────────────────────────────────────┘

# +
df = pd.DataFrame(
    {
        "title": title,
        "date": dt,
        "exlink": elink,
        "comments": comments,
        "category": category,
        "user": user,
        "popular": pop,
    }
)


tabloo.show(df)
# from operator import itemgetter
# df['user'] = df['user'] | p(map, p(itemgetter(0)), px) | p(list)

df["user"] = df["user"] | p(chain.from_iterable) | p(list)
df["exlink"] = df["exlink"] | px.str.replace(r"\(|\)", "", regex=True)
# -

# ### ┌─────────────────────────────────────────────────┐
# ### │ Functions for scraping more than one page       │
# ### └─────────────────────────────────────────────────┘

# +
import time


def get_page(url, sleep=False, prnt=False):
    response = requests.get(url)
    if not response.ok:
        print("Server Responded: ", response.status_code)
    else:
        soup = bs(response.text, "lxml")
        patt = re.compile(r'\d+')
        if sleep:
            time.sleep(3)
        if prnt:
            print(f'Number {patt.search(url).group(): ^5}done')
    return soup


def get_data(soup):
    pattern = re.compile(r"\([a-z0-9.\-]+[.](\w+)\)")

    try:
        title_ = [x.get_text(' ', strip=True) for x in soup.select("h2 span.story-title")]
        title = [pattern.sub('', x).strip() for x in title_]
    except:
        title = ""

    try:
        dated = [d.get_text(" ", strip=True) for d in soup.select("span.story-byline time")]
        date = [re.sub("on|@", "", x).strip() for x in dated]
    except:
        date = ""

    try:
        curls = {}
        ex = [x.get_text(' ', strip=True) for x in soup.select("h2 span.story-title")]

        for idx, u in enumerate(ex):
            if not pattern.search(u):
                curls[idx] = "Empty"
            else:
                curls[idx] = pattern.search(u).group()

        elink = list(curls.values())
    except:
        try:
            elink = [l.text.strip() for l in soup.select("h2 span span.no")]
        except:
            elink = ""

    try:
        comments = (
            [x.get_text() for x in soup.select("span.comment-bubble a")]
            | p(np.array)
            | px.astype("int"))
    except:
        comments = ""

    try:
        cat = ([b.get("alt") for b in soup.find_all("img")]
              | p(list, p(filter, None, px)))
        category = ([x.replace("Icon", "") for x in cat]
                    | p(filter, None)
                    | p(list))
    except:
        category = ""

    try:
        user = [u.get_text(" ", strip=True).replace("\n", "").replace("\t", "")
                for u in soup.select("span.story-byline")]
        user = [" ".join(a.split())
                | p(re.findall, r"Postedby\s(\w+)", px) for a in user]
    except:
        user = ""

    try:
        pop = [re.findall("'([a-zA-Z0-9,\s]*)'", prop["onclick"]) | px[1]
            for prop in soup.find_all("span", attrs={"alt": "Popularity"})]
    except:
        pop = ""

    temp =pd.DataFrame({
            "title": title,
            "date": date,
            "exlink": elink,
            "comments": comments,
            "category": category,
            "user": user,
            "popular": pop
        }
    )
    return temp
# -

# ### ┌─────────────────────────────────────────────────┐
# ### │ Scraping more than one page                     │
# ### └─────────────────────────────────────────────────┘

# +
from random import sample
from datetime import date

base_url = "https://slashdot.org/?page="
urls = [base_url + str(i) for i in range(1, 100)]

test_u = urls | p(sample, 2)

data = [get_data(get_page(x, sleep=True, prnt=True)) for x in urls] | p(pd.concat, px)

data = data.reset_index(drop=True)

data.to_csv(f'data/df-{date.today().strftime("%m%d%Y")}.csv',
            index=True, columns=data.columns.values)
# -


# ### ┌─────────────────────────────────────────────────┐
# ### │ Reading back in & cleaning                      │
# ### └─────────────────────────────────────────────────┘

# +
df = pd.read_csv(f'data/df-02272021.csv', index_col=0)
# df = pd.read_csv(f'data/df-{date.today().strftime("%m%d%Y")}.csv', index_col=0)

df.columns.values
df.dtypes

# Date
df['date'] = [' '.join(x.split(' ')[1:]) | p(dt.strptime, "%B %d, %Y %I:%M%p") for x in df['date']]
df["exlink"] = df["exlink"] | px.str.replace(r"\(|\)", "", regex=True)
df["category"] = df["category"].astype("category")
df["user"] = df["user"] | px.str.replace(r"\[|\]|\'", "", regex=True)
df["popular"] = df["popular"].astype("category")

df.dtypes
# -


# ### ┌─────────────────────────────────────────────────┐
# ### │ Scraping post pages for comments                │
# ### └─────────────────────────────────────────────────┘

# Main page to post page
test = urls[0]
soup = get_page(test)

# Internal links to other posts
ilink_ = [x['href'] for x in soup.select('h2 span a')] | p(filter, None) | p(list)
ilink = [re.sub("//", "", x) for x in ilink_[::2] if x.startswith("//")]

# post page
ppage = "https://hardware.slashdot.org/story/21/03/01/0019243/boston-dynamics-is-selling-its-70-pound-robot-dog-to-police-departments"

psoup = get_page(ppage)

# title / external link / comments
tec = [x.get_text() for x in psoup.select("h2 span a")]
tec

ptitle, pelink = [x.get_text(' ', strip=True) for x in psoup.select("h2 span.story-title a")]

# comment count
pcomm = [x.get_text() for x in psoup.select("span.comment-bubble")] | px[0]

# post score
pscore = [re.sub(r'\(|\)|Score:', '', x.get_text()) for x in psoup.select("span[id*='comment_score_']")]

# post scores with label
sc = [x.split(', ') for x in pscore if ',' in x]
sco, lab = list(zip(*sc))

# user and user id
users = [re.sub(r'\(|\)', '', x.get_text('', strip=True)).strip() for x in psoup.select('div.details span.by a')]
uid, username = users[::2], users[1::2]

# date
# [x.text for x in psoup.select('div.details span[id*=comment_otherdetails_].otherdetails')]

# comment titles
comm_title = [x.get_text('', strip=True) for x in psoup.select('div.title')] | px[1:]

# comment body
comm_body = [x.get_text('', strip=True) for x in psoup.select('div.commentBody')]

# top branch comments that have replies
top_comm = [idx for idx, x in enumerate(comm_title) if not x.startswith('Re')]

# the comments replying to top comments
reply_comm = [top_comm[idx] - top_comm[idx-1] for idx in range(1, len(top_comm))]
reply_comm.append(len(comm_title) - top_comm[-1])

temp = {top_comm[i]: reply_comm[i] for i in range(len(top_comm))}

all_reply_comm = {i: 0 for i in range(len(comm_title))}

# create a dictionary of comment_index: replies
for x in all_reply_comm.keys():
    if x in temp.keys():
        all_reply_comm[x] = temp[x]

all_reply_comm
