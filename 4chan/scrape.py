#!/usr/bin/env python

############################################################################
#    Author: Lucas Burns                                                   #
#     Email: burnsac@me.com                                                #
#   Created: 2021-05-21 13:09                                              #
############################################################################

# === imports {{{
from collections import defaultdict
from sspipe import p, px  # unix-like pipe
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tabloo  # data visualizer

from bs4 import BeautifulSoup as bs
import requests

# }}} === imports

# === beautifulsoup setup {{{
url = "https://boards.4chan.org/pol/"

headers = {
    "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9",
    "accept-encoding": "gzip, deflate, br",
    "accept-language": "en-US,en;q=0.9,es;q=0.8",
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
# }}} === beautifulsoup setup

# === helper functions {{{

# list(dict.fromkeys(items))
def uniset(inp):
    """
    awk !seen[$0]++ or perl -ne 'print if !$seen{$_}++'
    """
    seen = set()
    seen_add = seen.add
    return [x for x in inp if not (x in seen or seen_add(x))]


# }}} === helper functions

#############################################
# Section: Scraping a One Main Page         #
#############################################
# === Section: Scraping Single Page {{{

# board: e.g., /pol/ (there's only one, can't call '.text')
board = [s.get_text() for s in soup.select("div.boardTitle")][0]

# Anonymous (ID: 2o5j+xkQ)  05/21/21(Fri)20:29:50 No.3...
post_id_date = [s.get_text() for s in soup.select(".post")]
# [s for s in soup.select('.postMessage')]
# [s.get_text() for s in soup.select('.post.op')] # main reply
# [s.get_text() for s in soup.select('.post.reply')] # replies

# post id: sticky thread numbers
# [s['a'] for s in soup.select('span.postNum') if s.select('img.stickyIcon')]
sticky = []
for s in soup.select("span.postNum"):
    if s.select("img.stickyIcon"):
        for x in s.select("a"):
            sticky.append(re.sub("#(p|q)", "/", x["href"]).replace("thread/", ""))

# post: number of sticky
n_sticky = uniset([x.split("/")[0] for x in sticky])

# usernames: e.g., Anonymous
usernames = [s.get_text() for s in soup.select("span.name")]

# posterid: e.g., 88OlJHyW
posteruid = [s.get_text() for s in soup.select("span.posteruid")]
posterid = [re.sub(r"ID(?=:)|[(): ]", "", s) for s in posteruid]

# country_flag: e.g., Austria
flag = [s["title"] for s in soup.select("span.flag")]

# image_title: e.g., check catalog.jpg
img_title = [s.get_text() for s in soup.select(".fileText a")]
# [s.get_text() for s in soup.select('.fileText')]

# image size in KB
img_size = []
for s in soup.select(".fileText"):
    try:
        img_size.append(re.search(r"(?<=\()\d+\s(K|M)B", s.get_text()).group())
    except AttributeError:
        img_size.append(re.search(r"(?<=\()\d+\s(K|M)B", s.get_text()))

#  img_size = [x for x in img_size if x is not None]
#  img_size = ['NA' if x is None else x for x in img_size] # put in try except?

# convert mb to kb
img_size = [
    "{} KB".format(float(re.sub("\s?MB", "", x)) * 1000) if "MB" in x else x
    for x in img_size
]

# image dimensions: e.g, 400x400
img_dim = []
for s in soup.select(".fileText"):
    try:
        img_dim.append(re.search(r"(?<=,\s)\d+x\d+", s.get_text()).group())
    except AttributeError:
        img_dim.append(re.search(r"(?<=,\s)\d+x\d+", s.get_text()))

# Thread: e.g., thread/322626957#q322631216 -- #p = link; #q = reply
thread_post = [
    re.sub(r"#(q|p)", "/", s["href"]).replace("thread/", "")
    for s in soup.select("span.postNum a")
] | p(uniset)

# Thread: & posts ids
thread, post = map(list, zip(*(s.split("/") for s in thread_post)))

# Thread Post Dict: dict ids: e.g., thread : [p1, p2]
thread_d = defaultdict(list)
for x in [s.split("/") for s in thread_post]:
    thread_d[x[0]].append(x[1])

# OP: Anonymous ## Mod    05/31/20(Sun)15:07:39 No. ...
post_op = [s.get_text() for s in soup.select(".post.op")]
print(post_op[1])

# OP: post id
op_id = [re.search(r"(?<=No.)\d+", s).group(0) for s in post_op]

# number of replies
replies = [
    re.search(r"\d+(?=\sreplies)", x.get_text()).group(0)
    for x in soup.select("span.summary")
]

[x["href"] for x in soup.select("span.summary a")]

for x in soup.select("span.summary"):
    print(x.select("a"), x.get_text())

#  {x : x.get_text() for x in soup.select('span.summary')}

# }}} === Section: Scraping Single Page
