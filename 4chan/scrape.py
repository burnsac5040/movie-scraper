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
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#  import tabloo  # data visualizer

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
def uniset(inp):  # {{{
    """
    awk !seen[$0]++ or perl -ne 'print if !$seen{$_}++'
    """
    seen = set()
    seen_add = seen.add
    return [x for x in inp if not (x in seen or seen_add(x))]  # }}}


def rm_na(ltr, t2repl):  # {{{
    """
    Replace na/null values in a list (ltr = list 2 trans; t2repl = text 2 replace)
    """
    ltr = np.asarray(ltr)
    return list(np.where(np.logical_or(ltr != ltr, ltr == None), t2repl, ltr))  # }}}


def mb2kb(l2trans):  # {{{
    """
    Convert MB to KB
    """
    def lsub(x):  # {{{
        return re.sub(r"\s?(MB|KB)", "", x)  # }}}
    return [
        float(lsub(x)) * 1000 if x.endswith("MB") else float(lsub(x))
        for x in l2trans
    ]  # }}}
# }}} === helper functions

#############################################
# Section: Scraping a One Main Page         #
#############################################


def gboard(soup):  # {{{
    """
    board name: e.g., /pokemon/
    """
    return [s.get_text() for s in soup.select("div.boardTitle")][0]  # }}}

# == posts == {{{

def gwhole_post(soup):  # {{{
    """
    Anonymous (ID: 2o5j+xkQ)  05/21/21(Fri)20:29:50 No.3...
    """
    return [s.get_text() for s in soup.select(".post")]  # }}}


def gsticky(soup):  # {{{
    """
    Return hash of thread nums (k) with T/F if sticky (v)
    """
    sticky = {}
    def tsub(x):
        return re.sub("#(p|q)", "/", x["href"]).replace("thread/", "")
    for s in soup.select("span.postNum.desktop"):
        if s.select("img.stickyIcon"):
            for x in s.select("a"):
                sticky[tsub(x)] = 1
        else:
            for s in s.select("a"):  # CHECK: maybe only one for loop?
                sticky[tsub(s)] = 0
    return dict(uniset([(x[0].split("/")[0], x[1]) for x in sticky.items()]))  # }}}

sticky = gsticky(soup)

# }}} == post ==

# == user == {{{

def gusernames(soup):  # {{{
    """
    usernames: e.g., Anonymous
    """
    return [s.get_text() for s in soup.select("span.name")]  # }}}

usernames = gusernames(soup)

def guid(soup):  # {{{
    """
    posterid: e.g., 88OlJHyW
    """
    posteruid = [s.get_text() for s in soup.select("span.posteruid")]
    return [re.sub(r"ID(?=:)|[(): ]", "", s) for s in posteruid]  # }}}

uid = guid(soup)


def gflag(soup):  # {{{
    """
    country_flag: e.g., Austria
    """
    return [s["title"] for s in soup.select("span.flag")]  # }}}

flag = gflag(soup)
# }}} == user ==

# == image == {{{

def gimg_name(soup):  # {{{
    """
    image_title: e.g., check catalog.jpg
    """
    return [s.get_text() for s in soup.select(".fileText a")]  # }}}
# [s.get_text() for s in soup.select('.fileText')]


def gimg_size(soup):  # {{{
    """
    @return image size in kilobytes
    """
    # NOTE: not used
    def lsub(x, sub):
        return float(re.sub(r"\s?"+sub, "", x))
    img_size = []
    for s in soup.select(".fileText"):
        try:
            img_size.append(re.search(r"(?<=\()\d+\s(K|M)B", s.get_text()).group())
        except AttributeError:
            img_size.append(re.search(r"(?<=\()\d+\s(K|M)B", s.get_text()))

    rm_na(img_size, "0 KB")
    ii = []
    for s in img_size:
        if s is None:
            ii.append('0 KB')
        elif "MB" in s:
            ii.append(mb2kb(rm_na(s, "0 KB")))
        else:
            ii.append(s)
    # maybe clean up
    return [float(re.sub(r"\s?(MB|KB)", "", str(x))) for x in img_size]  #}}}

#  ii = []
#  for s in img_size:
#      if s is None:
#          ii.append('0 KB')
#      elif "MB" in s:
#          ii.append(mb2kb(rm_na(s, "0 KB")))
#      else:
#          ii.append(s)
#  # maybe clean up
#  return [float(re.sub(r"\s?(MB|KB)", "", str(x))) for x in img_size]  #}}}


img_size = gimg_size(soup)
# }}}

#  img_size = [x for x in img_size if x is not None]
#  img_size = ['NA' if x is None else x for x in img_size] # put in try except?

# image dimensions: e.g, 400x400
img_dim = []
for s in soup.select(".fileText"):
    try:
        img_dim.append(re.search(r"(?<=,\s)\d+x\d+", s.get_text()).group())
    except AttributeError:
        img_dim.append(re.search(r"(?<=,\s)\d+x\d+", s.get_text()))
# }}} == image ==

# == thread == {{{
# thread: e.g., thread/322626957#q322631216 -- #p = link; #q = reply
thread_post = [
    re.sub(r"#(q|p)", "/", s["href"]).replace("thread/", "")
    for s in soup.select("span.postNum a")
] | p(uniset)

# thread: & posts ids
thread, post = map(list, zip(*(s.split("/") for s in thread_post)))

# thread post dict: dict ids: e.g., thread : [p1, p2]
thread_d = defaultdict(list)
for x in [s.split("/") for s in thread_post]:
    thread_d[x[0]].append(x[1])
# }}} == thread ==

# == OG post == {{{
# OP: Anonymous ## Mod    05/31/20(Sun)15:07:39 No. ...
post_op = [s.get_text() for s in soup.select(".post.op")]

# OP: post id
op_id = [re.search(r"(?<=No.)\d+", s).group(0) for s in post_op]
# }}} == OG post ==

# == replies == {{{
# save list to guarantee correct thread; probably better way to do this
replies = u" ".join(str(x) for x in soup.select("span.summary"))
replies_bs = bs(replies, "html.parser")

replies = [
    re.search(r"\d+(?=\sreplies)", x.get_text()).group(0)
    for x in replies_bs.select("span.summary")
]

reply_thread_n = [
    x["href"].replace("thread/", "") for x in replies_bs.select("span.summary a")
]

if len(replies) == len(thread):
    replies_d = dict(zip(reply_thread_n, replies))
# }}} == replies ==

#  {x : x.get_text() for x in soup.select('span.summary')}
