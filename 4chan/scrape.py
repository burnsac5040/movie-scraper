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
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#  import tabloo  # data visualizer

from bs4 import BeautifulSoup as bs
import requests
# }}} === imports


# sspipe examples {{{
[1, 2, 3, 4] | p(map, p([p(str), px%2])) | p(dict)
[1, 2, 3, 4] | p(lambda l: reduce(lambda x, y: x+y, l))
dict([(1,1),(4,2),(2,3)]).items() / p(sorted) | p(OrderedDict)
# }}}

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

def get_page(url, headers="", sleep=False, prnt=False):
    response = requests.get(url, headers=headers)
    if not response.ok:
        print("Server Responded: ", response.status_code)
        return
    else:
        soup = bs(response.text, "html.parser")
        patt = re.compile(r'\d+')
        if sleep:
            time.sleep(3)
        if prnt:
            print(f'Number {patt.search(url).group(): ^5}done')
    return soup

soup = get_page(url, headers)
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
    # is None doesn't work?
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


# }}} == post ==

# == user == {{{
def gusernames(soup):  # {{{
    """
    usernames: e.g., Anonymous
    """
    return [s.get_text() for s in soup.select("span.name")]  # }}}


def guid(soup):  # {{{
    """
    posterid: e.g., 88OlJHyW
    """
    posteruid = [s.get_text() for s in soup.select("span.posteruid")]
    return [re.sub(r"ID(?=:)|[(): ]", "", s) for s in posteruid]  # }}}


def gflag(soup):  # {{{
    """
    country_flag: e.g., Austria
    """
    return [s["title"] for s in soup.select("span.flag")]  # }}}

# }}} == user ==

# == image == {{{


def gimg_name(soup):  # {{{
    """
    image_title: e.g., check catalog.jpg
    """
    return [s.get_text() for s in soup.select(".fileText a")]  # }}}


def gimg_size(soup):  # {{{
    """
    @return image size in kilobytes
    """
    img_size = []
    for s in soup.select(".fileText"):
        try:
            img_size.append(re.search(r"(?<=\()\d+\s(K|M)B", s.get_text()).group())
        except AttributeError:
            img_size.append(re.search(r"(?<=\()\d+\s(K|M)B", s.get_text()))
    return img_size | p(rm_na, '0 KB') | p(mb2kb)  # }}} much more elegant IMO
    #  return mb2kb(rm_na(img_size, '0 KB'))

gg = gimg_size(soup)


def gimg_dim(soup):  # {{{
    """
    Return width, height of image
    """
    img_dim = []
    for s in soup.select(".fileText"):
        try:
            img_dim.append(re.search(r"(?<=,\s)\d+x\d+", s.get_text()).group())
        except AttributeError:
            img_dim.append(re.search(r"(?<=,\s)\d+x\d+", s.get_text()))

    return [x.split("x") for x in img_dim] | p(lambda l: list(zip(*l)))  # }}}

w, h = gimg_dim(soup)

# }}} == image ==

def gthread_post(soup):  # {{{
    """
    #' @return data.table; thread_id & all post_ids
    """
    #  [re.sub(r"#(q|p)", "/", s["href"].replace("thread/", "") for s in ...]
    thread_id, post_id = (     # ? more readable
        [s["href"] | p(re.sub, r"#(q|p)", "/", px) | px.replace("thread/", "")
         for s in soup.select("span.postNum a")]
        | p(uniset)
        | p(map, px.split("/"))
        | p(lambda l: list(zip(*l)))
    )
    return thread_id, post_id  # }}}

ff, xx = gthread_post(soup)


#    ddf <- data.frame(thread_id,post_id)
#
#    # stackoverflow.com/questions/33523320
#    dtt <- setDT(ddf)[, do.call(paste, c(.SD, list(collapse=', '))), thread_id]
#    dt1 <- cSplit(dtt, 'V1', sep='[ ,]+', fixed=FALSE, stripWhite=TRUE)
#    setnames(dt1, 2:ncol(dt1), rep(names(ddf)[-1], 7))
#    return(dt1)
#  }

# == thread == {{{
# thread: e.g., thread/322626957#q322631216 -- #p = link; #q = reply

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
