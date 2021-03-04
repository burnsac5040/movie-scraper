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
# # !pip install youtube_transcript_api youtube_channel_transcript_api textstat

# +
from youtube_channel_transcript_api import YoutubeChannelTranscripts
from youtube_transcript_api import YouTubeTranscriptApi
from googleapiclient.discovery import build

import textstat

import pandas as pd
import numpy as np

# +
r_external_ids = {
    'StevenCrowder': 'UCIveFvW-ARp_B_RckhweNJw',
    'Ben Shapiro': 'UCnQC_G5Xsjhp9fEJKuIcrSw',
    'Mark Dice': 'UCzUV5283-l5c0oKRtyenj6Q',
    'The Rubin Report': 'UCJdKr0Bgd_5saZYqLCa9mng',
    'PragerU': 'UCZWlSUNDvCCS1hBiXV0zKcA',
    'Lauren Southern': 'UCla6APLHX6W3FeNLc8PYuvg',
    'Hunter Avallone': 'UCDgchsbJnrX604K-xWsd-fQ',
    "Don't Walk, Run! Productions": 'UCwpDBy43upJR7LZLrgnKkPA',
    'Lauren Chen': ':"UCLUrVTVTA3PnUFpYvpfMcpg',
    'Paul Joseph Watson': 'UCittVh8imKan0_5KohzDbpg',
    'Candace Owens': 'UCL0u5uz7KZ9q-pe-VC8TY-w',
    'Tim Pool': 'UCG749Dj4V2fKa143f8sE60Q'
}

l_external_ids = {
    # 'The Young Turks': 'UC1yBKRuGpC1tSM73A0ZjYjQ', 40,000 videos
    'LiberalViewer': 'UCqPKOi9bksw0pNS0KsJ9tDQ',
    'Vaush': 'UC1E-JS8L0j1Ei70D9VEFrPQ',
    # 'The Majority Report w/ Sam Seder': 'UC-3jIAlnQmbbVMV6gR7K8aQ',
    'The Damage Report': 'UC19roQQwv4o4OuBj3FhQdDQ',
    'David Pakman Show': 'UCvixJtaXuNdMPUGdOPcY8Ag',
    'Destiny': 'UC554eY5jNUfDq3yDOJYirOQ',
    'The Jimmy Dore Show': 'UC3M7l8ved_rYQ45AVzS0RGA',
    # 'Secular Talk': 'UCldfgbzNILYZA4dmDt4Cd6a', 14,000 videos
    # 'Thom Hartmann Program': 'UCbjBOso0vpWgDht9dPIVwhQ', 12,000 videos
    'ContraPoints': 'UCNvsIonJdJ5E4EXMa65VYpA',
    'hbomberguy': 'UClt01z1wHHT7c5lKcU8pxRQ',
    'Shaun': 'UCJ6o36XL0CpYb6U5dNBiXHQ',
    'Innuendo Studios': 'UC5fdssPqmmGhkhsJi4VcckA', # mrskimps
    'potholer54': 'UCljE1ODdSF7LS9xx9eWq0GQ',
    'Philosophy Tube': 'UC2PA-AKmVpU6NKCGtZq_rKQ'
}

mapping = {
    'StevenCrowder': 'crowder',
    'Ben Shapiro': 'shapiro',
    'Mark Dice': 'dice',
    'The Rubin Report': 'rubin',
    'PragerU': 'prager',
    'Lauren Southern': 'southern',
    'Hunter Avallone': 'avallone',
    "Don't Walk Run! Productions": 'walkrun',
    'Lauren Chen': 'chen',
    'Paul Joseph Watson': 'watson',
    'Candace Owens': 'owens',
    'Tim Pool': 'pool',
    'LiberalViewer': 'viewer',
    'Vaush': 'vaush',
    'The Damage Report': 'damage',
    'Destiny': 'destiny',
    'ContraPoints': 'contra',
    'hbomberguy': 'hbomb',
    'Shaun': 'shaun',
    'Innuendo Studios': 'innuendo',
    'potholer54': 'pothole',
    'Philosophy Tube': 'philo'
}


# +
def get_transcript(channel_name, var_name, text=True):
    var = YoutubeChannelTranscripts(channel_name,
                     'XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')
    video_data, video_error = var.get_transcripts(just_text=text)
    print('Finished getting transcripts ...')

    if len(video_data) == 0:
        print('No transcripts acquired, using video_error ...')
        df = pd.DataFrame(video_error, columns=['title', 'id']).set_index('id')
        df.to_csv(f'/content/drive/MyDrive/Colab Notebooks/youtube_transcripts/{var_name}_df.csv', columns=df.columns.values)
        print(f'Finished. Saved {var_name}_df.csv with {str(len(video_error))} rows')
    else:
        print('Using the transcripts acquired ...')
        df = pd.DataFrame(video_data).T
        df.to_csv(f'/content/drive/MyDrive/Colab Notebooks/youtube_transcripts/{var_name}_df.csv')
        print(f'Finished. Saved {var_name}_df.csv with {str(len(video_data))} rows')
        
    return df  


def get_stats(channel_id, var_name):
    key = 'XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX'
    youtube = build('youtube', 'v3', developerKey=key)
    id = channel_id

    statdata = youtube.channels().list(part='statistics', id=id).execute()
    stats = statdata['items'][0]['statistics']

    x = {var_name: {}}
    for k, v in stats.items():
        x[var_name].setdefault(k, []).append(v)

    orig.update(x)

    return orig


# +
df = get_transcript(channel_name="StevenCrowder", var_name='crowder')
df = get_transcript(channel_name="Ben Shapiro", var_name='shapiro')
df = get_transcript(channel_name="Mark Dice", var_name='dice')
df = get_transcript(channel_name="PragerU", var_name='prager')
df = get_transcript(channel_name="The Rubin Report", var_name='rubin')
df = get_transcript(channel_name="Lauren Southern", var_name='southern')
df = get_transcript(channel_name="Lauren Chen", var_name='chen')
df = get_transcript(channel_name="Paul Joseph Watson", var_name='watson')
df = get_transcript(channel_name="Candice Owens", var_name='ownes')
df = get_transcript(channel_name="Hunter Avallone", var_name='avallone')
df = get_transcript(channel_name="Don't Walk, Run Productions!", var_name='walkrun')
df = get_transcript(channel_name="Tim Pool", var_name='pool')

df = get_transcript(channel_name="LiberalViewer", var_name='viewer')
df = get_transcript(channel_name="Vaush", var_name='vaush')
# -

# ### DataFrames

# +
import os

os.listdir('data/')

# +
chen = pd.read_csv('data/chen_df.csv')
chen['user'] = 'Lauren Chen'
chen['orient'] = 'right'

prager = pd.read_csv('data/prager_df.csv')
prager['user'] = 'PragerU'
prager['orient'] = 'right'

crowder = pd.read_csv('data/crowder_df.csv')
crowder = crowder.rename(columns={'Unnamed: 0': 'id'})
crowder['user'] = 'StevenCrowder'
crowder['orient'] = 'right'

southern = pd.read_csv('data/southern_df.csv')
southern['user'] = 'Lauren Southern'
southern['orient'] = 'right'

watson = pd.read_csv('data/watson_df.csv')
watson['user'] = 'Paul Joseph Watson'
watson['orient'] = 'right'

dice = pd.read_csv('data/dice_df.csv')
dice['user'] = 'Mark Dice'
dice['orient'] = 'right'

rubin = pd.read_csv('data/rubin_df.csv')
rubin['user'] = 'The Rubin Report'
rubin['orient'] = 'right'

owens = pd.read_csv('data/owens_df.csv')
owens['user'] = 'Candace Owens'
owens['orient'] = 'right'

avallone = pd.read_csv('data/avallone_df.csv')
avallone['user'] = 'Hunter Avallone'
avallone['orient'] = 'right'

walkrun = pd.read_csv('data/walkrun_df.csv')
walkrun['user'] = "Don't Walk, Run! Productions"
walkrun['orient'] = 'right'

shapiro = pd.read_csv('data/shapiro_df.csv')
shapiro['user'] = 'Ben Shapiro'
shapiro['orient'] = 'right'

####################################################################################

viewer = pd.read_csv('data/viewer_df.csv')
viewer = viewer.rename(columns={'Unnamed: 0': 'id'})
viewer['user'] = 'LiberalViewer'
viewer['orient'] = 'left'

vaush = pd.read_csv('data/vaush_df.csv')
vaush['user'] = 'Vaush'
vaush['orient'] = 'left'

# +
dfs = [chen, prager,crowder,southern,watson, dice, rubin, owens, avallone, walkrun,shapiro, viewer, vaush]

pd.concat(dfs)
# -

# ### Statistics

# +
# orig = get_stats(r_external_ids['Lauren Chen'], str(dfs[0]))
orig = get_stats(r_external_ids['PragerU'], 'prager')
d = get_stats(r_external_ids['StevenCrowder'], 'crowder')
d = get_stats(r_external_ids['Lauren Southern'], 'southern')
# d = get_stats(r_external_ids['Paul Joseph Watson'], 'watson')
d = get_stats(r_external_ids['Mark Dice'], 'dice')
d = get_stats(r_external_ids['The Rubin Report'], 'rubin')
d = get_stats(r_external_ids['Candace Owens'], 'owens')
d = get_stats(r_external_ids['Hunter Avallone'], 'avallone')
d = get_stats(r_external_ids["Don't Walk, Run! Productions"], 'walkrun')
d = get_stats(r_external_ids['Ben Shapiro'], 'shapiro')
d = get_stats(r_external_ids['Tim Pool'], 'pool')

d = get_stats(l_external_ids['LiberalViewer'], 'viewer')
d = get_stats(l_external_ids['Vaush'], 'vaush')
# d = get_stats(l_external_ids['The Damage Report'], 'damage')
d = get_stats(l_external_ids['ContraPoints'], 'contra')
d = get_stats(l_external_ids['hbomberguy'], 'hbomb')
d = get_stats(l_external_ids['Shaun'], 'shaun')
d = get_stats(l_external_ids['Innuendo Studios'], 'innuendo')
d = get_stats(l_external_ids['potholer54'], 'pothole')

orig
# -

l_external_ids.keys()

# +
key = 'XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX'
youtube = build('youtube', 'v3', developerKey=key)
id = 'PrisonPlanetLive'

statdata = youtube.channels().list(part='statistics', id=id).execute()
statdata

# x = {var_name: {}}
# for k, v in stats.items():
#     x[var_name].setdefault(k, []).append(v)
# -


