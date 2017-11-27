# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 08:22:56 2017

@author: baradhwaj
"""

##Import all packages

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import os
# Reading file and performing cleanup

#players = pd.read_csv("../input/wta/players.csv",  encoding='latin', index_col=0)
os.getcwd()
os.chdir('G:\Python\Learning\Self Learning\Free Datasets\WTA')
players = pd.read_csv("players.csv",thousands =',',encoding='latin',index_col=0)

# Top column is misaligned.
players.index.name = 'ID'
players.columns = ['First' , 'Last', 'Hand', 'DOB', 'Country']
# Parse date data to dates.
players = players.assign(DOB=pd.to_datetime(players['DOB'],format='%Y%m%d'))
# Handidness is reported as U if unknown; set np.nan instead.
players = players.assign(Hand=players['Hand'].replace('U', np.nan))
# Included year of birth  column for graph 
players['YOB'] = players['DOB'].apply(lambda b: b.year)

# Players after 80  and right
print('After 80 and right hand:')
print(players.loc[(players["DOB"] >  "1980-01-01") & (players["Hand"] =="R" ) ,["First","Last"]])
# players after 90
print('After 90: ',players.loc[(players["DOB"] >  "1990-01-01") & (players["Hand"] =="L"),["First","Last"]])
# Before 70:
print('Before 70: ',players.loc[(players["DOB"] <  "1970-01-01")& (players["Hand"] =="L") ,["First","Last"]])
# Players representing USA
#print('USA Players')
print(players.loc[players["Country"]  == "USA" ,["First","Last"]])
## Graphs : 
print('Graph:')
players.groupby(['YOB', 'Hand']) \
 .size().unstack('Hand').replace(np.NaN, 0).plot(title="Players' hand over the years",figsize=(20,10))
players.groupby('Country') \
            .size() \
            .sort_values(ascending=False)[:10] \
            .plot.bar(title="Top 10 countries with the most player", 
                  figsize=(20,10))
            

#### Matches:
#matches = pd.read_csv("../input/wta/matches.csv", encoding='latin1', index_col=0)
os.chdir('G:\Python\Learning\Self Learning\Free Datasets\WTA')
matches = pd.read_csv("matches.csv",thousands =',')
## Most Wins - Top 10. : 
print('Most Wins')
print(matches.groupby(['winner_name']).winner_name.count().nlargest(10))
top10Winners = matches.groupby(['winner_name']).winner_name.count().nlargest(10)

grouped = matches.groupby(['winner_name','surface'])['winner_name'].count().reset_index(name='count')
print(grouped)

print(grouped[grouped.winner_name.isin(top10Winners)])
print(grouped[np.in1d(grouped.winner_name,top10Winners)])