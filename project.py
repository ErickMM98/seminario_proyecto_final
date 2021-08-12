# -*- coding: utf-8 -*-
"""
Created on Thu Aug 12 01:37:56 2021

Analisis de streamers en Twitch
@author: Erick Muñiz Morales
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 
import numpy as np
import os

from sklearn.decomposition import PCA

path_file = 'data/twitchdata-update.csv'
data = pd.read_csv(path_file)

print(data.columns)

"""
Index(['Channel', 
       'Watch time(Minutes)', 
       'Stream time(minutes)',
       'Peak viewers', 
       'Average viewers', 
       'Followers', 
       'Followers gained',
       'Views gained', 
       'Partnered', 
       'Mature', 
       'Language'],
      dtype='object')
"""


#-----------------------------------------------------------------------------
# ------------------- ESTADÍSTICOS -------------------------------------------
#-----------------------------------------------------------------------------

#data['']
#print(data.dtypes)

#Tenemos 770 personas mayores
#print(data['Mature'].value_counts())

#Tenemos 770 personas mayores
#print(data['Partnered'].value_counts())


#-----------------------------------------------------------------------------
# ------------------- PCA SOBRE LAS CONTINUAS --------------------------------
#-----------------------------------------------------------------------------

data_pca = data.iloc[:,range(2,9)]
print(data_pca.describe())

#a = data.groupby(by = 'Language').mean()
#print(a.sort_values(by = 'Average viewers', ascending=False))
