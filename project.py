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
from sklearn.preprocessing import StandardScaler

path_file = 'data/twitchdata-update.csv'
data = pd.read_csv(path_file)

partenered = data['Partnered']
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
print(data.dtypes)

#Tenemos 770 personas mayores
#print(data['Mature'].value_counts())

#Tenemos 770 personas mayores
#print(data['Partnered'].value_counts())


#-----------------------------------------------------------------------------
# ------------------- PCA SOBRE LAS CONTINUAS --------------------------------
#-----------------------------------------------------------------------------

data_pca = data.iloc[:,list(range(1,8)) + [9]]
data_pca = data_pca[partenered == True]
print(data_pca.columns)
desribe_data_pca = data_pca.describe()

print((data_pca < 0).sum().sum())
#sns.pairplot(data_pca, hue = 'Mature')

center_data = StandardScaler(with_std=False).fit(data_pca).transform(data_pca)

pca = PCA().fit(center_data)
pca.



#a = data.groupby(by = 'Language').mean()
#print(a.sort_values(by = 'Average viewers', ascending=False))
