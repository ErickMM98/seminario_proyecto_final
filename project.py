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

data_pca['Watch time(Minutes)'] = np.log(data_pca['Watch time(Minutes)'])
data_pca['Views gained'] = np.log(data_pca['Views gained'])
data_pca['Stream time(minutes)'] = np.log(data_pca['Stream time(minutes)'])
data_pca['Peak viewers'] = np.log(data_pca['Peak viewers'])
data_pca['Average viewers'] = np.log(data_pca['Average viewers'])
data_pca['Followers'] = np.log(data_pca['Followers gained'])

print(data_pca.columns)
desribe_data_pca = data_pca.describe()

print((data_pca < 0).sum().sum())

#fig, ax = plt.subplots()
#fig.set_size_inches([15,15])
g = sns.pairplot(data_pca, hue = 'Mature')
g.set_size_inches(15,15)
center_data = StandardScaler(with_std=True).fit(data_pca).transform(data_pca)

pca = PCA().fit(center_data)
#pca.
pca_transform = pca.transform(center_data)
print(pca.explained_variance_ratio_)

foo_df = pd.DataFrame({'PC1': pca_transform[:,0],
                       'PC2': pca_transform[:,1],
                       'user': data[partenered == True]['Channel'],
                       'Mature': data[partenered == True]['Mature'],
                       'language': data[partenered == True]['Language']})

sns.scatterplot(x = 'PC1', y='PC2', 
                hue = 'Mature',
                data = foo_df)

print(pd.DataFrame(pca.components_))



#a = data.groupby(by = 'Language').mean()
#print(a.sort_values(by = 'Average viewers', ascending=False))
