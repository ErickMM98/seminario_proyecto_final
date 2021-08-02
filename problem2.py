# -*- coding: utf-8 -*-
"""
Created on Sun Aug  1 21:30:22 2021

@author: Erick Muñiz Morales

Problem 2. From a clinical trial, we have 12 patients with HIV infection. After treatment, the disease
progressed in 6 patients (1) and in 6 patients the infection did not progress (0). Four measurements
are taken in the 12 patients (Age, sugar levels, T cell levels and Cholesterol). Which measurement
can be used as a marker to describe progression of the disease? Which will be the criteria to predict
the progression? The data can be found in „problem2.csv (x_age, x_sugar, x_Tcell, x_cholesterol,
outcome). Arrange the data and briefly explain your results. The variable “y” (target) is a vector of 0
and 1 to represent the progression.
"""
import pandas as pd 
import seaborn as sns
import numpy.polynomial.polynomial as poly
import matplotlib.pyplot as plt 

#plt.style.use('seaborn')
plt.style.use('fivethirtyeight')

data = pd.read_csv('data/problem2.csv').T
data = pd.DataFrame({'age': data.index[1:],
                     'cholesterol': data[0][1:],
                     'sugar': data[1][1:],
                     'Tcell': data[2][1:],
                     'y': data[3][1:]}).reset_index()
data = data.drop(['index'], axis = True)
data = data.sort_values(by='age')

fig, ax = plt.subplots(2,2)
sns.scatterplot(data=data, x="age", y="y", hue="y", ax = ax[0][0])
sns.scatterplot(data=data, x="cholesterol", y="y", hue="y", ax = ax[0][1])
sns.scatterplot(data=data, x="sugar", y="y", hue="y", ax = ax[1][0])
sns.scatterplot(data=data, x="Tcell", y="y", hue="y", ax = ax[1][1])

ax[0][0].get_legend().remove()
ax[0][1].get_legend().remove()
ax[1][0].get_legend().remove()
ax[0][1].set_yticks((0,1))
ax[0][0].set_yticks((0,1))
ax[1][1].set_yticks((0,1))
ax[1][0].set_yticks((0,1))
#ax[1][1].get_legend().remove()

#ax[0][1].scatter()