# -*- coding: utf-8 -*-
"""
Created on Sun Aug  1 21:09:25 2021


@author: Erick Muñiz Morales

Primer parte del proyecto. 

Problem 1. Using the data set called „problem1.csv (x_training, y_training)“ :
a) Find the polynomial that fits the best the training data
b) Using the AIC criteria, find the best polynomial that can fit the data.
c) Cross validate the polynomial with the data set called “problem1.csv (x_test, y_test)”
"""

import pandas as pd
import matplotlib.pyplot as plt

# =============================================================================
# plt.rcParams.update({
#     "text.usetex": True,
#     "font.family": "serif",
#     "font.serif": ["Palatino"],
# })
# =============================================================================

data = pd.read_csv('data/problem1.csv')

X_train = data['X_training']
Y_train = data['Y_training']

fig, ax = plt.subplots()

ax.scatter( X_train, Y_train, color = 'red')
plt.show()