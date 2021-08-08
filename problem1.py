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
import numpy as np
import numpy.polynomial.polynomial as poly
plt.style.use('fivethirtyeight')





data = pd.read_csv('data/problem1.csv')

X_train = data['X_training']
Y_train = data['Y_training']

X_test = data['X_test']
Y_test = data['Y_test']

x = np.linspace(-5,5, 100)

#PRIMER GRÁFICO
#Se ven los primeros ajustes
fig, ax = plt.subplots()
fig.set_size_inches([10,10])
#plt.cm.rainbow(0)
for degree in range(11):
    coef = poly.polyfit(X_train, Y_train, 
                    degree + 1,rcond=None,w=None) # Assuming a Polynomyal degree 8
    model = poly.Polynomial(coef)
    if degree > 4 and degree < 10:
        ax.plot(x, model(x), marker ='$ff$', linestyle = ':',linewidth=2.2,
                label = 'degree = {}'.format(degree+1), alpha = 0.8)
    else:
        ax.plot(x, model(x), marker ='.', linestyle = '--',linewidth=2.2,
                label = 'degree = {}'.format(degree+1), alpha = 0.8)
ax.scatter( X_train, Y_train, color = 'red', label = 'train', marker ='s', zorder = -1)
ax.scatter( X_test, Y_test, color = 'blue', label = 'test', marker ='s', alpha = 0.9)
ax.legend(loc = 'upper left')
ax.set_ylim([-10,60])
ax.set_ylim([Y_train.min(),Y_test.max()])
ax.set_xlim([X_train.min(),X_test.max()])
ax.set_title("Comparación de los polinomios.")
ax.set_xlabel("x")
ax.set_ylabel("y")
fig.show()
fig.savefig('images/comparacionplonomios.pdf')



#Cross Validation 

from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
test = test.dropna()
poly_features = PolynomialFeatures(degree=grade)
X_poly = poly_features.fit_transform(test)
poly = LinearRegression()
cross_val_score(poly, X_poly, test["Y_test"], cv=5)

