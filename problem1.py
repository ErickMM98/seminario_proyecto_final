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
import math


def f(x):
    return x**5 - 30*x**3  -10 # + stats.norm.rvs(loc=0,scale= .01,size=len(x))
    
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


#---------------- SOBRE AIC

Pol_Max = 16 # Highest degree polynomial we are going to check.
# empty arrays
RSSv = [] 
RSSv = np.zeros(Pol_Max-1)
AICv = []
AICv = np.zeros(Pol_Max-1)

x = np.arange(0,10,0.5)
for i in np.arange(0,Pol_Max-1):
    coef = poly.polyfit(X_train, Y_train, i+1)
    model = poly.Polynomial(coef)
    
    RSSv[i]=np.sum((Y_train-model(X_train))**2)
    
    AICv[i]=len(X_train)*math.log10(RSSv[i]/len(X_train))+ 2*len(coef)*len(X_train)/(len(X_train)-len(coef)-1)
        
fig, (ax1, ax2) = plt.subplots(1, 2)
fig.set_size_inches([15,7])
ax1.plot(np.arange(0,Pol_Max-1,1)+1, RSSv,'r',alpha=0.3)
ax1.set_yscale('log')
#ax1.legend(loc='best', frameon=False)
ax1.set(xlabel='Grado del polinomio', ylabel='RSS')
ax1.set(xlim=(0, Pol_Max))
ax1.set_title('')
# Second #figure
fig.tight_layout(pad=5.0)
ax2.plot(np.arange(0,Pol_Max-1,1)+1, AICv,'b',alpha=0.3)
ax2.set(xlabel='Grado del polinomio', ylabel='RSS')
#ax2.legend(loc='best', frameon=False)
ax2.set(xlabel='Grado del polinomio', ylabel='AIC')
ax2.set(xlim=(0, Pol_Max))
ax2.set_title('')
fig.savefig('images/rss.pdf')
plt.show()


#Tenemos un grado 4
grade = np.argmin(AICv)+1

#Cross Validation 
Pol_Max = 11 # Highest degree polynomial we are going to check.
# empty arrays
RSSv = [] 
RSSv = np.zeros(Pol_Max-1)
AICv = []
AICv = np.zeros(Pol_Max-1)

for i in np.arange(0,Pol_Max-1):
    coef = poly.polyfit(X_train, Y_train, i+1)
    model = poly.Polynomial(coef)
    
    RSSv[i]=np.sum((Y_test-model(X_test))**2)
    
    AICv[i]=len(X_test)*math.log10(RSSv[i]/len(X_test))+ 2*len(coef)*len(X_test)/(len(X_test)-len(coef)-1)


fig, (ax1, ax2) = plt.subplots(1, 2)
fig.set_size_inches([15,7])
ax1.plot(np.arange(0,Pol_Max-1,1)+1, RSSv,'r',alpha=0.3)
ax1.set_yscale('log')
#ax1.legend(loc='best', frameon=False)
ax1.set(xlabel='Grado del polinomio', ylabel='RSS')
ax1.set_title("Prueba de bondad sobre el conjunto de prueba.")
ax1.set(xlim=(0, Pol_Max))
ax1.set_title('')
# Second #figure
fig.tight_layout(pad=5.0)
ax2.plot(np.arange(0,Pol_Max-1,1)+1, AICv,'b',alpha=0.3)
ax2.set(xlabel='Grado del polinomio', ylabel='RSS')
#ax2.legend(loc='best', frameon=False)
ax2.set(xlabel='Grado del polinomio', ylabel='AIC')
ax2.set_title("Prueba de bondad sobre el conjunto de prueba.")
ax2.set(xlim=(0, Pol_Max))
ax2.set_title('')
fig.savefig('images/rss_test.pdf')
plt.show()



Pol_Max = 8
grade = []
RSS_train = []
AIC_train = []
RSS_test = []
AIC_test = []
j = 0
for i in np.arange(2,Pol_Max-1):
    coef = poly.polyfit(X_train, Y_train, i)
    model = poly.Polynomial(coef)
    
    grade.append(i)
    RSS_train.append(np.sum((Y_train-model(X_train))**2))
    AIC_train.append(len(X_train)*math.log10(RSS_train[j]/len(X_train))+ 2*len(coef)*len(X_train)/(len(X_train)-len(coef)-1))
    
    RSS_test.append(np.sum((Y_test-model(X_test))**2))
    AIC_test.append(len(X_test)*math.log10(RSS_test[j]/len(X_test))+ 2*len(coef)*len(X_train)/(len(X_test)-len(coef)-1))
    
    j+=1
    
data_to_present = pd.DataFrame(data = {'grapo': grade,
                                       'RSS_train': RSS_train,
                                       'AIC_train':AIC_train,
                                       'RSS_test': RSS_test,
                                       'AIC_test':AIC_test})

print(data_to_present.to_latex(index=False))


#Polinomio final

fig, ax = plt.subplots()
x = np.linspace(-2.5,X_test.max(), 100)
coef = poly.polyfit(X_train, Y_train, 4)
print(coef)
model = poly.Polynomial(coef)
ax.scatter( X_train, Y_train, color = 'red', label = 'train', marker ='s', zorder = -1)
ax.scatter( X_test, Y_test, color = 'blue', label = 'test', marker ='s', alpha = 0.9)
ax.plot(x, model(x), marker ='$ff$', linestyle = ':',linewidth=2.2,
                label = 'degree = {}'.format(4), alpha = 0.8)
ax.legend(loc = 'upper left')
ax.set_ylim([-10,60])
#ax.set_ylim([Y_train.min(),Y_test.max()])
#ax.set_xlim([X_train.min(),X_test.max()])
ax.set_title("Polinomio final de grado 4.")
ax.set_xlabel("x")
ax.set_ylabel("y")
fig.show()
fig.savefig('images/ajustefinal.pdf')

"""
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
test = test.dropna()
poly_features = PolynomialFeatures(degree=grade)
X_poly = poly_features.fit_transform(test)
poly = LinearRegression()
cross_val_score(poly, X_poly, test["Y_test"], cv=5)
"""
