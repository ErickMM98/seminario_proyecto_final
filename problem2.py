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
import numpy as np
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
fig.set_size_inches([10,10])
sns.scatterplot(data=data, x="age", y="y", hue="y", ax = ax[0][0])
sns.scatterplot(data=data, x="cholesterol", y="y", hue="y", ax = ax[0][1])
sns.scatterplot(data=data, x="sugar", y="y", hue="y", ax = ax[1][0])
sns.scatterplot(data=data, x="Tcell", y="y", hue="y", ax = ax[1][1])

ax[0][0].get_legend().remove()
ax[0][0].xaxis.set_tick_params(rotation=45)
ax[0][1].get_legend().remove()
ax[1][0].get_legend().remove()
ax[0][1].set_yticks((0,1))
ax[0][0].set_yticks((0,1))
ax[1][1].set_yticks((0,1))
ax[1][0].set_yticks((0,1))

fig.savefig("images/distribuciones.pdf")

#SEPARAMOS POR 

fig, [[ax0,ax1],[ax2,ax3]] = plt.subplots(2,2)
fig.set_size_inches([10,10])
sns.scatterplot(data=data, x="sugar", y="Tcell", hue="y", ax = ax0)
sns.scatterplot(data=data, x="cholesterol", y="Tcell", hue="y", ax = ax1)
sns.scatterplot(data=data, x="sugar", y="age", hue="y", ax = ax2)
sns.scatterplot(data=data, x="cholesterol", y="age", hue="y", ax = ax3)

fig.savefig("images/scaterplotgoods.pdf")

#---------------------------------------------------------------------------
# VEAMOS SI HAY SIGNIFICANCIA EN LAS CLASES POR LAS VARIABLES 
#---------------------------------------------------------------------------
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from scipy.special import expit

fig, ax = plt.subplots(2,2)
fig.set_size_inches([15,15])
listax = list(ax[0]) +  list(ax[1])
print(listax)
#ax.set_title("Ajustado")
i=0


for columname in data.columns:
    if columname != 'y':
        x_foo_foo = data[columname].astype(float)
        X_foo = np.array(data[columname].astype(float)).reshape(-1, 1)
    
        y_foo = data['y'].astype(int)
    
    
        clf = LogisticRegression(random_state=0).fit(X_foo, y_foo)
        score = clf.score(X_foo, y_foo)
        
        
        #print(clf.score(X_train, y_train))
        #print(clf.score(X_test, y_test))
    
        alpha = clf.intercept_
        beta = clf.coef_.T
        alpha = alpha[0]
        beta = beta[0][0]
        
        #X_train_tranform = np.array(X_foo @ beta + alpha)
    
        x_foo = np.linspace( np.array(X_foo).min() - 50,
                             np.array(X_foo).max() + 50,
                             100)
        x_foo = x_foo.reshape(-1, 1)
        #x_foo_foo = beta * x_foo_foo + alpha
        
        #loss = expit(x_foo_foo)
        ax = listax[i]
        ax.set_title(columname)
        ax.scatter(X_foo, clf.predict_proba(X_foo)[:,1], label = 'score = {}'.format(score),
               color = 'b')
        #ax.set_yticks((0,1))
        ax.scatter(x_foo,clf.predict_proba(x_foo)[:,1],marker='x',color='g',linewidth=.1)
        #ax.plot(x_foo_foo, loss, linewidth=3, color = 'g', alpha = 0.5)
        ax.legend()
        
        i += 1

#ax.legend()
#ax.set_ylabel("Diagnóstico.")
fig.savefig("images/significanciavariables.pdf")


#---------------------------------------------------------------------------
#--------------------------------------------------------------------------
#GENERAMOS EL MODELO SOBRE LAS VARIABLES SUGAR Y TCELL
#--------------------------------------------------------------
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from scipy.special import expit

X = data[['sugar', 'Tcell']]

y = data['y'].astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

print(X_train.shape)
print(X_test.shape)
clf = LogisticRegression(random_state=0).fit(X_train, y_train)
clf.fit(X_train, y_train)
print(clf.score(X_train, y_train))
print(clf.score(X_test, y_test))
#clf.predict(X_test)

print(confusion_matrix(y_test, clf.predict(X_test)))
print(f1_score(y_test, clf.predict(X_test)))

alpha = clf.intercept_
beta = clf.coef_.T
X_train_tranform = np.array(X_train @ beta + alpha)
X_ttest_tranform = np.array(X_test @ beta + alpha)

x_foo = np.linspace(min( [X_train_tranform.min(),
                          X_ttest_tranform.min()] ),
                    max( [X_train_tranform.max(),
                          X_ttest_tranform.max()] ),
                    100)


#print(beta)
loss = expit(x_foo)

fig, ax = plt.subplots()
ax.set_title("Ajustado")
ax.scatter(X_train_tranform, clf.predict_proba(X_train)[:,1], label = 'Train',
           color = 'b')
ax.scatter(X_ttest_tranform, clf.predict_proba(X_test)[:,1], label = 'Test',
           color = 'r')
ax.plot(x_foo, loss, linewidth=3, color = 'g', alpha = 0.5)

ax.legend()
ax.set_ylabel("Diagnóstico.")
fig.savefig("images/ajustado.pdf")
#ax.scatter(x_foo,clf.predict_proba(x_foo)[:,1],marker='x',color='g',linewidth=.1)


#---------------------------------------------------------------------------
#--------------------------------------------------------------------------
#GENERAMOS EL MODELO SOBRE LAS VARIABLES TCELL
#--------------------------------------------------------------
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from scipy.special import expit

X = data[['Tcell']]

y = data['y'].astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

print(X_train.shape)
print(X_test.shape)
clf = LogisticRegression(random_state=0).fit(X_train, y_train)
clf.fit(X_train, y_train)
print(clf.score(X_train, y_train))
print(clf.score(X_test, y_test))
#clf.predict(X_test)

print(confusion_matrix(y_test, clf.predict(X_test)))
print(f1_score(y_test, clf.predict(X_test)))

alpha = clf.intercept_
beta = clf.coef_.T
X_train_tranform = np.array(X_train @ beta + alpha)
X_ttest_tranform = np.array(X_test @ beta + alpha)

x_foo = np.linspace(min( [X_train_tranform.min(),
                          X_ttest_tranform.min()] ),
                    max( [X_train_tranform.max(),
                          X_ttest_tranform.max()] ),
                    100)


print(alpha)
print(beta)
loss = expit(x_foo)

fig, ax = plt.subplots()
ax.set_title("Ajustado")
ax.scatter(X_train_tranform, clf.predict_proba(X_train)[:,1], label = 'Train',
           color = 'b')
ax.scatter(X_ttest_tranform, clf.predict_proba(X_test)[:,1], label = 'Test',
           color = 'r')
ax.plot(x_foo, loss, linewidth=3, color = 'g', alpha = 0.5)

ax.legend()
ax.set_ylabel("Diagnóstico.")
fig.savefig("images/ajustado_tcell.pdf")
#ax.scatter(x_foo,clf.predict_proba(x_foo)[:,1],marker='x',color='g',linewidth=.1)



