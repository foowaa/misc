# -*- coding: utf-8 -*-
"""
@author: cltian
"""
import numpy as np
from matplotlib import pyplot as plt
from sklearn.gaussian_process import GaussianProcess
from sklearn import linear_model
import pandas as pd
from pandas import read_csv
#from sklearn import preprocessing

#read from csv file
path = r"..\dataset\d4.csv"
target = read_csv(path)
#target['cat'].fillna(target['cat'].median())


gp = GaussianProcess( theta0=1e-2, thetaL=1e-4, thetaU=1e-1,
                     random_start=100, normalize=True)
                    
X=target.loc[0:8,['fert','land','mach','popu','ppa','lp','cat']]
y = target.loc[0:8,['pro']]
gp.fit(X,y)

#mesh the data to create the smooth regression curve
df = pd.DataFrame({'A' :np.linspace(4120,6000,20),
                   'G' :np.linspace(32000,37000,20),
                   'B' :np.linspace(52000,110000,20),
                   'C' :np.linspace(48200,54000,20),
                   'D' :np.linspace(43000,61000,20),
                   'E' :np.linspace(0,1.3,20),
                   'F' :np.linspace(32000,55000,20)})

                                   
#Xp=target.loc[13:14,['fert','mach','popu','ppa','lp','cat']]
#yp=target.loc[13:14,['land']]
xp=df.loc[:,['A','G','B','C','D', 'E','F']]
y_pred, MSE = gp.predict(xp, eval_MSE=True)
#X_scaled = preprocessing.scale(y_pred)
sigma = np.sqrt(MSE)

#plot the result

#Fertilizers
fig = plt.figure()

plt.plot(X['fert'],y,'r.',markersize=10,label='fertilizer')
plt.plot(df['A'],y_pred,'r-')
plt.fill(np.concatenate([df['A'],df['A'][::-1]]), np.concatenate([y_pred-1.960*sigma,y_pred+1.960*sigma[::-1]]),
        alpha=.3, fc='b', ec='None')
plt.xlabel('Fertilizers')
plt.ylabel('Production')

#Machinery
fig = plt.figure()

plt.plot(X['mach'],y,'r.',markersize=10)
plt.plot(df['B'],y_pred,'r-')
plt.fill(np.concatenate([df['B'],df['B'][::-1]]), np.concatenate([y_pred-1.960*sigma,y_pred+1.960*sigma[::-1]]),
        alpha=.3, fc='b', ec='None')
plt.xlabel('Machinery')
plt.ylabel('Production')


#Population
fig = plt.figure()

plt.plot(X['popu'],y,'r.',markersize=10)
plt.plot(df['C'],y_pred,'r-')
plt.fill(np.concatenate([df['C'],df['C'][::-1]]), np.concatenate([y_pred-1.960*sigma,y_pred+1.960*sigma[::-1]]),
        alpha=.3, fc='b', ec='None')
plt.xlabel('Agricultural Population ')
plt.ylabel('Production')

#Prodution/area
fig = plt.figure()

plt.plot(X['ppa'],y,'r.',markersize=10)
plt.plot(df['D'],y_pred,'r-')
plt.fill(np.concatenate([df['D'],df['D'][::-1]]), np.concatenate([y_pred-1.960*sigma,y_pred+1.960*sigma[::-1]]),
        alpha=.3, fc='b', ec='None',)
plt.xlabel('Production per h.a. ')
plt.ylabel('Production')



#catastrophy
fig = plt.figure()

plt.plot(X['cat'],y,'r.',markersize=10)
plt.plot(df['F'],y_pred,'r-')
plt.fill(np.concatenate([df['F'],df['F'][::-1]]), np.concatenate([y_pred-1.960*sigma,y_pred+1.960*sigma[::-1]]),
        alpha=.3, fc='b', ec='None')
plt.xlabel('Catastrophy')
plt.ylabel('Production')

#land
fig = plt.figure()

plt.plot(X['land'],y,'r.',markersize=10)
plt.plot(df['G'],y_pred,'r-')
plt.fill(np.concatenate([df['G'],df['G'][::-1]]), np.concatenate([y_pred-1.960*sigma,y_pred+1.960*sigma[::-1]]),
        alpha=.3, fc='b', ec='None')
plt.xlabel('Land')
plt.ylabel('Production')


#OLS
X1=target.loc[0:8,['pro']]
y1 = target.loc[0:8,['price']]
clf = linear_model.LinearRegression()
clf.fit(X1, y1)

xc = target.loc[9:10,['fert','land','mach','popu','ppa','lp','cat']]
"""
plt.figure()
plt
"""
print clf.coef_
print clf.intercept_
coef = clf.coef_
intercept = clf.intercept_
sigma = sigma*coef*coef
y_pred=y_pred*coef+intercept
#Fertilizers
fig = plt.figure()

plt.plot(X['fert'],y1,'r.',markersize=10,label='fertilizer')
plt.plot(df['A'],y_pred,'r-')
plt.fill(np.concatenate([df['A'],df['A'][::-1]]), np.concatenate([y_pred-1.960*sigma,y_pred+1.960*sigma[::-1]]),
        alpha=.3, fc='b', ec='None')
plt.xlabel('Fertilizers')
plt.ylabel('Price')

#Machinery
fig = plt.figure()

plt.plot(X['mach'],y1,'r.',markersize=10)
plt.plot(df['B'],y_pred,'r-')
plt.fill(np.concatenate([df['B'],df['B'][::-1]]), np.concatenate([y_pred-1.960*sigma,y_pred+1.960*sigma[::-1]]),
        alpha=.3, fc='b', ec='None')
plt.xlabel('Machinery')
plt.ylabel('Price')


#Population
fig = plt.figure()

plt.plot(X['popu'],y1,'r.',markersize=10)
plt.plot(df['C'],y_pred,'r-')
plt.fill(np.concatenate([df['C'],df['C'][::-1]]), np.concatenate([y_pred-1.960*sigma,y_pred+1.960*sigma[::-1]]),
        alpha=.3, fc='b', ec='None')
plt.xlabel('Agricultural Population ')
plt.ylabel('Price')

#Prodution/area
fig = plt.figure()

plt.plot(X['ppa'],y1,'r.',markersize=10)
plt.plot(df['D'],y_pred,'r-')
plt.fill(np.concatenate([df['D'],df['D'][::-1]]), np.concatenate([y_pred-1.960*sigma,y_pred+1.960*sigma[::-1]]),
        alpha=.3, fc='b', ec='None',)
plt.xlabel('Production per h.a. ')
plt.ylabel('Price')

#low-price
fig = plt.figure()

plt.plot(X['lp'],y1,'r.',markersize=10)
plt.plot(df['E'],y_pred,'r-')
plt.fill(np.concatenate([df['E'],df['E'][::-1]]), np.concatenate([y_pred-1.960*sigma,y_pred+1.960*sigma[::-1]]),
        alpha=.3, fc='b', ec='None')
plt.xlabel('Lowest Price')
plt.ylabel('Price')


#catastrophy
fig = plt.figure()

plt.plot(X['cat'],y1,'r.',markersize=10)
plt.plot(df['F'],y_pred,'r-')
plt.fill(np.concatenate([df['F'],df['F'][::-1]]), np.concatenate([y_pred-1.960*sigma,y_pred+1.960*sigma[::-1]]),
        alpha=.3, fc='b', ec='None')
plt.xlabel('Catastrophy')
plt.ylabel('Price')

#land
fig = plt.figure()

plt.plot(X['land'],y1,'r.',markersize=10)
plt.plot(df['G'],y_pred,'r-')
plt.fill(np.concatenate([df['G'],df['G'][::-1]]), np.concatenate([y_pred-1.960*sigma,y_pred+1.960*sigma[::-1]]),
        alpha=.3, fc='b', ec='None')
plt.xlabel('Land')
plt.ylabel('Price')



