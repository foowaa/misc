# -*- coding: utf-8 -*-
"""
Created on Sat Sep 19 08:32:35 2016

@author: cltian
"""

import numpy as np
from matplotlib import pyplot as plt
from sklearn.gaussian_process import GaussianProcess
import pandas as pd
from pandas import read_csv
#from sklearn import preprocessing

#read from csv file
path = r"..\dataset\11nn.csv"
target = read_csv(path)


gp = GaussianProcess(corr='cubic', theta0=1e-2, thetaL=1e-4, thetaU=1e-1,
                     random_start=100, normalize=True)
                    
X=target.loc[0:13,['fert','mach','popu','ppa','lp','cat']]
y = target.loc[0:13,['land']]
gp.fit(X,y)

                             
Xp=target.loc[14,['fert','mach','popu','ppa','lp','cat']]
yp=target.loc[14,['land']]
#xp=df.loc[:,['A','B','C','D', 'E','F']]
y_pred, MSE = gp.predict(Xp, eval_MSE=True)
#X_scaled = preprocessing.scale(y_pred)
sigma = np.sqrt(MSE)
y_pred_n = y_pred*1.05
#find the best lP

n = 0
while(True):
    old = gp.predict(Xp)
    target.loc[14,'lp'] += 0.01
    new = gp.predict(target.loc[14,['fert','mach','popu','ppa','lp','cat']])
    n += 1
    if(new>=y_pred_n):
        break
    if(n>100):
        print target.loc[14,'lp']
        break

print target.loc[14,'lp']


