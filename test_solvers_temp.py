import pytest

import numpy as np

import awesomediff as ad


#import numpy as np
import pandas as pd

from sklearn import datasets
from sklearn.linear_model import LinearRegression
#from sklearn.linear_model import Ridge

#import awesomediff as ad



def solve_ols_exact(X,y):

    pass


if __name__=="__main__":
        
    X = [[3,-1,2,8],[1,-3,2,6],[3,2,3,8],[4,4,3,8],[1,2,5,8]]
    y = [45,26,50,60,47]

    if False:

        print(X)
        print(ad.transpose(X,check=False))
        print(ad.standardize(X,check=False,return_stats=True))

    if False:
        
        #reg = ad.LinearRegression(fit_intercept=True,standardize=True,solver='gradient_descent',learning_rate=0.3,abs_tol=1e-3,max_iter=10000,verbose=True)
        reg = ad.RidgeRegression(fit_intercept=True,standardize=True,solver='gradient_descent',learning_rate=0.1,abs_tol=1e-4,max_iter=10000,verbose=True,l2_penalty=0)
        reg.fit(X,y)
    
        print(reg.predict(X))
    
        print(reg.intercept,reg.coefs)
    
        print(reg.score(X,[45,26,50,60,47]))

    if True:

        dataset = datasets.load_boston()
        X = dataset['data']
        y = dataset['target']
        
        # Standardize data:
        feature_mean = X.mean(axis=0)
        feature_stdev = X.std(axis=0)
        X = (X-feature_mean)/feature_stdev

        print(X.shape)
        print(y.shape)
        
        for ad_reg in [
            ad.LinearRegression(fit_intercept=True,solver='gradient_descent',learning_rate=0.01,max_iter=10000,verbose=False),
            ad.LassoRegression(fit_intercept=True,solver='gradient_descent',learning_rate=0.01,max_iter=10000,verbose=False,l1_penalty=0.1),
            ad.RidgeRegression(fit_intercept=True,solver='gradient_descent',learning_rate=0.01,max_iter=10000,verbose=False,l2_penalty=0.1),
        ]:

            ad_reg.fit(X,y)
            print("R2 score:",ad_reg.score(X,y),"\n")
            print("intercept:",ad_reg.intercept,"\n")
            print("coefs:",ad_reg.coefs,"\n")
