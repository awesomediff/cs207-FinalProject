import pytest
  
import awesomediff as ad
import numpy as np


### Test uni_Newton
def test_uni_Newton_func():
    def root_finding(a):
        return a**2 + 2*a + 1
    root = ad.uni_Newton(root_finding, 100, max_iter=100, epsilon=1e-06)
    y_val = root_finding(root)
    assert (np.isclose(y_val, 0, atol=1e-06))

    def root_finding(a): #function with no root
        return a**2 + 2*a + 2
    root = ad.uni_Newton(root_finding, 100) #reach max iteration, and return none
    assert root is None

    def root_finding(a): #bad starting point: derivative = 0
        return -a**2 + 1
    root = ad.uni_Newton(root_finding, 0) #return None
    assert root is None

    def root_finding(a):
        return ad.sin(a)
    root = ad.uni_Newton(root_finding, 3, max_iter=100, epsilon=1e-08)
    y_val = root_finding(root)
    assert (np.isclose(y_val.val, 0, atol=1e-07))

    #check input format
    with pytest.raises(ValueError):
        ad.uni_Newton(root_finding, 's')

    def root_finding(a,b): #function with two inputs
        return a**2 + 2*a + 2
    with pytest.raises(ValueError):
        ad.uni_Newton(root_finding, 3)

def test_linear_regression():

    X = np.array([[-1,4,-1,-3],[1,-3,1,2],[0,2,0,6],[-2,-5,1,-6],[3,5,1,3],[-3,-5,-3,-3]])
    y = np.array([13,-2.5,-1,6,15.5,-23.5])

    reg = ad.LinearRegression(fit_intercept=False,standardize=False,max_iter=1000,learning_rate=0.01)
    reg.fit(X,y)
    ad_result = reg.coefs
    sklearn_intercept = None
    sklearn_coefs = [-1.38918162,2.79817086,6.6314567,-1.40219314]
    sklearn_score = 0.9571515374025673
    assert reg.intercept is None
    assert np.isclose(ad_result,sklearn_coefs,atol=0.1).all()
    assert np.isclose(reg.score(X,y),sklearn_score,atol=0.01).all()

    reg = ad.LinearRegression(fit_intercept=False,standardize=True,max_iter=2000,learning_rate=0.05)
    reg.fit(X,y)
    ad_result = reg.coefs
    sklearn_intercept = None
    sklearn_coefs = [-1.67457727,11.41779384,9.45887647,-6.28158263]
    sklearn_score = 0.9900476542631219
    assert reg.intercept is None
    assert np.isclose(ad_result,sklearn_coefs,atol=0.1).all()
    assert np.isclose(reg.score(X,y),sklearn_score,atol=0.01).all()

    reg = ad.LinearRegression(fit_intercept=True,standardize=False,max_iter=1000,learning_rate=0.01)
    reg.fit(X,y)
    ad_result = reg.coefs
    sklearn_intercept = 2.709096070445851
    sklearn_coefs = [-0.84916566,2.75129781,6.46763409,-1.51732197]
    sklearn_score = 0.999466215107025
    assert np.isclose(reg.intercept,sklearn_intercept,atol=0.1)
    assert np.isclose(ad_result,sklearn_coefs,atol=0.1).all()
    assert np.isclose(reg.score(X,y),sklearn_score,atol=0.01).all()

    reg = ad.LinearRegression(fit_intercept=True,standardize=True,max_iter=2000,learning_rate=0.05)
    reg.fit(X,y)
    ad_result = reg.coefs
    sklearn_intercept = 1.2500000000000004
    sklearn_coefs = [-1.67457727,11.41779384,9.45887647,-6.28158263]
    sklearn_score = 0.999466215107025
    assert np.isclose(reg.intercept,sklearn_intercept,atol=0.1)
    assert np.isclose(ad_result,sklearn_coefs,atol=0.1).all()
    assert np.isclose(reg.score(X,y),sklearn_score,atol=0.01).all()
