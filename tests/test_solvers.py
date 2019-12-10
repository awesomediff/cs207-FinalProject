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

def test_helpers():

    # Check inputs:
    X = [[1,2],[3,4]]
    y1 = [[1,2]]
    y2 = [1,2,3]
    with pytest.raises(AssertionError):
        X,y = ad.solvers._check_inputs(X,y2)

    
    # Transpose:
    M = [[1,2,3],[4,5,6]]
    Mt = ad.transpose(M,check=True)
    assert Mt == [[1,4],[2,5],[3,6]]

    # Mean and variance:
    assert ad.mean([1,2,3,4,5])==np.mean([1,2,3,4,5])
    assert ad.variance([1,2,3,4,5])==np.var([1,2,3,4,5])

    # Standardization:
    X = np.array([[-1,4,-1,-3],[1,-3,1,2],[0,2,0,6],[-2,-5,1,-6],[3,5,1,3],[-3,-5,-3,-3]])
    ad_result = np.array( ad.standardize(X,check=True) ).flatten()
    np_result = ( (X-X.mean(axis=0))/X.std(axis=0) ).flatten()
    assert np.isclose( ad_result , np_result ).all()

    # MSE:
    y_true = np.array([2,4,25,4,25,63,44])
    y_pred = np.array([3,6,27,5,22,66,45])
    ad_mse = ad.mean_squared_error(y_true,y_pred)
    np_mse = ((y_true-y_pred)**2).mean()
    assert ad_mse == np_mse

    # RSS:
    y_true = np.array([2,4,25,4,25,63,44])
    y_pred = np.array([3,6,27,5,22,66,45])
    ad_rss = ad.sum_square_residuals(y_true,y_pred)
    np_rss = ((y_true-y_pred)**2).sum()
    assert ad_rss == np_rss

    # TSS:
    y_true = np.array([2,4,25,4,25,63,44])
    ad_tss = ad.total_sum_squares(y_true)
    np_tss = ((y_true-y_true.mean())**2).sum()
    assert ad_tss == np_tss

    # R2 score:
    y_true = np.array([2,4,25,4,25,63,44])
    y_pred = np.array([3,6,27,5,22,66,45])
    ad_r2 = ad.r2_score(y_true,y_pred)
    np_r2 = 1 - ad.sum_square_residuals(y_true,y_pred)/((y_true-y_true.mean())**2).sum()
    assert ad_r2 == np_r2

def test_superclasses():

    X = np.array([[-1,4,-1,-3],[1,-3,1,2],[0,2,0,6],[-2,-5,1,-6],[3,5,1,3],[-3,-5,-3,-3]])
    y = np.array([13,-2.5,-1,6,15.5,-23.5])
    model_params = dict()
    weights = []
    placeholder_argument = None

    model = ad.Model()
    with pytest.raises(NotImplementedError):
        model._pack_weights(model_params,placeholder_argument)
    with pytest.raises(NotImplementedError):
        model._unpack_weights(model_params,weights)
    with pytest.raises(NotImplementedError):
        model._loss(model_params,weights,X,y)
    with pytest.raises(NotImplementedError):
        model._predict(model_params,weights,X)
    with pytest.raises(NotImplementedError):
        model.predict(X)
    with pytest.raises(NotImplementedError):
        model.fit(X,y)
    with pytest.raises(NotImplementedError):
        model.score(X,y)

    initial_weights = []

    solver = ad.Solver(model)
    with pytest.raises(NotImplementedError):
        solver.solve(model_params,initial_weights,X,y)

def test_linear_regression():

    # Test with and without intercept; with and without standardization.

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

def test_lasso_regression():

    X = np.array([[-1,4,-1,-3],[1,-3,1,2],[0,2,0,6],[-2,-5,1,-6],[3,5,1,3],[-3,-5,-3,-3]])
    y = np.array([13,-2.5,-1,6,15.5,-23.5])

    reg = ad.LassoRegression(fit_intercept=True,standardize=True,max_iter=2000,learning_rate=0.05,l1_penalty=1.0)
    reg.fit(X,y)
    assert reg.score(X,y)>0.99

def test_ridge_regression():

    X = np.array([[-1,4,-1,-3],[1,-3,1,2],[0,2,0,6],[-2,-5,1,-6],[3,5,1,3],[-3,-5,-3,-3]])
    y = np.array([13,-2.5,-1,6,15.5,-23.5])

    reg = ad.RidgeRegression(fit_intercept=True,standardize=True,max_iter=2000,learning_rate=0.05,l2_penalty=1.0)
    reg.fit(X,y)
    assert reg.score(X,y)>0.99
