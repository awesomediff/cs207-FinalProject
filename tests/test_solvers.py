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
