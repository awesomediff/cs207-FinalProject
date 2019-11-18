import pytest

import awesomediff as ad
import numpy as np

def test_operations():
    
    # Addition and multiplication:
    a = 5.0
    x = ad.variable(a)
    alpha = 2.0
    beta = 3.0

    f = alpha * x + beta
    assert f.val == 13
    assert f.der == 2

    f = x * alpha + beta
    assert f.val == 13
    assert f.der == 2

    f = beta + alpha * x
    assert f.val == 13
    assert f.der == 2

    f = beta + x * alpha
    assert f.val == 13
    assert f.der == 2
    
    # Power:
    a = 5.0
    x = ad.variable(a)
    f = x**2
    
    assert f.val == 25
    assert f.der == 10
    
    a= 3.0
    d = 2.0
    x = ad.variable(a, d)
    f = x**3
    assert f.val == 27
    assert f.der == 54

    with pytest.raises(ValueError):
        # power should be a number
        x**'n'
    
    #division(truediv)
    x = ad.variable(2)
    y = ad.variable(4)
    f = x/y
    assert f.val == 0.5
    assert f.der == 0.125

    x = ad.variable(2)
    with pytest.raises(ZeroDivisionError):
        # cannot perform division by zero
        x/0

    with pytest.raises(ValueError):
        # should be a scalar or instance variable
        x/'n'
    
    x = ad.variable(2,2)
    f = x/2
    assert f.val == 1
    assert f.der == 1
    
    #division(rtruediv)
    x = ad.variable(4)
    f = 4/x
    assert f.val == 1
    assert f.der == -0.25

def test_equal():
    x = ad.variable(3.0)
    y = ad.variable(3.0)

    assert x==y

    x = ad.variable(3.0)
    y = ad.variable(4.0)
    assert x!=y

    alpha = 2.0
    beta = 3.0
    f1= alpha * x + beta
    f2 = x * alpha + beta
    f3 = beta + alpha * x
    f4 = beta + x * alpha

    assert f1==f2
    assert f1==f3
    assert f1==f4

    f5 = beta * x + alpha
    assert f1!=f5

    z = 3.0
    assert x!=z