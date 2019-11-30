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

    #rpow
    a = 3
    x = ad.variable(a)
    f = 2**x
    assert f.val == 8
    assert f.der == 2**a * np.log(2)

    with pytest.raises(ValueError):
        # exponent should be a number
        's'**x

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

def test_notequal():
    x = ad.variable(3.0)
    y = ad.variable(4.0)

    assert x!=y

    alpha = 7.0
    beta = 8.0
    f1= alpha * x + beta

    f5 = beta * x + alpha
    assert f1!=f5

    assert f1!=3

def test_less_than():
    x = ad.variable(3.0)
    y = ad.variable(4.0)

    a = 1

    assert x<y

def test_greater_than():
    x = ad.variable(3.0)
    y = ad.variable(4.0)
    z = ad.variable(4.0)

    assert not(y>z)
    assert y>x
    assert z>x

def test_less_equal():
    x = ad.variable(3.0)
    y = ad.variable(4.0)
    z = ad.variable(4.0)

    assert x<=y
    assert z<=y
    assert y<=z

def test_greater_equal():
    x = ad.variable(3.0)
    y = ad.variable(3.0)
    z = ad.variable(5.0)

    assert y>=x
    assert x>=y
    assert z>=x
    assert z>=y