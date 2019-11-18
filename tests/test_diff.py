import pytest

import awesomediff as ad
import numpy as np

def test_simple_functions_1():
    
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

def test_single_variable_functions():
    
    # test add, pow, rmul, sin
    func11_val = lambda x: ((x**3) + 4*x) / np.sin(3)
    func11_der = lambda x: ((3 * x**2) + 4) / np.sin(3)
    
    a11 = 6
    x11 = ad.variable(a11)
    
    f11 = ((x11**3) + 4*x11) / ad.sin(3)
    assert f11.val == func11_val(a11)
    assert f11.der == func11_der(a11)
    
    
    # test rtruediv, sqrt, pow
    func12_val = lambda x: np.sqrt(17) / np.exp(2*x)
    func12_der = lambda x: -2*np.sqrt(17) * np.exp(-2*x)
    
    a12 = 2
    x12 = ad.variable(a12)
    
    f12 = ad.sqrt(17) / ad.exp(2*x12)
    assert f12.val == func12_val(a12)
    assert f12.der == func12_der(a12)
    
    
    # test log, pow, sub, mul
    func13_val = lambda x: np.log(5**2 - 2*x**2)
    func13_der = lambda x: (-4*x) / (25 - 2*x**2)
    
    a13 = 1/2
    x13 = ad.variable(a13)
    
    f13 = ad.log(5**2 - 2*x13**2)
    assert f13.val == func13_val(a13)
    assert f13.der == func13_der(a13)
    
    
    # test sin, cos, truediv, sub, mul
    func14_val = lambda x: np.sin(x) / (3 - 2*np.cos(x))
    func14_der = lambda x: (-2 + 3*np.cos(x)) / (3 - 2*np.cos(x))**2
    
    a14 = np.pi/4
    x14 = ad.variable(a14)
    
    f14 = ad.sin(x14) / (3 - 2*ad.cos(x14))
    assert np.isclose(f14.val, func14_val(a14))
    assert np.isclose(f14.der, func14_der(a14))
    
    
    # test sin, cos, truediv, sub, mul
    func15_val = lambda x: 3*x**(-4) - x**2 * np.tan(x)
    func15_der = lambda x: (-12/x**5) - 2*x*np.tan(x) - x**2*((np.tan(x))**2 + 1)
    
    a15 = 0.7
    x15 = ad.variable(a15)
    
    f15 = 3*x15**(-4) - x15**2 * ad.tan(x15)
    assert np.isclose(f15.val, func15_val(a15))
    assert np.isclose(f15.der, func15_der(a15))
