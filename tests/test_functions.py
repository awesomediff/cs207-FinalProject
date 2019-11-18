import numpy as np
import pytest

import awesomediff as ad


def test_simpleFunc1():
    def f1(x):
        return 2*x*np.exp(x)+np.sqrt(x)

    def f1_dx(x):
        return 2*x*np.exp(x)+2*np.exp(x)+1/(2*np.sqrt(x))

    x1 = ad.variable(22)
    f = 2*x1*ad.exp(x1)+ad.sqrt(x1)
    assert f.val==f1(22)
    assert f.der == f1_dx(22)

def test_simpleFunc1():
    def f1(x):
        return 2*x*np.exp(x)+np.sqrt(x)

    def f1_dx(x):
        return 2*x*np.exp(x)+2*np.exp(x)+1/(2*np.sqrt(x))

    x1 = ad.variable(22)
    f = 2*x1*ad.exp(x1)+ad.sqrt(x1)
    assert f.val==f1(22)
    assert f.der == f1_dx(22)

def test_simpleFunc2():
    def f1(x):
        return x**4-30/x

    def f1_dx(x):
        return 4*x**3+30/x**2

    x1 = ad.variable(46)
    f = x1**4-30/x1
    assert f.val==f1(46)
    assert f.der == f1_dx(46)

def test_simpleFunc3():
    def f1(x):
        return np.sin(x)/3+np.cos(x)/x

    def f1_dx(x):
        return -np.sin(x)/x-np.cos(x)/x**2+np.cos(x)/3

    x1 = ad.variable(83)
    f = ad.sin(x1)/3+ad.cos(x1)/x1
    assert f.val==f1(83)
    assert f.der == f1_dx(83)

def test_simpleFunc4():
    def f1(x):
        return 17*np.log(x)+25+1/x

    def f1_dx(x):
        return 17/x-1/x**2

    x1 = ad.variable(39)
    f = 17*ad.log(x1)+25+1/x1
    assert f.val==f1(39)
    assert f.der == f1_dx(39)

def test_simpleFunc5():
    def f1(x):
        return 254*np.sqrt(x)-np.tan(x)+1

    def f1_dx(x):
        return 127/np.sqrt(x)-(1/np.cos(x))**2

    x1 = ad.variable(65)
    f = 254*ad.sqrt(x1)-ad.tan(x1)+1
    assert f.val==f1(65)
    assert f.der == f1_dx(65)
    
    
    
def test_simpleFunc6():
    func11_val = lambda x: ((x**3) + 4*x) / np.sin(3)
    func11_der = lambda x: ((3 * x**2) + 4) / np.sin(3)
    
    a11 = 6
    x11 = ad.variable(a11)
    
    f11 = ((x11**3) + 4*x11) / ad.sin(3)
    assert f11.val == func11_val(a11)
    assert f11.der == func11_der(a11)
    
    
def test_simpleFunc7():
    func12_val = lambda x: np.sqrt(17) / np.exp(2*x)
    func12_der = lambda x: -2*np.sqrt(17) * np.exp(-2*x)
    
    a12 = 2
    x12 = ad.variable(a12)
    
    f12 = ad.sqrt(17) / ad.exp(2*x12)
    assert f12.val == func12_val(a12)
    assert f12.der == func12_der(a12)
    
    
def test_simpleFunc8():
    func13_val = lambda x: np.log(5**2 - 2*x**2)
    func13_der = lambda x: (-4*x) / (25 - 2*x**2)
    
    a13 = 1/2
    x13 = ad.variable(a13)
    
    f13 = ad.log(5**2 - 2*x13**2)
    assert f13.val == func13_val(a13)
    assert f13.der == func13_der(a13)
    
    
def test_simpleFunc9():
    func14_val = lambda x: np.sin(x) / (3 - 2*np.cos(x))
    func14_der = lambda x: (-2 + 3*np.cos(x)) / (3 - 2*np.cos(x))**2
    
    a14 = np.pi/4
    x14 = ad.variable(a14)
    
    f14 = ad.sin(x14) / (3 - 2*ad.cos(x14))
    assert np.isclose(f14.val, func14_val(a14))
    assert np.isclose(f14.der, func14_der(a14))
    
    
def test_simpleFunc10():
    func15_val = lambda x: 3*x**(-4) - x**2 * np.tan(x)
    func15_der = lambda x: (-12/x**5) - 2*x*np.tan(x) - x**2*((np.tan(x))**2 + 1)
    
    a15 = 0.7
    x15 = ad.variable(a15)
    
    f15 = 3*x15**(-4) - x15**2 * ad.tan(x15)
    assert np.isclose(f15.val, func15_val(a15))
    assert np.isclose(f15.der, func15_der(a15))
