import numpy as np
import pytest

import awesomediff as ad

def test_sin():

    # Sine of variable:
    f1 = ad.sin(np.pi)
    assert np.isclose(f1.val,0.0)
    assert np.isclose(f1.der,0.0)

    # Sine of variable:
    x2 = ad.variable(np.pi)
    f2 = ad.sin(x2)
    assert np.isclose(f2.val,0.0)
    assert np.isclose(f2.der,-1.0)

    # Sine of variable times constant:
    x3 = ad.variable(0.0)
    f3 = ad.sin(x3)*5
    assert np.isclose(f3.val,0.0)
    assert np.isclose(f3.der,5.0)

    # Sine of constant times variable:
    x4 = ad.variable(np.pi)
    f4 = ad.sin(x4*2)
    assert np.isclose(f4.val,0.0)
    assert np.isclose(f4.der,2.0)

def test_cos():

    # Cosine of variable:
    f1 = ad.cos(np.pi*3/4)
    assert np.isclose(f1.val,-np.sqrt(2)/2)
    assert np.isclose(f1.der,0)

    # Cosine of variable:
    x2 = ad.variable(np.pi)
    f2 = ad.cos(x2)
    assert np.isclose(f2.val,-1.0)
    assert np.isclose(f2.der,0.0)

    # Cosine of variable times constant:
    x3 = ad.variable(0.0)
    f3 = ad.cos(x3)*5
    assert np.isclose(f3.val,5.0)
    assert np.isclose(f3.der,0.0)

    # Cosine of constant times variable:
    x4 = ad.variable(np.pi)
    f4 = ad.cos(x4*0.5)
    assert np.isclose(f4.val,0.0)
    assert np.isclose(f4.der,-0.5)

def test_tan():

    # Tangent of variable:
    f1 = ad.tan(np.pi)
    assert np.isclose(f1.val,0.0)
    assert np.isclose(f1.der,0.0)

    # Tangent of variable:
    x2 = ad.variable(np.pi)
    f2 = ad.tan(x2)
    assert np.isclose(f2.val,0.0)
    assert np.isclose(f2.der,1.0)

    # Tangent of variable times constant:
    x3 = ad.variable(np.pi/4)
    f3 = ad.tan(x3)*5
    assert np.isclose(f3.val,5.0)
    assert np.isclose(f3.der,10.0)

    # Tangent of constant times variable:
    x4 = ad.variable(np.pi*2)
    f4 = ad.tan(x4*0.5)
    assert np.isclose(f4.val,0.0)
    assert np.isclose(f4.der,0.5)

def test_log():
    # log of a scalar
    f1 = ad.log(1)
    assert np.allclose(f1.val, 0)

    f2 = ad.log(10)
    assert np.allclose(f2.val, np.log(10))

    # log of a variable
    x3 = ad.variable(1)
    f3 = ad.log(x3)
    assert np.allclose(f3.val, 0)
    assert np.allclose(f3.der, 1)

    x4 = ad.variable(3)
    f4 = ad.log(x4)*5+1
    assert np.allclose(f4.val, np.log(3)*5+1)
    assert np.allclose(f4.der, 5/3)

def test_sqrt():
    # square root of a scalar
    f1 = ad.sqrt(81)
    assert np.allclose(f1.val, 9)

    # square root of a variable
    x2 = ad.variable(49)
    f2 = ad.sqrt(x2)
    assert np.allclose(f2.val, 7)
    assert np.allclose(f2.der, 1/14)

    x3 = ad.variable(64)
    f3 = 5+2*ad.sqrt(x3)
    assert np.allclose(f3.val, 21)
    assert np.allclose(f3.der, 1/8)

def test_exp():
    # exponential of a scalar
    f1 = ad.exp(10)
    assert np.allclose(f1.val, np.exp(10))

    # exponential of a variable
    x2 = ad.variable(5)
    f2 = ad.exp(x2)
    assert np.allclose(f2.val, np.exp(5))
    assert np.allclose(f2.der, np.exp(5))

    x3 = ad.variable(4)
    f3 = 5+2*ad.exp(x3)
    assert np.allclose(f3.val, 5+2*np.exp(4))
    assert np.allclose(f3.der, 2*np.exp(4))
    
    
def test_sinh():
    # sinh of a scalar
    f1 = ad.sinh(2)
    assert np.allclose(f1.val, (np.exp(2)-np.exp(-2))/2)
    
    # sinh of a variable
    x2 = ad.variable(4)
    f2 = ad.sinh(x2)
    assert np.allclose(f2.val, (np.exp(4)-np.exp(-4))/2)
    assert np.allclose(f2.der, (np.exp(4)+np.exp(-4))/2)
    
    x3 = ad.variable(-1)
    f3 = ad.sinh(x3)
    assert np.allclose(f3.val, (np.exp(-1)-np.exp(1))/2)
    assert np.allclose(f3.der, (np.exp(-1)+np.exp(1))/2)
    
    
def test_cosh():
    # cosh of a scalar
    f1 = ad.cosh(0)
    assert np.allclose(f1.val, (np.exp(0)+np.exp(0))/2)
    
    # cosh of a variable
    x2 = ad.variable(2)
    f2 = ad.cosh(x2)
    assert np.allclose(f2.val, (np.exp(2)+np.exp(-2))/2)
    assert np.allclose(f2.der, (np.exp(2)-np.exp(-2))/2)
    
    x3 = ad.variable(10)
    f3 = ad.cosh(x3)
    assert np.allclose(f3.val, (np.exp(10)-np.exp(-10))/2)
    assert np.allclose(f3.der, (np.exp(10)+np.exp(-10))/2)
    

def test_tanh():
    th = lambda x: ((np.exp(x)-np.exp(-x))/2) / ((np.exp(x)+np.exp(-x))/2)
    th_der = lambda x: 1 / ((np.exp(x)+np.exp(-x))/2)**2
    # tanh of a scalar
    f1 = ad.tanh(3)
    assert np.allclose(f1.val, th(3))
    
    # tanh of a variable
    x2 = ad.variable(-2)
    f2 = ad.tanh(x2)
    assert np.allclose(f2.val, th(-2))
    assert np.allclose(f2.der, th_der(-2))
    
    x3 = ad.variable(0)
    f3 = ad.tanh(x3)
    assert np.allclose(f3.val, th(0))
    assert np.allclose(f3.der, th_der(0))

def test_logistic():
    L = 1
    k = 1
    x0 = 1
    logistic = lambda x: L/(1 + np.exp(-k*(x-x0)))
    log_der = lambda x: (1-logistic(x))*logistic(x)

    f1 = ad.logistic(5)
    assert np.allclose(f1.val, logistic(5))

    x2 = ad.variable(10)
    f2 = ad.logistic(x2)
    assert np.allclose(f2.val, logistic(10))
    assert np.allclose(f2.der, log_der(10))

    x3 = ad.variable(-24)
    f3 = ad.logistic(x3)
    assert np.allclose(f3.val, logistic(-24))
    assert np.allclose(f3.der, log_der(-24))

    L = 2
    k = 1
    x0 = 1

    x3 = ad.variable(-24)
    f3 = ad.logistic(x3)
    assert np.allclose(f3.val, logistic(-24))
    assert np.allclose(f3.der, log_der(-24))

def test_arcsin():
    arcsin = lambda x: np.arcsin(x)
    arcsin_der = lambda x: 1/np.sqrt(1-x**2)


    f1 = ad.arcsin(np.pi/4)
    assert np.allclose(f1.val, arcsin(np.pi/4))

    x2 = ad.variable(0)
    f2 = ad.arcsin(x2)
    assert np.allclose(f2.val, arcsin(0))
    assert np.allclose(f2.der, arcsin_der(0))

def test_arccos():
    arccos = lambda x: np.arccos(x)
    arccos_der = lambda x: -1/np.sqrt(1-x**2)


    f1 = ad.arccos(np.pi/4)
    assert np.allclose(f1.val, arccos(np.pi/4))

    x2 = ad.variable(0)
    f2 = ad.arccos(x2)
    assert np.allclose(f2.val, arccos(0))
    assert np.allclose(f2.der, arccos_der(0))

def test_arctan():
    arctan = lambda x: np.arctan(x)
    arctan_der = lambda x: 1/(1+x**2)


    f1 = ad.arctan(np.pi/4)
    assert np.allclose(f1.val, arctan(np.pi/4))

    x2 = ad.variable(0)
    f2 = ad.arctan(x2)
    assert np.allclose(f2.val, arctan(0))
    assert np.allclose(f2.der, arctan_der(0))










    


    
