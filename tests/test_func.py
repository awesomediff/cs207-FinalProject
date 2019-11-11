import numpy as np
import pytest

from awesomediff import AutoDiff

def test_sin():

    # Sine of variable:
    f1 = AutoDiff.sin(np.pi)
    assert np.allclose(f1.val,0.0)
    assert np.allclose(f1.der,0.0)

    # Sine of variable:
    x2 = AutoDiff.variable(np.pi)
    f2 = AutoDiff.sin(x2)
    assert np.allclose(f2.val,0.0)
    assert np.allclose(f2.der,-1.0)

    # Sine of variable times constant:
    x3 = AutoDiff.variable(0.0)
    f3 = AutoDiff.sin(x3)*5
    assert np.allclose(f3.val,0.0)
    assert np.allclose(f3.der,5.0)

    # Sine of constant times variable:
    x4 = AutoDiff.variable(np.pi)
    f4 = AutoDiff.sin(x4*2)
    assert np.allclose(f4.val,0.0)
    assert np.allclose(f4.der,2.0)

def test_cos():

    # Cosine of variable:
    f1 = AutoDiff.cos(np.pi*3/4)
    assert np.allclose(f1.val,-np.sqrt(2)/2)
    assert np.allclose(f1.der,0)

    # Cosine of variable:
    x2 = AutoDiff.variable(np.pi)
    f2 = AutoDiff.cos(x2)
    assert np.allclose(f2.val,-1.0)
    assert np.allclose(f2.der,0.0)

    # Cosine of variable times constant:
    x3 = AutoDiff.variable(0.0)
    f3 = AutoDiff.cos(x3)*5
    assert np.allclose(f3.val,5.0)
    assert np.allclose(f3.der,0.0)

    # Cosine of constant times variable:
    x4 = AutoDiff.variable(np.pi)
    f4 = AutoDiff.cos(x4*0.5)
    assert np.allclose(f4.val,0.0)
    assert np.allclose(f4.der,-0.5)

def test_tan():

    # Tangent of variable:
    f1 = AutoDiff.tan(np.pi)
    assert np.allclose(f1.val,0.0)
    assert np.allclose(f1.der,0.0)

    # Tangent of variable:
    x2 = AutoDiff.variable(np.pi)
    f2 = AutoDiff.tan(x2)
    assert np.allclose(f2.val,0.0)
    assert np.allclose(f2.der,1.0)

    # Tangent of variable times constant:
    x3 = AutoDiff.variable(np.pi/4)
    f3 = AutoDiff.tan(x3)*5
    assert np.allclose(f3.val,5.0)
    assert np.allclose(f3.der,10.0)

    # Tangent of constant times variable:
    x4 = AutoDiff.variable(np.pi*2)
    f4 = AutoDiff.tan(x4*0.5)
    assert np.allclose(f4.val,0.0)
    assert np.allclose(f4.der,0.5)

def test_log():
    # log of a scalar
    f1 = AutoDiff.log(1)
    assert np.allclose(f1.val, 0)

    f2 = AutoDiff.log(10)
    assert np.allclose(f2.val, np.log(10))

    # log of a variable
    x3 = AutoDiff.variable(1)
    f3 = AutoDiff.log(x3)
    assert np.allclose(f3.val, 0)
    assert np.allclose(f3.der, 1)

    x4 = AutoDiff.variable(3)
    f4 = AutoDiff.log(x4)*5+1
    assert np.allclose(f4.val, np.log(3)*5+1)
    assert np.allclose(f4.der, 5/3)

def test_sqrt():
    # square root of a scalar
    f1 = AutoDiff.sqrt(81)
    assert np.allclose(f1.val, 9)

    # square root of a variable
    x2 = AutoDiff.variable(49)
    f2 = AutoDiff.sqrt(x2)
    assert np.allclose(f2.val, 7)
    assert np.allclose(f2.der, 1/14)

    x3 = AutoDiff.variable(64)
    f3 = 5+2*AutoDiff.sqrt(x3)
    assert np.allclose(f3.val, 21)
    assert np.allclose(f3.der, 1/8)

def test_exp():
    # exponential of a scalar
    f1 = AutoDiff.exp(10)
    assert np.allclose(f1.val, np.exp(10))

    # exponential of a variable
    x2 = AutoDiff.variable(5)
    f2 = AutoDiff.exp(x2)
    assert np.allclose(f2.val, np.exp(5))
    assert np.allclose(f2.der, np.exp(5))

    x3 = AutoDiff.variable(4)
    f3 = 5+2*AutoDiff.exp(x3)
    assert np.allclose(f3.val, 5+2*np.exp(4))
    assert np.allclose(f3.der, 2*np.exp(4))