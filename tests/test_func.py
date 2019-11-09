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
