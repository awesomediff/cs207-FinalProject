import numpy as np
import pytest

import awesomediff as ad
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
