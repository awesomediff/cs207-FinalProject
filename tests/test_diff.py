import pytest
from awesomediff import AutoDiff

def test_simple_functions_1():
    
    # Addition and multiplication:
    a = 5.0
    x = AutoDiff.variable(a)
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

def test_equal():
    x = AutoDiff.variable(3.0)
    y = AutoDiff.variable(3.0)

    assert x==y

    x = AutoDiff.variable(3.0)
    y = AutoDiff.variable(4.0)
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


