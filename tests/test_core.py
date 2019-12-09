#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 10:18:11 2019

@author: claireyang
"""

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
    
# test univariate single function
def test_univariate_single_func():
    
    def func1(x):
        f1 = (ad.sin(x))**2
        return f1
    
    output_value, jacobian = ad.evaluate(func=func1, vals=np.pi/4)
    
    assert np.isclose(output_value, (np.sin(np.pi/4))**2)
    np.testing.assert_allclose(jacobian, np.array([1]))


# test univariate vector function
def test_univariate_vector_func():
    
    def func2(x):
        f1 = x**2 - 3*x
        f2 = 2**x
        return [f1,f2]
    
    output_value, jacobian = ad.evaluate(func=func2, vals=1)
    
    np.testing.assert_allclose(output_value, np.array([-2,2]))
    np.testing.assert_allclose(jacobian, np.array([[-1], [np.log(4)]]))
    
    
# test multivariate single function
def test_multivariate_single_func():
    
    def func3(x,y):
        f1 = (3*x**2 - y) / (5*y)
        return f1

    output_value, jacobian = ad.evaluate(func=func3, vals=[2,2])

    assert output_value == 1
    np.testing.assert_allclose(jacobian, np.array([1.2, -0.6]))   
        

# test multivariate vector function
def test_multivariate_vector_func1():
    def func4(x,y,z):
        f1 = x**2 + 2*y - 7*z
        f2 = 3*x + z**2
        f3 = 3*y - 2*z
        return [f1,f2,f3]
 
    output_value, jacobian = ad.evaluate(func=func4,vals=[2,3,4])
    
    np.testing.assert_allclose(output_value, np.array([-18,22,1]))
    np.testing.assert_allclose(jacobian, np.array([[4,2,-7],[3,0,8],[0,3,-2]]))


def test_multivariate_vector_func2():
    def func5(x,y):
        f1 = x*y + ad.cos(x)
        f2 = x*y + ad.sin(y)
        return [f1,f2]
 
    output_value, jacobian = ad.evaluate(func=func5,vals=[np.pi/3, np.pi/2])
    
    np.testing.assert_allclose(output_value, np.array([(np.pi/3)*(np.pi/2)+0.5, (np.pi/3)*(np.pi/2)+1]))
    np.testing.assert_allclose(jacobian, np.array([[(np.pi/2)-np.sin(np.pi/3), np.pi/3],[np.pi/2, (np.pi/3)+np.cos(np.pi/2)]]))

 
# test raising error for incorrect input for 'vals'
def test_incorrect_input_vals1():
    
    def func1(x):
        f1 = (ad.sin(x))**2
        return f1
    
    with pytest.raises(RuntimeError):
        # a list of two scalars passed in for 'vals' when expecting only one scalar
        output_value, jacobian = ad.evaluate(func=func1, vals=[np.pi/4,np.pi/2])
        

def test_incorrect_input_vals2():
    
    def func3(x,y):
        f1 = (3*x**2 - y) / (5*y)
        return f1
    
    with pytest.raises(RuntimeError):
        output_value, jacobian = ad.evaluate(func=func3, vals=2)


def test_incorrect_input_vals3():
    
    def func4(x,y,z):
        f1 = x**2 + 2*y - 7*z
        f2 = 3*x + z**2
        f3 = 3*y - 2*z
        return [f1,f2,f3]
    
    with pytest.raises(RuntimeError):
        output_value, jacobian = ad.evaluate(func=func4,vals=[2,3])
    
    
def test_incorrect_input_val4():
    
    def func5(x,y):
        f1 = ad.exp(x)
        f2 = ad.exp(x) + y**2
        return [f1,f2]
        
    # expecting two arguments, but passed in three values
    with pytest.raises(RuntimeError):
        output_value, jacobian = ad.evaluate(func=func5, vals=[1,2,3])


# test raising error for incorrect input for 'seed'
def test_incorrect_input_seed1():
    
    def func1(x):
        f1 = (ad.sin(x))**2
        return f1
    
    with pytest.raises(ValueError):
        # raise error when seed value for a univariate function is given as a list instead of a scalar
        output_value, jacobian = ad.evaluate(func=func1, vals=np.pi/4, seed=[1])
        

def test_incorrect_input_seed2():
    
    def func3(x,y):
        f1 = (3*x**2 - y) / (5*y)
        return f1
    
    with pytest.raises(AssertionError):
        # raise error when seed is not given in correct format (expecting a list of two lists, each of length = 2)
        output_value, jacobian = ad.evaluate(func=func3, vals=[2,2], seed=[[1,0,2],[0,1]])
    

def test_incorrect_input_seed3():
    
    def func3(x,y):
        f1 = (3*x**2 - y) / (5*y)
        return f1
    
    with pytest.raises(AssertionError):
        # raise error when seed is not given in correct format (expecting a list of two lists)
        output_value, jacobian = ad.evaluate(func=func3, vals=[2,2], seed=[[1,0]])


# test correct jacobian is returned when user manually inputs seed
def test_correct_input_seed():
    
    def func4(x,y,z):
        f1 = x**2 + 2*y - 7*z
        f2 = 3*x + z**2
        f3 = 3*y - 2*z
        return [f1,f2,f3]
 
    output_value, jacobian = ad.evaluate(func=func4,vals=[2,3,4], seed=[[1,0,0],[0,1,0],[0,0,1]])
    
    np.testing.assert_allclose(output_value, np.array([-18,22,1]))
    np.testing.assert_allclose(jacobian, np.array([[4,2,-7],[3,0,8],[0,3,-2]]))




