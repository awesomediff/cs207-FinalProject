#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 10:00:16 2019

@author: claireyang
"""

import numpy as np
import pytest

import awesomediff as ad



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
def test_multivariate_vector_func():
    def func4(x,y,z):
        f1 = x**2 + 2*y - 7*z
        f2 = 3*x + z**2
        f3 = 3*y - 2*z
        return [f1,f2,f3]
 
    output_value, jacobian = ad.evaluate(func=func4,vals=[2,3,4])
    
    np.testing.assert_allclose(output_value, np.array([-18,22,1]))
    np.testing.assert_allclose(jacobian, np.array([[4,2,-7],[3,0,8],[0,3,-2]]))

        
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
        
        
        
        
        
        
    



