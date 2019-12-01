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
    assert jacobian == 1



