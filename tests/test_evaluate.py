#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 10:00:16 2019

@author: claireyang
"""

import numpy as np
import pytest

import awesomediff as ad


def func2(x,y):
    f1 = x**2 - 3*y
    return f1


output_vals, jacobian_matrix = ad.evaluate(func=func2, vals=[2,1], seed=[[1,0],[0,1]])
print(output_vals)
print(jacobian_matrix)