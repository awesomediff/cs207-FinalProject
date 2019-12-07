from awesomediff.core import variable
from awesomediff.core import evaluate
from awesomediff.func import sin
from awesomediff.func import cos
from awesomediff.func import tan
from awesomediff.func import log
from awesomediff.func import sqrt
from awesomediff.func import exp
from awesomediff.func import sinh
from awesomediff.func import cosh
from awesomediff.func import tanh
import numpy as np
from inspect import signature


def gradientDescent(func, initial, rate=0.01, precision=0.00001, iteration = 2000):
	# df = func
	count = 0
	current = initial
	while (step>precision and count <iteration):
		last = current
		current = current - rate*func(last)
		step = abs(current-last)
		count+=1

	return current


# Newton-Raphson Method
def uni_Newton(func, initial, max_iter=200, epsilon=1e-06):
	'''

	:param func: univariate function
	:param initial: starting point(scalar)
	:param max_iter: max iteration
	:param epsilon: change in function value < epsilon (stopping condition)
	:return: if root is found, return the root. Otherwise None

	def root_finding(a):
    	return a**2 + 2*a + 1
    root = uni_Newton(root_finding, 50)
    root_finding(root) #gives something close to 0
	'''

	# Check Input formats
	sig = signature(func)  # function should take only one scalar input
	if len(sig.parameters) != 1:
		raise ValueError('The function should be uni-variate')

	# check the initial point is a scalar
	try:
		int(initial)
	except ValueError:
		print('The input must be a scalar')

	current_x = int(initial)
	for i in range(max_iter):
		func_val, func_der = evaluate(func, current_x)

		if np.abs(func_val) <= epsilon:
			print('Root Approximation Found!', ' root = ', current_x)
			return current_x

		if i == max_iter - 1:
			print(
				'Max iteration reached, failed to find the root. The function may not have a root or try increase iteration numbers')
			return None

		# check if it's a bad derivative(0)
		if np.abs(func_der[0]) <= 10 ** (-15):
			print('Bad Starting Point: Derivative of the function = 0 at some point')
			return None
			break

		current_x = current_x - func_val / func_der[0]

