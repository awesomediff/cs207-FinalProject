from inspect import signature

import numpy as np

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


def mean_squared_error(y_true,y_pred):

    assert len(y_true)==len(y_pred)

    loss = 0
    for true,pred in zip(y_true,y_pred):
        loss += (true-pred)**2
    loss = loss / len(y_true)
    return loss


def _check_inputs(X,y):
    """Converts a matrix X into a list of lists and a vector y into a list."""

    new_y = []
    new_X = []

    for val in y:
        try:
            new_y.append( float(val) )
        except:
            new_y.append( float(val[0]) )

    assert len(X)==len(y), "Dimensions of X and y do not match."

    for row in X:
        new_row = []
        for val in row:
            new_row.append( float(val) )
        new_X.append( new_row )

    return new_X,new_y


class Solver:

    def __init__(self):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def step(self):
        raise NotImplementedError

    def solve(self):
        raise NotImplementedError


class GradientDescent(Solver):
    
    def __init__(self,model,learning_rate=0.01,rel_tol=1e-3,abs_tol=1e-6,max_iter=1000,random_seed=None,verbose=False):
        
        self.model = model
        self.learning_rate = learning_rate
        self.rel_tol = rel_tol
        self.abs_tol = abs_tol
        self.max_iter = max_iter
        self.random_seed = random_seed
        self.verbose = verbose

    def solve(self,initial_weights,X,y):

        # Set random seed:
        if self.random_seed is not None:
            random.seed(self.random_seed)
        else:
            random.seed()
        alpha = self.learning_rate
        def loss_func(*weights):
            return self.model._loss(weights,X,y)
            
        prev_loss = None
        prev_weights = initial_weights

        converged = False
        iteration = 0
        while not converged:
            iteration += 1
            loss,grad = evaluate(loss_func,prev_weights)
            weights = [w - alpha * gr for w,gr in zip(prev_weights,grad)]
            if self.verbose:
                print("Step {}: loss={}; grad={}".format(iteration,loss,grad))
                #print("Step {}: loss={}".format(iteration,loss))
            # Check for stopping conditions:
            if iteration >= self.max_iter:
                converged = True
                break
            if (prev_loss is not None) and (abs(loss-prev_loss) < self.abs_tol):
                converged = True
                break
            if (prev_loss is not None) and (abs((loss-prev_loss)/prev_loss) < self.rel_tol):
                converged = True
                break
            prev_loss = loss
            prev_weights = weights
        
        if not converged:
            print("Warning: Gradient descent did not converge.")
        
        # Un-set random seed:
        random.seed()
            
        return weights


class Model:

    def __init__(self):
        """Initialize the model."""
        pass

    def _pack_weights(self,placeholder):
        """Create a single list of weights to pass to the solver."""
        raise NotImplementedError

    def _unpack_weights(self,weights):
        """Extract weights from list returned by solver."""
        raise NotImplementedError

    def _loss(self,weights,X,y):
        """Calculate the loss between y and the predictions made with X."""
        raise NotImplementedError

    def _predict(self,weights,X):
        """Calculates the """
        raise NotImplementedError

    def predict(self,X):
        """Predict X."""
        raise NotImplementedError

    def fit(self,X,y):
        """Fit the model to X and y."""
        raise NotImplementedError

    def score(self,X,y):
        """Return the score of y and the predictions made with X."""
        raise NotImplementedError


class LinearRegression(Model):

    def __init__(self,fit_intercept=True,solver='gradient_descent',**solver_kwargs):

        valid_solvers = ['gradient_descent']
        if solver=='gradient_descent':
            self.solver = GradientDescent(model=self,**solver_kwargs)
        else:
            raise ValueError("Solver must be one of the following: {}".format(", ".join(valid_solvers)))
        
        self.fit_intercept = fit_intercept
        self.intercept = None
        self.coefs = []

    def _pack_weights(self,intercept,coefs):
        """Creates a single list of weights to pass to solver."""

        if self.fit_intercept:
            return [intercept]+coefs
        else:
            return coefs

    def _unpack_weights(self,weights):
        """Extracts relevant parameters to pass to solver"""

        if (self.fit_intercept==True) and len(weights)==1:
            intercept = weights[0]
            coefs = []
        elif (self.fit_intercept==True) and len(weights)>1:
            intercept = weights[0]
            coefs = weights[1:]
        elif (self.fit_intercept==False):
            intercept = 0
            coefs = weights
        
        return intercept,coefs

    def _loss(self,weights,X,y):

        return mean_squared_error( y , self._predict(weights,X) )
    
    def _predict(self,weights,X):

        intercept,coefs = self._unpack_weights(weights)
        predictions = []
        for vals in X:
            pred = intercept
            for x,coef in zip(vals,coefs):
                pred += x*coef
            predictions.append(pred)
        return predictions

    def predict(self,X):
        
        assert len(X[0])==len(self.coefs), "X does not match dimensions of fitted model."
        predictions = []
        for vals in X:
            pred = self.intercept if self.fit_intercept else 0
            for x,coef in zip(vals,self.coefs):
                pred += x*coef
            predictions.append(pred)
        return predictions

    def fit(self,X,y):

        X,y = _check_inputs(X,y)
        n_features = len(X[0])
        n_weights = n_features+1 if self.fit_intercept==True else n_features
        initial_weights = [random.uniform(0,1) for _ in range(n_weights)]
        weights = self.solver.solve(initial_weights,X,y)

        # Expose final values to user:
        intercept,coefs = self._unpack_weights(weights)
        self.intercept = intercept if self.fit_intercept else None
        self.coefs = coefs


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

