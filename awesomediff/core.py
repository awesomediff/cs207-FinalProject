import math
import numpy as np
from inspect import signature

class variable:

    def __init__(self,val,der=1):
        """
            Initialize a variable object
            with a specified value and derivative.
            If no derivative is specified,
            a seed of 1 is assumed.
        """

        self._val = val
        self._der = der

    @property
    def val(self):
        return self._val

    @property
    def der(self):
        return self._der
    
    def __repr__(self):
        """
            Define representation of variable object:
        """
        return "awesomediff.variable(val={},der={})".format(self.val,self.der)

    def __neg__(self):
        """
            Overload negation.
            Returns a variable object with:
            val : negated value.
            der : negated derivative.
        """
        return variable(val=-self.val,der=-self.der)

    def __add__(self,other):
        """
            Overload addition.
            Returns a variable object with:
            val : sum of values.
            der : sum of derivatives.
        """
        self_val = self.val
        self_der = self.der
        try:
            # Assume other object is a variable:
            other_val = other.val
            other_der = other.der
        except:
            # If not, use this value and derivative 0.
            try:
                float(other)
            except:
                raise ValueError("{} is not a number.".format(other))
            other_val = other
            other_der = 0  # Derivative of a constant is zero.
        # Calculate new values (simple summation):
        new_val = self_val + other_val
        new_der = self_der + other_der
        # Return variable with new value and derivative:
        return variable(val=new_val,der=new_der)

    def __radd__(self,other):
        """
            Overload reverse addition by calling __add__().
        """
        return self.__add__(other)

    def __mul__(self,other):
        """
            Overload multiplication.
            Returns a variable object with:
            val : product of values.
            der : result of product rule
        """
        self_val = self.val
        self_der = self.der
        try:
            # Assume other object is a variable:
            other_val = other.val
            other_der = other.der
        except:
            # If not, uses this value and derivative 0.
            try:
                float(other)
            except:
                raise ValueError("{} is not a number.".format(other))
            other_val = other
            other_der = 0  # Derivative of a constant is zero.
        # Calculate new values (applying product rule):
        new_val = self_val * other_val
        new_der = (self_der * other_val) + (self_val * other_der)
        # Return variable with new value and derivative:
        return variable(val=new_val,der=new_der)

    def __rmul__(self,other):
        """
            Overload reverse multiplication
            Calls __mul__.
        """
        return self.__mul__(other)

    def __sub__(self,other):
        """
            Overload subtraction.
            Implemented using negation and addition.
        """
        return self.__add__(-other)

    def __rsub__(self,other):
        """
            Overload reverse subtraction.
            Implemented using negation and addition.
        """
        return self.__neg__().__add__(other)

    def __truediv__(self, other):
        try:
            # assume other is an instance variable
            new_val = self.val / other.val
            new_der = (self.der * other.val - self.val * other.der) / other.val ** 2
        except:
            # assume in the form of variable/scaler
            try:
                float(other)
            except:
                raise ValueError("{} is not a number or instance variable.".format(other))

            if float(other) == 0:
                raise ZeroDivisionError('Cannot perform division by zero')
            new_val = self.val / float(other)
            new_der = self.der / float(other)

        return variable(val=new_val, der=new_der)

    def __rtruediv__(self, other):
        new_val = other / self.val
        new_der = -other * self.val ** (-2)  # other*self.pow(-1)
        return variable(val=new_val, der=new_der)

    def __eq__(self, other):
        """
            Overload equal
            Check if the thing compared to is a variable and has the same vale
        """
        return isinstance(other, variable) and self.val == other.val

    def __ne__(self, other):
        """
            Overload not equal
            Check if the thing compared to is a variable and does not have the same vale
        """
        return not isinstance(other, variable) or not (self.val == other.val)

    def __lt__(self, other):
        """
            Overload less than
        """
        try:
            return self.val<other.val
        except:
            raise AttributeError("Can only compare varibles.")

    def __gt__(self, other):
        """
            Overload greater than
        """
        try:
            return self.val>other.val
        except:
            raise AttributeError("Can only compare varibles.")

    def __le__(self, other):
        """
            Overload less than or equal to
        """
        try:
            return self.val<=other.val
        except:
            raise AttributeError("Can only compare varibles.")

    def __ge__(self,other):
        """
            Overload greater than or equal to
        """
        try:
            return self.val>=other.val
        except:
            raise AttributeError("Can only compare varibles.")

    def __pow__(self, other):
        try:
            # check whether other is a number
            new_val = self.val ** other
        except:
            raise ValueError("{} must be a number.".format(other))
        new_der = other * self.val ** (other - 1) * self.der
        return variable(val=new_val, der=new_der)

    def __rpow__(self, other):
        try:
            new_val = other** self.val
        except:
            raise ValueError("{} must be a number.".format(other))
        new_der = other**self.val * np.log(other)
        return variable(val=new_val, der=new_der)
    
    
    
    
    
def _build_jacobian(outputs):
    """
    INPUT:
        outputs: a list of awesomediff.variable objects
    OUTPUT:
        returns n x m jacobian matrix where n is the number of functions
        and m is the total number of variables
        
    EXAMPLE:
    outputs = [awesomediff.variable(val=5, der=[4]), awesomediff.variable
    (val=0.5, der=[0.5])]
    
    >>> _build_jacobian(outputs)
    [[4]
     [0.25]]
    
    """
    n = len(outputs) # get number of functions
    m = len(outputs[0].der) # get number of variables
    jacobian = np.zeros((n,m))
    for i, output in enumerate(outputs):
         jacobian[i,:] = output.der
    
    return jacobian
    

def evaluate(func,vals,seed=None):
    """
        A wapper function that evaluates a user-defined function
        using awesomediff.variable objects.

        INPUTS:
            func: A user-defined function of n variables and m functions
                that will be evaluated to find the value and derivative 
                at a specified point.
                The user-defined function may take one or more variables 
                as arguments and may return one or more function outputs (as a list).
                Functions may include basic operations (+,*,etc), 
                as well as any special functions supported by awesomediff.
                
            vals: A scalar (for univariate function) or a list of scalars
                (for multivariate function) at which to evaluate the function 
                and its derivative. 
                
                
            seed: A scalar (for univariate function) or a list of lists (for multivariate
                function) containing seed values for each variable. The lengths of the 
                outer list and inner list must equal the number of arguments taken by `func`
                (i.e. the number of variables). Defaults to a seed of 1 for each 
                variable if no seed value is provided. 
                
                EXAMPLE: 
                To set seed of variables x, y, z at 1, 2, and 3, respectively, user
                passes the following list of lists as seed:
                [[1,0,0],[0,2,0],[0,0,3]]
                The length of outer list and the lengths of each of the inner lists
                must be equal to the total number of variables (in this example, 3)
        ...

        RETURNS:
            output_value: value of function at a specified point. A scalar for 
            a function, a vector of scalars for a vector function
            
            jacobian: the jacobian of function. A vector of scalar or scalars 
            for univariate and multivariate functions, respectively. A matrix of
            scalars for univariate and multivariate vector functions
        ...

        EXAMPLE:
        >>> def parametric_ellipse(a,b,t):
                x = a * ad.cos(t)
                y = b * ad.sin(t)
        >>>     return [x, y]
        >>> output_value, jacobian = ad.evaluate(func=parametric_ellipse,vals=[4,2,0])
        >>> output_value
        np.array([4, 0])    # [a*cos(t), b*sin(t)]
        >>> jacobian
        np.array([[1,0,0],[0,0,4]]) #[[cos(t), 0, -a*sin(t)],[0 , sin(t), a*cos(t)]]

    """
    
    ## get number of arguments that were passed into func
    sig = signature(func)
    num_vars = len(sig.parameters) # num_vars is the total number of variables
    
    ## check user input for vals and convert it to numpy array
    # user input for vals will be a scalar for univariate function and 
    # a list of scalars for multivariate function
    try:
        float(vals) # if vals is a scalar
        vals = np.array([vals])
    except:
        if isinstance(vals, list): # if vals is a list
            vals = np.array(vals)
    
    # check length of vals is equal to num_vars
    if len(vals) != num_vars:
        raise ValueError("number of values passed in does not agree with number of variables")
    
    
    ## check user input for seed
    # user input for seed will be a scalar for univariate function and 
    # a list of scalars for multivariate function
    
    # if user doesn't pass in value for seed argument, create seed with value of 1
    if seed == None:
        if num_vars == 1: #if univariate function
            seed = np.array([1])
        else: #if multivariate function
            seed_matrix = np.identity(num_vars)
        
    #if user passes in value for seed argument
    else: 
        try:
            float(seed) # if seed is a scalar
            seed = np.array([seed])
        except:
            if isinstance(seed, list): # if seed is a list 
                # check length of outer list
                assert len(seed) == num_vars, "length of outer list must equal number of variables"
                
                # check length of inner list
                for s in seed:
                    try: 
                        len(s)
                    except: # seed for univariate function given as a list
                        raise ValueError("seed for univariate function should be given as a scalar")
                    assert len(s) == num_vars, "length of inner list must equal number of variables"
    
                seed_matrix = np.array(seed)
                
            else: # if seed is not provided as a scalar or a list
                raise ValueError("seed must be provided as a scalar or a list")
                                   
    
    ## evaluate the user-defined function passed in
    
    # instantiate awesomediff.variable objects
    if num_vars == 1: # if univariate function
        var = variable(val=vals[0], der=seed)
        inputs = [var] 
    else: # if multivariate function
        inputs = []
        for i,v in enumerate(vals):
            var = variable(val=v,der=seed_matrix[i,:])
            inputs.append(var)
    
    # pass awesomediff.variable objects into user-defined function
    # outputs will be an awesomediff.variable object or a list of 
    # awesomediff.variable objects storing values and derivatives of functions
    outputs = func(*inputs)
    
    # create output_value and jacobian
    try: # if a vector-function
        len(outputs)
        output_value = np.zeros(len(outputs))
        for i,output in enumerate(outputs):
            output_value[i] = output.val
        jacobian = _build_jacobian(outputs)
        
    except: # if not a vector function 
        output_value = outputs.val
        jacobian = outputs.der
        
    return output_value, jacobian
    
    
    