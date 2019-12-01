import numpy as np

from core import variable
from inspect import signature


def _build_seed_matrix(seeds):
    """
    INPUT:
        seed: 1-d array of seed values passed in by the user in the 'evaluate' function
    OUTPUT:
        if seed is n elements long, it returns a n x n array where each row represents
        the seed vector for a variable
        
    EXAMPLE:
    _build_seed_matrix([1,1,2])
    >>>
    [[1,0,0],
    [0,1,0],
    [0,0,2]]
    """
    n = len(seeds)
    seed_matrix = np.identity(n)
    for i,seed in zip(range(n),seeds):
        seed_matrix[i,i] = seed
    
    return seed_matrix
    

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
                
            vals: A scalar (for univariate function) or a list of lists
                (for multivariate function) at which to evaluate the function 
                and its derivative. The lengths of the outer list and inner list
                must be equal to the number of arguments taken by `func`
                
            seed: A list of seed values for evaluating the derivatives
                Defaults to a seed of 1 for each variable if no seed value is provided
        ...

        RETURNS:
            output_value: value of function at a specified point. A scalar for 
            a function, a vector of scalars for a vector function
            
            jacobian: the jacobian of function. A scalar for a univariate function,
            a vector of scalars for a multivariate function, a matrix of scalars
            for univariate or multivariate vector functions
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
            #seed = np.ones(num_vars)
            #seed_matrix = _build_seed_matrix(seed)
        
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
                    assert len(s) == num_vars, "length of inner list must equal number of variables"
    
                seed_matrix = np.array(seed)
                
            else: # if seed is not provided as a scalar or a list
                raise ValueError("seed must be provided as a scalar or a list")
                               
    
    # check length of seed is equal to num_vars
    #if len(seed) != num_vars:
        #raise ValueError("number of seed values passed in does not agree with number of variables")
    
    # if function is multivariate, build seed matrix
    #if len(seed) > 1:
        #seed_matrix = _build_seed_matrix(seed)
    
    
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
    
    

## Demo cases of using evaluate function ##
    
# =============================================================================
# # single-variable function 
# def func1(x):
#     f1 = x**2 - 3
#     return f1
# 
# output_vals, jacobian_matrix = evaluate(func=func1, vals=2, seed=1)
# print(output_vals)
# print(jacobian_matrix)
# =============================================================================


# multi-variable function
def func2(x,y):
    f1 = x**2 - 3*y
    return f1


output_vals, jacobian_matrix = evaluate(func=func2, vals=[2,1], seed=[[1,0],[0,1,3]])
print(output_vals)
print(jacobian_matrix)


# =============================================================================
# # vector function of single variable
# def func3(x):
#     f1 = 4*x - 3
#     f2 = x / 4
#     return [f1,f2]
#     
# output_vals, jacobian_matrix = evaluate(func=func3, vals=2)
# print(output_vals)
# print(jacobian_matrix)
# 
# 
# # vector function of multiple variables
# def func4(x,y,z):
#     f1 = x**2 + 2*y - 7*z
#     f2 = 3*x + z**2
#     f3 = 3*y - 2*z
#     return [f1,f2,f3]
# 
# evaluate(func=func4,vals=[2,3,4], seed=[1,2,1])
# 
# output_vals, jacobian_matrix = evaluate(func=func4,vals=[2,3,4])
# print(output_vals)
# print(jacobian_matrix)
# =============================================================================
    

    
    

