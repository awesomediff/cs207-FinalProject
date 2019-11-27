import numpy as np

from core import variable

def _make_matrix(v):
    """
        Converts a scalar, vector, or matrix into column vector.
        Raises a ValueError if matrix has more than one dimension that is not zero or one.
    """
    try:
        # Try to access length (if scalar, will fail into `except` block):
        n = len(v)  # If scalar, will fail into `except` block.
        try:
            # Try to reshape to vector (if too many dimensions, will fail into `except` block):
            v = np.array(v)
            old_shape = v.shape
            new_shape = [dim for dim in old_shape if dim>1]
            assert len(new_shape) in [1,2], "Object has too many dimensions: {}".format(new_shape)
            new_shape = (new_shape[0],1) if len(new_shape)==1 else new_shape
            v = v.flatten().reshape(new_shape)
            #v = [float(x) for x in v]
            #v = np.array(v)
        except:
            # Object with len but non-numeric values:
            raise ValueError("Unable to coerce {} to matrix.".format(type(v)))
    except:
        # Scalar:
        v = np.array([[v]])

    return v

def _make_vector(v):
    
    v = _make_matrix(v)
    try:
        old_shape = v.shape
        new_shape = [dim for dim in old_shape if dim>1]
        if len(new_shape)==0:
            v = v.reshape(0)
        else:
            assert len(new_shape)==1, "Object has too many dimensions: {}".format(new_shape)
            v = v.reshape(new_shape)
    except:
        raise ValueError("Unable to coerce {} to vector.".format(type(v)))
    
    return v

def evaluate(func,vals,seed=None,labels=None):
    """
        A wapper function that evaluates a user-defined function
        using awesomediff.variable objects.

        INPUTS:
            func:  A user-defined function of n inputs and m outputs
                that will be evaluated to find the value and derivative 
                at a specified point.
                The function may take one or more inputs and 
                may return one or more outputs (as a tuple).
                The function may include basic operations (+,*,etc), 
                as well as any special functions supported by awesomediff.
                May be a named function or a lambda function.
            vals: A vector of value at which to evaluate the function
                and its derivative.
                Its length must be equal to the number of arguments taken by `func`.
                May be a list or a numpy array.
            seed: A matrix of seed values for evaluating the drivatives
                Defaults to the n-by-n identity matrix
                (where n is the length of the specified vector of values),
                which computes the full Jacobian.
                May be a list of lists or a matrix.
                Each row corresponds to the seed at which a variable will be computed.
                Must have as many rows as there are inputs to `func`.
            labels: An optional vector of strings that correspond 
                to the names of the variables being passed to `func`.
        ...

        RETURNS:
            An awesomediff.variable object if `func` is a scalar function,
            or a tuple of awesomediff.variables if `func` is a vector function.
            Each aweseomdiff.variable will have the value and derivatives
            corresponding to the function evaluated at the specified point.
        ...

        EXAMPLE:
        >>> def parametric_ellipse(a,b,t):
                x = a * ad.cos(t)
                y = b * ad.sin(t)
        >>>     return (x, y)
        >>> x,y = ad.evaluate(func=parametric_ellipse,vals=[4,2,0])
        >>> x.val
        4                   # a*cos(t)
        >>> x.der
        np.array([1,0,0])   # [ cos(t), 0, -a*sin(t) ]
        >>> y.val
        0                   # b*sin(t)
        >>> y.der
        np.array([0,0,4])   # [ 0 , sin(t), a*cos(t) ]

    """
    
    #vals = _make_vector(vals)
    #seed = _make_matrix(seed)
    
# =============================================================================
#     inputs = []
#     for i,v in enumerate(vals):
#         var = variable(val=v,der=seed[i,:])
#         inputs.append( var )
#     outputs = func(*inputs)
# =============================================================================

    #self._func = func
    #self._vals = vals
    #self._seed = seed
    #self._n = n
    #self._m = m
    #self._inputs = inputs
    #self._outputs = outputs

# =============================================================================
#     return outputs
# =============================================================================

x = variable(3, der=1)
print(x)


#fn.sin(1)

