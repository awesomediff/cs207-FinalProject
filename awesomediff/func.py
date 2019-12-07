import math
import numpy as np

from awesomediff.core import variable

def sin(x):
    """
        Helper function that calculates the sin of a variable or number.

        INPUTS:
            x : awesomediff.variable object or a number.

        OUTPUT:
            awesomediff.variable
    """
    
    try:
        # Assume object is a variable:
        val = x.val
        der = x.der
    except:
        # If not, treat it as a constant.
        try:
            float(x)
        except:
            raise ValueError("{} is not a number.".format(x))
        val = x
        der = 0  # Derivative of a constant is zero.
    # Calculate new value an derivative:
    new_val = np.sin(val)
    new_der = np.cos(val)*der
    # Return variable with new value an derivative:
    return variable(val=new_val,der=new_der)

def cos(x):
    """
        Helper function that calculates the sin of a variable or number.

        INPUTS:
            x : awesomediff.variable object or a number.

        OUTPUT:
            awesomediff.variable
    """
    
    try:
        # Assume object is a variable:
        val = x.val
        der = x.der
    except:
        # If not, treat it as a constant.
        try:
            float(x)
        except:
            raise ValueError("{} is not a number.".format(x))
        val = x
        der = 0  # Derivative of a constant is zero.
    # Calculate new value an derivative:
    new_val = np.cos(val)
    new_der = -np.sin(val)*der
    # Return variable with new value an derivative:
    return variable(val=new_val,der=new_der)

def tan(x):
    """
        Helper function that calculates the sin of a variable or number.

        INPUTS:
            x : awesomediff.variable object or a number.

        OUTPUT:
            awesomediff.variable
    """
    
    try:
        # Assume object is a variable:
        val = x.val
        der = x.der
    except:
        # If not, treat it as a constant.
        try:
            float(x)
        except:
            raise ValueError("{} is not a number.".format(x))
        val = x
        der = 0  # Derivative of a constant is zero.
    # Calculate new value an derivative:
    new_val = np.tan(val)
    new_der = ((1/np.cos(val))**2)*der
    # Return variable with new value an derivative:
    return variable(val=new_val,der=new_der)

def arcsin(x):
    """
        Helper function that calculates the arcsine of a variable or number.

        INPUTS:
            x : awesomediff.variable object or a number.

        OUTPUT:
            awesomediff.variable
    """
    try:
        # Assume object is a variable:
        val = x.val
        der = x.der
    except:
        # If not, treat it as a constant.
        try:
            float(x)
        except:
            raise ValueError("{} is not a number.".format(x))
        val = x
        der = 0  # Derivative of a constant is zero.
    # Calculate new value an derivative:
    new_val = np.arcsin(val)
    new_der = 1/np.sqrt(1-val**2)*der
    print("new val", new_val)
    # Return variable with new value an derivative:
    return variable(val=new_val,der=new_der)

def arccos(x):
    """
        Helper function that calculates the arccos of a variable or number.

        INPUTS:
            x : awesomediff.variable object or a number.

        OUTPUT:
            awesomediff.variable
    """
    try:
        # Assume object is a variable:
        val = x.val
        der = x.der
    except:
        # If not, treat it as a constant.
        try:
            float(x)
        except:
            raise ValueError("{} is not a number.".format(x))
        val = x
        der = 0  # Derivative of a constant is zero.
    # Calculate new value an derivative:
    new_val = np.arccos(val)
    new_der = -1/np.sqrt(1-val**2)*der
    # Return variable with new value an derivative:
    return variable(val=new_val,der=new_der)

def arctan(x):
    """
        Helper function that calculates the arctan of a variable or number.

        INPUTS:
            x : awesomediff.variable object or a number.

        OUTPUT:
            awesomediff.variable
    """
    try:
        # Assume object is a variable:
        val = x.val
        der = x.der
    except:
        # If not, treat it as a constant.
        try:
            float(x)
        except:
            raise ValueError("{} is not a number.".format(x))
        val = x
        der = 0  # Derivative of a constant is zero.
    # Calculate new value an derivative:
    new_val = np.arctan(val)
    new_der = 1/(1+val**2)*der
    # Return variable with new value an derivative:
    return variable(val=new_val,der=new_der)

def log(x):
    """
        Helper function that calculates the natural log of a variable or number.

        INPUTS:
            x : AutoDiff.variable object or a number.

        OUTPUT:
            AutoDiff.variable
    """
    try:
        # x.val = np.log(x.val)
        # x.der = (1/x.val)*x.der
        new_val = np.log(x.val)
        new_der = (1/x.val)*x.der
        # return x
        return variable(val=new_val,der=new_der)
    except:
        new_val = np.log(x)
        new_der = 0
        return variable(val=new_val,der=new_der)

def logb(x, b):
    """
        Helper function that calculates the log of a variable or number with any base.

        INPUTS:
            x : AutoDiff.variable object or a number.
            b: a number

        OUTPUT:
            AutoDiff.variable
    """   
    try:
        new_val = np.log(x.val)/np.log(b)
        new_der = (1/(np.log(b)*x.val))*x.der
        return variable(val=new_val,der=new_der)
    except:
        new_val = np.log(x)/np.log(b)
        new_der = 0
        return variable(val=new_val,der=new_der)

def sqrt(x):
    """
        Helper function that calculates the square root of a variable or number.

        INPUTS:
            x : AutoDiff.variable object or a number.

        OUTPUT:
            AutoDiff.variable
    """
    try:
        new_val = np.sqrt(x.val)
        new_der = (0.5/np.sqrt(x.val))*x.der
        return variable(val=new_val,der=new_der)
    except:
        new_val = np.sqrt(x)
        new_der = 0
        return variable(val=new_val,der=new_der)

def exp(x):
    """
        Helper function that calculates the exponential of a variable or number.

        INPUTS:
            x : AutoDiff.variable object or a number.

        OUTPUT:
            AutoDiff.variable
    """
    try:
        new_val = np.exp(x.val)
        new_der = np.exp(x.val)*x.der
        return variable(val=new_val,der=new_der)
    except:
        new_val = np.exp(x)
        new_der = 0
        return variable(val=new_val,der=new_der)


def sinh(x):
    """
        Helper function that calculates hyperbolic sine of a variable or number
        
        INPUTS:
            x : AutoDiff.variable object or a number.

        OUTPUT:
            AutoDiff.variable
    """
    return (exp(x) - exp(-x)) / 2

def cosh(x):
    """
        Helper function that calculates hyperbolic cosine of a variable or number
        
        INPUTS:
            x : AutoDiff.variable object or a number.

        OUTPUT:
            AutoDiff.variable
    """
    return (exp(x) + exp(-x)) / 2


def tanh(x):
    """
        Helper function that calculates hyperbolic tangent of a variable or number
        
        INPUTS:
            x : AutoDiff.variable object or a number.

        OUTPUT:
            AutoDiff.variable
    """
    return sinh(x) / cosh(x)
    
def logistic(x, L=1, k=1, x0=1):
    """
        Helper function that calculates the logistic of a variable or number.

        INPUTS:
            x : AutoDiff.variable object or a number.

        OUTPUT:
            AutoDiff.variable
    """
    try:
        val = x.val
        der = x.der
        logistic = L/(1+np.exp(-k*(val-x0)))
        new_val = logistic
        new_der = logistic*(1-logistic)*der
        return variable(val=new_val,der=new_der)
    except:
        new_val = L/(1+np.exp(-k*(x-x0)))
        new_der = 0
        return variable(val=new_val,der=new_der)  




