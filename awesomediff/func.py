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
