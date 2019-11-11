import math
import numpy as np

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
        return "AutoDiff.variable(val={},der={})".format(self.val,self.der)

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
        print(self,other,-self.__sub__(-other))
        """
            Overload reverse subtraction.
            Implemented using negation and addition.
        """
        return self.__neg__().__add__(other)

    def __div__(self,other):
        raise NotImplementedError

    def __rdiv__(self,other):
        raise NotImplementedError

def sin(x):
    """
        Helper function that calculates the sin of a variable or number.

        INPUTS:
            x : AutoDiff.variable object or a number.

        OUTPUT:
            AutoDiff.variable
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
            x : AutoDiff.variable object or a number.

        OUTPUT:
            AutoDiff.variable
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
            x : AutoDiff.variable object or a number.

        OUTPUT:
            AutoDiff.variable
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
        return np.log(x)

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
        return np.sqrt(x)

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
        return np.exp(x)
