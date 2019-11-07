class variable:

    def __init__(self,val,der=1):
        """
            Initialize a variable object
            with a specified value and derivative.
            If no derivative is specified,
            a seed of 1 is assumed.
        """

        self.val = val
        self.der = der

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
