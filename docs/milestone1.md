Milestone 1
===========

## Introduction

AwesomeDiff is a Python 3 package designed by and for Data Scientists to help implement solving algorithms in Machine Learning. The package provides an implementation of the Forward Mode of Automatic Differentiation using operator overloading. AwesomeDiff uses AD to calculate the gradients of commonly used loss functions in Machine Learning, such as mean square errors (MSE).

Applications of Automatic Differentiation (AD) are numerous and play a key role in a wide range of fields, from demography to finance. We focus on applications in Machine Learning (ML), specifically the need to quickly and accurately compute the gradient of loss functions (i.e. its partial derivative with respect to each of the variables of interest).

Many Machine Learning algorithms estimate the parameters of a model by starting with a guess, evaluating a loss function (which represents the distance between that guess and an ideal solution), and using the gradient of the loss function to determine in which direction to search for a better guess. This process is iterative, and may require computing the gradient of the loss function thousands of times until the algorithm reaches an acceptable solution. For this reason, Data Scientists need ways to calculate derivatives both efficiently and accurately.

There are several approaches for solving derivatives, but there is a tradeoff between accuracy and computational complexity:
- Numerical approximations typically rely evaluating the function at small intervals and using the change in value to estimate the slope. These approaches are generally easy to implement but suffer from approximation errors, especially for functions with high curvature.
- Symbolic approaches use an analytical definition of the derivative to produce exact results by may be extremely complicated or expensive to evaluate. Furthermore many Machine Learning algorithms rely on functions that have no closed-form derivative, making symbolic approaches impossible.
- In the context of Machine Learning, AD offers an ideal compromise between these two approaches: it makes use of the fact that computer programs define functions as a series of elementary operations (addition, exponentiation, etc.) which have known symbolic derivatives, and pairs the evaluation of the derivative with the evaluation of the function itself. AD can evaluate the derivative at a specified point with the same accuracy as function value.


## Background


## How to Use *AwesomeDiff*


## Software Organization
- An overview of how we are organizing our software package. 
  * Directory structure  
  ```
   cs207-FinalProject\
        awesomediff\
            AutoDiff.py
            Efunc.py
            LossFunction.py
        tests\
            __init__.py
            test_diff.py
            test_efunc.py
            test_loss.py
        docs\
            README.md
        .gitignore
        .travis.yml
        README.md
        setup.cfg
        requirements.txt
  ```
  * Modules
    - `AutoDiff`
      - The main module that defines an AutoDiff object. It determines the properties of deriving a derivate using automatic differentiation.
    - `Efunc`
      - Contains elementary math functions including `sin`, `cos`, `log`, `exp`. These functions are written on the basis of the `numpy` package.
    - `LossFunction`
      - The module for the advanced feature (we haven’t decided what advanced feature we are going to implement now).
    - `test_diff`
      - Contains tests for different cases when using the package.  
  * Test
    - The tests of the package are located in `tests` directory.
    - We use Travis CI to run tests automatically, and we use CodeCov for checking code coverage automatically.
    - The `README` file presents badges that show the tests performance and code coverage monitored by Travis CI and CodeCov.
  * Distribution  
    - The `awesomediff` package will be available on PyPI.
    - The installation code can be found in the ["How to use `awesomediff`"](#how-to-use-awesomediff) section.
  * Dependency
    - `awesomediff` is dependent on the `numpy` package for elementary math functions.
  * Package
    - We are not using a framework because this package is relatively straightforward. We will follow the templates and tutorials on PyPI for the packaging process

## Implementation

#### `AutoDiff` Module
The module `AutoDiff` consists of the class `Variable`. `Variable` class can be used to instantiate variables of a function for which the user wishes to compute the derivative. When instantiating the `Variable` class, the user will input the value of the variable at which to calculate the function's derivative and the total number of variables in the function.

###### Example
To calculate the derivative of f(x,y) at x = a and y = b, the user will first need to instantiate `Variable` objects. 

```
x = Variable(a, 2)
y = Variable(b, 2)
```
The second parameter, 2, indicates that the function we wish to calculate the derivative for is a function of 2 variables. 


##### `Variable` Class Attributes
`Variable` class have two attributes: `self.val` and `self.der`, which keep track of the value of the elementary function and the value of the elementary function’s derivative, respectively. 

`self.val` is initialized with an int or a float that the user passed in as the first parameter of `Variable` class. ` 

`self.der` is a numpy array containing all the partial derivative values of the elementary function. The length of `self.der` will be initialized with the user’s input for the number of variables in the function of interest. 

For example, when calculating the derivative of f(x,y), `self.der` will have a length of 2 where the first element is the partial derivative value of the elementary function with respect to x and the second element is the partial derivative value of the elementary function with respect to y.

##### `Variable` Class Methods
`Variable` class overloads the following operations:  
`__add__`  
`__sub__`  
`__mul__`  
`__truediv__`  
`__pow__`  
`Variable` class has the corresponding `r` methods to make the above operations commutative.   


##### Elementary Operations
The module should not only be able to compute derivatives of variables that have been added, subtracted, multiplied, divided, or exponentiated by a scalar but also compute derivatives of sum, product, division, or powers of variables (e.g. derivative of x + y, x / y, x * y, x^y). To achieve the latter, each of the operation methods defined in `Variable` handles operations between a scalar and `Variable` object separately from operations between two `Variable` objects

###### Example
```
def __add__(x, y):
	# if x is a Variable object while y is a scalar:
		# value of derivative is unchanged
	# if self and other are both Variable objects:
		# value of derivative is sum of derivative values for x and y 
```

#### `EFunc` Module
`Efunc` module contains functions defining each of the following elementary functions:  

`sqrt`  
`exp`  
`ln`  
`log` (logarithm of any chosen base)  
`sin`, `cos`, `tan`  
`arcsin`, `arccos`, `arctan`  

`Efunc` module will rely on the numpy package to evaluate the above elementary functions. Functions for each of the elementary function will define how these operations should be doen on scalars as well as `Variable` objects. Take for example, the sin function in the `Efunc` module. 

```
def sin(x):
# if x is a scalar:
	# return np.sin(x)
# if x is a Variable object:
	# val = np.sin(x.val)
	# der = np.cos(x.val) * x.der
	# f = Variable(val)
	# f.der = der
	# return f – returns a Variable object
```

`Efunc` module is imported into `AutoDiff` module so that when `AutoDiff` module alone is imported by the user, the user can directly use the elementary functions on the `Variable` objects. 

`AutoDiff` module contains a function that handles vector functions. A vector function can be thought of as a list of functions. Therefore we can use the `Variable` class to compute derivatives of each of the component functions and return the outputs as a multi-dimensional array. 

