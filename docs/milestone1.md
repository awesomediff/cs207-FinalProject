Milestone 1
===========

## Introduction

AwesomeDiff is a Python 3 package designed by and for Data Scientists to help implement solving algorithms in Machine Learning. The package provides an implementation of the Forward Mode of Automatic Differentiation using operator overloading. AwesomeDiff uses AD to calculate the gradients of commonly used loss functions in Machine Learning, such as mean square errors (MSE).

Applications of Automatic Differentiation (AD) are numerous and play a key role in a wide range of fields, from demography to finance. We focus on applications in Machine Learning (ML), specifically the need to quickly and accurately compute the gradient of loss functions (i.e. its partial derivative with respect to each of the variables of interest).

Many Machine Learning algorithms estimate the parameters of a model by starting with a guess, evaluating a loss function (which represents the distance between that guess and an ideal solution), and using the gradient of the loss function to determine in which direction to search for a better guess. This process is iterative, and may require computing the gradient of the loss function thousands of times until the algorithm reaches an acceptable solution. For this reason, Data Scientists need ways to calculate derivatives both efficiently and accurately.

There are several approaches for solving derivatives, but there is a tradeoff between accuracy and computational complexity:
- Numerical approximations typically rely evaluating the function at small intervals and using the change in value to estimate the slope. These approaches are generally easy to implement but suffer from approximation errors, especially for functions with high curvature.
- Symbolic approaches use an analytical definition of the derivative to produce exact results by may be extremely complicated or expensive to evaluate. Furthermore, many Machine Learning algorithms rely on functions that have no closed-form derivative, making symbolic approaches impossible.
- In the context of Machine Learning, AD offers an ideal compromise between these two approaches: it makes use of the fact that computer programs define functions as a series of elementary operations (addition, exponentiation, etc.) which have known symbolic derivatives, and pairs the evaluation of the derivative with the evaluation of the function itself. AD can evaluate the derivative at a specified point with the same accuracy as function value.


## Background
### Graph structure of calculations:

Computer programs treat mathematical functions as a series of elementary operations (addition, exponentiation, etc). We can think of complicated functions as a graph consisting of nodes (representing the quantities being operated on) and edges (representing the operations being performed on those quantities). The output of one operation may form the input of another operation, which provides a way to represent nested operations. 
The user (or the program calling the function) uses input nodes (for example, `x` and `y`) to specify the point at which the function should be evaluated, and the value of the function at that point (for example `f(x,y)`) is represented by a final output node. 
For example, the loss function MSE can be represented with the following graph:
![MSE](https://github.com/awesomediff/cs207-FinalProject/blob/milestone1/docs/MSE.png?raw=true)

### Chain rule 
AD makes use of the fact that the elementary operations have known symbolic derivatives, and it makes use of the chain rule to iteratively update/compute derivatives of potentially complex programs.
![Chain Rule](https://github.com/awesomediff/cs207-FinalProject/blob/milestone1/docs/Chain_rule.png?raw=true)

As the function is evaluated at a given point, the forward mode of AD also evaluates the derivative of each elementary operation and applies the chain rule to keep track of compute the derivative of the function at that point. AD also has a reverse mode, which relies on the same principles but applies the chain rule in the opposite direction of traversing the calculation graph.

We can think of this in terms of the calculation graph described above: as the program traverses the graph to get from input(s) to output(s), and computes the value at each node, the forward mode simultaneously computes the value of the functions derivative at each node. The derivative value associated with the output node is (to machine precision) the value of the functions derivative at the point where the function is being evaluated. The AD approach can also be used to calculate the gradient or Jacobian of functions of multiple variables.

![Vector Input Example](https://github.com/awesomediff/cs207-FinalProject/blob/milestone1/docs/Vector%20Input%20Example.png?raw=true)

By pairing the evaluation of the derivative with the evaluation of the function itself, AD achieves machine precision.

## How to Use *AwesomeDiff*
### Installation
* The package will be available on 'PyPI.' 
    - You can either install the package in your local environment or in a virtual environment.  
* IF you have a Python3 environment with numpy installed ready to go, the awesomediff package can be installed using the following code:
```
pip install awesomediff
```
* If you want to install the package in a virtual environment, set up the virtual environment using:
```
conda create -n awesomediff python=3.6 anaconda
source activate awesomediff
git clone https://github.com/awesomediff/cs207-FinalProject.git
cd cs207-FinalProject
```
Then install the dependencies using
```
pip install -r requirements.txt
```

### Usage
- To use awesomediff, the user should import the library first using code similar to this:
```python
import awesomediff as ad
```
- Generally, the user should initialize an auto differentiation object (known as a `Variable` object in the package) for each variable they need in the function first using the module. 
- The user can form functions using the objects initialized to create more complex objects with elementary math operations and functions.
- The elementary math operations and functions can be performed in the same way as the operations in numpy.
- The constructor of a 'Variable' object takes in a scalar or a vector.
- The user can calculate the derivative(s) and the output value(s) at a targeting evaluation point.

Below are some example scenarios to demonstrate how the module works:  
  
#### I. Scalar function

This is the simplest case scenario. The AutoDiff object takes in a function that only involved elementary operations with scalers. In the below example, the AutoDiff object calculates the derivative at 3 for f(x) = 3x+15
```python
import awesomediff as ad

# instantiate an AutoDiff object at the given scalar point
x = ad.variable(5.0)

# create the function needed for differentiation
scalarFunc = 5*x**2

# get the derivative and output value 
print(scalarFunc.val())
print(scalarFunc.der())
```

The AutoDiff object can also take in functions that involve sine, cosine, and exponential terms. The next example shows that the AutoDiff handles these functions the same way as using numpy math functions.
```python
import awesomediff as ad

x = ad.variable(3.0)

# a function with an exponential term
# the exponential function is used in the same way as the exponential funciton in numpy
funcExp = ad.exp(x)*3+11

print(funcExp.val(), funcExp.der())
```

#### II. Vector function

The AutoDiff object can take in vectors as inputs:
```python
import awesomediff as ad
# instantiate an AutoDiff object with vector inputs
x = ad.variable([1,2,3])

# create the function needed for differentiation
funcVector = 5*x**2

# print out output values and derivatives
print(funcVector.val(), funcVector.der())
```

#### III. Scalar and vector functions with multiple variables

Differentiate multivariable functions can be used the same way with scalar and vector inputs.  
```python
import awesomediff as ad

# instantiate two AutoDiff objects with scalar inputs
x = ad.variable(3.0)
y = ad.variable(22.0)

# a function with more than one variable
funcMulti1 = 3*x+24*y

print(funcMulti1.val(), ffuncMulti1.der())

# instantiate two AutoDiff objects with vector inputs
w = ad.variable([1,2,3])
z = ad.variable([4,5,6])

funcMulti2 = 3*w+24*z

print(funcMulti2.val(), funcMulti2.der())

```

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
    - The [`README`](../README.md) file presents badges that show the tests performance and code coverage monitored by Travis CI and CodeCov.
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

