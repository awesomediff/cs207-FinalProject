Milestone 1
===========

## Introduction

`awesomediff` is a Python 3 package designed by and for Data Scientists to help implement solving algorithms in Machine Learning. The package provides an implementation of the Forward Mode of Automatic Differentiation using operator overloading. `awesomediff` uses AD to calculate the gradients of commonly used loss functions in Machine Learning, such as mean square errors (MSE).

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
![MSE](resources/MSE.png?raw=true)

### Chain rule

AD makes use of the fact that the elementary operations have known symbolic derivatives, and it makes use of the chain rule to iteratively update/compute derivatives of potentially complex programs.
![Chain Rule](resources/Chain_rule.png?raw=true)

As the function is evaluated at a given point, the forward mode of AD also evaluates the derivative of each elementary operation and applies the chain rule to keep track of compute the derivative of the function at that point. AD also has a reverse mode, which relies on the same principles but applies the chain rule in the opposite direction of traversing the calculation graph.

We can think of this in terms of the calculation graph described above: as the program traverses the graph to get from input(s) to output(s), and computes the value at each node, the forward mode simultaneously computes the value of the functions derivative at each node. The derivative value associated with the output node is (to machine precision) the value of the functions derivative at the point where the function is being evaluated. The AD approach can also be used to calculate the gradient or Jacobian of functions of multiple variables.

![Vector Input Example](resources/Vector%20Input%20Example.png?raw=true)

By pairing the evaluation of the derivative with the evaluation of the function itself, AD achieves machine precision.

## How to Use `awesomediff`

### Installation

* The package will be available on PyPI.
    - You can either install the package in your local environment or in a virtual environment.
* If you have a Python3 environment with `numpy` installed ready to go, the `awesomediff` package can be installed using the following code:
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

- To use `awesomediff`, the user should import the library first using code similar to this:
```python
import awesomediff as ad
```
- Generally, the user should initialize an auto differentiation object (known as a `variable` object in the package) first, for each variable needed in the function, using the module.
- The user can form functions using the objects initialized to create more complex objects with elementary math operations and functions.
- The elementary math operations and functions can be performed in the same way as the operations in `numpy`.
- The constructor of a `variable` object takes in a scalar or a vector.
- The user can calculate the derivative(s) and the output value(s) at a targeting evaluation point.

Below are some example scenarios to demonstrate how the module works:

#### I. Scalar function

This is the simplest case scenario. The `awesomediff.variable` takes in a function that only involved elementary operations with scalers. In the below example, the `awesomediff.variable` object calculates the derivative at 3 for f(x) = 3x+15
```python
import awesomediff as ad

# instantiate an awesomediff.variable object at the given scalar point
x = ad.variable(5.0)

# create the function needed for differentiation
scalarFunc = 5*x**2

# get the derivative and output value
print(scalarFunc.val())
print(scalarFunc.der())
```

The `awesomediff.variable` can also take in functions that involve sine, cosine, and exponential terms. The next example shows that `awesomediff` handles these functions the same way as using `numpy` math functions.
```python
import awesomediff as ad

x = ad.variable(3.0)

# a function with an exponential term
# the exponential function is used in the same way as the exponential funciton in numpy
funcExp = ad.exp(x)*3+11

print(funcExp.val(), funcExp.der())
```

#### II. Vector function

The `awesomediff.variable` can take in vectors as inputs:
```python
import awesomediff as ad
# instantiate an awesomediff.variable with vector inputs
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

# instantiate two awesomediff.variables with scalar inputs
x = ad.variable(3.0)
y = ad.variable(22.0)

# a function with more than one variable
funcMulti1 = 3*x+24*y

print(funcMulti1.val(), ffuncMulti1.der())

# instantiate two awesomediff.variables with vector inputs
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
            core.py
            func.py
            solvers.py
        tests\
            __init__.py
            test_diff.py
            test_func.py
            test_solvers.py
        docs\
            README.md
        .gitignore
        .travis.yml
        README.md
        setup.cfg
        requirements.txt
  ```
  * Modules
    - `core.py`
      - The main module that defines the `variable` and `function` classes. It determines the properties of deriving a derivate using automatic differentiation.
    - `func.py`
      - Contains elementary math functions including `sin`, `cos`, `log`, `exp`. These functions are written on the basis of the `numpy` package.
    - `solvers.py`
      - A module for the advanced features, which fill contain solvers to implement machine learning loss functions.
  * Testing
    - The tests of the package are located in `tests` directory:
      - `test_diff.py`
      - `test_func.py`
      - `test_solvers.py`
    - The tests follow the naming conventions of the `pytest` package, and can be executed by running `pytest` from anywhere in the package directory.
    - We also use Travis CI's GitHub integration to run tests automatically when new commits are pushed.
    - Similarly, we use CodeCov to automatically check for code coverage in the GitHub repository. We seek to maintain at least 90% code coverage.
    - The [`README`](../README.md) file presents badges that show the tests performance and code coverage monitored by Travis CI and CodeCov.
  * Distribution
    - The `awesomediff` package will be available on PyPI.
    - The installation code can be found in the ["How to use `awesomediff`"](#how-to-use-awesomediff) section.
  * Dependency
    - `awesomediff` is dependent on the `numpy` package for elementary math functions.
  * Package
    - We are not using a framework because this package is relatively straightforward. We will follow the templates and tutorials on PyPI for the packaging process

## Implementation Details

### `awesomediff` package structure

The automatic differentiation functionality is stored in the `core` and `func` modules. The applications to machine learning cost functions are stored in the `solvers` modules. Additionally, we provide testing modules for each of these implementation modules.

The `awesomediff` package abstracts these different modules and allows users to access all functionality directly from the `awesomediff` namespace. We recommend the following import convention:
```python
import awesomediff as ad
```

The core data structure of the `awesomediff` package is the `variable` object, which represents the value and derivative of a function and defines how it should behave when [elementary operations](#Elementary-Operations) are performed on it (perhaps in conjunction with another function).

We also provide a `function` class, which provides a wrapper for performing functions of multiple variables and with multiple input values. We expect users to interact mainly with the `function` class, although they may implement simple functions directly with the `variable` class.

The `awesomediff` package also provides tools for data scientists to apply AD to Machine Learning tasks. The `solvers` module contains loss functions and solving routines to implement optimization tasks such as gradient descent.

_(**Note:** As of November 2019, multivariate functionality and machine learning applications have not yet been implemented. Please see the [Roadmap](#Roadmap-for-Future-Development) section below for more details on how we plan to implement these features.)_

### The `variable` Class

The `variable` object is the core data structure of `awesomediff`'s AD functionality. It represents a node in the execution graph of a function, and stores the variable and derivate of the function at that node. It also defines how a `variable` should behave when an [elementary operation](#Elementary-Operations) elementary function is performed on it (perhaps in conjunction with a numeric object or another `variable` object).

All elementary operations performed on `variable` instances return new `variable` instances with appropriate values and derivatives, which allows them to propagate through a function graph and guarantees that the result will be a `variable`, with a value and derivative that the user can access.

`aweseomdiff`'s implementation of the forward mode of AD does not explicitly define a function graph. Instead, it overloads Python's operators (and defines special functions for common operations like `log` and `sin`), and allows Python's interpreter to build the function graph implicitly as it evaluates the functions.

For simple functions, users can create `variable` objects with a specified value and derivative seed. The seed may be a scalar, or a vector of length `n`, where `n` is the number of variables in the function. (For more complex functions, we recommend using a `function` objects, which handles the creation of `variable` objects to avoid dimensionality issues.)

The `variable` class is immutable. It has two attributes, `.val` and `.der`, which store the value and derivative of the function. (In the current implementation, both are scalars; future implementations will store derivatives as a vector.)

The example below demonstrates how a `variable` can be created and used in functions to calculate the derivative and variable of any univariate function of [supported elementary operations](#Elementary-Operations).

#### Demo 1: Univariate functions

**Evaluating `f(x)=3*sin(0.5*x)^2` at `x=5`.**

```python
import math
import awesomediff as ad

# Method 1:
x = ad.variable(val=math.pi)
f = 3*ad.sin(0.5*x)**2
print(f.val)  # 3
print(f.der)  # 0

# Method 2:
calc_area = lambda r: math.pi*r**2  # Area of a circle.
radius = ad.variable(10)
area = calc_area(radius)
print(area.val)  # 314.1592653589793==100*math.pi
print(area.der)  # 62.83185307179586==20*math.pi

# Method 3:
def exp_cdf(x,rate):
  # CDF of exponential distribution:
  return 1-ad.exp(-rate*x)
x = ad.variable(0.5)
result = exp_cdf(x=x,rate=5)
cdf = result.val
pdf = result.der
print(cdf)  # 0.9179150013761012
print(pdf)  # 0.410424993119494
```

### The `function` Class

_(**Note:** The `function` class is not yet implemented. It will be added as part of the multivariate functionality described in the [Roadmap](#Roadmap-for-Future-Development) section below.)_

The `function` class takes as inputs a function (defined which uses any elementary operations supported by `awesomediff`), a list of values (possibly vectors) at which to evaluate the function, and a matrix representing the seed of the derivative. It then evaluates the function and derivative using the specified values and seed, and provides methods for the user to access the results. Optionally, users can provide a list of labels (such as 'x' and 'y') that can be used to access values and partial derivatives for a specific variable.

The `function` class takes as arguments a user-defined function of `n` inputs and `m` outputs, a list of `n` values (each corresponding to the initial value of one of the input variables) and an `n`-by-`n` seed for the derivative.

By default, the seed is is set to an identity matrix, which will produce the partial derivative with respect to each of the variables.

The `function` class verifies that the inputs have coherent dimensions, then wraps each value in a `variable`, evaluates the function on those variables, and stores the values and derivatives for the user to access.

#### Demo 2: Multivariate functions (future)

To calculate the derivative of f(x,y) at x=5 and y=4, the user can instantiate `variable` objects directly or use a `function` object.

```python
# Method 1:
x = ad.variable(val=5, seed=[1,0])
y = ad.variable(val=4, seed=[0,1])
f = 2*x + y^2
print(f.val)  # 26
print(f.der)  # [2,8]

# Method 2:
func = lambda x,y: 2*x + y^2
f = ad.function(func=func, vals=[5,4], seed=[[1,0],[0,1]], labels=['x','y'] )
print(f.val)  # 26
print(f.der)  # [2,8]
print(f.der['x'])  # 2
print(f.der['y'])  # 8
```

_(Note: The current implementation exposes the derivative and variable to the user as properties. In future implementations, we plan to update the interface so that users can also access derivatives with a function that allows them to specify which variable's derivative to return. The example above assumes for simplicity that the user provides labels for each variable which are uses as the keys of a dictionary that stores the derivatives.)_

### Elementary Operations

The module should not only be able to compute derivatives of variables that have been added, subtracted, multiplied, divided, or exponentiated by a scalar but also compute derivatives of sum, product, division, or powers of variables (e.g. derivative of `x + y`, `x / y`, `x * y`, `x**y`). To achieve the latter, each of the operation methods defined in `variable` handles operations between a scalar and `variable` object separately from operations between two `variable` objects.

The `variable` class overloads the following unary operations:
- `__neg__`

The `variable` class overloads the following binary operations (and their respective reverse methods):
- `__add__`
- `__sub__`
- `__mul__`
- `__truediv__`
- `__pow__`

Each binary operator acts on the `variable` instance and can accept a scalar or a another `variable` instance as its other argument. For example, addition is overloaded in the following way:

```python
def __add__(x, y):
  try:
    # If other is a variable object:
    other_val = other.val
    other_der = other.der
  except:
    # If other is a scalar:
    other_val = other
    other_der = 0  # A scalar has zero derivative.
  new_val = self.val + other_val
  new_der = self.der + other_der  # Derivative of sum is sum of derivatives.
  result = ad.variable(new_val,new_der)
  # Return a variable object:
  return result
```

The `func` module contains functions defining each of the following elementary functions (as well as their corresponding reverse operations):

- `sqrt`
- `exp`
- `ln`
- `log` (logarithm of any chosen base)
- `sin`, `cos`, `tan`
- `arcsin`, `arccos`, `arctan` (not implemented yet)

The `func` module relies on the `numpy` package to evaluate the above elementary functions. The methods in the `func` module define how to perform elementary operations on scalars as well as `variable` objects. 

For example, the `sin` function uses `numpy.sin` and `numpy.cos` to calculate value and derivative respectively. When a scalar input is detected, it is given a null derivative (i.e. treated as a constant). Like most `awesomediff` functions, `sin` returns a `variable` object with the updated value and derivative (perhaps zero). 

```python
def sin(x):
  try:
    # If x is a variable object:
    val = np.sin(x.val)
    der = np.cos(x.val) * x.der
  except:
    # If x is a scalar:
    val = np.sin(x)
    der = 0
  result = ad.variable(val,der)
  # Return a variable object:
  return result
```

## Roadmap for Future Development

### Multivariate and Vectorized Implementations

We are working on these features in a [development branch](https://github.com/awesomediff/cs207-FinalProject/blob/docs/docs/milestone2.md) and hope to roll them out soon.

Our multivariate implementation will store the derivatives in `numpy` arrays. This will allow us to store the derivative with respect to multiple seeds. We will make use of vectorized arithmetic with `numpy` to enable quick calculations. Most of our existing operators will easily work with vectors instead of scalars. 

For binary operations, we will verify that the derivatives of each `variable` have the same dimensions. If that is not the case, we will return an error. We will also provide a `function` class that acts as a wrapper to build and manage `variable` instances with the appropriate dimensions for the input values and seed the user provides.

We discuss this this functionality of this `function` class and sketch out a demo of how the user will interact with it in the[`function` Class section (above)](#The-function-Class).

### Machine Learning Applications
