Documentation
===========

## Introduction

`awesomediff` is a Python 3 package designed for data scientists to help implement solving algorithms in Machine Learning. The package provides an implementation of the Forward Mode of Automatic Differentiation using operator overloading. `awesomediff` uses AD to calculate the gradients of commonly used loss functions in Machine Learning. Our package offers a special focus on optimization of regression problem and can handle both Ridge and Lasso regularization cases through gradient descent. The package also provides additional root-finding feature utilizing Newton’s method.  

Applications of Automatic Differentiation (AD) are numerous and play a key role in a wide range of fields, from demography to finance. We focus on applications in Machine Learning (ML), specifically the need to quickly and accurately compute the gradient of loss functions (i.e. its partial derivative with respect to each of the variables of interest).

Many Machine Learning algorithms estimate the parameters of a model by starting with a guess, evaluating a loss function (which represents the distance between that guess and an ideal solution), and using the gradient of the loss function to determine in which direction to search for a better guess. This process is iterative, and may require computing the gradient of the loss function thousands of times until the algorithm reaches an acceptable solution. For this reason, data scientists need ways to calculate derivatives both efficiently and accurately.

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

Univariate function case: given the function f(u(t)), we want the derivative of f with respect to t, by applying the chain rule, we get:

![Chain Rule](resources/Chain_rule.png?raw=true)

As the function is evaluated at a given point, the forward mode of AD also evaluates the derivative of each elementary operation and applies the chain rule to keep track of compute the derivative of the function at that point. AD also has a reverse mode, which relies on the same principles but applies the chain rule in the opposite direction of traversing the calculation graph.

We can think of this in terms of the calculation graph described above: as the program traverses the graph to get from input(s) to output(s), and computes the value at each node, the forward mode simultaneously computes the value of the functions derivative at each node. The derivative value associated with the output node is (to machine precision) the value of the functions derivative at the point where the function is being evaluated. The AD approach can also be used to calculate the gradient or Jacobian of functions of multiple variables.

For vector of multi-variate functions, 
![vector functions](resources/vector_of_func.png?raw=true)
the chain rule states that:
![jacobian](resources/jacobian.png?raw=true)


By pairing the evaluation of the derivative with the evaluation of the function itself, AD achieves machine precision.

## Efficiency

Before talking about the efficiency of the automatic differentiation, we can look at the drawbacks of  symbolic differentiation or numerical differentiation. Both of them can be used to compute mathematical expression. A symbolic differentiation program finds the derivative of a given formula with respect to a specified variable, producing a new formula as its output, but applying it for higher order derivatives cannot always ensure a small computational cost.  Numerical differentiation or finite differences suffers round-off errors. Therefore as the number of operation increases, the derivative calculation is not precise to machine precision and it also has stability issues. 

Automatic differentiation, on the other hand, escapes the limitations posed by symbolic differentiation and numerical differentiation. It exploits the idea that the computing the derivative of a function, no matter how complex, can be broken down into a sequence of elementary arithmetic operations (addition, subtraction, multiplication, division) and elementary functions (sin, cos, log, exp, etc.). By drawing the computation graph and applying the chain rule repeatedly to these operations derivatives, the value of the function can be evaluated accurately to machine precision. It is exact, and the speed is comparable to hand-coding derivative. Thus, we design this automatic differentiation package to implement AD, which aims to allow differentiation to occur seamlessly, while users can just focus on their programming tasks.


## How to Use `awesomediff`

### Installation

* The package is available on PyPI.
    - You can either install the package in your local environment or in a virtual environment.

`awesomediff` package can be installed using the following code:
```
pip install awesomediff
```

* For developers, you can install the package by getting our Github repository following these steps:
* Clone the project's git repository to your machine:
```
git clone https://github.com/awesomediff/cs207-FinalProject.git
```
Then install the dependencies:
```
pip3 install -r requirements.txt
```
If you want to run the testing with `pytest`, run the following from the root:
```
pytest tests/
```

* If you want to install the package in a virtual environment, set up the virtual environment in the cloned directory using:
```
pip install virtualenv
virtualenv -p python3 venv
source venv/bin/activate
```
Inside the virtual environment, you can either install the package using pip or clone the Github repository with the same steps as above.

### Usage

- To use `awesomediff`, the user should import the library first using code similar to this:
```python
import awesomediff as ad
```
- To compute the value and derivative/jacobian of a function at a specified set of values, the user will first define a function using elementary operations supported by `awesomediff` and pass it to the `evaluate` function along with a list of values of variables at which to evaluate. 
- While it is advised that the user call `evaluate` to compute the value and derivative/jacobian of a function, the user may also directly instantiate an auto differentiation object (known as a `variable` object in the package) and access the value and derivative via the variable object's `val` and `der` attributes, respectively. The user can form functions using the objects initialized to create more complex objects with elementary math operations and functions.
- The elementary math operations and functions can be performed in the same way as the operations in `numpy`.

Below are some examples to demonstrate how the module works:

```python
import math
import awesomediff as ad
```

###### Case 1: Evaluate value and derivative of area = pi * r^2,  r = 10

```python
def calc_area(r):
  area = math.pi*r**2
  return area

area_circle, circumference = ad.evaluate(func=calc_area, vals=10)

# area when radius = 10
>>> print("area at radius = 10:", area_circle)
314.159 

# derivative of area (i.e. circumference) when radius = 10
>>> print("circumference at radius = 10:", circumference) 
62.832
```

In the case of a single univariate function, user can also directly instantiate `variable` object to calculate the output value and derivative
```python
# area of a circle
calc_area = lambda r: math.pi*r**2  

# instantiate variable object 
radius = ad.variable(10)

# pass in variable object into lambda function
area = calc_area(radius)

# area when radius = 10
>>> print("area at radius = 10:", area.val)  
314.159 

# derivative of area (i.e. circumference) when radius = 10
>>> print("circumference at radius = 10:", area.der)
62.832
```

###### Case 2: Evaluate value and partial derivatives of 1 - e^(-rx), x = 0.5, r = 5

```python
# define function
def function(x,r):
    func1 = 1-ad.exp(-r*x)
    return func1

# pass function and list of values at which to evaluate 
output_value, partial_ders = ad.evaluate(func=function, vals=[0.5, 5])

# value of function
>>> print("output value at x = 0.5, r = 5", output_value) 
0.9179

# derivative of function 
>>> print("partial derivatives at x = 0.5, r = 5", partial_ders)
[0.4104, 0.0410]  # [e^(-rx)r, e^(-rx)x]
```

###### Case 3: Evaluate value and jacobian of f = [[xy + cos(x)], [x + y + cos(y)]]

```python
# define function
def function(x,y):
    func1 = x * y + ad.cos(x)
    func2 = x + y + ad.cos(y)
    return [func1, func2]

# pass function and list of values at which to evaluate 
output_value, jacobian = ad.evaluate(func=function, vals=[1, 1])

# value of function
print("output value at x = 1, y = 1", output_value)
>>> [1.5403, 2.5403] # [value of func1, value of func2]

# derivative of function
print("jacobian at x = 1, y = 1", jacobian) # [[y-sin(x), x], [y, x+cos(y)]]
>>> [[0.1585, 1],
     [1, 0.1585]]
```



## Software Organization

- An overview of how we will eventually organize our software package.
  * Directory structure
  ```
   cs207-FinalProject\
        awesomediff\
            __init__.py
            core.py
            func.py
            solvers.py
        tests\
            __init__.py
            test_core.py
            test_func.py
            test_solvers.py
        docs\
            resources\
                Chain_rule.png
                MSE.png
                Vector Input Example.png
            milestone1.md
            milestone2.md
            documentation.md
        univariate_demos.ipynb
        .gitignore
        .travis.yml
        README.md
        setup.cfg
        setup.py
        LICENSE.txt
        requirements.txt
  ```
  * Modules
    - `core.py`
      - The main module that defines the `variable` class. The model also contains the `evaluate` function to handle functions directly. It determines the properties of deriving a derivate using automatic differentiation.
    - `func.py`
      - Contains elementary math functions including trig functions, inverse trig functions, exponentials, hyperbolic functions, logistic function, logarithms, and square root. These functions are written on the basis of the `numpy` package.
    - `solvers.py`
      - A module for the advanced features, which include solvers to implement machine learning loss functions and Newton's method.
  * Testing
    - The test files correspond to the files in the module.
    - The tests of the package are located in `tests` directory:
      - `test_core.py`
      - `test_func.py`
      - `test_solvers.py`
    - The tests follow the naming conventions of the `pytest` package, and can be executed by running `pytest` from anywhere in the package directory.
    - We also use Travis CI's GitHub integration to run tests automatically when new commits are pushed.
    - Similarly, we use CodeCov to automatically check for code coverage in the GitHub repository. We seek to maintain at least 90% code coverage.
    - The [`README`](../README.md) file presents badges that show the tests performance and code coverage monitored by Travis CI and CodeCov.
  * Distribution
    - The `awesomediff` package is available on PyPI.
    - The installation code can be found in the ["How to use `awesomediff`"](#how-to-use-awesomediff) section.
  * Dependency
    - `awesomediff` is dependent on the `numpy` package for elementary math functions. We also use `pytest` for testing.
 

## Implementation Details

### `awesomediff` package structure

The automatic differentiation functionality is stored in the `core` and `func` modules. The applications to machine learning cost functions and Newton's method are stored in the `solvers` modules. Additionally, we provide testing modules for each of these implementation modules.

The `awesomediff` package abstracts these different modules and allows users to access all functionality directly from the `awesomediff` namespace. We recommend the following import convention:
```python
import awesomediff as ad
```

The core data structure of the `awesomediff` package is the `variable` object, which represents the value and derivative of a function and defines how it should behave when [elementary operations](#Elementary-Operations) are performed on it (perhaps in conjunction with another function).

We also provide `evaluate` function, stored in the `core` module. The `evaluate` function allows users to compute the values and the jacobian of univariate and multivariate functions as well as a vector of univariate and multivariate functions. We expect users to call the `evaluate` function rather than directly instantiating `variable` objects to compute the values and derivatives of functions (although they are not precluded from doing so). 

The `awesomediff` package also provides tools for data scientists to apply AD to Machine Learning tasks. The `solvers` module contains loss functions and solving routines to implement optimization tasks such as gradient descent.


### The `variable` Class

The `variable` object is the core data structure of `awesomediff`'s AD functionality. It represents a node in the execution graph of a function, and stores the variable and derivate of the function at that node. It also defines how a `variable` should behave when an [elementary operation](#Elementary-Operations) elementary function is performed on it (perhaps in conjunction with a numeric object or another `variable` object).

All elementary operations performed on `variable` instances return new `variable` instances with appropriate values and derivatives, which allows them to propagate through a function graph and guarantees that the result will be a `variable`, with a value and derivative that the user can access.

`aweseomdiff`'s implementation of the forward mode of AD does not explicitly define a function graph. Instead, it overloads Python's operators (and defines special functions for common operations like `log` and `sin`), and allows Python's interpreter to build the function graph implicitly as it evaluates the functions.

For simple functions, users can create `variable` objects with a specified value (`val`) and derivative seed (`seed`). `val` expects a scalar. `seed` may be a scalar, or a numpy array of length `n`, where `n` is the number of variables in the function. (For computing the value and derivative/jacobian of more complex functions, we recommend calling `evaluate`, which handles the creation of `variable` objects internally). 

The `variable` class is immutable. It has two attributes, `.val` and `.der`, which store the value and derivative of the function.

The example below demonstrates how a `variable` can be created and used in functions to calculate the derivative and variable of any univariate function of [supported elementary operations](#Elementary-Operations).

###### Example: Evaluate value and derivative of f(x) = 3 * sin(0.5x)^2,  x = pi

```python
# instantiate variable object
x = ad.variable(val=math.pi)

# define function with x, which is a variable object
f = 3*ad.sin(0.5*x)**2

# value of f(x) at x = pi
print("value at x = pi:", f.val) # 3.0  

# derivative of f(x) at x = pi
print("derivative at x = pi:", f.der) # 1.837e-16
```


### The `evaluate` function

The `evaluate` function takes as inputs 1) `func`: a user-defined function that involves any elementary operations supported by `awesomediff`, and 2) `vals`: a value at which to evaluate the function (a list of values if the function is multivariate). `vals` expects a scalar (float, int) or a list of scalars. `evaluate` returns the output value and the jacobian of the function evaluated at the specified value as arrays.  

`evaluate` also takes an optional argument `seed` if the user wishes to set the seeds of variables to a value other than 1. Otherwise, `evaluate` uses a default value of 1 for seed. For further details on how to provide an input for `seed`, see [Input for Seed](#Input-for-Seed).  


The example below steps through how a user would call `evaluate` to compute the output values and the jacobian of a multivariate vector function at a specified set of values. 

In the case of vector functions, the user-defined function should return the component functions as a list. For example, for a vector function composed of f1 and f2, the function should return f1 and f2 in a single list. 
```python
def function(x,y):
  f1 = ad.sin(x) + 2*y
  f2 = 2*x + ad.cos(y)
  return [f1,f2]
```

Once the user defines a function at which to evaluate the output value and the jacobian, the user calls `evaluate` in the following manner. To evaluate the ouput value and the jacobian of the multivariate vector function (as defined above) at x = 0 and y = 1, 
```python
output_value, jacobian = ad.evaluate(func=function, vals=[0, 1])
```
`evaluate` assumes that the order in which the values are provided in the list passed to `vals` corresponds to the order in which the variables were declared in the user-defined function. Here it is assumed that the user wishes to evaluate the multivariate vector function at x = 0 and y = 1. `evaluate` verifies that the number of values (i.e. the length of the list passed to `vals`) agrees with the number of variables in the function. 

`evaluate` returns the following:
```python
output_value
>>> [2, 0.5403]
jacobian
>>> [[1, 2],
     [2, -0.8415]]
```
The first row of the jacobian contains the partial derivatives of f1 with respect to x and with respect to y in the first and second columns, respectively. The second row of the jacobian contains the partial derivatives of f2 with respect to x and with respect to y in the first and second columns, respectively. In general, the jacobian has a dimension of m x n where m = number of component functions in a vector function and n = number of variables. In this example, the jacobian has the dimension 2 x 2 since the vector function consists of two component functions, f1 and f2 of two variables. 

For a single univariate function, the jacobian is a scalar (a numpy array of a single value), and for a single multivariate function, the jacobian is a numpy row vector. For a univariate vector function, the jacobian is a numpy column vector, and for a multivariate vector function, the jacobian is a 2-dimensional numpy array. 


#### Input for Seed
If the user does not pass in an input for the `seed` argument, it is assumed that the seed for each variable is 1. For example, for a function of two variables x and y, it is assumed that the seed for x = [1, 0] and the seed for y = [0, 1]. If the user instead wishes to specify the seed of x to be 2 and the seed of y to be 3, the user will input the seed as [[2, 0], [0, 3]], a list of lists:

```python
output_value, jacobian = ad.evaluate(func=function, vals=[0, 1], seed=[[2, 0], [0, 3]])
```
In general, the length of the outer list and the lengths of each of the inner lists for seed should equal the total number of variables of the function passed. 


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

The `variable` class overloads the following comparison operators:
- `__eq__`
- `__ne__`
- `__lt__`
- `__gt__`
- `__le__`
- `__ge__`

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

- `sin`, `cos`, `tan`
- `arcsin`, `arccos`, `arctan` 
- `exp`
- `sinh`, `cosh`, `tanh`
- `sqrt`
- `logistic`


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

### Machine Learning Toolkit

The Awesomediff Team's interest in automatic differentiation is driven by Data Scientists' need for efficient and accurate ways of repeatedly evaluating derivatives as part of optimization problems in Machine Learning.

The awesomediff package includes a Machine Learning Toolkit that uses gradient descent to solve linear regressions and perform root finding with Newton's method.

#### Gradient Descent Solver

<center><img src="resources/gradient-descent.png?raw=true" width="50%" /></center>

Gradient descent is the algorithm at the heart of many machine learning techniques. It provides a very straightforward way of finding a local minimum of differentiable cost functions (and in the case of convex functions, this is guaranteed to be a global minimum).

Gradient Descent can be used to solve many types of models, including regression models, that have a cost function and a series of weights. The algorithm begins with randomly initialized weights, uses the model to calculate a solution based on those weights, then evaluates the cost associated with that solution.

Once the cost has been found, it looks for a new solution with a lower cost by taking a step along the gradient of the cost function to choose new weights. The size of the step is controlled by a learning rate. This process is repeated iteratively until a stopping condition has been met.

The algorithm may stop when it exceeds a maximum number of iterations (indicating that it failed to converge on a solution) or when the difference in cost is below a certain absolute or relative threshold (indicated that it has found a point acceptably close to a minimum).

The `GradientDescent` class included in `awesomediff`'s toolkit implements this algorithm and allows the user to specify a learning rate and stopping conditions.

Note: It is not generally necessary to create `GradientDescent` objects directly, (although advanced users may with to use the class to create their own solvers, as described in the section [below](#How-to-Contribute-to-awesomediff)); instead, they are initialized under the hood in `Model` objects that the user creates. Creating linear regression models that make use of the gradient descent algorithm is documented in the following section.

#### Linear Regression Models

Linear regressions model data using an equation of the following form `Y=β₀+β₁X+β₂X+...+βₚ` , where `X` is an `n`-by-`p` matrix of predictor variables and `Y` is an `n`-length vector of response variables.

The simplest regression model, ordinary least squares (OLS) regression, finds the values of `β` that minimize the mean squared error (MSE) of the predictions. More advanced models, such as Ridge and Lasso regression, add a regularization term which penalizes large coefficient values. This helps reduce the problem of overfitting and helps find solutions that generalize better to new data.

##### Ordinary Least Squares Regression

The OLS regression takes the following initialization parameters:

- `fit_intercept` : A boolean indicating whether or not to include an intercept term.
- `standardize` : A boolean indicating whether or not to standardize the data (mean=0,stdev=1) before fitting.
- `solver` : A string indicating which solver to use (currently only `'gradient_descent'` is supported).
- `solver_kwargs` : A dictionary of optional arguments to pass to the solver.

OLS demo:

```python
X = np.array([[-1,4,-1,-3],[1,-3,1,2],[0,2,0,6],[-2,-5,1,-6],[3,5,1,3],[-3,-5,-3,-3]])
y = np.array([13,-2.5,-1,6,15.5,-23.5])

print("\nWith intercept:")
reg = ad.LinearRegression(fit_intercept=True,learning_rate=0.01,max_iter=3000)
reg.fit(X,y)
equation = "y = {:.3f}".format(reg.intercept)
for i,b in enumerate(reg.coefs):
    equation += " + {:.3f} * x{}".format(b,i+1)
equation = equation.replace("+ -","- ")
print(equation)
print("R^2 =",reg.score(X,y))

print("\nWithout intercept:")
reg = ad.LinearRegression(fit_intercept=False,learning_rate=0.01,max_iter=3000)
reg.fit(X,y)
equation = "y = "
for i,b in enumerate(reg.coefs):
    equation += " + {:.3f} * x{}".format(b,i+1)
equation = equation.replace("+ -","- ").replace("= +","=")
print(equation)
print("R^2 =",reg.score(X,y))
```

Output:

```
With intercept:
y = 2.711 - 0.841 * x1 + 2.750 * x2 + 6.461 * x3 - 1.519 * x4
R^2 = 0.9994659005096189

Without intercept:
y =  - 1.314 * x1 + 2.785 * x2 + 6.569 * x3 - 1.414 * x4
R^2 = 0.9571266824223628
```

##### Lasso Regression (L1 norm regularization)

In addition to the parameters described above, the Ridge regression construction takes an additional `l1_penalty` parameter, specifying the penalty applied to the L1 norm of the coefficients.

Lasso regression example:

```python
X = np.array([[-1,4,-1,-3],[1,-3,1,2],[0,2,0,6],[-2,-5,1,-6],[3,5,1,3],[-3,-5,-3,-3]])
y = np.array([13,-2.5,-1,6,15.5,-23.5])

print("\nWith intercept:")
reg = ad.LassoRegression(fit_intercept=True,learning_rate=0.01,max_iter=3000,l1_penalty=1.0)
reg.fit(X,y)
equation = "y = {:.3f}".format(reg.intercept)
for i,b in enumerate(reg.coefs):
    equation += " + {:.3f} * x{}".format(b,i+1)
equation = equation.replace("+ -","- ")
print(equation)
print("R^2 =",reg.score(X,y))

print("\nWithout intercept:")
reg = ad.LassoRegression(fit_intercept=False,learning_rate=0.01,max_iter=3000,l1_penalty=1.0)
reg.fit(X,y)
equation = "y = "
for i,b in enumerate(reg.coefs):
    equation += " + {:.3f} * x{}".format(b,i+1)
equation = equation.replace("+ -","- ").replace("= +","=")
print(equation)
print("R^2 =",reg.score(X,y))
```

Output:

```
With intercept:
y = 2.713 - 0.828 * x1 + 2.748 * x2 + 6.450 * x3 - 1.521 * x4
R^2 = 0.9994642054709294

Without intercept:
y =  - 1.369 * x1 + 2.795 * x2 + 6.614 * x3 - 1.406 * x4
R^2 = 0.9571496985554787
```

##### Ridge Regression (L2 norm regularization)

In addition to the parameters described above, the Ridge regression construction takes an additional `l2_penalty` parameter, specifying the penalty applied to the L2 norm of the coefficients.

Ridge regression example (using the standardize parameter: the solver converges better when data is transformed such that each feature has a mean of `0` and a standard deviation of `1`):
```python
X = np.array([[-1,4,-1,-3],[1,-3,1,2],[0,2,0,6],[-2,-5,1,-6],[3,5,1,3],[-3,-5,-3,-3]])
y = np.array([13,-2.5,-1,6,15.5,-23.5])

print("\nWith intercept:")
reg = ad.RidgeRegression(fit_intercept=True,standardize=True,learning_rate=0.05,max_iter=3000,l2_penalty=1.0)
reg.fit(X,y)
equation = "y = {:.3f}".format(reg.intercept)
for i,b in enumerate(reg.coefs):
    equation += " + {:.3f} * x{}".format(b,i+1)
equation = equation.replace("+ -","- ")
print(equation)
print("R^2 =",reg.score(X,y))

print("\nWithout intercept:")
reg = ad.RidgeRegression(fit_intercept=False,standardize=True,learning_rate=0.05,max_iter=3000,l2_penalty=1.0)
reg.fit(X,y)
equation = "y = "
for i,b in enumerate(reg.coefs):
    equation += " + {:.3f} * x{}".format(b,i+1)
equation = equation.replace("+ -","- ").replace("= +","=")
print(equation)
print("R^2 =",reg.score(X,y))
```

Output:

```
With intercept:
y = 1.244 - 1.410 * x1 + 11.222 * x2 + 9.271 * x3 - 6.280 * x4
R^2 = 0.9993211387084784

Without intercept:
y =  - 1.402 * x1 + 11.219 * x2 + 9.267 * x3 - 6.282 * x4
R^2 = 0.9898979933989382
```

#### Newton's Method

In numerical analysis, Newton's method is a root-finding algorithm which which produces successively better approximations to the roots (or zeroes) of a real-valued function. The most basic version starts with a single-variable function f defined for a real variable x, the function's derivative f′, and an initial guess x0 for a root of f. If the function satisfies sufficient assumptions and the initial guess is close, then 

![Newton](resources/Newton.png?raw=true). 

Our package provides this root-finding algorithm. The user can input a function and a starting point, and optionally specify stopping conditions(max iteration or change in function value), the function is expected to return one of the roots of the function if it has one.  There are two cases where the function cannot find a root. In the first case, the function will return None if we reach the max iteration but the change in function value is less than default/specified epsilon. In the second case, the function also returns None when the function derivative is 0. 

Below is an illustration of how to implement Newton's method through uni_Newton function from Awesomediff package. If no root is found, a message explaining what might be the possible cause will be displayed.

```python
def root_finding(a):
    return a**2 + 2*a + 1
    
root = uni_Newton(root_finding, 50)
>>> 9.689480066299438e-06
```

### How to Contribute to `awesomediff`

In order to facilitate extensions of `awesomediff` functionality, we have have provided super-classes for solvers and models.

**Solvers** take an instance of a `Model` subclass in their constructor, and implement a `solve` method which iteratively invokes the model's `_predict` and `_loss` functions.

```python
class Solver:

    """A superclass for defining a solver."""

    def __init__(self,model,**solver_params):
        self.model = model
        pass

    def solve(self,model_params,initial_weights,X,y):
        raise NotImplementedError
```

**Models** implement expose the `fit`, `predict`, and `score` methods to the user, and internally implement `_predict` and `_loss` (which are invoked by the solver). Since models may use various configurations for their weights (for example, regressions have coefficients as well as an optional intercept term), the model may use the `_pack_weights` and `_unpack_weights` methods to transit weights to and from the solver as a single list.

```python
class Model:

    def __init__(self):
        """Initialize the model."""
        pass

    @classmethod
    def _pack_weights(cls,model_params,placeholder):
        """Create a single list of weights to pass to the solver."""
        raise NotImplementedError

    @classmethod
    def _unpack_weights(cls,model_params,weights):
        """Extract weights from list returned by the solver."""
        raise NotImplementedError

    @classmethod
    def _loss(cls,model_params,weights,X,y):
        """Calculate the loss between y and the predictions made with X using specified parameters and weights."""
        raise NotImplementedError

    @classmethod
    def _predict(cls,model_params,weights,X):
        """Predict X using specified parameters and weights."""
        raise NotImplementedError

    def predict(self,X):
        """Predict X."""
        raise NotImplementedError

    def fit(self,X,y):
        """Fit the model to X and y."""
        raise NotImplementedError

    def score(self,X,y):
        """Return the score of y and the predictions made with X."""
        raise NotImplementedError
```


