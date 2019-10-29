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


