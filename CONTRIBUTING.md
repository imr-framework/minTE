# Contributing to minTE
:thumbsup: :tada: First, thanks for contributing! :thumbsup: :tada:

The following is a set of guidelines for contributing to the minTE repository. These are mostly guidelines, not rules. Use your best judgment, and feel free to propose changes to this document in a pull request.

## Table of contents
1. [Contributing](#contributing-to-the-software)
2. [Reporting Issues](#reporting-issues)
3. [Seeking Support](#seeking-support)

## Contributing to the Software

If you would like to contribute to the repository, please Fork and make a pull request with adequate documentation of functionality. In addition, please take a look at the Code of Conduct and documentation guide below: 

**Code of Conduct**

This project and everyone participating in it is governed by the [minTE Code of Conduct](https://github.com/imr-framework/minTE/blob/main/CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report unacceptable behavior to imr.framework2018@github.com.

**PEP Style Guide for Python Code**

Read through the [Style Guide for Python Code](https://www.python.org/dev/peps/pep-0008/) and follow the coding conventions. If you notice any of the code not adhering to PEP8, submit a pull request or open an issue.

**Source code header**

Insert and fill out the following header comments at the top of every script in your source code:
```
Institution : (your university/research facility)
Version : 1.0.0 
```
Note that "Version" refers to the release you developed your code on. You can add more fields if needed. 

**Documenting source code**

Please add a top-level description of code functionality in each script. In addition, document every class and method in your source code using the [Numpy docstring guide](https://numpydoc.readthedocs.io/en/latest/format.html). An example is shown below.

```python
def addition(a, b):
  """
  Addition of two numbers.
  
  Parameters
  ----------
  a : int, float
    Operand 1
  b : int, float
    Operand 2
    
  Returns
  -------
  Arthimetic sum of operand 1 and operand 2.
  """
  return a + b
```
## Reporting Issues
Please report issues in Github Issues, following the given format for either "Bug Report" or "Feature Request". 

## Seeking Support
For support in setting up and using the minTE repository, please feel free to either use Github Issues or email Gehua Tong at gt2407@columbia.edu. 
