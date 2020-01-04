"""
creation Date : Nov.11/2019
Title         : Hw5_main.py
TeamName      : SyntaxError
TeamMembers   : Jeffery
              : Oon
              : Aboli
              : Emeka
              : Paola
              : Junjie
"""

# importing necessary packages
import numpy as np
from scipy import optimize as opt
import datetime
import matplotlib.pyplot as plt
import math


# use recursion to find factorial
def find_factorial(x):
    try:
        if x == 1:
            return 1
        elif x < 1 or not (isinstance(x, int)):
            return 'no float & negatives'
        else:
            return x * find_factorial(x - 1)
    except Exception:
        '''it is a very board clause, but still applicable'''
        return 'invalid input'


# Q1
def gradient_optimizer(f, x0, eps, t, num_iter):
    """
    the first optimizer call the scipy package

    :param f:        objective function that we are opt
    :type f:         function
    :param x0:       start value, this should be an numpy array
    :type x0:        np.array
    :param eps:      eps is a small number that you use to test that $x_n$ is close to 0
    :type eps:
    :param t:        t is a small scalar to control the size of the gradient when we take a step.
    :type t:
    :param num_iter: num_iter is the maximum number of iterations
    :type            int

    """

    for i in range(num_iter):
        grad = opt.approx_fprime(x0, f, 0.0001)
        if np.all(grad < eps):
            return x0
        else:
            x0 = x0 - t * grad


# Q1 Bonus;
'''
    code provided by Grp Member Emeka
    solved the parameter mismatch using *args
'''


def gradient_optimizer2(f, eps, t, max_iter, *function_vars):
    """
    gradient_optimizer2 will be imported/implemented in class `cross entropy`

    :param f:             objective function
    :type f:              func
    :param eps:           objective value; approaching value
    :type eps:            float
    :param t:             scalar
    :type t:              float
    :param max_iter:      # of iterations
    :type max_iter:       int
    :param function_vars: other @para input
    :type function_vars:  @para
    :return:              gradient
    :rtype:               ndarray
    """

    def approx_grad(f, eps, *function_vars):
        grad = []
        xpluseps = function_vars[0].copy()
        xminuseps = function_vars[0].copy()
        if len(function_vars) > 1:
            var = list(function_vars[1:])
        for i in range(len(function_vars[0])):
            xpluseps[i] = xpluseps[i] + eps
            xminuseps[i] = xminuseps[i] - eps

            if len(function_vars) > 1:
                der = (f(xpluseps, *var) - f(xminuseps, *var)) / (2 * eps)
            else:
                der = (f(xpluseps) - f(xminuseps)) / (2 * eps)
            grad.append(der)
            xpluseps[i] = xpluseps[i] - eps
            xminuseps[i] = xminuseps[i] + eps
        if len(function_vars) > 1:
            return np.array(grad)[:, np.newaxis]
        else:
            return np.array(grad)

    varz = list(function_vars)

    for i in range(max_iter):
        grad = approx_grad(f, eps, *varz)

        if np.all(abs(grad) < eps):
            return np.array(varz[0])
        else:
            varz[0] = np.array(varz[0]) - (t * grad)
    grad = approx_grad(f, eps, *varz)
    if np.all(abs(grad) < eps):
        return np.array(varz[0])
    else:
        return None

# test function:
# def f(x):
#     return x[0] ** 2 * x[1] ** 2
#
#
# gradient_optimizer(f, [2, 2], 0.001, 0.01, 10000)
