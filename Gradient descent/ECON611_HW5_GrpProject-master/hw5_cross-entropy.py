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

import math
import numpy as np
from hw5 import gradient_optimizer2, gradient_optimizer
import matplotlib.pyplot as plt


# Simulate test Sample Data for function and class test
# simulate data set #1
def sample_data():
    numpos = 100
    numneg = 100

    mupos = [1.0, 1.0]
    covpos = np.array([[1.0, 0.0], [0.0, 1.0]])
    muneg = [-1.0, -1.0]
    covneg = np.array([[1.0, 0.0], [0.0, 1.0]])

    Xpos = np.ones((numpos, 3))

    for i in range(numpos):
        # @para@: mean & variance; size |--> iterations
        Xpos[i, 0:2] = np.random.multivariate_normal(mupos, covpos)
    Ypos = np.ones((numpos, 1))

    Xneg = np.ones((numneg, 3))

    for i in range(numneg):
        Xneg[i, 0:2] = np.random.multivariate_normal(muneg, covneg)
    Yneg = np.zeros((numneg, 1))

    X = np.concatenate((Xpos, Xneg))
    Y = np.concatenate((Ypos, Yneg))

    return X, Y, Xpos, Xneg


sample_x, sample_y, sample_xpos, sample_ypos = sample_data()

'''
    define `cross entropy class` here
    As demonstrated by TA, because `Sigmoid`, `L` & `fit_approx functions` are taking values;
    hence we separate them from `C.R.` class
'''
class cross_entropy(object):
    beta = 0   # class_level var; called and updated in `fit_approx()`
    beta2 = 0  # class level var; called and updated in `fit()`

    def __init__(self, x, y):
        """
            class initiation;
            self.beta created base on given ndarray `x`
        """

        self.x = x
        self.y = y
        self.beta = np.zeros((np.shape(x)[1], 1))

    def fit(self, eps=0.001, t=.01, iteration=1000):
        """
            :return: beta2
            :rtype:  ndarray
        """

        # let's looping
        for i in range(iteration):
            grad = np.dot(self.x.T, self.y - self.in_cls_sigmoid())
            if np.all(np.abs(grad) < eps):
                # check cond --> objective MIN found
                return self.beta
            else:
                self.beta = self.beta + t * grad
        return self.beta

    # in our original code, we separate all the code outside the class
    # in case if you really want all the functions inside the class
    def in_cls_sigmoid(self):
        return 1 / (1 + np.exp(-1 * np.dot(self.x, self.beta)))

    def in_cls_L_func(self):
        return -(np.sum(self.y * np.log(self.in_cls_sigmoid()) + (1 - self.y) * np.log(1 - self.in_cls_sigmoid())))


def sigmoid(beta, x):
    return 1 / (1 + np.exp(-1 * np.dot(x, beta)))


def L(beta, x, y):
    return -(np.sum(y * np.log(sigmoid(beta, x)) + (1 - y) * np.log(1 - sigmoid(beta, x))))


def fit_approx(L, beta, x, y, eps=0.0001, t=0.01, max_iter=10000):
    return gradient_optimizer2(L, eps, t, max_iter, *[beta, x, y])


# test function:

test_obj = cross_entropy(sample_x, sample_y)

res = test_obj.fit()
print(res)

res2 = fit_approx(L, test_obj.beta, sample_x, sample_y)
print(res2)
