from statistics import mean
import numpy as np
from math import exp


def distance1(x, y):
    return abs(x - y)


def distance2(x, y):
    return (x - y) ** 2


def sigmoid(x):
    return 1 / (1 + exp(-x))


def _sum_func(func_list, inputs, list_variable=True):
    # lambda function that evaluates a function with an argument
    if list_variable:
        evaluate = lambda f, args: f(args)  # in case args should be treated as a list variable
    else:
        evaluate = lambda f, args: f(*args)  # in case args should be treated as a list of arguments

    accu = 0

    for i, func in enumerate(func_list):
        accu += evaluate(func, inputs[i])

    return accu


def sum(X):
    y = 0
    for x in X:
        y += x

    return y


def density(x):
    K = len(x)
    return 1 / ((1 / K) * sum(x))


def ard(x, nghbrs):
    K = len(nghbrs)
    densities = [density] * K

    return density(x) / ((1 / K) * _sum_func(densities, nghbrs))


def var_explained(diagonal, indices):
    """
    Calculates the explained variance from the values in the diagonal of the SVD matrix.
    Note that the indices starts from one, to make the indexing more intuitive wrt. the phrasing in the exam.
    :param diagonal: values of the diagonal in the SVD matrix
    :param indices: position of the chosen values in the diagonal (index starting from 1)
    :return: Explained variance
    """
    denom = [d ** 2 for d in diagonal]
    num = [denom[i - 1] for i in indices]

    return sum(num) / sum(denom)

# TODO: Implement Jaccard Coefficient
# TODO: Implement Impurity Gain + Impurity measures
