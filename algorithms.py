import numpy as np
from statistics import mean
from copy import copy
from functions import *


def _delegate_points(mu, data):
    mu = np.array(mu)
    groups = [[] for _ in range(3)]  # define a list of lists with ability to append

    for d in data:
        dist = np.absolute(mu - d).tolist()  # calculate the distance to each mean
        c_index = dist.index(min(dist))  # determine the index of the cluster to the smallest distance
        groups[c_index].append(d)  # assign data point to a cluster

    return groups


def _calculate_means(clusters):
    means = []
    for c in clusters:
        means.append(mean(c))

    return means


def _nested_equals(nl1, nl2):
    """
    Compares two nested lists index-wise
    :param nl1: nested list
    :param nl2: nested list
    :return: boolean
    """
    try:
        iter = len(nl1)
    except TypeError:
        return False

    if (nl1 is None) or (nl2 is None):
        return False

    for i in range(iter):
        if nl1[i] != nl2[i]:  # compare each list in the nested list
            return False

    return True


def k_means(mu, data):
    """
    Given some data and initial means, this method calculates the kmeans clusters and means, performing k-means algo.
    :param mu: list of initial means
    :param data: list of values
    :return: list of clustered values and their correpsonding means
    """
    groups = []
    n_groups = None

    while not _nested_equals(groups, n_groups):
        groups = copy(n_groups)
        n_groups = _delegate_points(mu, data)
        mu = _calculate_means(n_groups)

    return groups, mu


def jaccard(Z, Q):
    N = len(Z)
    S_mat = [[0] * N for _ in range(N)]
    D_mat = [[0] * N for _ in range(N)]

    # compute D and S matrices
    for i in range(N):
        for j in range(N):
            d_z = delta(Z[i], Z[j])
            d_q = delta(Q[i], Q[j])
            S_mat[i][j] = d_z * d_q
            D_mat[i][j] = (1 - d_z) * (1 - d_q)

    # compute D and S values
    S = 0
    D = 0

    for i in range(N - 1):
        j = i + 1
        while j < N:
            S += S_mat[i][j]
            D += D_mat[i][j]
            j += 1

    return S / ((1 / 2) * N * (N - 1) - D)


Z = [1, 1, 2, 2, 2, 3, 3, 3, 3, 3]
Q = [1, 1, 3, 1, 1, 1, 1, 3, 3, 2]

print(jaccard(Z, Q))
