import numpy as np
from statistics import mean
from copy import copy


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
        if nl1[i] != nl2[i]:    # compare each list in the nested list
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

