from statistics import mean


def distance1(x, y):
    return abs(x - y)


def distance2(x, y):
    return (x - y) ** 2


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


o = [1.7, 2.2]
o1 = [0.9, 2.1]
o2 = [1.7, 1.8]

neighbours = [o1, o2]

print(ard(o, neighbours))
