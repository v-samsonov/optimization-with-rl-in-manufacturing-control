import numpy as np


def permute_rows(x, random_generator):
    '''
    x is a np array
    '''
    ix_i = np.tile(np.arange(x.shape[0]), (x.shape[1], 1)).T
    ix_j = random_generator.random(x.shape).argsort(axis=1)
    return x[ix_i, ix_j]


def uni_instance_gen(n_j, n_m, low, high, random_generator):
    times = random_generator.integers(low=low, high=high, size=(n_j, n_m))
    machines = np.expand_dims(np.arange(1, n_m+1), axis=0).repeat(repeats=n_j, axis=0)
    machines = permute_rows(machines, random_generator)
    return times, machines


def override(fn):
    """
    override decorator
    """
    return fn


