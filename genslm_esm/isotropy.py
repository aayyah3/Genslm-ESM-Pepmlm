import numpy as np
from numba import jit, prange
from numba.typed import List as numbaList


# This gets rid of numba type warnings
def _numba_list_decorator(func):
    def wrapper(list_data):
        return func(numbaList(list_data))

    return wrapper


@jit(nopython=True, fastmath=True)
def self_similarity(emb):
    # emb is (n x d) matrix
    n = emb.shape[0]

    # Normalize matrix rows
    for i in range(n):
        emb[i] /= np.linalg.norm(emb[i])

    # Compute the double sum as a matrix product
    mat = emb @ emb.T
    np.fill_diagonal(mat, 0)
    return np.sum(mat) / (n * (n - 1))


@_numba_list_decorator
@jit(nopython=True, parallel=True, fastmath=True)
def isotropy(dataset):
    num_examples = len(dataset)
    selfsim = 0.0
    for i in prange(num_examples):
        selfsim += self_similarity(dataset[i])
    return 1 - selfsim / num_examples


@_numba_list_decorator
@jit(nopython=True, parallel=True, fastmath=True)
def isotropy_return_all(dataset):
    num_examples = len(dataset)
    result = np.empty(num_examples)
    for i in prange(num_examples):
        result[i] = 1 - self_similarity(dataset[i])
    return result
