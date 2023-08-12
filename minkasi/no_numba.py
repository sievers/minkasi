"""
Replacement for numba stuff when numba isn't installed.
"""
import numpy as np


def prange(*args, **kwargs):
    return np.arange(*args, **kwargs)


# https://stackoverflow.com/questions/69824610/dummy-function-decorator-with-arguments
def __noop(*args0, **kwargs0):
    def wrapper(func):
        def inner(*args, **kwargs):
            return func(*args, **kwargs)

        return inner

    return wrapper


autojit = __noop
generated_jit = __noop
guvectorize = __noop
jit = __noop
jitclass = __noop
njit = __noop
vectorize = __noop
