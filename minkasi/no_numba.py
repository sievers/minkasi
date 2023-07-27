"""
Replacement for numba stuff when numba isn't installed.
"""
import numpy as np
import functools


def prange(*args, **kwargs):
    return np.arange(*args, **kwargs)


# Decorator sub based on https://github.com/ptooley/numbasub
def optional_arg_decorator(fn):
    @functools.wraps(fn)
    def wrapped_decorator(*args, **kwargs):
        # If no arguments were passed...
        if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
            return fn(args[0])
        else:

            def real_decorator(decoratee):
                return fn(decoratee, *args, **kwargs)

            return real_decorator

    return wrapped_decorator


@optional_arg_decorator
def __noop(func, *args, **kwargs):
    return func


autojit = __noop
generated_jit = __noop
guvectorize = __noop
jit = __noop
jitclass = __noop
njit = __noop
vectorize = __noop
