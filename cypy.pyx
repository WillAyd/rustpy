cimport cython
from libc.limits cimport LLONG_MIN
import numpy as np
from numpy cimport ndarray, int64_t
import time


# cythonize -3 cypy.pyx
# gcc -g3 -Wall -Werror -fPIC -shared -I/home/willayd/mambaforge/envs/scratchpad/include/python3.11 -DNPY_NO_DEPRECATED_API=0 -I/home/willayd/mambaforge/envs/scratchpad/lib/python3.11/site-packages/numpy/core/include cypy.c -o cypy.so -L/home/willayd/mambaforge/envs/scratchpad/lib/ -lpython3.11
@cython.boundscheck(False)
@cython.wraparound(False)
cdef ndarray[int64_t, ndim=1] _find_max(ndarray[int64_t, ndim=2] values):
    cdef:
        ndarray[int64_t, ndim=1] out
        int64_t val, colnum, rownum
        Py_ssize_t N, K

    N, K = (<object>values).shape
    out = np.zeros(K, dtype=np.int64)    
    with nogil:
        for colnum in range(K):
            val = LLONG_MIN  # imperfect assumption, but no INT64_T_MIN from numpy
            for rownum in range(N):
                if val <= values[rownum, colnum]:
                    val = values[rownum, colnum]

            out[colnum] = val

    return out


def find_max(ndarray[int64_t, ndim=2] values):
    cdef ndarray[int64_t, ndim=1] result
    start = time.time_ns()
    result = _find_max(values)
    end = time.time_ns()
    duration = (end - start) / 1_000_000
    print(f"cypy took {duration} milliseconds")
    return result
    
