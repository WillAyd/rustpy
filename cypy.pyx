cimport cython
from libc.limits cimport LLONG_MIN
import numpy as np
from numpy cimport ndarray, int64_t
import time

@cython.boundscheck(False)
@cython.wraparound(False)
cdef ndarray[int64_t, ndim=1] _find_max(ndarray[int64_t, ndim=2] values):
    cdef:
        ndarray[int64_t, ndim=1] out
        int64_t val, colnum, rownum, new_val
        Py_ssize_t N, K

    N, K = (<object>values).shape
    out = np.zeros(K, dtype=np.int64)    
    with nogil:
        for colnum in range(K):
            val = LLONG_MIN  # imperfect assumption, but no INT64_T_MIN from numpy
            for rownum in range(N):
                new_val = values[rownum, colnum]
                if val < new_val:
                    val = new_val

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
    
