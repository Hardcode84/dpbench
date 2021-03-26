# Copyright (C) 2017-2018 Intel Corporation
#
# SPDX-License-Identifier: MIT


import base_pair_wise
import numpy as np
import numba
import numba_dpcomp

@numba_dpcomp.jit(nopython=True,parallel=True,fastmath=True)
def pw_distance(X1,X2):
    M = X1.shape[0]
    N = X2.shape[0]
    O = X1.shape[1]
    D = np.empty((M, N))
    for i in numba.prange(M):
        for j in range(N):
            d = 0.0
            for k in range(O):
                tmp = X1[i, k] - X2[j, k]
                d += tmp * tmp
            D[i, j] = np.sqrt(d)
    return D

base_pair_wise.run("Numba par_for", pw_distance)
