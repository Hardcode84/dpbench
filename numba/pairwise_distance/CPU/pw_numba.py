# Copyright (C) 2017-2018 Intel Corporation
#
# SPDX-License-Identifier: MIT


import base_pair_wise
import numpy as np
import numba

@numba.jit(nopython=True,parallel=True,fastmath=True)
def pw_distance(X1,X2):
    #return np.sqrt((np.square(X1 - X2.reshape((X2.shape[0],1,X2.shape[1])))).sum(axis=2))
    x1 = np.sum(np.square(X1), axis=1) #X1=4*3 -> 4*1
    x2 = np.sum(np.square(X2), axis=1) #X2=4*3 -> 4*1
    x3 = x1.reshape((x1.size,1))
    D = -2 * np.dot(X1, X2.T)
    D = D + x3 #x1[:,None] Not supported by Numba
    D = D + x2
    return np.sqrt(D)

base_pair_wise.run("Numba FastMath", pw_distance)
