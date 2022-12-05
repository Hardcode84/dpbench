# Copyright 2022 Intel Corp.
#
# SPDX-License-Identifier: Apache-2.0

import numba_dpcomp as nb
import numpy as np


@nb.njit(parallel=True, fastmath=True)
def _pairwise_distance(X1, X2, D):
    x1 = np.sum(np.square(X1), axis=1)
    x2 = np.sum(np.square(X2), axis=1)
    np.dot(X1, X2.T, D)
    # D *= -2 TODO: inplace ops doesn't work as intended
    D[:] = D * -2
    x3 = x1.reshape(x1.size, 1)
    np.add(D, x3, D)
    np.add(D, x2, D)
    np.sqrt(D, D)


def pairwise_distance(X1, X2, D):
    _pairwise_distance(X1, X2, D)
