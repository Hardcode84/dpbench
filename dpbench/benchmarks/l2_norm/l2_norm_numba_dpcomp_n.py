# Copyright 2022 Intel Corp.
#
# SPDX-License-Identifier: Apache-2.0

import numba_dpcomp as nb
import numpy as np


@nb.njit(parallel=True, fastmath=True)
def _l2_norm(a, d):
    sq = np.square(a)
    sum = sq.sum(axis=1)
    d[:] = np.sqrt(sum)


def l2_norm(a, d):
    _l2_norm(a, d)
