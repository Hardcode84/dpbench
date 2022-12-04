# Copyright 2022 Intel Corp.
#
# SPDX-License-Identifier: Apache-2.0

import math

import numpy as np
import numba_dpcomp.mlir.kernel_impl as nb

@nb.kernel
def l2_norm_kernel(a, d):
    i = nb.get_global_id(0)
    O = a.shape[1]
    d[i] = 0.0
    for k in range(O):
        d[i] += a[i, k] * a[i, k]
    d[i] = np.sqrt(d[i])


def l2_norm(a, d):
    l2_norm_kernel[a.shape[0], nb.DEFAULT_LOCAL_SIZE](a, d)
