# Copyright 2022 Intel Corp.
#
# SPDX-License-Identifier: Apache 2.0

import logging
from typing import Any, Callable, Dict

import numpy as np
import pkg_resources

from .framework import Framework


class NumbaDpcompFramework(Framework):
    """A class for reading and processing framework information."""

    def __init__(self, fname: str, fconfig_path: str = None):
        """Reads framework information.
        :param fname: The framework name.
        """

        super().__init__(fname, fconfig_path)

    def imports(self) -> Dict[str, Any]:
        """Returns a dictionary any modules and methods needed for running
        a benchmark."""
        import dpctl

        return {"dpctl": dpctl}

    def copy_to_func(self) -> Callable:
        """Returns the copy-method that should be used
        for copying the benchmark arguments to device."""

        if self.sycl_device:
            import dpctl.tensor as dpt

            def _copy_to_func_impl(ref_array):

                if ref_array.flags["C_CONTIGUOUS"]:
                    order = "C"
                elif ref_array.flags["F_CONTIGUOUS"]:
                    order = "F"
                else:
                    order = "K"
                return dpt.asarray(
                    obj=ref_array,
                    dtype=ref_array.dtype,
                    device=self.sycl_device,
                    copy=None,
                    usm_type=None,
                    sycl_queue=None,
                    order=order,
                )

            return _copy_to_func_impl
        else:
            return np.copy

    def copy_from_func(self) -> Callable:
        """Returns the copy-method that should be used
        for copying results back to NumPy (host) from
        any array created by the framework possibly on
        a device memory domain."""

        if self.sycl_device:
            import dpctl.tensor as dpt

            return dpt.asnumpy
        else:
            return np.copy

    def execute(self, impl_fn: Callable, input_args: Dict):
        return impl_fn(**input_args)

    def version(self) -> str:
        """Returns the numba-dpex version."""

        try:
            return pkg_resources.get_distribution("numba_dpcomp").version
        except pkg_resources.DistributionNotFound:
            logging.exception("No version information exists for framework")
            return "unknown"
