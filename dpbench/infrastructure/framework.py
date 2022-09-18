# Copyright 2021 ETH Zurich and the NPBench authors. All rights reserved.
# Copyright 2022 Intel Corp.
#
# SPDX-License-Identifier: BSD-3-Clause

import json
import pathlib
import warnings
from typing import Callable

import dpctl
import numpy as np
import pkg_resources

from dpbench.infrastructure import utilities


class Framework(object):
    """A class for reading and processing framework information."""

    def __init__(self, fname: str, fconfig_path: str = None):
        """Reads framework information.
        :param fname: The framework name.
        """

        self.fname = fname

        frmwrk_filename = "{f}.json".format(f=fname)
        frmwrk_path = None

        if fconfig_path:
            frmwrk_path = pathlib.Path(fconfig_path).joinpath(frmwrk_filename)
        else:
            parent_folder = pathlib.Path(__file__).parent.absolute()
            frmwrk_path = parent_folder.joinpath(
                "..", "configs", "framework_info", frmwrk_filename
            )

        try:
            with open(frmwrk_path) as json_file:
                self.info = json.load(json_file)["framework"]
        except Exception as e:
            print(
                "Framework JSON file {f} could not be opened.".format(
                    f=frmwrk_filename
                )
            )
            raise (e)

        try:
            self.sycl_device = self.info["sycl_device"]
            dpctl.SyclDevice(self.sycl_device)
        except KeyError:
            pass
        except dpctl.SyclDeviceCreationError as sdce:
            warnings.warn(
                "Could not create a Sycl device using filter {} string".format(
                    self.info["sycl_device"]
                )
            )
            print(sdce)
            raise sdce

    def device_filter_string(self) -> str:
        """Returns the sycl device's filter string if the framework has an
        associated sycl device."""

        try:
            return dpctl.SyclDevice(self.device).get_filter_string()
        except Exception:
            return ""

    def version(self) -> str:
        """Returns the framework version."""
        try:
            return pkg_resources.get_distribution(self.fname).version
        except pkg_resources.DistributionNotFound:
            return "unknown"

    def copy_to_func(self) -> Callable:
        """Returns the copy-method that should be used
        for copying the benchmark arguments from host
        to device."""
        return np.copy

    def copy_from_func(self) -> Callable:
        """Returns the copy-method that should be used
        for copying the benchmark arguments from device
        to host."""
        return np.copy

    def validator(self) -> Callable:
        """Returns a function that compares two lists of arrays and validates if
        the arrays in each list have data that are either the same or close
        enough.

        The function signature is:
            validator(ref_out, fw_out) -> bool

        where,
            ref_out is a list of arrays with the reference results of a specific
            benchmark
            fw_out is a list of arrays array with the results generated by the
            framework's implementation of the benchmark
        """

        def _validator(ref, test):
            return utilities.validate(ref, test, framework=self.fname)

        return _validator

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        return self.fname == other.fname

    def __hash__(self):
        return hash((self.fname))
