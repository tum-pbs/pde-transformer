import numpy as np
import h5py
import sys
from abc import ABC, abstractmethod
from itertools import groupby

from .logging import info, success, warn, fail 
from typing import Type

class NormStrategy(ABC):
    @abstractmethod
    def normalize(self, data):
        pass

    @abstractmethod
    def normalize_rev(self, data):
        pass

class StdNorm(NormStrategy):
    """Normalizes fields/constants using only the standard deviation."""

    def __init__(self, norm_attrs):
        self.std = norm_attrs["std"]
        if np.any(self.std < 1e-10):
            warn("Standard deviation used for normalization contains near-zero entries.")

    def normalize(self, data):
        return data / self.std

    def normalize_rev(self, data):
        return data * self.std


class MeanStdNorm(NormStrategy):
    """Normalizes fields/constants using both mean and standard deviation. Ignores vector fields and treats them like scalar fields, thus does not use the field scheme."""

    def __init__(self, norm_attrs):
        self.mean = norm_attrs["mean"]
        self.std = norm_attrs["std"]
        if np.any(self.std < 1e-10):
            warn("Standard deviation used for normalization contains near-zero entries.")

    def normalize(self, data):
        return (data - self.mean) / self.std

    def normalize_rev(self, data):
        return data * self.std + self.mean


class MinMaxNorm(NormStrategy):
    """Scales fields/constants to a min-max range."""

    def __init__(self, norm_attrs, **kwargs):
        self.min = norm_attrs["min"]
        self.max = norm_attrs["max"]
        self.min_val = kwargs.get("min_val", 0)
        self.max_val = kwargs.get("max_val", 1)
        if self.min_val >= self.max_val:
            warn("Min is greater than or equal to max.")

        if np.any((self.max - self.min) < 1e-10):
            warn("Largest and smallest value found in data are too close for min-max normalization.")

    def normalize(self, data):
        return (data - self.min) / (self.max - self.min) * (self.max_val - self.min_val) + self.min_val

    def normalize_rev(self, data):
        return (data - self.min_val) / (self.max_val - self.min_val) * (self.max - self.min) + self.min