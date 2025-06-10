import numpy as np
import h5py
import sys
from abc import ABC, abstractmethod
from itertools import groupby

from .utilities import get_const_sim, get_meta_data
from .logging import info, success, warn, fail

NORM_DATA_ARR = [
    "norm_fields_sca_mean",
    "norm_fields_sca_std",
    "norm_fields_std",
    "norm_fields_sca_min",
    "norm_fields_sca_max",
    "norm_const_mean",
    "norm_const_std",
    "norm_const_min",
    "norm_const_max",
]


class NormStrategy(ABC):

    @abstractmethod
    def normalize(self, data, const=False):
        pass

    @abstractmethod
    def normalize_rev(self, data, const=False):
        pass

    # def normalize_const(self, const):
    #     """Constant normalization always uses mean and standard deviation across all variants."""
    #     return (
    #         const - self.const_mean
    #     ) / self.const_std  # TODO handling zero const_std?

    def check_norm_data(dset):
        """Checks whether the norm data is complete."""
        return all(key in dset for key in NORM_DATA_ARR)

    def calculate_norm_data(dset):
        """Calculate and cache norm data."""
        clear_cache(dset)

        meta = get_meta_data(dset)
        num_sca_fields = meta["num_sca_fields"]

        # calculate the starting indices for fields
        field_indices = [0]
        for _, f in groupby(meta["fields_scheme"]):
            field_indices.append(field_indices[-1] + len(list(f)))  # TODO

        # slim means that for vector (non-scalar) fields the std must first be broadcasted to the original size
        fields_std_slim = [0] * meta["num_fields"]

        fields_sca_std = np.full((num_sca_fields,) + (1,) * meta["num_spatial_dim"], 0)
        fields_sca_mean = np.full((num_sca_fields,) + (1,) * meta["num_spatial_dim"], 0)

        fields_sca_min = np.full(
            (num_sca_fields,) + (1,) * meta["num_spatial_dim"], float("inf")
        )
        fields_sca_max = np.full(
            (num_sca_fields,) + (1,) * meta["num_spatial_dim"], -float("inf")
        )

        const_stacked = []

        # sequential loading of sims, norm data will be combined in the end
        for s in dset["sims/"]:

            sim = dset["sims/" + s]

            axis = (0,) + tuple(range(2, 2 + meta["num_spatial_dim"]))
            fields_sca_std = np.add(
                fields_sca_std, np.std(sim, axis=axis, keepdims=True)[0]
            )
            fields_sca_mean = np.add(
                fields_sca_mean, np.mean(sim, axis=axis, keepdims=True)[0]
            )

            fields_sca_min = np.minimum(
                fields_sca_min, np.min(sim, axis=axis, keepdims=True)[0]
            )
            fields_sca_max = np.maximum(
                fields_sca_max, np.max(sim, axis=axis, keepdims=True)[0]
            )

            for f in range(meta["num_fields"]):
                field = sim[:, field_indices[f] : field_indices[f + 1], ...]

                # vector norm
                field_norm = np.linalg.norm(field, axis=1, keepdims=True)

                # frame dim + spatial dims
                axis = (0,) + tuple(range(2, 2 + meta["num_spatial_dim"]))

                # std over frame dim and spatial dims
                fields_std_slim[f] += np.std(field_norm, axis=axis, keepdims=True)[0]

            const_stacked.append(get_const_sim(dset, int(s[3:])))

        fields_sca_mean = np.array(fields_sca_mean) / meta["num_sims"]
        fields_sca_std = np.array(fields_sca_std) / meta["num_sims"]

        # TODO overall std is calculated by averaging the std of all sims, efficient but mathematically not correct
        fields_std = []
        for f in range(meta["num_fields"]):
            field_std_avg = fields_std_slim[f] / meta["num_sims"]
            field_len = field_indices[f + 1] - field_indices[f]
            fields_std.append(
                np.broadcast_to(  # broadcast to original field dims
                    field_std_avg,
                    (field_len,) + (1,) * meta["num_spatial_dim"],
                )
            )
        fields_std = np.concatenate(fields_std, axis=0)

        # caching norm data
        dset["norm_fields_sca_mean"] = fields_sca_mean
        dset["norm_fields_sca_std"] = fields_sca_std
        dset["norm_fields_std"] = fields_std
        dset["norm_fields_sca_min"] = fields_sca_min
        dset["norm_fields_sca_max"] = fields_sca_max
        dset["norm_const_mean"] = np.mean(const_stacked, axis=0, keepdims=False)
        dset["norm_const_std"] = np.std(const_stacked, axis=0, keepdims=False)
        dset["norm_const_min"] = np.min(const_stacked, axis=0, keepdims=False)
        dset["norm_const_max"] = np.max(const_stacked, axis=0, keepdims=False)

    def load_norm_data(self, dset, sel_const):
        """Makes normalization data available as attributes."""

        self.sel_const = sel_const
        self.meta = get_meta_data(dset)

        # load normalization data
        # [()] reads the entire array TODO
        self.fields_sca_mean = dset["norm_fields_sca_mean"][()]
        self.fields_sca_std = dset["norm_fields_sca_std"][()]
        self.fields_std = dset["norm_fields_std"][()]
        self.fields_sca_min = dset["norm_fields_sca_min"][()]
        self.fields_sca_max = dset["norm_fields_sca_max"][()]
        self.const_mean = dset["norm_const_mean"][()]
        self.const_std = dset["norm_const_std"][()]
        self.const_min = dset["norm_const_min"][()]
        self.const_max = dset["norm_const_max"][()]

        # do basic checks on shape
        if self.fields_std.shape[0] != self.meta["sim_shape"][1]:
            raise ValueError(
                "Inconsistent number of fields between normalization data and simulation data."
            )

        if self.const_mean.shape[0] != self.meta["num_const"]:
            raise ValueError(
                "Mean data of constants does not match shape of constants."
            )

        if self.const_std.shape[0] != self.meta["num_const"]:
            raise ValueError("Std data of constants does not match shape of constants.")

        # filter norm data for selected constants
        if sel_const:
            indices = [
                i
                for i, const in enumerate(self.meta["const"])
                if const in self.sel_const
            ]
            self.const_std = self.const_std[indices]
            self.const_mea = self.const_mean[indices]


class StdNorm(NormStrategy):
    """Normalizes fields/constants using only the standard deviation."""

    def __init__(self, dset, sel_const, const=False):

        self.load_norm_data(dset, sel_const)
        self.std = self.const_std if const else self.fields_std

        if (self.std < 1e-10).any():
            warn(
                "Standard deviation used for normalization contains near-zero entries."
            )

    def normalize(self, data):
        return data / self.std

    def normalize_rev(self, data):
        return data * self.std

        # if (const_std < 10e-10).any():
        #     const_norm = np.zeros_like(const)
        # else:
        #     const_norm = (const - const_mean) / const_std


class MeanStdNorm(NormStrategy):
    """Normalizes fields/constants using both mean and standard deviation. Ignores vector fields and treats them like scalar fields, thus does not use the field scheme."""

    def __init__(self, dset, sel_const, const=False):
        self.load_norm_data(dset, sel_const)
        self.mean = self.const_mean if const else self.fields_sca_mean
        self.std = self.const_std if const else self.fields_sca_std

        if (self.std < 1e-10).any():
            warn(
                "Standard deviation used for normalization contains near-zero entries."
            )

    def normalize(self, data):
        return (data - self.mean) / self.std

    def normalize_rev(self, data):
        return data * self.std + self.mean


class MinMaxNorm(NormStrategy):
    """Scales fields/constants to a min-max range."""

    def __init__(self, dset, sel_const, const=False, min_val=0, max_val=1):
        self.min_val = min_val
        self.max_val = max_val

        self.load_norm_data(dset, sel_const)
        self.min = self.const_min if const else self.fields_sca_min
        self.max = self.const_max if const else self.fields_sca_max

        if min_val == max_val:
            warn("Min and max specified for normalization must be different.")

        if (self.max - self.min < 1e-10).any():
            warn(
                "Largest and smallest value found in data are too close for min-max normalization."
            )

    def normalize(self, data):
        return (data - self.min) / (self.max - self.min) * (
            self.max_val - self.min_val
        ) + self.min_val

    def normalize_rev(self, data):
        return ((data - self.min_val) / (self.max_val - self.min_val)) * (
            self.max - self.min
        ) + self.min


def clear_cache(dset):

    for key in NORM_DATA_ARR:
        dset.pop(key, None)


STR_TO_NORM_STRAT = {
    "std": (StdNorm, {}),
    "mean-std": (MeanStdNorm, {}),
    "zero-to-one": (MinMaxNorm, {"min_val": 0, "max_val": 1}),
    "minus-one-to-one": (MinMaxNorm, {"min_val": -1, "max_val": 1}),
}


def get_norm_strat_from_str(str, dset, sel_const, const=False):
    if str not in STR_TO_NORM_STRAT.keys():
        suggestions = ", ".join(STR_TO_NORM_STRAT.keys())
        fail(
            f"Unknown normalization strategy `{str}`. Supported strategies are: {suggestions}."
        )
        sys.exit(0)

    norm_class, args = STR_TO_NORM_STRAT.get(str)
    return norm_class(dset, sel_const, const, **args)
