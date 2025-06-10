from ..pbdl_dataloader.dataset import Dataset as PBDLDataset
from ..pbdl_dataloader.utilities import get_sel_const_sim
import numpy as np


class VariableDtDataset(PBDLDataset):
    r'''
    Dataset for loading data with variable time steps.
    This class inherits from PBDLDataset, see pbdl.torch.dataset for more information on the other parameters.

    Args:
        maximum_dt: maximum time delta between two frames
        seed: seed for the random number generator. If none a random seed is used. Ensure that a fixed seed is used for test sets.
    '''
    def __init__(self, maximum_dt:int, seed:int=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.maximum_dt = maximum_dt
        self.rng = np.random.default_rng(seed)

        assert self.num_frames > self.time_steps * maximum_dt, \
                "The dataset {} with {} simulation time steps is too small for the given time steps {} and maximum dt stride {}.".format(
                self.dset_name, self.num_frames, self.time_steps, maximum_dt)
        assert maximum_dt > 1, "The maximum time delta must be greater than 1."

    def __getitem__(self, idx):
        """
        The data provided has the shape (channels, spatial dims...).

        Returns:
            numpy.ndarray: Input data (without constants)
            tuple: Constants
            numpy.ndarray: Target data
            tuple: Non-normalized constants (only if solver flag is set)
        """
        if idx >= len(self):
            raise IndexError

        # create input-target pairs with random time interval
        if self.sel_sims:
            sim_idx = self.sel_sims[idx // self.samples_per_sim]
        else:
            sim_idx = idx // self.samples_per_sim

        sim = self.dset["sims/sim" + str(sim_idx)]
        const = get_sel_const_sim(self.dset, sim_idx, self.sel_const)

        input_frame_idx = (
            self.trim_start + (idx % self.samples_per_sim) * self.step_size
        )

        # random temporal step stride
        time_step_stride = self.rng.integers(1, self.maximum_dt, endpoint=True)
        # reduce stride if the target frame index is out of bounds at the end of each simulations (slight bias towards smaller strides)
        while input_frame_idx + self.time_steps * time_step_stride >= self.num_frames - self.trim_end:
            time_step_stride -= 1
        if time_step_stride < 1:
            raise ValueError("This should not happen. Time step stride is too small for input frame {} and time steps {}.".format(input_frame_idx, self.time_steps))
        target_frame_idx = input_frame_idx + self.time_steps * time_step_stride

        if self.intermediate_time_steps:
            target = sim[input_frame_idx : target_frame_idx + 1 : time_step_stride]
            input = target[0]
            target = target[1:]
        else:
            input = sim[input_frame_idx]
            target = sim[target_frame_idx - time_step_stride + 1]

        const_nnorm = const

        # normalize
        if self.norm_strat_data:
            input = self.norm_strat_data.normalize(input)

            if self.intermediate_time_steps:
                target = np.array(
                    [self.norm_strat_data.normalize(frame) for frame in target]
                )
            else:
                target = self.norm_strat_data.normalize(target)

        if self.norm_strat_const:
            # const = self.norm_strat_const.normalize(const, const=True)
            # const=True gives an unexpected keyword argument error
            # TODO check if it is fine to not use that keyword
            const = self.norm_strat_const.normalize(const)

        return (
            input,
            target,
            tuple(const),
            tuple(const_nnorm),
            time_step_stride,
        )
    