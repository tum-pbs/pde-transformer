from abc import ABC
from typing import Dict, Tuple, Optional, Union

from torch import nn
import torch
from torch.nn.functional import avg_pool2d

from ...utils import instantiate_from_config


class AbstractTask(ABC):

    def __init__(self):
        pass

    def prepare_data(self, data: Dict, prob: float) -> dict:
        pass

def generate_channel_mask(channel: torch.Tensor, prob: float, patch_size: int) -> list[torch.Tensor]: # noqa

    batch_size = channel.shape[0]

    num_channel_tokens = (channel.shape[2] // patch_size) * (channel.shape[3] // patch_size)
    num_channel_tokens = int(num_channel_tokens)

    mask = torch.ones(batch_size, num_channel_tokens,
                      device=channel.device)

    mask = torch.bernoulli(prob * mask)

    return mask

def generate_empty_channel_mask(channels: torch.Tensor, patch_size: int) -> list[torch.Tensor]: # noqa

    return generate_channel_mask(channels, prob=0.0, patch_size=patch_size)


def generate_full_channel_mask(channels: list[torch.Tensor], patch_size: int) -> list[torch.Tensor]: # noqa

    return generate_channel_mask(channels, prob=1.0, patch_size=patch_size)

def generate_token_mask(tokens: torch.Tensor, prob: float) -> torch.Tensor:

        batch_size = tokens.shape[0]
        num_tokens = tokens.shape[1]

        mask = torch.ones(batch_size, num_tokens, device=tokens.device)

        torch.bernoulli(prob * mask)

        return mask

def generate_empty_token_mask(tokens: torch.Tensor) -> torch.Tensor:

        return generate_token_mask(tokens, prob=0.0)

def generate_full_token_mask(tokens: torch.Tensor) -> torch.Tensor:

        return generate_token_mask(tokens, prob=1.0)


class GenerativeModeling(AbstractTask):

    def __init__(self, patch_size: int, num_timesteps: int, task_idx: int):
        super(GenerativeModeling, self).__init__()
        self.patch_size = patch_size
        self.num_timesteps = num_timesteps
        self.task_idx = task_idx

    def prepare_data(self, data: dict, prob: float) -> dict:

        channels = data['channels']
        data['channels'] = []

        for channel in channels:

            channel_id = channel['channel_id']
            channel = channel['channel']
            num_timesteps = channel.shape[1]
            time_mask = generate_empty_channel_mask(channels=channel, patch_size=self.patch_size)
            time_masks = [time_mask] * num_timesteps

            task_emb = (torch.zeros(channel.shape[0], device=channel.device)
                        + self.task_idx)
            task_emb = task_emb.long()

            simulation_time = torch.einsum('t, b -> b t',
                                           torch.arange(0, num_timesteps, device=channel.device),
                                           data['time_step_stride'])

            dict_ = {'data': list(channel.transpose(0,1)),
                     'mask': time_masks,
                     'channel_id': channel_id,
                     'pde': data['pde'],
                     'constants': data['constants'],
                     'constants_class': data['constants_class'],
                     'time_step_stride': data['time_step_stride'],
                     'simulation_time': simulation_time,
                     'task_emb': task_emb
                    }

            data['channels'].append(dict_)

        return data

class Interpolation(AbstractTask):

    def __init__(self, patch_size: int, num_timesteps: int, task_idx: int):
        super(Interpolation, self).__init__()
        self.patch_size = patch_size
        self.num_timesteps = num_timesteps
        self.task_idx = task_idx

    def prepare_data(self, data: dict, prob: float) -> dict:
        channels = data['channels']
        data['channels'] = []

        for channel in channels:
            channel_id = channel['channel_id']
            channel = channel['channel']
            num_timesteps = channel.shape[1]

            time_inactive = generate_empty_channel_mask(channels=channel, patch_size=self.patch_size)
            time_active = generate_full_channel_mask(channels=channel, patch_size=self.patch_size)

            time_masks = [time_active]
            time_masks += [time_inactive] * (self.num_timesteps - 2)
            time_masks += [time_active]

            task_emb = (torch.zeros(channel.shape[0], device=channel.device)
                        + self.task_idx)
            task_emb = task_emb.long()

            simulation_time = torch.einsum('t, b -> b t',
                                           torch.arange(0, num_timesteps, device=channel.device),
                                           data['time_step_stride'])

            dict_ = {'data': list(channel.transpose(0, 1)),
                     'mask': time_masks,
                     'channel_id': channel_id,
                     'pde': data['pde'],
                     'constants': data['constants'],
                     'constants_class': data['constants_class'],
                     'time_step_stride': data['time_step_stride'],
                     'simulation_time': simulation_time,
                     'task_emb': task_emb
                     }

            data['channels'].append(dict_)

        return data

class ForwardPrediction(AbstractTask):

    def __init__(self, patch_size: int, num_timesteps: int, m: int, n: int, task_idx: int):
        """
        m to n forward prediction task
        Args:
            :param patch_size:
        """
        super(ForwardPrediction, self).__init__()
        self.patch_size = patch_size
        self.m = m
        self.n = n
        self.num_timesteps = num_timesteps
        self.task_idx = task_idx

        assert self.m + self.n == self.num_timesteps, "m + n must equal num_timesteps"

    def prepare_data(self, data: dict, prob: float) -> dict:
        channels = data['channels']
        data['channels'] = []

        for channel in channels:
            channel_id = channel['channel_id']
            channel = channel['channel']
            num_timesteps = channel.shape[1]

            time_inactive = generate_empty_channel_mask(channels=channel, patch_size=self.patch_size)
            time_active = generate_full_channel_mask(channels=channel, patch_size=self.patch_size)

            time_masks = [time_active] * (self.num_timesteps - self.n)
            time_masks += [time_inactive] * self.n

            task_emb = (torch.zeros(channel.shape[0], device=channel.device)
                        + self.task_idx)
            task_emb = task_emb.long()

            simulation_time = torch.einsum('t, b -> b t',
                                           torch.arange(0, num_timesteps, device=channel.device),
                                           data['time_step_stride'])

            dict_ = {'data': list(channel.transpose(0, 1)),
                     'mask': time_masks,
                     'channel_id': channel_id,
                     'pde': data['pde'],
                     'constants': data['constants'],
                     'constants_class': data['constants_class'],
                     'time_step_stride': data['time_step_stride'],
                     'simulation_time': simulation_time,
                     'task_emb': task_emb
                     }

            data['channels'].append(dict_)

        return data


class BackwardPrediction(AbstractTask):

    def __init__(self, patch_size: int, num_timesteps: int, m: int, n: int, task_idx: int):
        """
        m to n forward prediction task
        Args:
            :param patch_size:
        """
        super(BackwardPrediction, self).__init__()
        self.patch_size = patch_size
        self.m = m
        self.n = n
        self.num_timesteps = num_timesteps
        self.task_idx = task_idx

        assert self.m + self.n == self.num_timesteps, "m + n must equal num_timesteps"

    def prepare_data(self, data: dict, prob: float) -> dict:
        channels = data['channels']
        data['channels'] = []

        for channel in channels:
            channel_id = channel['channel_id']
            channel = channel['channel']
            num_timesteps = channel.shape[1]

            time_inactive = generate_empty_channel_mask(channels=channel, patch_size=self.patch_size)
            time_active = generate_full_channel_mask(channels=channel, patch_size=self.patch_size)

            time_masks = [time_inactive] * (self.num_timesteps - self.n)
            time_masks += [time_active] * self.n

            task_emb = (torch.zeros(channel.shape[0], device=channel.device)
                        + self.task_idx)
            task_emb = task_emb.long()

            simulation_time = torch.einsum('t, b -> b t',
                                           torch.arange(0, num_timesteps, device=channel.device),
                                           data['time_step_stride'])

            dict_ = {'data': list(channel.transpose(0, 1)),
                     'mask': time_masks,
                     'channel_id': channel_id,
                     'pde': data['pde'],
                     'constants': data['constants'],
                     'constants_class': data['constants_class'],
                     'time_step_stride': data['time_step_stride'],
                     'simulation_time': simulation_time,
                     'task_emb': task_emb
                     }

            data['channels'].append(dict_)

        return data

TASK_REGISTER = {
    'GenerativeModeling6':    1,
    'ForwardPrediction1to5':  2,
    'ForwardPrediction5to1':  3,
    'ForwardPrediction3to3':  4,
    'BackwardPrediction1to5': 5,
    'BackwardPrediction5to1': 6,
    'BackwardPrediction3to3': 7,
    'Interpolation6':         8
}

def ForwardPrediction1to5(patch_size: int):
    return ForwardPrediction(patch_size=patch_size, num_timesteps=6, m=1, n=5,
                             task_idx=TASK_REGISTER['ForwardPrediction1to5'])

def ForwardPrediction5to1(patch_size: int):
    return ForwardPrediction(patch_size=patch_size, num_timesteps=6, m=5, n=1,
                             task_idx=TASK_REGISTER['ForwardPrediction5to1'])

def ForwardPrediction3to3(patch_size: int):
    return ForwardPrediction(patch_size=patch_size, num_timesteps=6, m=3, n=3,
                             task_idx=TASK_REGISTER['ForwardPrediction3to3'])

def BackwardPrediction1to5(patch_size: int):
    return BackwardPrediction(patch_size=patch_size, num_timesteps=6, m=1, n=5,
                              task_idx=TASK_REGISTER['BackwardPrediction1to5'])

def BackwardPrediction5to1(patch_size: int):
    return BackwardPrediction(patch_size=patch_size, num_timesteps=6, m=5, n=1,
                              task_idx=TASK_REGISTER['BackwardPrediction5to1'])

def BackwardPrediction3to3(patch_size: int):
    return BackwardPrediction(patch_size=patch_size, num_timesteps=6, m=3, n=3,
                              task_idx=TASK_REGISTER['BackwardPrediction3to3'])

def Interpolation6(patch_size: int):
    return Interpolation(patch_size=patch_size, num_timesteps=6,
                         task_idx=TASK_REGISTER['Interpolation6'])

def GenerativeModeling6(patch_size: int):
    return GenerativeModeling(patch_size=patch_size, num_timesteps=6,
                              task_idx=TASK_REGISTER['GenerativeModeling6'])

def get_task_from_str(task: str, patch_size: int) -> AbstractTask:
    init_dict = {'target': f'src.core.masked_modeling_v2.tasks.{task}',
                 'params': {'patch_size': patch_size}}
    return instantiate_from_config(init_dict)



class TaskMasking(nn.Module):

    def __init__(self, tasks: list[Union[str, AbstractTask]], weighting, patch_size: int, downsampling_factor: int = 1):
        super(TaskMasking, self).__init__()
        self.patch_size = patch_size

        self.downsampling_factor = downsampling_factor
        self.initialize_tasks(tasks, weighting)

    def get_task(self, task: str) -> AbstractTask:
        return get_task_from_str(task, patch_size=self.patch_size)

    def get_random_task(self, batch_idx: int):

        # sample from probabilities with seed batch_idx
        command_idx = torch.multinomial(self.weighting, 1,
                                        generator=torch.Generator().manual_seed(batch_idx))

        return command_idx[0].item()

    def initialize_tasks(self, tasks: list[Union[str, AbstractTask]], weighting: str):

        self.tasks: list[AbstractTask] = [] # noqa
        self.task_names: list[str] = [] # noqa

        for task in tasks:
            try:
                if isinstance(task, AbstractTask):
                    self.tasks.append(task)
                    self.task_names.append(task.__class__.__name__)
                else:
                    self.tasks.append(get_task_from_str(task, patch_size=self.patch_size))
                    self.task_names.append(task)
            except Exception as e:
                print(f"Could not initialize task {task}: {e}")

        if len(self.tasks) == 0:
            raise ValueError("No tasks initialized")

        if isinstance(weighting, str):

            if weighting == 'uniform':
                self.weighting = torch.Tensor([1/len(self.tasks) for _ in self.tasks]) # noqa
            else:
                raise ValueError(f"Weighting {weighting} not implemented")

        else:

            self.weighting = weighting # noqa

    def get_masks(self, data: dict, task_idx: int, prob: float) -> dict:

        return self.tasks[task_idx].prepare_data(data, prob)

    def get_inputs_impl(self, batch: dict, task_idx: int, prob: float, window: int):

        data = self.get_data(batch)
        masks = self.get_masks(data, task_idx, prob)

        data.update(masks)

        return data

    def get_inputs(self, batch: dict, batch_idx: int, task: Optional[int] = None):

        if task is None:
            task = self.get_random_task(batch_idx)

        prob = torch.rand(1, device=batch['data'].device).item()

        data = self.get_inputs_impl(batch, task, prob, batch_idx)
        data['task_name'] = self.task_names[task]
        data['task_idx'] = task # noqa

        return data

    def get_channels(self, batch: dict, idx: int) -> Tuple[torch.Tensor, torch.Tensor]: # noqa
        """
        Get the data for a specific channel at a specific time step.

        Args:
            batch: batch data
            idx: index of the time step

        Returns:
            tuple: channels, simulation times, channel ids
        """

        channel_data: torch.Tensor = batch['data'][:, :, idx]

        if self.downsampling_factor > 1:
            channel_data = avg_pool2d(channel_data, self.downsampling_factor)

        if len(batch['physical_metadata']['Fields'].shape) == 1:
            channel_id = batch['physical_metadata']['Fields'].long()
        else:
            channel_id = batch['physical_metadata']['Fields'][:, idx].long()

        return channel_data, channel_id


    def get_data(self, batch: dict):

        # timesteps = batch['data'].shape[1]

        num_channels = batch['data'].shape[2]

        data_temp = []

        for idx in range(num_channels):

            channel, channel_id = self.get_channels(batch, idx)
            data_temp.append({'channel': channel, 'channel_id': channel_id})

        pde = batch['physical_metadata']['PDE']
        constants = batch['constants']
        constants_class = batch['physical_metadata']['Constants']
        time_step_stride = batch['time_step_stride'].flatten()

        result = {} # noqa

        result['channels'] = data_temp
        result['pde'] = pde
        result['constants'] = constants
        result['constants_class'] = constants_class
        result['time_step_stride'] = time_step_stride

        return result




