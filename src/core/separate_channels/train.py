from pathlib import Path
from typing import Tuple, Dict, Any, Optional

import torch
import torch.nn as nn
from lightning.pytorch.utilities.types import STEP_OUTPUT
from sympy.stats.rv import probability
from tensorboard.summary.v1 import image
from timm.scheduler import TanhLRScheduler
from torch.nn.functional import unfold
from tqdm import tqdm

from core.masked_modeling_v1.tasks import AbstractTask
from core.masked_modeling_v2.tasks import TaskMasking
from sampler.scheduler import OdeEulerScheduler
from src.utils import instantiate_from_config

import numpy as np

def accumulate_embeddings_and_values(*args):

    embedding_list = []
    value_list = []

    for dict_ in args:
        for key, value in dict_.items():

            embedding_list.append(value['embedding'])
            value_list.append(value['value'])

    return embedding_list, value_list

def get_weighting_function(weighting: str):
    """
    Get weighting function for the loss
    :param weighting: type
    :return:
    """

    def identity(t, w):
        return w

    def constant(t, w):
        return torch.ones_like(t)

    if weighting == "id":
        return identity
    elif weighting == "constant":
        return constant
    else:
        raise ValueError(f"Weighting function {weighting} not implemented")

def sample_time(device, num: int = 1, t0: float = 0.0,
                t1: float = 1.0, eps=1e-5) -> torch.Tensor:
    """
    Sample time points between t0 and t1 based on batch size for variance reduction
    :param device: device
    :param num: batch size
    :param t0: minimum time
    :param t1: maximum time
    :param eps: make sure to not sample exactly at t0 or t1
    :return:
    """

    t0 += eps
    t1 -= eps
    times = torch.linspace(t0, t1, num+1, device=device)[:num]
    uniform = torch.rand((num,), device=device)
    uniform = uniform * ((t1 - t0) / num)
    time_samples = times + uniform

    return time_samples

def to_patch_representation(images: torch.Tensor, patch_size: int = 16):
    if len(images.shape) == 3:
        images = images.unsqueeze(1)
    batch_size, channels, height, width = images.shape
    kh, kw = patch_size, patch_size  # kernel size
    dh, dw = patch_size, patch_size  # stride
    patches = images.unfold(2, kh, dh).unfold(3, kw, dw)
    patches = patches.contiguous().view(batch_size, -1, channels * kh * kw)

    return patches


def from_patch_representation(patches: torch.Tensor, height: int, width: int, patch_size: int = 16):
    batch_size = patches.shape[0]
    unfold_shape = (batch_size, -1, height // patch_size, width // patch_size, patch_size, patch_size)
    patches_orig = patches.view(unfold_shape)
    output_h = unfold_shape[2] * unfold_shape[4]
    output_w = unfold_shape[3] * unfold_shape[5]
    patches_orig = patches_orig.permute(0, 1, 2, 4, 3, 5).contiguous()
    patches_orig = patches_orig.view(batch_size, -1, output_h, output_w)

    return patches_orig


def get_masked_model_inputs(images: list[torch.tensor] , tokens: torch.tensor,
                            mask_images: list[torch.tensor], mask_tokens: torch.tensor,
                            t: torch.tensor, sigma_min: float, patch_size: int) -> Tuple:

    tokens_id = tokens[:, 0]
    tokens_value = tokens[:, 1]

    heights = [img.shape[2] for img in images]
    widths = [img.shape[3] for img in images]

    epsilon_images = [torch.randn(size=img.shape, device=img.device, dtype=img.dtype)
                        for img in images]
    epsilon_tokens = torch.randn(size=tokens_value.shape, device=tokens_value.device, dtype=tokens_value.dtype)

    psi_t = (1 - (1 - sigma_min) * t)
    psi_t_images = [torch.einsum("a,abcd->abcd", psi_t, epsilon_img) for epsilon_img
                    in epsilon_images]

    psi_t_images = [psi_t_img + torch.einsum("a, abcd->abcd", t, img)
                    for psi_t_img, img in zip(psi_t_images, images)]

    psi_t_images = [to_patch_representation(psi_t_img, patch_size=patch_size)
                    for psi_t_img in psi_t_images]

    images = [to_patch_representation(img, patch_size=patch_size) for img in images]

    psi_t_images = [(1 - mask_img) * psi_t_img + mask_img * img for psi_t_img, mask_img, img in
                    zip(psi_t_images, mask_images, images)]

    psi_t_images = [from_patch_representation(psi_t_img, height=height, width=width, patch_size=patch_size)
                    for psi_t_img, height, width in zip(psi_t_images, heights, widths)]

    psi_t_tokens = torch.einsum("a,ab->ab", psi_t, epsilon_tokens)
    psi_t_tokens = psi_t_tokens + torch.einsum("a, ab->ab", t, tokens_value)

    psi_t_tokens = (1 - mask_tokens) * psi_t_tokens + mask_tokens * tokens_value

    psi_t_tokens = torch.stack([tokens_id, psi_t_tokens], dim=1)

    return psi_t_images, psi_t_tokens, epsilon_images, epsilon_tokens

def get_masked_channel_inputs(channels: list[torch.Tensor],
                              mask: list[torch.Tensor],
                              t: torch.Tensor,
                              sigma_min: float, patch_size: int,
                              generator = None) -> Tuple:

    heights = [channel.shape[1] for channel in channels]
    widths = [channel.shape[2] for channel in channels]

    epsilon_images = [torch.randn(size=channel.shape, device=channel.device,
                                  dtype=channel.dtype, generator=generator)
                      for channel in channels]

    # epsilon_images = [torch.randn(size=channel.shape, device=channel.device,
    #                               dtype=channel.dtype)
    #                   for channel in channels]

    psi_t = (1 - (1 - sigma_min) * t)

    psi_t_channels = [torch.einsum("a, abc->abc", psi_t, epsilon_img) for epsilon_img
                    in epsilon_images]

    psi_t_channels = [psi_t_channel + torch.einsum("a, abc->abc", t, channel)
                    for psi_t_channel, channel in zip(psi_t_channels, channels)]

    psi_t_channels = [to_patch_representation(psi_t_channel, patch_size=patch_size)
                    for psi_t_channel in psi_t_channels]

    channels = [to_patch_representation(channel, patch_size=patch_size) for channel in channels]

    psi_t_channels = [torch.einsum('ab, abc -> abc', (1 - mask_channel), psi_t_channel) +
                      torch.einsum('ab, abc -> abc', mask_channel, channel)
                      for psi_t_channel, mask_channel, channel in zip(psi_t_channels, mask, channels)]

    psi_t_channels = [from_patch_representation(psi_t_img, height=height,
                                                width=width, patch_size=patch_size)[:, 0]
                      for psi_t_img, height, width in zip(psi_t_channels, heights, widths)]

    return psi_t_channels, epsilon_images


def get_model_input(batch: dict, t: torch.Tensor, sigma_min: float,
                    patch_size: int, generator = None) -> Tuple:

    batch_size = batch['channels'][0]['channel_id'].shape[0]
    device = batch['channels'][0]['channel_id'].device

    channel_inputs = []
    channel_targets = []
    channel_epsilons = []
    channel_masks = []
    channel_pdes = []
    channel_constants = []
    channel_constants_class = []
    channel_idx = []
    channel_time_step_strides = []
    channel_task_embs = []

    heights = [channel['data'][0].shape[1] for channel in batch['channels']]
    widths = [channel['data'][0].shape[2] for channel in batch['channels']]

    for channel in batch['channels']:

        channel_input, epsilon = get_masked_channel_inputs(channel['data'], channel['mask'],
                                                           t, sigma_min, patch_size, generator=generator)

        channel_target = [(img - (1 - sigma_min) * eps) for img, eps
                         in zip(channel['data'], epsilon)]

        channel_target = torch.stack(channel_target, dim=1)

        channel_masks.append(channel['mask'])

        channel_targets.append(channel_target)
        channel_inputs.append(torch.stack(channel_input, dim=1))
        channel_epsilons.append(epsilon)
        channel_constants.append(channel['constants'])
        channel_constants_class.append(channel['constants_class'])
        channel_pdes.append(channel['pde'])
        channel_idx.append(channel['channel_id'])
        channel_time_step_strides.append(channel['time_step_stride'])
        channel_task_embs.append(channel['task_emb'])

    simulation_time = [torch.zeros(batch_size).to(device)] * len(channel_inputs)
    t_in = [t] * len(channel_inputs)

    return (channel_inputs, channel_targets, channel_epsilons, channel_masks, channel_pdes, channel_constants,
            channel_constants_class, channel_idx, channel_time_step_strides, channel_task_embs, simulation_time, t_in)

class MaskedFlowMatching(nn.Module):
    """
    Objective for conditional optimal transport matching the flow.
    See Lipman et al. 2023 "Flow Matching for Generative Modeling" for details.
    """

    def __init__(self, sigma_min: float = 1e-4, weighting: str = 'constant', patch_size: int = 4):

        super().__init__()

        self.patch_size = patch_size
        self.sigma_min = sigma_min
        self.weighting_function = get_weighting_function(weighting)
        self.start_time = 0.0
        self.end_time = 1.0

        self.scheduler = OdeEulerScheduler(num_train_timesteps=100)

    def predict(self, data: dict, model: nn.Module, num_inference_steps: int = 100,
                generator: torch.Generator = None, **kwargs) -> list[torch.Tensor]:

        self.scheduler.set_timesteps(num_inference_steps)

        t = self.scheduler.timesteps[0]

        batch_size = data['pde'].shape[0]
        device = data['pde'].device

        t = t.repeat(batch_size).to(device)

        channel_inputs, _, _, channel_masks, channel_pdes, channel_constants, \
            channel_constants_class, channel_idx, channel_time_step_strides, channel_task_embs, simulation_time, _ = \
            get_model_input(data, t=t, sigma_min=self.sigma_min, patch_size=self.patch_size,
                            generator = generator)

        heights = [channel['data'][0].shape[1] for channel in data['channels']]
        widths = [channel['data'][0].shape[2] for channel in data['channels']]

        channel_masks = [torch.stack(mask_img, dim=1) for mask_img in channel_masks]

        channel_masks = [from_patch_representation(mask_img, height=int(height / self.patch_size),
                                                   width=int(width / self.patch_size), patch_size=1)
                         for mask_img, height, width in zip(channel_masks, heights, widths)]

        # upscale masks by self.patch_size
        channel_masks = [torch.nn.functional.interpolate(mask_img, scale_factor=self.patch_size, mode='nearest')
                         for mask_img in channel_masks]

        for t in self.scheduler.timesteps:

            t_in = t.repeat(batch_size).to(device)
            t_in = [t_in] * len(channel_inputs)

            prediction = model.forward(x=channel_inputs,
                                       simulation_time=simulation_time,
                                       channel_type=channel_idx,
                                       pde_type=channel_pdes,
                                       pde_parameters=channel_constants,
                                       pde_parameters_class=channel_constants_class,
                                       simulation_dt=channel_time_step_strides,
                                       task=channel_task_embs,
                                       t=t_in)

            channel_predictions = prediction.sample

            # channel_masks = [to_patch_representation(mask_img, patch_size=self.patch_size)
            #                     for mask_img in channel_masks]

            # channel_predictions = [(1 - mask_img) * prediction_img +
            #                        mask_img * target_img
            #                        for prediction_img, mask_img, target_img in
            #                        zip(channel_predictions, channel_masks, channel_inputs)]

            updated_channel_inputs = []

            for prediction, channel_mask, channel_input in zip(channel_predictions, channel_masks, channel_inputs):

                x0 = self.scheduler.step(prediction, t, channel_input, generator=generator).prev_sample

                x0 = (1 - channel_mask) * x0 + channel_mask * channel_input

                updated_channel_inputs.append(x0)

            channel_inputs = updated_channel_inputs

        return channel_inputs



    def loss(self, model: nn.Module, batch: dict, **kwargs) -> Tuple[torch.Tensor, Dict]:

        batch_size = batch['channels'][0]['channel_id'].shape[0]
        device = batch['channels'][0]['channel_id'].device

        t = sample_time(device, batch_size, t0=self.start_time,
                        t1=self.end_time)

        channel_inputs = []
        channel_targets = []
        channel_epsilons = []
        channel_masks = []
        channel_pdes = []
        channel_constants = []
        channel_constants_class = []
        channel_idx = []
        channel_time_step_strides = []
        channel_task_embs = []

        heights = [channel['data'][0].shape[1] for channel in batch['channels']]
        widths = [channel['data'][0].shape[2] for channel in batch['channels']]

        for channel in batch['channels']:

            channel_input, epsilon = get_masked_channel_inputs(channel['data'], channel['mask'],
                                                               t, self.sigma_min, self.patch_size)

            channel_target = [(img - (1 - self.sigma_min) * eps) for img, eps
                             in zip(channel['data'], epsilon)]

            channel_target = torch.stack(channel_target, dim=1)

            # channel_target = to_patch_representation(channel_target, patch_size=self.patch_size)

            channel_masks.append(channel['mask'])

            channel_targets.append(channel_target)
            channel_inputs.append(torch.stack(channel_input, dim=1))
            channel_epsilons.append(epsilon)
            channel_constants.append(channel['constants'])
            channel_constants_class.append(channel['constants_class'])
            channel_pdes.append(channel['pde'])
            channel_idx.append(channel['channel_id'])
            channel_time_step_strides.append(channel['time_step_stride'])
            channel_task_embs.append(channel['task_emb'])

        simulation_time = [torch.zeros(batch_size).to(device)] * len(channel_inputs)
        t_in = [t] * len(channel_inputs)
        prediction = model.forward(x=channel_inputs,
                                   simulation_time=simulation_time,
                                   channel_type=channel_idx,
                                   pde_type=channel_pdes,
                                   pde_parameters=channel_constants,
                                   pde_parameters_class=channel_constants_class,
                                   simulation_dt=channel_time_step_strides,
                                   task=channel_task_embs,
                                   t=t_in)

        channel_predictions = prediction.sample

        # channel_predictions = [to_patch_representation(img, patch_size=self.patch_size)
        #                      for img in channel_predictions]

        channel_masks = [torch.stack(mask_img, dim=1) for mask_img in channel_masks]

        channel_masks = [from_patch_representation(mask_img, height=int(height / self.patch_size),
                                                   width=int(width / self.patch_size), patch_size=1)
                            for mask_img, height, width in zip(channel_masks, heights, widths)]

        # upscale masks by self.patch_size
        channel_masks = [torch.nn.functional.interpolate(mask_img, scale_factor=self.patch_size, mode='nearest')
                         for mask_img in channel_masks]

        # channel_masks = [to_patch_representation(mask_img, patch_size=self.patch_size)
        #                     for mask_img in channel_masks]

        channel_predictions = [(1 - mask_img) * prediction_img +
                               mask_img * target_img
                             for prediction_img, mask_img, target_img in
                             zip(channel_predictions, channel_masks, channel_targets)]

        loss_images = [torch.mean((target_img - prediction_img) ** 2, dim=[1,2])
                       for target_img, prediction_img in
                       zip(channel_targets, channel_predictions)]

        # channel_predictions = [from_patch_representation(prediction_img, height=height, width=width, patch_size=self.patch_size)
        #                          for prediction_img, height, width in
        #                          zip(channel_predictions, heights, widths)]
        #
        # channel_targets = [
        #     from_patch_representation(target_img, height=height, width=width, patch_size=self.patch_size)
        #     for target_img, height, width in
        #     zip(channel_targets, heights, widths)]

        loss_images = torch.stack(loss_images, dim=1) # [batch_size, num_channels]
        loss_images = loss_images.mean(dim=1)

        loss_raw = loss_images.mean()
        loss = loss_raw / (loss_raw.detach() + 1e-6)

        return loss, {'loss_raw': loss_raw.item()}

import lightning

class FlowMatching(lightning.LightningModule):

    def __init__(self, model: dict,
                 tasks: list[str],
                 ckpt_path=None,
                 ignore_keys=None,
                 monitor: str = "val/loss_epoch",
                 frequency: int = 500,
                 steps: int = 20,
                 weighting: str = 'uniform',
                 patch_size: int = 4,
                 timesteps: int = 6):

        super().__init__()

        self.monitor = monitor
        self.frequency = frequency
        self.patch_size = patch_size
        self.model = instantiate_from_config(model)

        self.timesteps = timesteps

        self.fm = MaskedFlowMatching(patch_size=patch_size)
        self.steps = steps

        self.task_masking = TaskMasking(tasks, weighting, patch_size=patch_size)
        self.save_hyperparameters()

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

    def init_from_ckpt(self, path: str, ignore_keys:list[str]=None):

        if ignore_keys is None:
            ignore_keys = list()

        if Path(path).is_dir():
            path = Path(path).joinpath("last.ckpt")
        else:
            path = Path(path)

        if path.is_file():
            sd = torch.load(path, map_location="cpu")["state_dict"]
            keys = list(sd.keys())
            for k in keys:
                for ik in ignore_keys:
                    if k.startswith(ik):
                        print("Deleting key {} from state_dict.".format(k))
                        del sd[k]
            self.load_state_dict(sd, strict=False)
            print(f"Restored from {path}")

    # Useful function when getting the error https://github.com/Lightning-AI/pytorch-lightning/issues/17212
    def on_after_backward(self):
        for name, param in self.named_parameters():
            if param.grad is None:
                print(name)

    def shared_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Tuple[torch.Tensor, dict]:

        data = self.task_masking.get_inputs(batch, batch_idx)

        loss, info_dict = self.fm.loss(self.model, data)
        info_dict['task_name'] = data['task_name']

        return loss, info_dict

    def training_step(self, batch, batch_idx):

        loss, info_dict = self.shared_step(batch, batch_idx)

        task_name = info_dict['task_name']
        loss_raw = info_dict['loss_raw']

        self.log(f"{task_name}/loss", loss_raw, prog_bar=True,
                 logger=True, on_step=False, on_epoch=True, sync_dist=True)

        self.log("val/loss", loss_raw, prog_bar=True,
                 logger=True, on_step=True, on_epoch=True, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):

        loss, info_dict = self.shared_step(batch, batch_idx)

        task_name = info_dict['task_name']
        loss_raw = info_dict['loss_raw']

        self.log(f"val/{task_name}/loss", loss_raw, prog_bar=True,
                 logger=True, on_step=False, on_epoch=True, sync_dist=True)

        self.log("val/loss", loss_raw, prog_bar=True,
                 logger=True, on_step=True, on_epoch=True, sync_dist=True)

        return {"val/loss": loss}

    def configure_optimizers(self):

        opt = torch.optim.AdamW(self.parameters(), lr=self.learning_rate,
                                weight_decay=0.0)

        scheduler = TanhLRScheduler(opt, t_initial=self.frequency, cycle_limit=10000)

        return [opt], [] # [{"scheduler": scheduler, "interval": "epoch"}]

    def predict_step(self, data,
                     num_inference_steps=100,
                     generator=None, **kwargs):


        return self.fm.predict(data, self.model, num_inference_steps=num_inference_steps,
                               generator=generator, **kwargs)


    def predict(self, batch, device, num_frames=20, generator=None,
                output_type='numpy', num_inference_steps=100, return_dict=True, batch_dim=True):

        task: AbstractTask = self.task_masking.get_task('ForwardPrediction5to1')

        batch['data'] = batch['data'].to(device)
        batch['constants'] = batch['constants'].to(device)
        batch['constants_norm'] = batch['constants_norm'].to(device)
        batch['time_step_stride'] = batch['time_step_stride'].to(device)
        batch['physical_metadata'] = {k: v.to(device) for k, v in batch['physical_metadata'].items()}

        with torch.no_grad():

            data = self.task_masking.get_data(batch)

            frames = torch.stack([channel['channel'][:,:self.timesteps - 1] for channel in data['channels']], dim=1)
            reference = torch.stack([channel['channel'][:,:num_frames] for channel in data['channels']], dim=1)

            frames = list(torch.permute(frames, (2, 0, 1, 3, 4)))
            reference = list(torch.permute(reference, (2, 0, 1, 3, 4)))

            for channel in data['channels']:
                channel['channel'] = channel['channel'][:,:self.timesteps]

            masks = task.prepare_data(data, prob=0.0)

            data.update(masks)

            if generator is None:
                generator = (torch.Generator(device=device)
                             .manual_seed(2024))

            batch_size = data['channels'][0]['channel_id'].shape[0]

            for _ in tqdm(range(num_frames - self.timesteps + 1)):

                x0 = self.predict_step(data, generator=generator,
                                       num_inference_steps=num_inference_steps)

                # save latest time step to frames
                frames.append(torch.stack(x0, dim=1)[:, :, -1])

                # shift time dimension of each channel by one
                for channel, predicted_channel in zip(data['channels'], x0):

                    # shift time dimension of channel by one
                    x0 = torch.concat([predicted_channel[:, 1:],
                                       torch.zeros_like(predicted_channel[:, :1])], dim=1)

                    # set to zero where channel_id is zero
                    mask = channel['channel_id'].flatten() > 0
                    x0 = torch.einsum('b...,b->b...', x0, mask.float())

                    x0 = x0.swapaxes(0, 1)
                    channel['data'] = list(x0)


            vid = np.array(torch.stack(frames).cpu())
            vid = np.swapaxes(vid, 0, 1)

            reference = np.array(torch.stack(reference).cpu())
            reference = np.swapaxes(reference, 0, 1)

            if not batch_dim:
                vid = vid[0]
                reference = reference[0]

        return vid, reference


    def test_step(self, *args: Any, **kwargs: Any) -> STEP_OUTPUT:

        return