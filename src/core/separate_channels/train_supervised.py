from pathlib import Path
from typing import Tuple, Dict, Any, Optional

import torch
import torch.nn as nn
from lightning.pytorch.utilities.types import STEP_OUTPUT
from tqdm import tqdm

from src.core.separate_channels.tasks import TaskMasking, ForwardPrediction
from src.core.separate_channels.utils import from_patch_representation, to_patch_representation

from src.utils import instantiate_from_config

import numpy as np

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
                              patch_size: int) -> list[torch.Tensor]:

    heights = [channel.shape[1] for channel in channels]
    widths = [channel.shape[2] for channel in channels]

    zero_images = [torch.zeros_like(channel, device=channel.device,
                                  dtype=channel.dtype) for channel in channels]

    zero_images = [to_patch_representation(zero_image, patch_size=patch_size)
                   for zero_image in zero_images]

    channels = [to_patch_representation(channel, patch_size=patch_size)
                for channel in channels]

    output = [torch.einsum('ab, abc -> abc', (1 - mask_channel), zero_image) +
                      torch.einsum('ab, abc -> abc', mask_channel, channel)
                      for zero_image, mask_channel, channel in zip(zero_images, mask, channels)]

    output = [from_patch_representation(out, height=height,
                                        width=width, patch_size=patch_size)[:, 0]
                      for out, height, width in zip(output, heights, widths)]

    return output

def get_model_input(batch: dict, patch_size: int) -> Tuple:

    batch_size = batch['channels'][0]['channel_id'].shape[0]
    device = batch['channels'][0]['channel_id'].device

    channel_inputs = []
    channel_targets = []

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

        channel_input = get_masked_channel_inputs(channel['data'], channel['mask'],
                                                           patch_size)

        channel_target = channel['data']
        channel_target = torch.stack(channel_target, dim=1)

        channel_masks.append(channel['mask'])

        channel_targets.append(channel_target)
        channel_inputs.append(torch.stack(channel_input, dim=1))

        channel_constants.append(channel['constants'])
        channel_constants_class.append(channel['constants_class'])
        channel_pdes.append(channel['pde'])
        channel_idx.append(channel['channel_id'])
        channel_time_step_strides.append(channel['time_step_stride'])
        channel_task_embs.append(channel['task_emb'])

    simulation_time = [torch.zeros(batch_size).to(device)] * len(channel_inputs)
    t_in = [torch.zeros(batch_size).to(device)] * len(channel_inputs)

    channel_masks = [torch.stack(mask_img, dim=1) for mask_img in channel_masks]

    channel_masks = [from_patch_representation(mask_img, height=int(height / patch_size),
                                               width=int(width / patch_size), patch_size=1)
                     for mask_img, height, width in zip(channel_masks, heights, widths)]

    # upscale masks by self.patch_size
    channel_masks = [torch.nn.functional.interpolate(mask_img, scale_factor=patch_size, mode='nearest')
                     for mask_img in channel_masks]

    return (channel_inputs, channel_targets, channel_masks, channel_pdes, channel_constants,
            channel_constants_class, channel_idx, channel_time_step_strides, channel_task_embs, simulation_time, t_in)

class MultiStepMSE(nn.Module):
    """
    Objective for supervised learning with a multi time step
    """

    def __init__(self, m: int = 5, n: int = 1, patch_size: int = 4,
                 normalize_channels: bool = False):

        super().__init__()
        self.normalize_channels = normalize_channels
        self.m = m
        self.n = n
        self.patch_size = patch_size

    def predict(self, batch: dict, model: nn.Module,
                generator: torch.Generator = None, **kwargs) -> list[torch.Tensor]:

        channel_inputs, _, channel_masks, channel_pdes, channel_constants, \
            channel_constants_class, channel_idx, channel_time_step_strides, channel_task_embs, simulation_time, t_in = \
            get_model_input(batch, patch_size=self.patch_size)

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

        channel_predictions = [(1 - mask_img) * prediction_img +
                               mask_img * target_img
                               for prediction_img, mask_img, target_img in
                               zip(channel_predictions, channel_masks, channel_inputs)]

        return channel_predictions

    def loss(self, model: nn.Module, batch: dict, **kwargs) -> Tuple[torch.Tensor, Dict]:

        channel_inputs, channel_targets, channel_masks, channel_pdes, channel_constants, \
            channel_constants_class, channel_idx, channel_time_step_strides, channel_task_embs, simulation_time, t_in = \
            get_model_input(batch, patch_size=self.patch_size)

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

        channel_predictions = [(1 - mask_img) * prediction_img +
                               mask_img * target_img
                               for prediction_img, mask_img, target_img in
                               zip(channel_predictions, channel_masks, channel_targets)]

        predictions = torch.stack(channel_predictions, dim=1)
        targets = torch.stack(channel_targets, dim=1)

        if self.normalize_channels:
            targets = ((targets - targets.mean(dim=(3, 4), keepdim=True)) /
                       (targets.std(dim=(3, 4), keepdim=True) + 1e-4))
            predictions = ((predictions - targets.mean(dim=(3, 4), keepdim=True)) /
                           (targets.std(dim=(3, 4), keepdim=True) + 1e-4))

        loss = torch.mean((targets[:, :, self.m:] - predictions[:, :, self.m:]) ** 2)

        # loss_images = [torch.mean((target_img[:,self.m:] - prediction_img[:,self.m:]) ** 2, dim=[1,2])
        #                for target_img, prediction_img in
        #                zip(channel_targets, channel_predictions)]
        #
        # loss_images = torch.stack(loss_images, dim=1) # [batch_size, num_channels]
        # loss_images = loss_images.mean(dim=1)
        #
        # loss = loss_images.mean()

        return loss, {}

import lightning

class Supervised(lightning.LightningModule):

    def __init__(self, model: dict,
                 ckpt_path=None,
                 ignore_keys=None,
                 monitor: str = "val/loss_epoch",
                 weighting: str = 'uniform',
                 patch_size: int = 4,
                 timesteps: int = 6,
                 optimizer: str = 'adamw',
                 normalize_channels: bool = False):

        super().__init__()

        self.monitor = monitor

        self.optimizer = optimizer

        self.patch_size = patch_size
        self.model = instantiate_from_config(model)

        self.timesteps = timesteps
        self.weighting = "constant"

        self.objective = MultiStepMSE(patch_size=patch_size, m=timesteps-1, n=1,
                                      normalize_channels=normalize_channels)

        task_list = [ForwardPrediction(patch_size=self.patch_size, num_timesteps=timesteps, m=timesteps-1, n=1, task_idx=0)]

        self.task_masking = TaskMasking(task_list, weighting, patch_size=patch_size)
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

    def shared_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Tuple[torch.Tensor, str]:

        data = self.task_masking.get_inputs(batch, batch_idx)

        loss, _ = self.objective.loss(self.model, data)

        return loss, data['task_name']

    def get_pipeline_args(self):
        return {
            "unet": self.model,
        }

    def training_step(self, batch, batch_idx):

        loss, task_name = self.shared_step(batch, batch_idx)

        self.log(f"{task_name}/loss", loss.item(), prog_bar=True,
                 logger=True, on_step=False, on_epoch=True, sync_dist=True)

        self.log("loss", loss.item(), prog_bar=True,
                 logger=True, on_step=True, on_epoch=True, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):

        loss, task_name = self.shared_step(batch, batch_idx)

        self.log(f"val/{task_name}/loss", loss.item(), prog_bar=True,
                 logger=True, on_step=False, on_epoch=True, sync_dist=True)

        self.log("val/loss", loss.item(), prog_bar=True,
                 logger=True, on_step=True, on_epoch=True, sync_dist=True)

        return {"val/loss": loss}

    def configure_optimizers(self):

        if self.optimizer == 'adamw':

            opt = torch.optim.AdamW(self.parameters(), lr=self.learning_rate,
                                    weight_decay=1e-15)
        elif self.optimizer == 'adabelief':

            from adabelief_pytorch import AdaBelief # noqa
            opt = AdaBelief(self.parameters(), lr=self.learning_rate)

        else:

            raise ValueError(f"Optimizer {self.optimizer} not supported;")

        return [opt]

    def predict_step(self, data,
                     num_inference_steps=100,
                     generator=None, **kwargs):


        return self.objective.predict(data, self.model, num_inference_steps=num_inference_steps,
                               generator=generator, **kwargs)


    def predict(self, batch, device, num_frames=20, generator=None,
                output_type='numpy', num_inference_steps=100, return_dict=True, batch_dim=True):

        task = ForwardPrediction(patch_size=self.patch_size, num_timesteps=self.timesteps,
                                               m=self.timesteps-1, n=1, task_idx=0)

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