from pathlib import Path
from typing import List

from diffusers.utils.torch_utils import randn_tensor
from tqdm import tqdm

from sampler.scheduler import OdeEulerScheduler
from src.objectives import TrainingObjective

from src.utils import instantiate_from_config
import torch.nn as nn
import lightning
import torch

import numpy as np

class SingleStepDiffusion(lightning.LightningModule):
    def __init__(self,
                 model,
                 objective,
                 ckpt_path=None,
                 ignore_keys=None,
                 image_key=0,
                 monitor=None,
                 downsample_factor: int = 1,
                 optimizer='adamw',
                 ):
        """
        Single step diffusion model for generative modeling.
        Args:
            model: model configuration
            objective: objective configuration
            ckpt_path: path to checkpoint
            ignore_keys: keys to ignore in checkpoint
            image_key: key for image in batch
            monitor: monitor configuration
            downsample_factor: downsample factor for input
        """

        super().__init__()

        self.image_key = image_key
        self.optimizer = optimizer

        # Score model
        self.model: nn.Module = instantiate_from_config(model)

        # Objective
        self.objective: TrainingObjective = (
            instantiate_from_config(objective))

        self.downsample_factor = downsample_factor

        if monitor is not None:
            self.monitor = monitor
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

        self.scheduler = OdeEulerScheduler(num_train_timesteps=100)

    def forward(self, x: torch.Tensor, t: torch.Tensor,
                **kwargs) -> torch.Tensor:

        return self.model(x, t, **kwargs)

    def get_pipeline_args(self):
        return {
            "unet": self.model,
        }

    # Useful function for getting the error https://github.com/Lightning-AI/pytorch-lightning/issues/17212
    # def on_after_backward(self):
    #     for name, param in self.named_parameters():
    #         if param.grad is None:
    #             print(name)

    def get_input(self, batch, batch_dim=True):

        data: torch.Tensor = batch["data"]
        meta_data_loading: dict = batch["loading_metadata"]
        meta_data_physical: dict = batch["physical_metadata"]

        if batch_dim:
            x: torch.Tensor = data[:, 0]
            y: torch.Tensor = data[:, 1:]

            task_idx = meta_data_physical['PDE'][:, 0]
            # task_idx = meta_data_loading['dataset_idx'].long()

        else:
            x: torch.Tensor = data[0]
            y: torch.Tensor = data[1:]
            task_idx = meta_data_physical['PDE']
            # task_idx = meta_data_loading['dataset_idx']

            x = torch.unsqueeze(x, 0)
            y = torch.unsqueeze(y, 0)

            if not torch.is_tensor(task_idx):
                task_idx = torch.tensor(task_idx)

            # task_idx = torch.unsqueeze(task_idx, 0)

        if self.downsample_factor > 1:
            # downsample with average pooling
            x = nn.functional.avg_pool2d(x, self.downsample_factor)

            num_batches = y.shape[0]
            y = y.reshape(-1, y.shape[-3], y.shape[-2], y.shape[-1])
            y = nn.functional.avg_pool2d(y, self.downsample_factor)
            y = y.reshape(num_batches, -1, y.shape[-3], y.shape[-2], y.shape[-1])

        return x, y, task_idx

    def test_step(self, batch, batch_idx):

        return 0, {}

    def init_from_ckpt(self, path: str, ignore_keys: List[str]=None):
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

    def training_step(self, batch, batch_idx):

        input_0, input_1, labels = self.get_input(batch)

        score_loss, log_dict = self.objective.loss(self.model,
                                                   {"data": input_1[:,-1], "y": input_0,
                                                    "class_labels": labels}, split="train")

        self.log("loss", score_loss.item(), prog_bar=True,
                 logger=True, on_step=True, on_epoch=True, sync_dist=True)

        self.log_dict(log_dict, prog_bar=False, logger=True,
                      on_step=True, on_epoch=False, sync_dist=True)

        return score_loss

    def validation_step(self, batch, batch_idx):

        input_0, input_1, labels = self.get_input(batch)

        score_loss, log_dict = self.objective.loss(self.model,
                                                   {"data": input_1[:,-1], "y": input_0,
                                                    "class_labels": labels}, split="train")

        self.log("val/loss", score_loss.item(), prog_bar=True,
                 logger=True, on_step=True, on_epoch=True)

        self.log_dict(log_dict, prog_bar=False, logger=True,
                      on_step=True, on_epoch=False)

        return self.log_dict

    def predict_step(self, previous_frame,
                     num_inference_steps=100,
                     generator=None, **kwargs):

        image_shape = previous_frame.shape

        if self.device.type == "mps":
            # randn does not work reproducibly on mps
            x0 = randn_tensor(image_shape, generator=generator)
            x0 = x0.to(self.device)
        else:
            x0 = randn_tensor(image_shape, generator=generator, device=self.device)

        input = torch.cat([x0, previous_frame], dim=1)

        # set step values
        self.scheduler.set_timesteps(num_inference_steps)

        for t in self.scheduler.timesteps:
            # 1. predict noise model_output
            model_output = self.forward(input, t, **kwargs).sample

            # 2. compute previous image: x_t -> x_t-1
            x0 = self.scheduler.step(model_output, t, x0, generator=generator).prev_sample

            input = torch.cat([x0, previous_frame], dim=1)

        return x0

    def postprocess_output(self, x0, physical_metadata):

        fields = physical_metadata['Fields'].to(x0.device)
        fields = fields > 0 # if field id > 0, then it corresponds to a physical field, padded otherwise
        # set padded values to zero

        if len(fields.shape) == 1:
            fields = fields.unsqueeze(0)

        x0 = torch.einsum('bchw, bc -> bchw', x0, fields)

        return x0

    def predict(self, batch, device, num_frames=20, generator=None,
                output_type='numpy', num_inference_steps=100, return_dict=True, batch_dim=True):

        with torch.no_grad():

            input_0, input_1, labels = self.get_input(batch, batch_dim=batch_dim)

            input_0 = input_0.to(device)
            input_1 = input_1.to(device)
            labels = labels.to(device)

            if generator is None:
                generator = (torch.Generator(device=input_0.device)
                             .manual_seed(2024))

            frames = [input_0.cpu()]
            previous_frame = input_0

            for _ in tqdm(range(num_frames)):

                x0 = self.predict_step(previous_frame, generator=generator, class_labels=labels,
                                       num_inference_steps=num_inference_steps)
                x0 = self.postprocess_output(x0, batch['physical_metadata'])
                previous_frame = x0
                frames.append(x0.cpu())

            vid = np.array([frame.numpy() for frame in frames])
            vid = np.swapaxes(vid, 0, 1)

            reference = np.array(torch.concat([frames[0].unsqueeze(1),
                                               input_1.cpu()], dim=1))

            if not batch_dim:
                vid = vid[0]
                reference = reference[0]

        return vid, reference

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

    def get_last_layer(self):
        return self.decoder.conv_out.weight

