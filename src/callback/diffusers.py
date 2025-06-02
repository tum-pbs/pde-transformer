from copy import copy
from typing import Union

from lightning import Callback, Trainer, LightningModule
from lightning.fabric.utilities import rank_zero_only

from omegaconf import DictConfig, OmegaConf

from src.callback import EMA
from src.callback.ema import EMAOptimizer
from src.utils import instantiate_from_config, get_pipeline

import os

class Diffusers(Callback):

    def __init__(self, pipeline: Union[DictConfig, OmegaConf], frequency: int, repo_id: str, save_dir: str, push_to_hub: bool=False,
                 disabled: bool=False, hub_private_repo: bool = True, save_ema: bool = True,
                 overwrite_output_dir: bool = True):
        super().__init__()

        if push_to_hub and "HUGGING_FACE_API_KEY" not in os.environ:
            print("Please set the HUGGING_FACE_API_KEY environment variable to push to the hub.")
            push_to_hub = False
            self.token = ''
        else:
            self.token = os.getenv("HUGGING_FACE_API_KEY")

        self.repo_id = repo_id
        self.pipeline = OmegaConf.to_container(pipeline)
        self.frequency = frequency
        self.save_dir = save_dir
        self.save_ema = save_ema
        self.push_to_hub = push_to_hub
        self.disabled = disabled
        self.hub_private_repo = hub_private_repo
        self.overwrite_output_dir = overwrite_output_dir


    def save_pipeline(self, pl_module, trainer):

        pipeline_instance = get_pipeline(self.pipeline, pl_module)

        pipeline_instance.save_pretrained(self.save_dir)

        if self.push_to_hub:

            pipeline_instance.push_to_hub(repo_id=self.repo_id, token=self.token,
                                          private=self.hub_private_repo,
                                          commit=f'Epoch: {trainer.current_epoch}')

        if self.save_ema:

            optimizers_ema = [optimizer for optimizer in trainer.optimizers if isinstance(optimizer, EMAOptimizer)]

            if len(optimizers_ema) > 0:

                ema: EMAOptimizer = optimizers_ema[0]

                with ema.swap_ema_weights():

                    pipeline_instance.save_pretrained(self.save_dir + '_ema')

    def check_frequency(self, check_idx):
        return ((check_idx + 1) % self.frequency) == 0

    @rank_zero_only
    def on_train_epoch_end(self, trainer, pl_module):
        if not self.disabled and self.check_frequency(pl_module.current_epoch):
            self.save_pipeline(pl_module, trainer)