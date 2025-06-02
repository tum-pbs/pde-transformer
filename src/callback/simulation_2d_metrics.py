from typing import Optional, Any, Dict

import torchmetrics
from lightning import Callback, Trainer, LightningModule
from lightning.fabric.utilities import rank_zero_only
import torch
from math import ceil
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from data.multi_module import get_subdatasets_from_dataloader
from src.callback.ema import EMAOptimizer
from src.metric.metric import get_metrics
from src.utils import instantiate_from_config, get_pipeline

import logging
log = logging.getLogger(__name__)

from lightning.fabric.utilities.rank_zero import rank_prefixed_message, _get_rank, rank_zero_info

import torch.nn as nn

class Simulation2DMetricLogger(Callback):
    def __init__(self, frequency: int, batch_size: int,
                 pipelines: DictConfig, log_first_step: bool = False,
                 test_only: bool = True,
                 metric_config: DictConfig = None):

        super().__init__()

        self.frequency = frequency
        self.pipelines = OmegaConf.to_container(pipelines)
        self.batch_size = batch_size
        self.disabled = False
        self.test_only = test_only
        self.log_first_step = log_first_step
        self.metric_config = metric_config
        self.temp_directory = "/tmp"

    def update_metrics_impl(self, pipelines, trainer, pl_module, dataloader):

        metrics_torch = {}

        for pipeline in pipelines.keys():
            metrics_torch[pipeline] = nn.ModuleList(get_metrics(
                self.metric_config)).to(pl_module.device)

        generator = torch.Generator(device=pl_module.device).manual_seed(trainer.global_rank)

        for pipeline in pipelines.keys():

            for batch in dataloader:

                frames, target, labels = pl_module.get_input(batch, batch_dim=True)
                num_frames = target.shape[1]

                frames = frames.to(pl_module.device)
                target = target.to(pl_module.device)
                labels = labels.to(pl_module.device)

                prediction = pipelines[pipeline](data=frames, num_frames=num_frames,
                                             generator=generator,
                                             output_type='numpy', return_dict=True,
                                             class_labels=labels).videos

                # first element is the input frame
                prediction = torch.from_numpy(prediction[:, 1:]).to(pl_module.device)

                for metric in metrics_torch[pipeline]:

                    metric.update(prediction, target, class_labels=labels)

        metric_dict = {"epoch": trainer.current_epoch}

        for pipeline in pipelines.keys():
            for metric in metrics_torch[pipeline]:

                metric_name = metric.__class__.__name__

                metric_value = metric.compute()

                if isinstance(metric_value, tuple):
                    metric_dict[f"{pipeline}_{metric_name}_mean"] = metric_value[0]
                    metric_dict[f"{pipeline}_{metric_name}_std"] = metric_value[1]

                else:
                    metric_dict[f"{pipeline}_{metric_name}"] = metric_value

                if trainer.global_rank == 0:

                    logdir = trainer.logger.experiment.config["runtime"]["logdir"]
                    metric.save(logdir + '/metrics/', f"{pipeline}_{metric_name}_{trainer.current_epoch}.csv")

            self.log_metrics(trainer, metric_dict)

    def update_metrics(self, trainer : Trainer, pl_module : LightningModule, dataloader,
                       use_ema: bool = True):

        optimizers_ema = [optimizer for optimizer in trainer.optimizers
                          if isinstance(optimizer, EMAOptimizer)]

        if use_ema and len(optimizers_ema) > 0:

            ema: EMAOptimizer = optimizers_ema[0]

            with ema.swap_ema_weights():

                pipelines = {key: get_pipeline(value, pl_module)
                             for key, value in self.pipelines.items()}

                self.update_metrics_impl(pipelines, trainer, pl_module, dataloader)

        else:

                pipelines = {key: get_pipeline(value, pl_module)
                            for key, value in self.pipelines.items()}

                self.update_metrics_impl(pipelines, trainer, pl_module, dataloader)

    @rank_zero_only
    def log_metrics(self, trainer: Trainer, metric_dict: Dict[str, Any]):
        trainer.logger.experiment.log(metric_dict)

    def check_frequency(self, check_idx):
        # check_idx + 1 since current_idx starts with 0
        return ((check_idx + 1) % self.frequency) == 0 or (check_idx == 0 and self.log_first_step)

    def on_test_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if not self.disabled:
            self.update_metrics(trainer, pl_module, trainer.test_dataloaders)

    def on_train_epoch_end(self, trainer, pl_module):

        if ((not self.disabled) and (not self.test_only) and
                self.check_frequency(pl_module.current_epoch)):

            if (dataloader := trainer.test_dataloaders) is None:
                trainer.test_loop.setup_data()
                dataloader = trainer.test_dataloaders

            self.update_metrics(trainer, pl_module, dataloader)

class Simulation2DMetricLoggerCustom(Callback):
    def __init__(self, frequency: int, batch_size: int,
                 num_inference_steps: int = 100, num_frames: int = 20,
                 trim_start: int = 1,
                 use_ema: bool = True,
                 log_first_step: bool = False, test_only: bool = False,
                 disabled: bool = False,
                 reference_boundary: bool = False,
                 metric_config: DictConfig = None):
        """

        :param frequency:
        :param batch_size:
        :param num_inference_steps:
        :param num_frames:
        :param trim_start:
        :param use_ema:
        :param log_first_step:
        :param test_only:
        :param disabled:
        :param reference_boundary:
        :param metric_config:
        """

        super().__init__()

        self.num_inference_steps = num_inference_steps
        self.num_frames = num_frames
        self.trim_start = trim_start
        self.frequency = frequency
        self.batch_size = batch_size
        self.disabled = disabled
        self.test_only = test_only
        self.log_first_step = log_first_step
        self.metric_config = metric_config
        self.temp_directory = "/tmp"
        self.reference_boundary = reference_boundary
        self.use_ema = use_ema

    def update_metrics_impl(self, trainer, pl_module, dataloader):

        metrics_torch = nn.ModuleList(get_metrics(
                self.metric_config)).to(pl_module.device)

        generator = torch.Generator(device=pl_module.device).manual_seed(trainer.global_rank)

        for batch in dataloader:

            class_labels = batch["loading_metadata"]["dataset_idx"].flatten().to(pl_module.device)

            prediction, target = pl_module.predict(batch, pl_module.device,
                                                 num_frames=self.num_frames,
                                                 num_inference_steps=self.num_inference_steps,
                                                 generator=generator,
                                                 reference_boundary=self.reference_boundary,
                                                 batch_dim=True)

            # first element is the input frame
            prediction = torch.from_numpy(prediction).to(pl_module.device)
            target = torch.from_numpy(target).to(pl_module.device)

            for metric in metrics_torch:

                metric.update(prediction[:,self.trim_start:], target[:,self.trim_start:], class_labels=class_labels)

        metric_dict = {"epoch": trainer.current_epoch}

        for metric in metrics_torch:

            metric_name = metric.__class__.__name__

            metric_value = metric.compute()

            if isinstance(metric_value, tuple):
                metric_dict[f"{metric_name}_mean"] = metric_value[0]
                metric_dict[f"{metric_name}_std"] = metric_value[1]

            else:
                metric_dict[f"{metric_name}"] = metric_value

            if trainer.global_rank == 0:
                if hasattr(trainer, "test_config_debug"):
                    logdir = trainer.test_config_debug["runtime"]["logdir"]
                else:
                    logdir = trainer.logger.experiment.config["runtime"]["logdir"]
                #logdir = trainer.config["runtime"]["logdir"]
                metric.save(logdir + '/metrics/', f"{metric_name}_{trainer.current_epoch}.csv")

        self.log_metrics(trainer, metric_dict)

    def update_metrics(self, trainer : Trainer, pl_module : LightningModule, dataloader,
                       use_ema: bool = False):

        optimizers_ema = [optimizer for optimizer in trainer.optimizers
                          if isinstance(optimizer, EMAOptimizer)]

        if use_ema and len(optimizers_ema) > 0:

            ema: EMAOptimizer = optimizers_ema[0]

            with ema.swap_ema_weights():
                print("Swapping to EMA weights..")
                self.update_metrics_impl(trainer, pl_module, dataloader)

        else:
                self.update_metrics_impl(trainer, pl_module, dataloader)

    @rank_zero_only
    def log_metrics(self, trainer: Trainer, metric_dict: Dict[str, Any]):
        trainer.logger.experiment.log(metric_dict)

    def check_frequency(self, check_idx):
        # check_idx + 1 since current_idx starts with 0
        return ((check_idx + 1) % self.frequency) == 0 or (check_idx == 0 and self.log_first_step)

    def on_test_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if not self.disabled:

            dataloader = trainer.test_dataloaders

            self.update_metrics(trainer, pl_module, dataloader, use_ema=self.use_ema)

    def on_train_epoch_end(self, trainer, pl_module):

        if ((not self.disabled) and (not self.test_only) and
                self.check_frequency(pl_module.current_epoch)):

            if (dataloader := trainer.test_dataloaders) is None:
                trainer.test_loop.setup_data()
                dataloader = trainer.test_dataloaders

            self.update_metrics(trainer, pl_module, dataloader, use_ema=self.use_ema)