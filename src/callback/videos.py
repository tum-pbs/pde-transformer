from typing import List, Optional, Union, Iterable

from lightning import Callback, Trainer, LightningModule
from lightning.fabric.utilities import rank_zero_only
from lightning_utilities.core.rank_zero import rank_zero_warn
from omegaconf import DictConfig, OmegaConf

import glob
import numpy as np
import os
import wandb
from PIL import Image
from tqdm import tqdm

import torch
import subprocess

import seaborn as sns
from matplotlib import pyplot as plt
from src.visualization import render_trajectory, render_vape_3d

from src.data.simulations_apebench.render import zigzag_alpha
from src.data.multi_module import get_subdatasets_from_dataloader
from src.utils import instantiate_from_config, get_pipeline

plt.ioff()

def gen_video_frame_2d(img, reference, normalization, std):

    AE_NORM_2D = 0.3

    norm_min, norm_max = normalization

    if norm_min is None or norm_max is None:
        norm_min = img.min()
        norm_max = img.max()

    img = (img - norm_min) / (norm_max - norm_min)
    reference = (reference - norm_min) / (norm_max - norm_min)
    diff = img - reference
    diff = diff / (std * AE_NORM_2D) + 0.5

    plt_cm = plt.get_cmap('twilight')
    img = (plt_cm(img)[:, :, :3] * 255).astype(np.uint8)
    reference = (plt_cm(reference)[:, :, :3] * 255).astype(np.uint8)
    diff_cm = plt.get_cmap('bwr')
    diff = (diff_cm(diff)[:, :, :3] * 255).astype(np.uint8)

    frame = np.concatenate([img, reference, diff], axis=1)

    return frame


def save_video(img, full_simulation, dataset_name, folder, savename, gen_video_frame_fn):

    if not os.path.exists(folder):
        os.makedirs(folder)

    full_simulation = full_simulation.cpu().numpy()
    img = img.cpu().numpy()

    std = np.std(full_simulation[0])

    if dataset_name == "mhd1024":
        vmin, vmax = -0.7, 0.7
    elif dataset_name == "isotropic1024coarse":
        vmin, vmax = -1.2, 1.2
    elif dataset_name in ["adv", "diff", "adv_diff", "disp", "hyp", "burgers", "kdv"]:
        vmin = -0.5
        vmax = 0.5
    elif dataset_name in ["fisher", "gs_alpha_test", "gs_beta_test", "gs_gamma_test", "gs_delta", "gs_epsilon_test",
                          "gs_theta", "gs_iota", "gs_kappa"]:
        vmin = 0.0
        vmax = 1.0
    else:
        # automatically determined via min and max of data
        vmin, vmax = None, None

    normalization = (vmin, vmax)

    for i in tqdm(range(len(img))):

        img_frame = img[i][0]
        reference = full_simulation[i][0]

        frame = gen_video_frame_fn(img_frame, reference, normalization, std)

        pil_img = Image.fromarray(frame)
        pil_img.save(folder + "/pic%02d.png" % i)

    current_dir = os.getcwd()
    os.chdir(folder)
    subprocess.call([
        'ffmpeg', '-y', '-framerate', '8', '-i', 'pic%02d.png', '-r', '30', '-pix_fmt', 'yuv420p',
        f'{savename}.mp4'
    ])
    for file_name in glob.glob("*.png"):
        os.remove(file_name)

    os.chdir(current_dir)

class VideoLogger(Callback):
    def __init__(self, frequency: int, pipelines: Union[DictConfig, OmegaConf], max_videos: int = 4,
                 num_frames: int = 10, clamp=True, increase_log_steps=True,
                 rescale=True, disabled=False, log_first_step=True, test_only=False,
                 log_images_kwargs=None, prepare_plots: Optional[DictConfig]=None):
        super().__init__()

        self.pipelines = OmegaConf.to_container(pipelines)
        self.rescale = rescale
        self.frequency = frequency
        self.max_videos = max_videos
        self.num_frames = num_frames
        self.test_only = test_only

        if not prepare_plots is None:
            self.prepare_plots = instantiate_from_config(prepare_plots)
        else:
            self.prepare_plots = torch.nn.Identity()

        self.log_steps = [2 ** n for n in range(int(np.log2(self.frequency)) + 1)]
        if not increase_log_steps:
            self.log_steps = [self.frequency]

        self.clamp = clamp
        self.disabled = disabled

        self.log_images_kwargs = log_images_kwargs if log_images_kwargs else {}
        self.log_first_step = log_first_step

    @rank_zero_only
    def log_video(self, trainer, pl_module):

        current_epoch = trainer.current_epoch

        pipelines = {key: get_pipeline(value, pl_module)
                     for key, value in self.pipelines.items()}

        is_train = pl_module.training

        if is_train:
            pl_module.eval()

        for pipeline in pipelines.keys():

            generator = torch.Generator(device=pl_module.device).manual_seed(trainer.global_rank)

            frames_list = []

            for batch in iter(trainer.val_dataloaders):

                frames_list.extend(list(pl_module.get_input(batch)[0]))

                if len(frames_list) > self.max_videos:
                    break

            frames_list = frames_list[:self.max_videos]
            frames_list = torch.stack(frames_list)

            videos = pipelines[pipeline](data=frames_list, num_frames=self.num_frames,
                                         generator=generator,
                                         output_type='numpy', return_dict=True).videos

            videos = torch.from_numpy(videos)
            logdir = trainer.logger.experiment.config["runtime"]["logdir"]
            for c, video in enumerate(videos):
                save_video(video, logdir + '/videos', f"pipeline-{pipeline}_e{current_epoch}_i{c}")

            # videos = [[self.prepare_plots(im).detach().cpu() for im in list(frames)] for frames in list(videos)]
            # videos = [np.array(vid) for vid in videos]

            # self.log_local(logdir, pipeline, list(images), current_epoch)

            trainer.logger.experiment.log({
                pipeline: [wandb.Video(logdir + f'/videos/pipeline-{pipeline}_e{current_epoch}_i{i}.mp4', format='mp4',)
                           for i in range(videos.shape[0])]
            })

        if is_train:
            pl_module.train()

    def check_frequency(self, check_idx):
        if ((check_idx + 1) % self.frequency) == 0 or (check_idx in self.log_steps) or (check_idx == 0 and self.log_first_step):
            try:
                self.log_steps.pop(0)
            except IndexError as e:
                print(e)
                pass
            return True
        return False

    def log_dataset(self, trainer: Trainer, pl_module: LightningModule):

        images_list = []

        # only visualize first dataset
        for batch in iter(trainer.train_dataloader):

            images_list.extend(list(pl_module.get_input(batch)))

            if len(images_list) > self.max_images:
                break

        images = [self.prepare_plots(im).detach().cpu() for im in images_list[:self.max_images]]

        logdir = trainer.logger.experiment.config["runtime"]["logdir"]
        self.log_local(logdir, 'data', images[:self.max_images], 0)

        trainer.logger.experiment.log({
            'data': [wandb.Image(im, caption=str(k)) for k, im in enumerate(images)]
        })

    def setup_test_dataloader(self, trainer):
        test_dataloader = trainer.test_dataloaders
        if test_dataloader is None:
            trainer.test_loop.setup_data()
            _ = trainer.test_dataloaders

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:

        self.setup_test_dataloader(trainer)
        if ((not self.disabled) and (not self.test_only) and self.log_first_step and
                pl_module.current_epoch == 0):
            self.log_video(trainer, pl_module)

    def on_train_epoch_end(self, trainer, pl_module):

        self.setup_test_dataloader(trainer)
        if ((not self.disabled) and (not self.test_only) and (pl_module.current_epoch > 0) and
                self.check_frequency(pl_module.current_epoch) and (self.max_videos > 0)):
            self.log_video(trainer, pl_module)



class MultiTaskVideoLogger(VideoLogger):

    data_type = '2d'

    @rank_zero_only
    def log_video(self, trainer, pl_module):

        current_epoch = trainer.current_epoch

        pipelines = {key: get_pipeline(value, pl_module)
                     for key, value in self.pipelines.items()}

        is_train = pl_module.training

        if is_train:
            pl_module.eval()

        test_dataloader = trainer.test_dataloaders
        subsets, dataset = get_subdatasets_from_dataloader(test_dataloader)

        for pipeline in pipelines.keys():

            generator = torch.Generator(device=pl_module.device).manual_seed(trainer.global_rank)

            # TODO introduce a batch size used for inference
            for d in range(len(subsets)):

                frames_list = []
                class_labels = []
                names_list = []
                full_simulation_list = []


                sample = subsets[d][0] # only first sample
                # simulation_name = sample["physical_metadata"]["PDE"]
                simulation_name = subsets[d].dataset.dset_name

                sample = dataset.process_sample(sample)

                frames, target, labels = pl_module.get_input(sample, batch_dim=False)

                names_list.append(simulation_name)
                frames_list.append(frames[0])
                class_labels.append(labels)
                full_simulation_list.append(torch.concatenate([frames, target[0]], dim=0))

                frames_list = torch.stack(frames_list)
                full_simulation_list = torch.stack(full_simulation_list)

                class_labels = torch.tensor(class_labels, device=pl_module.device, dtype=torch.long)

                videos = pipelines[pipeline](
                                             data=frames_list, num_frames=self.num_frames,
                                             generator=generator,
                                             output_type='numpy', return_dict=True,
                                             class_labels=class_labels
                                            ).videos

                videos = torch.from_numpy(videos)
                logdir = trainer.logger.experiment.config["runtime"]["logdir"]

                video_frame_fn = None
                if self.data_type == '2d':
                    video_frame_fn = gen_video_frame_2d
                elif self.data_type == '3d':
                    video_frame_fn = gen_video_frame_3d

                for c, video in enumerate(videos):

                    full_simulation = full_simulation_list[c]
                    savedir = logdir + '/videos/' + names_list[c]
                    savename = f'pipeline-{pipeline}_e{current_epoch}'
                    save_video(video, full_simulation, names_list[c], savedir, savename, video_frame_fn)
                    savename = f"{savedir}/{savename}.mp4"
                    trainer.logger.experiment.log({
                        f'{names_list[c]}_{pipeline}': wandb.Video(savename, format='mp4',)
                    })

        if is_train:
            pl_module.train()

    def on_test_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.setup_test_dataloader(trainer)
        if not self.disabled:
            self.log_video(trainer, pl_module)

class MultiTaskVideoLoggerCustom(Callback):

    data_type = '2d'

    def __init__(self, frequency: int, max_videos: int = 4,
                 num_frames: int = 10, num_inference_steps: int = 100, clamp=True, increase_log_steps=True,
                 rescale=True, disabled=False, log_first_step=False, test_only=False,
                 reference_boundary=False, trim:int=0,
                 log_images_kwargs=None, prepare_plots: Optional[DictConfig]=None):

        super().__init__()

        self.rescale = rescale
        self.frequency = frequency
        self.max_videos = max_videos
        self.num_frames = num_frames
        self.test_only = test_only
        self.reference_boundary = reference_boundary
        self.trim = trim
        self.steps_plot = 4

        if not prepare_plots is None:
            self.prepare_plots = instantiate_from_config(prepare_plots)
        else:
            self.prepare_plots = torch.nn.Identity()

        self.num_inference_steps = num_inference_steps
        self.log_steps = [2 ** n for n in range(int(np.log2(self.frequency)) + 1)]
        if not increase_log_steps:
            self.log_steps = [self.frequency]

        self.clamp = clamp
        self.disabled = disabled

        self.log_images_kwargs = log_images_kwargs if log_images_kwargs else {}
        self.log_first_step = log_first_step

    def setup_test_dataloader(self, trainer):
        test_dataloader = trainer.test_dataloaders
        if test_dataloader is None:
            trainer.test_loop.setup_data()
            _ = trainer.test_dataloaders

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:

        self.setup_test_dataloader(trainer)
        if ((not self.disabled) and (not self.test_only) and
                pl_module.current_epoch == 0 and self.log_first_step):
            self.log_video(trainer, pl_module)

    def on_train_epoch_end(self, trainer, pl_module):

        self.setup_test_dataloader(trainer)
        if ((not self.disabled) and (not self.test_only) and (pl_module.current_epoch > 0) and
                self.check_frequency(pl_module.current_epoch) and (self.max_videos > 0)):
            self.log_video(trainer, pl_module)

    def check_frequency(self, check_idx):
        if ((check_idx + 1) % self.frequency) == 0 or (check_idx in self.log_steps) or (check_idx == 0 and self.log_first_step):
            try:
                self.log_steps.pop(0)
            except IndexError as e:
                print(e)
                pass
            return True
        return False

    @rank_zero_only
    def log_video(self, trainer, pl_module):

        try:

            current_epoch = trainer.current_epoch

            is_train = pl_module.training

            if is_train:
                pl_module.eval()

            test_dataloader = trainer.test_dataloaders
            subsets, dataset = get_subdatasets_from_dataloader(test_dataloader)

            logdir = trainer.logger.experiment.config["runtime"]["logdir"]

            video_frame_fn = None
            if self.data_type == '2d':
                video_frame_fn = gen_video_frame_2d
            elif self.data_type == '3d':
                video_frame_fn = gen_video_frame_3d


            generator = torch.Generator(device=pl_module.device).manual_seed(trainer.global_rank)

            for d in range(len(subsets)):
                sample = subsets[d][0] # only first sample
                # simulation_name = sample["physical_metadata"]["PDE"]
                simulation_name = subsets[d].dataset.dset_name

                batch = dataset.process_sample(sample, add_batch_dim=True)

                video, reference = pl_module.predict(batch, pl_module.device,
                                                     num_frames=self.num_frames,
                                                     num_inference_steps=self.num_inference_steps,
                                                     reference_boundary=self.reference_boundary,
                                                     generator=generator, trim=self.trim)

                video = torch.from_numpy(video)[0]
                reference = torch.from_numpy(reference)[0]

                savedir_video = logdir + '/videos/' + simulation_name
                savedir_picture = logdir + '/pictures/' + simulation_name
                savename = f'e{current_epoch}'
                save_video(video, reference, simulation_name, savedir_video, savename, video_frame_fn)

                savename_video = f"{savedir_video}/{savename}.mp4"
                if self.data_type == '3d':
                    save_3d_visualization(video, reference, simulation_name, savedir_picture,
                                          current_epoch=current_epoch,
                                          steps_plot=self.steps_plot)
                    savename_pred = f"{savedir_picture}/pred_e{current_epoch}.png"
                    savename_ref = f"{savedir_picture}/ref_e{current_epoch}.png"
                    trainer.logger.experiment.log({
                        f'{simulation_name}_pred': wandb.Image(savename_pred),
                        f'{simulation_name}_ref': wandb.Image(savename_ref),
                        f'{simulation_name}_video': wandb.Video(savename_video, format="mp4")
                    })
                else:
                    trainer.logger.experiment.log({
                        f'{simulation_name}': wandb.Video(savename_video, format='mp4'),
                    })

            if is_train:
                pl_module.train()

        except Exception as e:

            rank_zero_warn(f"Error while logging video: {e}")
            raise e

    def on_test_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.setup_test_dataloader(trainer)
        if not self.disabled:
            self.log_video(trainer, pl_module)

class MultiTaskVideoLogger3D(MultiTaskVideoLoggerCustom):

    data_type = '3d'

