"""
This module provides functions for creating videos from datasets.
"""

# non-local package imports
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

# local class imports
from pbdl.loader import Dataloader


# TODO what if there are more that 2 spatial dim?
def create_preview_video(
    dataset_name,
    path="preview.mp4",
    channels=(0, 1),
    fps=30,
    sec=5,
    res_width=512,  # maintain width-height ratio
    cmap=plt.get_cmap("twilight"),
):

    # TODO what if fps * sec too large?
    loader = Dataloader(
        dataset_name,
        sel_sims=[0],
        disable_progress=True,
    )

    sim = loader.get_sim_raw(0)

    # skip animation if not enough frames
    if len(sim) < fps * sec:
        return

    frames = sim[: fps * sec]

    # add a second spatial dimension
    if loader.dataset.num_spatial_dim == 1:
        frames_ext = np.expand_dims(frames, axis=3)
        frames_ext = np.repeat(frames_ext, frames.shape[-1] // 2, axis=3)
        frames = frames_ext

    # take vector norm
    if len(channels) > 1:
        frames = np.sqrt(np.sum(frames[:, channels, ...] ** 2, axis=2))
    else:
        frames = frames[:, channels[0], ...]

    height = frames.shape[1]
    width = frames.shape[2]

    high_res_size = (res_width, int((height / width) * res_width))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video = cv2.VideoWriter("tmp.mp4", fourcc, float(fps), high_res_size)

    # normalize
    min, max = frames.min(), frames.max()
    frames = (frames - min) / (max - min)
    frames = cmap(frames)
    frames = (frames[:, :, :, :3] * 255).astype(np.uint8)

    for frame_count in range(len(frames)):
        high_res_frame = cv2.resize(
            frames[frame_count],
            high_res_size,
            interpolation=cv2.INTER_NEAREST,
        )

        video.write(high_res_frame)

    video.release()

    # TODO quick fix to convert to browser compatible video codec
    os.system(f"ffmpeg -y -i tmp.mp4 -vcodec libx264 -f mp4 {path}")
    os.remove("tmp.mp4")
