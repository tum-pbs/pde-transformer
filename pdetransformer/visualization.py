import os
import copy
import numpy as np
import matplotlib.pyplot as plt
import torch

from mpl_toolkits.axes_grid1.axes_rgb import RGBAxes, make_rgb_axes
from mpl_toolkits.axes_grid1 import ImageGrid
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

from vape4d import render
import seaborn as sns

cmap_nonlinear = sns.color_palette("icefire", as_cmap=True)


def triangle_wave(x, p):
    return 2 * np.abs(x / p - np.floor(x / p + 0.5))


def zigzag_alpha(cmap, min_alpha=0.2):
    """changes the alpha channel of a colormap to be linear (0->0, 1->1)

    Args:
        cmap (Colormap): colormap

    Returns:a
        Colormap: new colormap
    """
    if isinstance(cmap, ListedColormap):
        colors = copy.deepcopy(cmap.colors)
        for i, a in enumerate(colors):
            a.append((triangle_wave(i / (cmap.N - 1), 0.5) * (1 - min_alpha)) + min_alpha)
        return ListedColormap(colors, cmap.name)
    elif isinstance(cmap, LinearSegmentedColormap):
        segmentdata = copy.deepcopy(cmap._segmentdata)
        segmentdata["alpha"] = np.array([
            [0.0, 0.0, 0.0],
            [0.25, 1.0, 1.0],
            [0.5, 0.0, 0.0],
            [0.75, 1.0, 1.0],
            [1.0, 0.0, 0.0]]
        )
        return LinearSegmentedColormap(cmap.name, segmentdata)
    else:
        raise TypeError(
            "cmap must be either a ListedColormap or a LinearSegmentedColormap"
        )


def symmetric_min_max(arr):
    vmin = arr.min().item()
    vmax = arr.max().item()
    absmax = max(abs(vmin), abs(vmax))
    return -absmax, absmax


def render_vape_3d(volume, cmap, height, width, time=0.1, vmin=None, vmax=None):
    if vmin is None:
        vmin, _ = symmetric_min_max(volume)
    if vmax is None:
        _, vmax = symmetric_min_max(volume)

    img = render(
        np.array(volume),
        (cmap),
        height=height,
        width=width,
        time=time,
        background=(255, 255, 255, 255),
        distance_scale=10,
        vmin=vmin,
        vmax=vmax,
    )
    # gamma correction
    return np.power(img / 255.0, 2.4)


def render_trajectory(
        data: list,
        dimension: int,
        sim_id: int,
        title: str,
        time_steps: int,
        steps_plot: int | list[int],
        vmin: float = None,
        vmax: float = None,
):
    if isinstance(steps_plot, list):
        time_steps = steps_plot
    else:
        time_steps = list(range(0, time_steps, time_steps // steps_plot))

    if dimension == 2:
        modes = []
        for c in range(data[0].shape[0]):
            modes += ["channel%d" % c]

    elif dimension == 3:
        modes = ["x-mean", "y-mean", "z-mean", "x-slice", "y-slice", "z-slice"]

        if data[0].shape[0] == 1:
            modes += ["vape"]
        else:
            for c in range(data[0].shape[0]):
                modes += ["vape-c%d" % c]
    else:
        raise ValueError("Invalid dimension: %d" % dimension)

    title = f"{title} - Simulation {sim_id}"
    print("Rendering %s..." % title)

    fig = plt.figure(figsize=(1.5 * len(time_steps), 1.5 * len(modes)), dpi=200)
    fig.text(0.5, 0.90, title, fontsize=12, ha="center")

    grid = ImageGrid(
        fig,
        111,
        nrows_ncols=(len(modes), len(time_steps)),
        axes_pad=0.03,
        share_all=True,
        cbar_location="right",
        cbar_mode="single",
        cbar_size="3%",
        cbar_pad=0.1,
    )

    if vmin is None:
        vmin = float("inf")
        for t in range(len(time_steps)):
            loaded = data[time_steps[t]]
            vmin = min(vmin, np.min(loaded))

    if vmax is None:
        vmax = float("-inf")
        for t in range(len(time_steps)):
            loaded = data[time_steps[t]]
            vmax = max(vmax, np.max(loaded))

    for t in range(len(time_steps)):
        loaded = data[time_steps[t]]
        if dimension == 2:
            if isinstance(loaded, np.ndarray):
                loaded = loaded.transpose(1, 2, 0)
            else:
                loaded = torch.permute(loaded, (1, 2, 0))
        else:

            if isinstance(loaded, np.ndarray):
                loaded = loaded.transpose(1, 2, 3, 0)
            else:
                loaded = torch.permute(loaded, (1, 2, 3, 0))

        for m in range(len(modes)):
            if modes[m] == "x-mean":
                img = np.mean(loaded, axis=0)
            elif modes[m] == "y-mean":
                img = np.mean(loaded, axis=1)
            elif modes[m] == "z-mean":
                img = np.mean(loaded, axis=2)
            elif modes[m] == "x-slice":
                img = loaded[loaded.shape[0] // 2]
            elif modes[m] == "y-slice":
                img = loaded[:, loaded.shape[1] // 2]
            elif modes[m] == "z-slice":
                img = loaded[:, :, loaded.shape[2] // 2]
            elif "vape" in modes[m]:

                l = loaded.transpose(3, 0, 1, 2)

                img = render_vape_3d(l if modes[m] == "vape" else l[int(modes[m][-1]):int(modes[m][-1]) + 1],
                                     zigzag_alpha(cmap_nonlinear, 0.1),
                                     height=loaded.shape[0],
                                     width=loaded.shape[1],
                                     time=float(t) / len(time_steps),
                                     vmin=vmin,
                                     vmax=vmax)

            elif "channel" in modes[m]:
                img = loaded[:, :, int(modes[m][-1])]
            else:
                raise ValueError("Unknown mode: %s" % modes[m])

            ax = grid[m * len(time_steps) + t]
            if "vape" not in modes[m]:
                if dimension == 3 and img.shape[2] > 1:
                    img = img[:, :, 0]
                ax.imshow(img, cmap="viridis", vmin=vmin, vmax=vmax)
            else:
                ax.imshow(img)

            ax.set_ylabel(modes[m])
            ax.set_xlabel("t = %d" % time_steps[t])

            ax.set_xticks([])
            ax.set_yticks([])

    grid.cbar_axes[0].colorbar(grid[0].get_images()[0])

    return fig
