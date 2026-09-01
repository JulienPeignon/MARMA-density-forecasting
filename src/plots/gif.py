"""Animated predictive densities: frame rendering and GIF assembly."""

import glob
import os
import shutil

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from src.utils.setup_logger import setup_logger

logger = setup_logger()


def generate_moving_density_gif(
    predictive_density,
    grid_x,
    grid_y,
    quantiles,
    plot_predictive_density_given_yt,
    y_start,
    y_end,
    step=5,
    gif_path="moving_density.gif",
    total_duration=2.0,
    abs_thresh=1e-4,
):
    """Animate the predictive density as the conditioning value y_T moves.

    ``plot_predictive_density_given_yt`` renders one frame; ``step`` thins the
    conditioning grid and ``abs_thresh`` trims the x-range.
    """
    gif_parent = os.path.dirname(gif_path)
    frame_dir = os.path.join(gif_parent, "frames_gif")
    os.makedirs(frame_dir, exist_ok=True)

    max_frames = int(total_duration / 0.02)
    n_points = y_end - y_start + 1
    step = max(step, n_points // max_frames)
    indices = range(y_start, y_end + 1, step)

    xlim_min, xlim_max = grid_y[-1], grid_y[0]
    ylim_max = 0.0
    for y_T in indices:
        pdf_hat = predictive_density[y_T, :]
        mask = np.where(pdf_hat > abs_thresh)[0]
        if len(mask) > 0:
            xlim_min = min(xlim_min, grid_y[mask[0]])
            xlim_max = max(xlim_max, grid_y[mask[-1]])
            ylim_max = max(ylim_max, pdf_hat[mask].max())

    for i, y_T in enumerate(tqdm(indices, desc="Generating frames")):
        pdf_hat = predictive_density[y_T, :]
        fig = plot_predictive_density_given_yt(pdf_hat, grid_x[y_T], grid_y, quantiles)
        ax = fig.axes[0]
        ax.set_xlim(xlim_min, xlim_max)
        ax.set_ylim(0, ylim_max * 1.05)
        frame_path = os.path.join(frame_dir, f"frame-{i + 1:04d}.png")
        fig.savefig(
            frame_path, dpi=72, bbox_inches="tight", pil_kwargs={"compress_level": 9}
        )
        plt.close(fig)

    frame_files = sorted(glob.glob(os.path.join(frame_dir, "frame-*.png")))
    pp_sequence = frame_files + frame_files[-2:0:-1]

    pp_dir = os.path.join(gif_parent, "frames_gif_pp")
    os.makedirs(pp_dir, exist_ok=True)
    for j, src in enumerate(pp_sequence):
        shutil.copy2(src, os.path.join(pp_dir, f"pp-{j + 1:04d}.png"))

    logger.info(f"Ping-pong frames saved in {pp_dir} ({len(pp_sequence)} frames)")

    # Build GIF from ping-pong sequence
    logger.info("Creating GIF...")
    per_frame_ms = 1000.0 * float(total_duration) / max(1, len(pp_sequence))
    per_frame_ms = max(per_frame_ms, 20)

    with imageio.get_writer(gif_path, mode="I", duration=per_frame_ms) as writer:
        for filename in pp_sequence:
            image = imageio.imread(filename)
            writer.append_data(image)

    frames_gif_path = os.path.join(frame_dir, os.path.basename(gif_path))
    shutil.copy2(gif_path, frames_gif_path)

    logger.info(f"Saved GIF to {gif_path}")
    logger.info(f"Saved GIF copy to {frames_gif_path}")
    logger.info(f"Frames saved in {frame_dir}")
