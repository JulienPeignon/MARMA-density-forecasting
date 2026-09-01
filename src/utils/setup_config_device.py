"""Device selection, thread limits and seeding."""

import os
import random

import numpy as np
import torch

from src.utils.setup_logger import setup_logger


def setup_device():
    """Return the best available device: "cuda", "mps" or "cpu"."""
    logger = setup_logger()
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    torch.set_default_dtype(torch.float32)

    if device == "cuda":
        # auto-tune cuDNN kernels
        torch.backends.cudnn.benchmark = True

        # TF32 on Ampere and later
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        torch.autograd.set_detect_anomaly(False)
        torch.autograd.profiler.profile(False)
        torch.autograd.profiler.emit_nvtx(False)

    logger.info(f"Using {device} device")
    return device


def get_allowed_cpu_count() -> int:
    """Return the number of usable CPU cores."""
    logger = setup_logger()
    try:
        nbr_cpu = len(os.sched_getaffinity(0))
    except AttributeError:
        nbr_cpu = os.cpu_count() or 1
    logger.info(f"Using {nbr_cpu} CPUs")
    return nbr_cpu


def setup_config_device(cpu_count: int) -> int:
    """Set the PyTorch thread count from ``cpu_count``."""
    logger = setup_logger()

    n_process = max(1, int(3 * cpu_count // 5))

    torch.set_num_threads(n_process)
    logger.info(f"torch set up to use {n_process} processes")

    return n_process


def set_seed(seed: int = 42) -> None:
    """Seed Python, NumPy and PyTorch."""
    logger = setup_logger()

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if hasattr(torch, "use_deterministic_algorithms"):
        torch.use_deterministic_algorithms(True)

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    logger.info(f"Seed set to {seed}")
