"""Seed every RNG and force deterministic kernels, for reproducible runs."""

import os
import random
 
import numpy as np
import torch
 
 
def set_determinism(seed: int, warn_only: bool = True) -> None:
    """Make everything downstream of this call reproducible for a given seed."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ["PYTHONHASHSEED"] = str(seed)
 
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
 
    torch.use_deterministic_algorithms(True, warn_only=warn_only)
 
 
def seed_worker(worker_id: int) -> None:
    """DataLoader worker_init_fn for reproducible multi-worker loading."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
