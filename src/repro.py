import os
import random
 
import numpy as np
import torch
 
 
def set_determinism(seed: int, warn_only: bool = True) -> None:
    """Make everything downstream of this call reproducible for a given seed.
 
    Call it FIRST -- before building the model, the optimizer, or the
    DataLoader -- because it fixes the RNG state those objects draw from.
    """
    # This env var must be set before the first CUDA context is created for
    # CUBLAS determinism to take effect; setting it here covers the common
    # case where training starts after this call.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ["PYTHONHASHSEED"] = str(seed)
 
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)          # seeds CPU and all CUDA devices
    torch.cuda.manual_seed_all(seed)  # explicit, in case of multi-GPU
 
    # The two flags that were missing before: force cuDNN to pick a fixed
    # algorithm rather than benchmarking for the fastest one each run.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
 
    # Ask Torch to use deterministic kernels everywhere it can; warn (don't
    # crash) if some op has no deterministic version.
    torch.use_deterministic_algorithms(True, warn_only=warn_only)
 
 
def seed_worker(worker_id: int) -> None:
    """DataLoader worker_init_fn for reproducible multi-worker loading.
 
    Only needed when num_workers > 0. Derives each worker's seed from the
    torch base seed so the whole pipeline stays deterministic.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)