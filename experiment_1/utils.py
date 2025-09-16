import random
import numpy as np
import torch

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False  # allow autotune for 3D volumes
    torch.backends.cudnn.benchmark = True