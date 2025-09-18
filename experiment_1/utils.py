import random
import numpy as np
import torch

LABEL_COLS = [
    'Left Infraclinoid Internal Carotid Artery',
    'Right Infraclinoid Internal Carotid Artery',
    'Left Supraclinoid Internal Carotid Artery',
    'Right Supraclinoid Internal Carotid Artery',
    'Left Middle Cerebral Artery', 
    'Right Middle Cerebral Artery',
    'Anterior Communicating Artery', 
    'Left Anterior Cerebral Artery',
    'Right Anterior Cerebral Artery',
    'Left Posterior Communicating Artery',
    'Right Posterior Communicating Artery', 
    'Basilar Tip',
    'Other Posterior Circulation',
    'Aneurysm Present'
]

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False  # allow autotune for 3D volumes
    torch.backends.cudnn.benchmark = True