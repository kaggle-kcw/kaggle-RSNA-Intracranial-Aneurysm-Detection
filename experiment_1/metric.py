import pandas as pd
import numpy as np
import os, math, random, time
from tqdm import tqdm
import pydicom
import cv2
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from scipy import ndimage
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from torch.utils.data import Dataset, DataLoader, Subset
from typing import Callable, Optional, Tuple, Sequence, Dict, Any
from sklearn.metrics import roc_auc_score
from tqdm.auto import tqdm
from sklearn.model_selection import StratifiedKFold
@torch.no_grad()
def auc_per_label(y_true, y_prob):
    """Return list[float or np.nan] of AUROC per label; handles all-0/1 columns."""
    y_true = y_true.astype(np.float32)
    y_prob = y_prob.astype(np.float32)
    L = y_true.shape[1]
    aucs = []
    for i in range(L):
        col = y_true[:, i]
        if np.unique(col).size < 2:
            aucs.append(np.nan)   # undefined; will be ignored in means
            continue
        aucs.append(roc_auc_score(col, y_prob[:, i]))
    return aucs

def rsna_final_score(aucs, ap_index=0, other_indices=None):
    """
    Implements the RSNA 'Final Score' = 0.5 * (AUC_AP + mean(other AUCs)).
    - aucs: list of floats (per label)
    - ap_index: index for 'Aneurysm Present'
    - other_indices: which labels to average besides AP (defaults to all except AP)
    NaNs are ignored when averaging.
    """
    aucs = np.array(aucs, dtype=np.float32)
    ap = aucs[ap_index]
    if other_indices is None:
        other_indices = [i for i in range(len(aucs)) if i != ap_index]
    others = aucs[other_indices]
    others = others[~np.isnan(others)]
    if len(others) == 0 or np.isnan(ap):
        return np.nan
    return 0.5 * (float(ap) + float(others.mean()))

class AverageMeter:
    def __init__(self): 
        self.reset()

    def reset(self): 
        self.sum = 0.0
        self.n = 0

    def update(self, val, k=1): 
        self.sum += float(val) * k
        self.n += k
        
    @property
    def avg(self): return self.sum / max(1, self.n)