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

def get_auc_per_location_list(y_true, y_prob):
    y_true = y_true.astype(np.float32)
    y_prob = y_prob.astype(np.float32)
    L = y_true.shape[1]
    auc_per_location_list = []
    for i in range(L):
        curr_y_true_by_location = y_true[:, i]
        curr_y_prob_by_location = y_prob[:, i]
        
        auc_per_location_list.append(roc_auc_score(curr_y_true_by_location, curr_y_prob_by_location))
    return auc_per_location_list

def get_final_score(y_true, y_prob, auc_per_location_list):

    average_location_roc_auc_score = 0
    for i in range(len(auc_per_location_list)):
        average_location_roc_auc_score += auc_per_location_list[i]
    average_location_roc_auc_score /= len(auc_per_location_list)

    true_aneurysm_present_list = (np.any(y_true == 1, axis=1)).astype(int)
    prob_aneurysm_present_list = (np.any(y_prob > 0.5, axis=1)).astype(int)
    aneurysm_present_roc_auc_score = roc_auc_score(true_aneurysm_present_list, prob_aneurysm_present_list)
    
    final_score = 0.5 * (average_location_roc_auc_score + aneurysm_present_roc_auc_score)

    return final_score

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