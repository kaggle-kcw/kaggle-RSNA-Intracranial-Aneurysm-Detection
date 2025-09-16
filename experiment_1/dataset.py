import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from torch.utils.data import Dataset, DataLoader, Subset
from typing import Callable, Optional, Tuple, Sequence, Dict, Any
from sklearn.metrics import roc_auc_score
from tqdm.auto import tqdm
from sklearn.model_selection import StratifiedKFold
from data_preprocess import process_dicom_series_safe

class RSNAAneurysmDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        input_dir: str,
        target_shape: Tuple[int, int, int] = (32, 384, 384),  # (D,H,W)
        label_cols: Optional[Sequence[str]] = None,           # explicit multi-label columns (order respected)
    ):
        self.df = df.reset_index(drop=True).copy()
        self.input_dir = Path(input_dir)
        self.target_shape = target_shape
        self.label_cols = label_cols

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        series_instance_uid = str(row['SeriesInstanceUID'])
        series_path = self.input_dir / 'series' / series_instance_uid

        # Load volume: (D,H,W) uint8
        vol = process_dicom_series_safe(str(series_path), self.target_shape)  # (D,H,W) uint8
        vol_t = torch.from_numpy(vol)  # uint8 [D,H,W]
        vol_t = vol_t.float().div_(255.0) 

        # Labels
        if self.label_cols and all(c in row for c in self.label_cols):
            label_t = row[self.label_cols].values.astype(np.float32)
            label_t = torch.from_numpy(label_t)
        else:
            # Return a dummy tensor for labels if they don't exist (inference case)
            label_t = torch.empty(0)

        # Meta
        meta = {
            'series_instance_uid': series_instance_uid,
            'series_instance_uid_path': str(series_path)
        }

        return vol_t, label_t, meta

def collate(batch):
        vols, labels, metas = zip(*batch)
        vols = torch.stack(vols, dim=0)
        labels = torch.stack(labels, dim=0)
        return vols, labels, metas
