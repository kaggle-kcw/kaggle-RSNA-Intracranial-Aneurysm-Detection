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

class EffnetAneurysmClassifier(nn.Module):
    """
    Inputs:
        x: float tensor of shape [B, D, H, W] in [0,1]  (from your dataset)
    Behavior:
        - Builds 3-channel 2D input using axial/coronal/sagittal MIPs
        - Runs through a timm EfficientNet backbone
        - Returns either probabilities in [0,1] (default) or raw logits
    """
    def __init__(self, model_name: str = "efficientnet_b0", num_classes: int = 15,
                 pretrained: bool = True, return_logits: bool = False):
        super().__init__()
        self.return_logits = return_logits

        # Create backbone
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0, in_chans=3)
        feat_dim = self.backbone.num_features if hasattr(self.backbone, "num_features") else self.backbone.num_classes

        # Classification head
        self.head = nn.Linear(feat_dim, num_classes)

        # ImageNet normalization (expected by timm EfficientNet)
        # mean/std as buffers so they'll move with .to(device)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1), persistent=False)
        self.register_buffer("std",  torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1),  persistent=False)

    @staticmethod
    def _mip(x: torch.Tensor, dim: int) -> torch.Tensor:
        # maximum-intensity projection along dimension 'dim'
        return x.max(dim=dim).values

    def _make_mip_rgb(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, D, H, W] -> RGB-like [B, 3, H, W]
        R = axial   MIP (over depth D)      -> shape [B, H, W]
        G = coronal MIP (over width W)      -> shape [B, D, H] -> need to resize to [B, H, W]
        H = sagittal MIP (over height H)    -> shape [B, D, W] -> need to resize to [B, H, W]
        We'll resize coronal/sagittal to HxW with bilinear for a consistent 2D input.
        """
        B, D, H, W = x.shape
        # Axial (over D) -> [B, H, W]
        axial = self._mip(x, dim=1)

        # Coronal (over W): MIP along width gives [B, D, H], permute -> [B, 1, D, H], then resize to [H, W]
        coronal = self._mip(x, dim=3)                    # [B, D, H]
        coronal = coronal.unsqueeze(1)                   # [B, 1, D, H]
        coronal = F.interpolate(coronal, size=(H, W), mode="bilinear", align_corners=False)
        coronal = coronal.squeeze(1)                     # [B, H, W]

        # Sagittal (over H): MIP along height gives [B, D, W], permute -> [B, 1, D, W], resize to [H, W]
        sagittal = self._mip(x, dim=2)                   # [B, D, W]
        sagittal = sagittal.unsqueeze(1)                  # [B, 1, D, W]
        sagittal = F.interpolate(sagittal, size=(H, W), mode="bilinear", align_corners=False)
        sagittal = sagittal.squeeze(1)                    # [B, H, W]

        rgb = torch.stack([axial, coronal, sagittal], dim=1)  # [B, 3, H, W]
        return rgb

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expect x in [B, D, H, W], values in [0,1]
        assert x.ndim == 4, f"Expected [B, D, H, W], got {x.shape}"
        rgb = self._make_mip_rgb(x)              # [B, 3, H, W]
        rgb = (rgb - self.mean) / self.std       # ImageNet normalization

        feats = self.backbone(rgb)               # [B, feat_dim]
        logits = self.head(feats)                # [B, 15]
        if self.return_logits:
            return logits
        return torch.sigmoid(logits)             # probabilities in [0,1]

