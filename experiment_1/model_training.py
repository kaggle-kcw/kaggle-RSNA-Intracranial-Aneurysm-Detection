#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import os, math, random, time
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


# In[2]:


input_dir = "/home/khor/kaggle_kcw/kaggle-RSNA-Intracranial-Aneurysm-Detection/input/"
train_df = pd.read_csv(f"{input_dir}/train.csv") 
train_localizers_df = pd.read_csv(f"{input_dir}/train_localizers.csv")


# In[7]:


class DICOMPreprocessorKaggle:
    def __init__(self, target_shape: Tuple[int, int, int] = (32, 384, 384)):
        self.target_depth, self.target_height, self.target_width = target_shape

    def load_dicom_series(self, series_path: str) -> Tuple[List[pydicom.Dataset], str]:
        series_path = Path(series_path)
        series_name = series_path.name

        dicom_files = []
        for root, _, files in os.walk(series_path):
            for file in files:
                if file.endswith('.dcm'):
                    dicom_files.append(os.path.join(root, file))

        datasets = []
        for filepath in dicom_files:
            ds = pydicom.dcmread(filepath, force=True)
            datasets.append(ds)

        return datasets, series_name

    def extract_slice_info(self, datasets: List[pydicom.Dataset]) -> List[Dict]:
        slice_info = []

        for index, ds in enumerate(datasets):
            info = {
                'dataset': ds,
                'index': index,
                'instance_number': getattr(ds, 'InstanceNumber', index),
            }

            position = getattr(ds, 'ImagePositionPatient', None)
            if position is not None and len(position) >= 3:
                info['z_position'] = float(position[2])
            else:
                info['z_position'] = float(info['instance_number'])

            slice_info.append(info)

        return slice_info

    def get_windowing_params(self, ds: pydicom.Dataset) -> Tuple[Optional[float], Optional[float]]:
        """
        Get windowing parameters based on modality
        """
        modality = getattr(ds, 'Modality', 'CT')

        if modality == 'CT':
            # For CT, apply CTA (angiography) settings
            return "CT", "CT"

        elif modality == 'MR':
            # For MR, skip windowing (statistical normalization only)
            return None, None
        else:
            # Unexpected modality (safety measure), using CTA windowing
            return None, None

    def apply_windowing_or_normalize(self, img: np.ndarray, center: Optional[float], width: Optional[float]) -> np.ndarray:
        """
        Apply windowing or statistical normalization
        """
        if center is not None and width is not None:
            # Windowing processing (for CT/CTA)
            # Applied windowing: [{img_min:.1f}, {img_max:.1f}] → [0, 255]")

            # Statistical normalization (for CT as well)
            # Normalize using 1-99 percentiles
            p1, p99 = np.percentile(img, [1, 99])
            p1, p99 = 0, 500

            if p99 > p1:
                # Applied statistical normalization: [{p1:.1f}, {p99:.1f}] → [0, 255]
                normalized = np.clip(img, p1, p99)
                normalized = (normalized - p1) / (p99 - p1)
                result = (normalized * 255).astype(np.uint8)

                return result
            else:
                # Fallback: min-max normalization
                # Applied min-max normalization: [{img_min:.1f}, {img_max:.1f}] → [0, 255]
                img_min, img_max = img.min(), img.max()
                if img_max > img_min:
                    normalized = (img - img_min) / (img_max - img_min)
                    result = (normalized * 255).astype(np.uint8)
                    return result
                else:
                    # If image has no variation
                    return np.zeros_like(img, dtype=np.uint8)
        else:
            # Statistical normalization (for MR)
            # Normalize using 1-99 percentiles
            # Applied statistical normalization: [{p1:.1f}, {p99:.1f}] → [0, 255]
            p1, p99 = np.percentile(img, [1, 99])

            if p99 > p1:
                normalized = np.clip(img, p1, p99)
                normalized = (normalized - p1) / (p99 - p1)
                result = (normalized * 255).astype(np.uint8)

                return result
            else:
                # Fallback: min-max normalization
                # Applied min-max normalization: [{img_min:.1f}, {img_max:.1f}] → [0, 255]
                img_min, img_max = img.min(), img.max()
                if img_max > img_min:
                    normalized = (img - img_min) / (img_max - img_min)
                    result = (normalized * 255).astype(np.uint8)
                    return result
                else:
                    # If image has no variation
                    return np.zeros_like(img, dtype=np.uint8)

    def extract_pixel_array(self, ds: pydicom.Dataset) -> np.ndarray:
        """
        Extract 2D pixel array from DICOM and apply preprocessing (for 2D DICOM series)
        """
        img = ds.pixel_array.astype(np.float32)

        # For 3D volume case (multiple frames) - select middle frame
        if img.ndim == 3:
            # 3D DICOM in 2D processing - using middle frame
            frame_idx = img.shape[0] // 2
            img = img[frame_idx]

        # Convert color image to grayscale
        if img.ndim == 3 and img.shape[-1] == 3:
            img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)

        return img

    def resize_volume_3d(self, volume: np.ndarray) -> np.ndarray:
        """
        Resize 3D volume to target size
        """
        current_shape = volume.shape
        target_shape = (self.target_depth, self.target_height, self.target_width)

        if current_shape == target_shape:
            return volume

        # 3D resizing using scipy.ndimage
        zoom_factors = [
            target_shape[i] / current_shape[i] for i in range(3)
        ]

        # Resize with linear interpolation
        resized_volume = ndimage.zoom(volume, zoom_factors, order=1, mode='nearest')

        # Clip to exact size just in case
        resized_volume = resized_volume[:self.target_depth, :self.target_height, :self.target_width]

        # Padding if necessary
        pad_width = [
            (0, max(0, self.target_depth - resized_volume.shape[0])),
            (0, max(0, self.target_height - resized_volume.shape[1])),
            (0, max(0, self.target_width - resized_volume.shape[2]))
        ]

        if any(pw[1] > 0 for pw in pad_width):
            resized_volume = np.pad(resized_volume, pad_width, mode='edge')
        return resized_volume.astype(np.uint8)

    def process_series(self, series_path: str) -> np.ndarray:
        datasets, series_name = self.load_dicom_series(series_path)

        # Check first DICOM to determine 3D/2D
        first_ds = datasets[0]
        first_img = first_ds.pixel_array

        if len(datasets) == 1 and first_img.ndim == 3:
            # Case 1: Single 3D DICOM file
            return self._process_single_3d_dicom(first_ds)
        else:
            # Case 2: Multiple 2D DICOM files
            return self._process_multiple_2d_dicoms(datasets)


    def _process_single_3d_dicom(self, ds: pydicom.Dataset) -> np.ndarray:
        """
        Process single 3D DICOM file
        """
        volume = ds.pixel_array.astype(np.float32)
        window_center, window_width = self.get_windowing_params(ds)

        processed_slices = []
        for i in range(volume.shape[0]):
            slice_img = volume[i]
            processed_img = self.apply_windowing_or_normalize(slice_img, window_center, window_width)
            processed_slices.append(processed_img)

        volume = np.stack(processed_slices, axis=0)
        final_volume = self.resize_volume_3d(volume)

        return final_volume

    def _process_multiple_2d_dicoms(self, datasets: List[pydicom.Dataset]) -> np.ndarray:
        """
        Process multiple 2D DICOM files
        """
        slice_info = self.extract_slice_info(datasets)
        sorted_slices = sorted(slice_info, key=lambda x: x['z_position'])
        window_center, window_width = self.get_windowing_params(sorted_slices[0]['dataset'])
        processed_slices = []

        for slice_data in sorted_slices:
            ds = slice_data['dataset']
            img = self.extract_pixel_array(ds)
            processed_img = self.apply_windowing_or_normalize(img, window_center, window_width)
            resized_img = cv2.resize(processed_img, (self.target_width, self.target_height))

            processed_slices.append(resized_img)

        volume = np.stack(processed_slices, axis=0)
        final_volume = self.resize_volume_3d(volume)

        return final_volume


# In[8]:


def process_dicom_series_safe(series_path: str, target_shape: Tuple[int, int, int] = (32, 384, 384)) -> np.ndarray:
    preprocessor = DICOMPreprocessorKaggle(target_shape=target_shape)
    volume = preprocessor.process_series(series_path)
    return volume


# In[9]:


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


# In[ ]:


series_path = "/home/khor/kaggle_kcw/kaggle-RSNA-Intracranial-Aneurysm-Detection/input/series/1.2.826.0.1.3680043.8.498.10004044428023505108375152878107656647" 
volume = process_dicom_series_safe(series_path, (32, 384, 384))


# In[19]:


for series_name in os.listdir(f"{input_dir}/series/"):
    series_path = os.path.join(f"{input_dir}/series/", series_name)
    volume = process_dicom_series_safe(series_path, (32, 384, 384))
    print(f"Processed series: {series_name}, volume shape: {volume.shape}")
    del volume
    gc.collect()


# In[18]:


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
        label_t = row[self.label_cols].values.astype(np.float32)
        label_t = torch.from_numpy(label_t)

        # Meta
        meta = {
            'series_instance_uid': series_instance_uid,
            'series_instance_uid_path': str(series_path)
        }

        return vol_t, label_t, meta


# In[ ]:


input_dir = "/home/khor/kaggle_kcw/kaggle-RSNA-Intracranial-Aneurysm-Detection/input"
train_csv = os.path.join(input_dir, "train.csv")
train_df = pd.read_csv(train_csv)

train_ds = RSNAAneurysmDataset(
    df=train_df,
    input_dir=input_dir,
    target_shape=(32, 384, 384),
    label_cols=LABEL_COLS
)


# In[20]:


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


# In[21]:


NUM_LABELS = len(LABEL_COLS)
print(f"Number of labels: {NUM_LABELS}")


# In[ ]:


# ---------- utils
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False  # allow autotune for 3D volumes
    torch.backends.cudnn.benchmark = True

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
    def __init__(self): self.reset()
    def reset(self): self.sum = 0.0; self.n = 0
    def update(self, val, k=1): self.sum += float(val) * k; self.n += k
    @property
    def avg(self): return self.sum / max(1, self.n)


if 'val_loader' not in globals():
    # make a stratified split by 'aneurysm_present' (assumed first label)
    y_for_split = train_df[LABEL_COLS[0]].values
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    tr_idx, va_idx = next(iter(skf.split(train_df, y_for_split)))

    train_subset = Subset(train_ds, tr_idx)
    val_ds = RSNAAneurysmDataset(
        df=train_df.iloc[va_idx].reset_index(drop=True),
        input_dir=input_dir,
        target_shape=(32, 384, 384),
        label_cols=LABEL_COLS
    )

    def collate(batch):
        vols, labels, metas = zip(*batch)
        vols = torch.stack(vols, dim=0)
        labels = torch.stack(labels, dim=0)
        return vols, labels, metas

    train_loader = DataLoader(train_subset, batch_size=4, shuffle=True,
                              num_workers=4, pin_memory=True, collate_fn=collate, persistent_workers=True)
    val_loader   = DataLoader(val_ds,     batch_size=4, shuffle=False,
                              num_workers=4, pin_memory=True, collate_fn=collate, persistent_workers=True)

# ---------- model / opt / sched (yours, with additions)
set_seed(42)

model = EffnetAneurysmClassifier(
    model_name="efficientnet_b0",
    num_classes=NUM_LABELS,
    pretrained=True,
    return_logits=True
).cuda()

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)

# cosine decay with warmup
total_epochs = 10
warmup_epochs = 1
def lr_lambda(epoch):
    if epoch < warmup_epochs:
        return (epoch + 1) / max(1, warmup_epochs)
    progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
    return 0.5 * (1.0 + math.cos(math.pi * progress))
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

scaler = torch.amp.GradScaler()

# ---------- training / validation loop
save_dir = "./checkpoints"; os.makedirs(save_dir, exist_ok=True)
best_score = -1.0
patience, bad_epochs = 3, 0
grad_clip_norm = 2.0
accum_steps = 1  # set >1 if you want gradient accumulation

for epoch in range(total_epochs):
    # ---- train
    model.train()
    loss_meter = AverageMeter()
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{total_epochs} [train]")
    optimizer.zero_grad(set_to_none=True)

    for step, (vols, labels, _) in enumerate(pbar):
        vols = vols.cuda(non_blocking=True).float()
        labels = labels.cuda(non_blocking=True).float()

        with torch.amp.autocast(device_type= "cuda", dtype=torch.float16):
            logits = model(vols)
            loss = criterion(logits, labels) / accum_steps

        scaler.scale(loss).backward()

        if (step + 1) % accum_steps == 0:
            if grad_clip_norm is not None:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        loss_meter.update(loss.item() * accum_steps, k=vols.size(0))
        pbar.set_postfix(loss=f"{loss_meter.avg:.4f}", lr=f"{optimizer.param_groups[0]['lr']:.2e}")

    scheduler.step()

    # ---- validate
    model.eval()
    val_loss = AverageMeter()
    all_probs, all_trues = [], []
    with torch.no_grad(), torch.amp.autocast(device_type= "cuda", dtype=torch.float16):
        for vols, labels, _ in tqdm(val_loader, desc=f"Epoch {epoch+1}/{total_epochs} [valid]"):
            vols = vols.cuda(non_blocking=True).float()
            labels = labels.cuda(non_blocking=True).float()
            logits = model(vols)
            loss = criterion(logits, labels)
            val_loss.update(loss.item(), k=vols.size(0))

            probs = torch.sigmoid(logits).float().cpu().numpy()
            all_probs.append(probs)
            all_trues.append(labels.cpu().numpy())

    y_prob = np.concatenate(all_probs, axis=0)
    y_true = np.concatenate(all_trues, axis=0)

    per_label_auc = auc_per_label(y_true, y_prob)
    # RSNA uses 'Aneurysm Present' as the special label — assume it's LABEL_COLS[0]
    final = rsna_final_score(per_label_auc, ap_index=0)

    # logging
    readable_aucs = [None if np.isnan(x) else round(float(x), 4) for x in per_label_auc]
    print(f"\nEpoch {epoch+1}: train_loss={loss_meter.avg:.4f}  "
          f"val_loss={val_loss.avg:.4f}  final_score={None if np.isnan(final) else round(final,4)}")
    print("Per-label AUROC:", {LABEL_COLS[i]: readable_aucs[i] for i in range(len(readable_aucs))})

    # checkpointing / early stop
    score_for_ckpt = -1 if np.isnan(final) else final
    if score_for_ckpt > best_score:
        best_score = score_for_ckpt
        torch.save(
            {"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
             "scheduler": scheduler.state_dict(), "best_final": best_score, "label_cols": LABEL_COLS},
            os.path.join(save_dir, "best_by_final_score.pt")
        )
        bad_epochs = 0
        print(f"✅ New best final score: {best_score:.4f} (checkpoint saved).")
    else:
        bad_epochs += 1
        print(f"↩️  No improvement ({bad_epochs}/{patience}).")
        if bad_epochs >= patience:
            print("⏹️  Early stopping.")
            break


# In[ ]:


model_infer = EffnetAneurysmClassifier(
    model_name="efficientnet_b0",
    num_classes=NUM_LABELS,
    pretrained=False,
    return_logits=False   # directly output probs
).cuda()
model_infer.load_state_dict(torch.load("best.pt"))
model_infer.eval()

with torch.no_grad():
    vols, labels, metas = next(iter(train_loader))
    probs = model_infer(vols.cuda())  # [B, 15], each in [0,1]


# In[ ]:





# In[ ]:




