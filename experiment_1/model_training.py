import pandas as pd
import numpy as np
import os, math
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm
from sklearn.model_selection import StratifiedKFold

from data_preprocess import process_dicom_series_safe
from dataset import RSNAAneurysmDataset, collate
from model import EffnetAneurysmClassifier
from metric import AverageMeter, get_auc_per_location_list, get_final_score
from utils import set_seed, LABEL_COLS, find_unused_series_instance_uid_list

location_list = LABEL_COLS[:-1]  # All except 'Aneurysm Present'

set_seed(42)

MAIN_DIR = "/home/khor/kaggle_kcw/kaggle-RSNA-Intracranial-Aneurysm-Detection"
INPUT_DIR = f"{MAIN_DIR}/input"
train_df = pd.read_csv(f"{INPUT_DIR}/train.csv") 
train_localizers_df = pd.read_csv(f"{INPUT_DIR}/train_localizers.csv")

unused_series_instance_uid_list = find_unused_series_instance_uid_list(train_df, train_localizers_df)
train_df = train_df[~train_df["SeriesInstanceUID"].isin(unused_series_instance_uid_list)].reset_index(drop=True)    

TARGET_SHAPE = (32, 384, 384)

# Pre-process DICOMs to NumPy array
print("Starting DICOM pre-processing...")

PREPROCESSED_DIR = os.path.join(f"{MAIN_DIR}/experiment_1", "train_volumes")
os.makedirs(PREPROCESSED_DIR, exist_ok=True)

all_series_instance_uid_list = train_df['SeriesInstanceUID'].unique()
already_processed_series_instance_uid_list = {f.split('.npy')[0] for f in os.listdir(PREPROCESSED_DIR)}
series_instance_uid_to_process_list = [uid for uid in all_series_instance_uid_list if uid not in already_processed_series_instance_uid_list]

print(f"Processing {len(series_instance_uid_to_process_list)} new series...")
for curr_series_instance_uid in tqdm(series_instance_uid_to_process_list):
    curr_dicom_series_dir = os.path.join(INPUT_DIR, "series", curr_series_instance_uid)
    curr_volume = process_dicom_series_safe(curr_dicom_series_dir, TARGET_SHAPE)
    np.save(os.path.join(PREPROCESSED_DIR, f"{curr_series_instance_uid}.npy"), curr_volume)

train_ds = RSNAAneurysmDataset(
    df=train_df,
    input_dir=PREPROCESSED_DIR,
    target_shape=TARGET_SHAPE,
    label_cols=location_list
)

NUM_LABELS = len(location_list)
print(f"Number of curr_labels: {NUM_LABELS}")

y_for_split_column = 'Aneurysm Present'
y_for_split = train_df[y_for_split_column].values
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
train_ds_idx, val_ds_idx = next(iter(skf.split(train_df, y_for_split)))

train_subset = Subset(train_ds, train_ds_idx)
val_subset = Subset(train_ds, val_ds_idx)

train_subset_loader = DataLoader(train_subset, batch_size=32, shuffle=True,
                            num_workers=4, pin_memory=True, collate_fn=collate, persistent_workers=True)
val_subset_loader   = DataLoader(val_subset,     batch_size=32, shuffle=False,
                            num_workers=4, pin_memory=True, collate_fn=collate, persistent_workers=True)

model = EffnetAneurysmClassifier(
    model_name="efficientnet_b0",
    num_classes=NUM_LABELS,
    pretrained=True,
    return_logits=True
).cuda()

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)

# cosine decay with warmup
total_epochs = 200
warmup_epochs = 1
def lr_lambda(epoch):
    if epoch < warmup_epochs:
        return (epoch + 1) / max(1, warmup_epochs)
    progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
    return 0.5 * (1.0 + math.cos(math.pi * progress))
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

scaler = torch.amp.GradScaler()

# --- TensorBoard Setup ---
writer = SummaryWriter('runs/aneurysm_experiment_1')

# ---------- training / validation loop
save_dir = "./checkpoints"; os.makedirs(save_dir, exist_ok=True)
best_score = -1.0
patience, bad_epochs = 5, 0
grad_clip_norm = 2.0
accum_steps = 1
log_grad_steps = 1

for epoch in range(total_epochs):
    # ---- train
    model.train()
    train_loss_meter = AverageMeter()
    train_pbar = tqdm(train_subset_loader, desc=f"Epoch {epoch+1}/{total_epochs} [train]")
    optimizer.zero_grad(set_to_none=True)

    for curr_step, (curr_vols, curr_labels, _) in enumerate(train_pbar):
        curr_vols = curr_vols.cuda(non_blocking=True).float()
        curr_labels = curr_labels.cuda(non_blocking=True).float()

        with torch.amp.autocast(device_type= "cuda", dtype=torch.float16):
            curr_logits = model(curr_vols)
            curr_loss = criterion(curr_logits, curr_labels) / accum_steps

        scaler.scale(curr_loss).backward()

        if (curr_step + 1) % accum_steps == 0:
            if grad_clip_norm is not None:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)

            # Log gradients every N steps, AFTER unscaling and BEFORE optimizer step
            if (curr_step + 1) % log_grad_steps == 0:
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        writer.add_histogram(f'Gradients/{name}', param.grad, epoch * len(train_subset_loader) + curr_step)

            scaler.step(optimizer)
            scaler.update()
            
            # Log weights every N steps, AFTER the optimizer step
            if (curr_step + 1) % log_grad_steps == 0:
                for name, param in model.named_parameters():
                    writer.add_histogram(f'Weights/{name}', param, epoch * len(train_subset_loader) + curr_step)

            optimizer.zero_grad(set_to_none=True)

        train_loss_meter.update(curr_loss.item() * accum_steps, k=curr_vols.size(0))
        train_pbar.set_postfix(train_loss=f"{train_loss_meter.avg:.4f}", lr=f"{optimizer.param_groups[0]['lr']:.2e}")

    scheduler.step()

    # Log training metrics to TensorBoard
    writer.add_scalar('Loss/train', train_loss_meter.avg, epoch)
    writer.add_scalar('LearningRate', optimizer.param_groups[0]['lr'], epoch)

    # ---- validate
    model.eval()
    val_loss_meter = AverageMeter()
    all_probs, all_trues = [], []
    with torch.no_grad(), torch.amp.autocast(device_type= "cuda", dtype=torch.float16):
        for curr_vols, curr_labels, _ in tqdm(val_subset_loader, desc=f"Epoch {epoch+1}/{total_epochs} [valid]"):
            curr_vols = curr_vols.cuda(non_blocking=True).float()
            curr_labels = curr_labels.cuda(non_blocking=True).float()
            curr_logits = model(curr_vols)
            curr_loss = criterion(curr_logits, curr_labels)
            val_loss_meter.update(curr_loss.item(), k=curr_vols.size(0))

            curr_probs = torch.sigmoid(curr_logits).float().cpu().numpy()
            all_probs.append(curr_probs)
            all_trues.append(curr_labels.cpu().numpy())

    y_prob = np.concatenate(all_probs, axis=0)
    y_true = np.concatenate(all_trues, axis=0)

    auc_per_location_list = get_auc_per_location_list(y_true, y_prob)
    final_score = get_final_score(y_true, y_prob, auc_per_location_list)

    # Log validation metrics to TensorBoard
    writer.add_scalar('Loss/validation', val_loss_meter.avg, epoch)
    writer.add_scalar('Score/final_score', final_score, epoch)
    for i, auc in enumerate(auc_per_location_list):
        writer.add_scalar(f'AUC/{location_list[i]}', auc, epoch)

    print(f"\nEpoch {epoch+1}: train_loss={train_loss_meter.avg:.4f}  "
            f"val_loss_meter={val_loss_meter.avg:.4f}  final_score={final_score:.4f}")
    print("Per-label AUROC:", {location_list[i]: auc_per_location_list[i] for i in range(len(auc_per_location_list))})

    # checkpointing / early stop
    score_for_ckpt = final_score
    if score_for_ckpt > best_score:
        best_score = score_for_ckpt
        bad_epochs = 0
        print(f"✅ New best final_score score: {best_score:.4f} (checkpoint saved).")
        # Save local checkpoint
        torch.save(
            {"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(), "best_final": best_score, "label_cols": LABEL_COLS},
            os.path.join(save_dir, "best_by_final_score.pt")
        )
    else:
        bad_epochs += 1
        print(f"↩️  No improvement ({bad_epochs}/{patience}).")
        if bad_epochs >= patience:
            print("⏹️  Early stopping.")
            break

# Close the TensorBoard writer
writer.close()