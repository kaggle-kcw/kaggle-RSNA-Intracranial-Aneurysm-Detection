import pandas as pd
import numpy as np
import os, math
import gc
import torch
import mlflow
import mlflow.pytorch
import torch.nn as nn
import torch.nn.functional as F
import timm
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm
from sklearn.model_selection import StratifiedKFold

from data_preprocess import process_dicom_series_safe
from dataset import RSNAAneurysmDataset, collate
from model import EffnetAneurysmClassifier
from metric import AverageMeter, auc_per_label, rsna_final_score
from utils import set_seed

input_dir = "/home/khor/kaggle_kcw/kaggle-RSNA-Intracranial-Aneurysm-Detection/input/"
train_df = pd.read_csv(f"{input_dir}/train.csv") 
train_localizers_df = pd.read_csv(f"{input_dir}/train_localizers.csv")
    
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



input_dir = "/home/khor/kaggle_kcw/kaggle-RSNA-Intracranial-Aneurysm-Detection/input"
train_csv = os.path.join(input_dir, "train.csv")
train_df = pd.read_csv(train_csv)

train_ds = RSNAAneurysmDataset(
    df=train_df,
    input_dir=input_dir,
    target_shape=(32, 384, 384),
    label_cols=LABEL_COLS
)

NUM_LABELS = len(LABEL_COLS)
print(f"Number of labels: {NUM_LABELS}")

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

train_loader = DataLoader(train_subset, batch_size=4, shuffle=True,
                            num_workers=4, pin_memory=True, collate_fn=collate, persistent_workers=True)
val_loader   = DataLoader(val_ds,     batch_size=4, shuffle=False,
                            num_workers=4, pin_memory=True, collate_fn=collate, persistent_workers=True)

set_seed(42)

# --- MLflow Setup ---
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("RSNA Aneurysm Detection")

with mlflow.start_run() as run:
    print(f"MLflow Run ID: {run.info.run_id}")

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

    # Log hyperparameters
    mlflow.log_param("model_name", "efficientnet_b0")
    mlflow.log_param("target_shape", "(32, 384, 384)")
    mlflow.log_param("learning_rate", 2e-4)
    mlflow.log_param("weight_decay", 1e-4)
    mlflow.log_param("total_epochs", total_epochs)
    mlflow.log_param("warmup_epochs", warmup_epochs)
    mlflow.log_param("batch_size", 4)
    mlflow.log_param("seed", 42)

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

        # MLflow logging
        mlflow.log_metric("train_loss", loss_meter.avg, step=epoch)
        mlflow.log_metric("val_loss", val_loss.avg, step=epoch)
        if not np.isnan(final):
            mlflow.log_metric("final_score", final, step=epoch)

        # Log per-label AUCs
        for i, col in enumerate(LABEL_COLS):
            if not np.isnan(per_label_auc[i]):
                mlflow.log_metric(f"auc_{col.replace(' ', '_')}", per_label_auc[i], step=epoch)

        # checkpointing / early stop
        score_for_ckpt = -1 if np.isnan(final) else final
        if score_for_ckpt > best_score:
            best_score = score_for_ckpt
            bad_epochs = 0
            print(f"✅ New best final score: {best_score:.4f} (checkpoint saved).")
            # Save local checkpoint
            torch.save(
                {"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                 "scheduler": scheduler.state_dict(), "best_final": best_score, "label_cols": LABEL_COLS},
                os.path.join(save_dir, "best_by_final_score.pt")
            )
            # Log model to MLflow
            mlflow.pytorch.log_model(model, "model", registered_model_name="rsna-aneurysm-effnet-b0")
        else:
            bad_epochs += 1
            print(f"↩️  No improvement ({bad_epochs}/{patience}).")
            if bad_epochs >= patience:
                print("⏹️  Early stopping.")
                break
