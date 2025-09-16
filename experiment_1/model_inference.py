import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from dataset import RSNAAneurysmDataset, collate
from model import EffnetAneurysmClassifier

# --- Configuration ---
INPUT_DIR = "/home/khor/kaggle_kcw/kaggle-RSNA-Intracranial-Aneurysm-Detection/input"
TEST_CSV_PATH = os.path.join(INPUT_DIR, "kaggle_evaluation/test.csv")
CHECKPOINT_PATH = "./checkpoints/best_by_final_score.pt" # Path to your trained model checkpoint

TARGET_SHAPE = (32, 384, 384)
BATCH_SIZE = 4
NUM_WORKERS = 4

# These must match the labels used during training
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
NUM_LABELS = len(LABEL_COLS)

# --- Load Data ---
print(f"Loading test data from {TEST_CSV_PATH}...")
test_df = pd.read_csv(TEST_CSV_PATH)

test_ds = RSNAAneurysmDataset(
    df=test_df,
    input_dir=INPUT_DIR,
    target_shape=TARGET_SHAPE,
    label_cols=None  # No labels for test set
)

test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                         num_workers=NUM_WORKERS, pin_memory=True, collate_fn=collate,
                         persistent_workers=True)

print(f"Found {len(test_ds)} test cases.")

# --- Load Model ---
print(f"Loading model from {CHECKPOINT_PATH}...")
model_infer = EffnetAneurysmClassifier(
    model_name="efficientnet_b0",
    num_classes=NUM_LABELS,
    pretrained=False,
    return_logits=False   # Set to False to directly get probabilities
).cuda()

checkpoint = torch.load(CHECKPOINT_PATH)
model_infer.load_state_dict(checkpoint['model'])
model_infer.eval()

# --- Run Inference ---
print("Starting inference...")
all_probs = []
all_series_uids = []

with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.float16):
    for vols, _, metas in tqdm(test_loader, desc="Inference"):
        vols = vols.cuda(non_blocking=True).float()
        probs = model_infer(vols)  # [B, NUM_LABELS], each in [0,1]
        all_probs.append(probs.cpu().numpy())
        all_series_uids.extend([m['series_instance_uid'] for m in metas])

# --- Create Submission File ---
print("Creating submission file...")
y_prob_test = np.concatenate(all_probs, axis=0)
submission_df = pd.DataFrame(y_prob_test, columns=LABEL_COLS)
submission_df['SeriesInstanceUID'] = all_series_uids

# Reorder columns to have SeriesInstanceUID first
submission_df = submission_df[['SeriesInstanceUID'] + LABEL_COLS]

submission_df.to_csv("submission.csv", index=False)
print("submission.csv created successfully!")
