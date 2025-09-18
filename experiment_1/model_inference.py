# --- Configuration ---
INPUT_DIR = "/home/khor/kaggle_kcw/kaggle-RSNA-Intracranial-Aneurysm-Detection/input"

import sys 
sys.path.append('/kaggle/input/rsna-intracranial-aneurysm-detection-dataset')
sys.path.append(INPUT_DIR)

import os, gc, shutil
import polars as pl
import numpy as np
import torch

import kaggle_evaluation.rsna_inference_server

from data_preprocess import process_dicom_series_safe
from model import EffnetAneurysmClassifier
from utils import LABEL_COLS

CHECKPOINT_PATH = "./checkpoints/best_by_final_score.pt" # Path to your trained model checkpoint

TARGET_SHAPE = (32, 384, 384)

# --- Globals ---
model_infer = None
NUM_LABELS = len(LABEL_COLS)

def load_models():
    """Loads the inference model onto the GPU."""
    global model_infer
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
    print("Model loaded successfully.")

def _predict_inner(series_path: str) -> pl.DataFrame:
    """
    Core prediction logic for a single DICOM series.
    """
    # 1. Preprocess the DICOM series
    vol = process_dicom_series_safe(series_path, TARGET_SHAPE)  # (D,H,W) uint8
    vol_t = torch.from_numpy(vol).float().div_(255.0) # uint8 [D,H,W] -> float [D,H,W]
    vol_t = vol_t.unsqueeze(0) # Add batch dimension -> [1, D, H, W]

    # 2. Run inference
    with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.float16):
        vols_cuda = vol_t.cuda(non_blocking=True)
        probs = model_infer(vols_cuda)  # [1, NUM_LABELS]
        probs_np = probs.cpu().numpy()

    # 3. Format output as a Polars DataFrame
    predictions = pl.DataFrame(
        data=probs_np,
        schema=LABEL_COLS,
        orient='row'
    )
    return predictions

def predict(series_path: str) -> pl.DataFrame:
    """
    Top-level prediction function passed to the server.
    It calls the core logic and guarantees cleanup in a `finally` block.
    """
    try:
        # Call the internal prediction logic
        predictions = _predict_inner(series_path)
        return predictions
    except Exception as e:
        print(f"Error during prediction for {os.path.basename(series_path)}: {e}")
        print("Using fallback predictions.")
        # Return a fallback dataframe with the correct schema
        conservative_preds = [0.1] * len(LABEL_COLS)
        predictions = pl.DataFrame(
            data=[conservative_preds],
            schema=LABEL_COLS,
            orient='row'
        )
        return predictions.head(0) # Return empty dataframe as per submission guidelines
    finally:
        # This code is required to prevent "out of disk space" and "directory not empty" errors.
        # It deletes the shared folder and then immediately recreates it, ensuring it's
        # empty and ready for the next prediction.
        shared_dir = '/kaggle/shared'
        shutil.rmtree(shared_dir, ignore_errors=True)
        os.makedirs(shared_dir, exist_ok=True)

        # Also perform memory cleanup here
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

# ====================================================
# Main Execution
# ====================================================

# Load models at startup
load_models()
# Initialize the inference server with our main `predict` function.
inference_server = kaggle_evaluation.rsna_inference_server.RSNAInferenceServer(predict)

# Check if the notebook is running in the competition environment or a local session.
if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    inference_server.serve()
else:
    inference_server.run_local_gateway()