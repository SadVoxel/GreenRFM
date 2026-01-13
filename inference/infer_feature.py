"""
Unified feature extraction script for LeanRad pipeline.
Supports CT (Abd, Chest, Merlin, CT-Rate) and MRI (Knee, Spine).
"""

import os
import argparse
import sys
import glob
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import BertTokenizer

# Ensure we can import from LeanRad
# If running as script, add parent directory to path if needed, 
# but relying on installed package or PYTHONPATH is better.
# Assuming LeanRad structure is valid.

# Add parent directory to path to allow imports from LeanRad
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from models.clip import CTCLIP
    from models.mr_model import MultiPlaneNet
    from data import (
        AHKneeDataset, 
        AHSpineDataset, 
        AHAbdDataset, 
        AHChestInferenceDataset,
        MerlinDataset, 
        CTRATEDataset,
        RadChestCTDataset
    )
except ImportError as e:
    print(f"Import Error: {e}")
    print("Please ensure you are running this from the root of the workspace or have LeanRad in PYTHONPATH.")
    sys.exit(1)


def get_dataset(config):
    """Factory function for initializing datasets based on name."""
    name = config["dataset"].lower()
    
    if name == "ah_knee":
        return AHKneeDataset(
            img_dir=config["data_folder"],
            label_csv=config["labels_csv"],
            report_xlsx=config["reports_csv"],
            train_mode=False
        )
        
    elif name == "ah_spine":
        return AHSpineDataset(
            img_dir=config["data_folder"],
            label_csv=config["labels_csv"],
            report_xlsx=config["reports_csv"],
            train_mode=False
        )

    elif name == "ah_abd":
        return AHAbdDataset(
            data_folder=config["data_folder"],
            reports_csv=config["reports_csv"],
            labels_csv=config["labels_csv"],
            split="test"
        )

    elif name == "ah_chest":
        return AHChestInferenceDataset(
            data_folder=config["data_folder"],
            reports_csv=config["reports_csv"],
            labels_csv=config["labels_csv"]
        )

    elif name == "merlin":
        return MerlinDataset(
            data_folder=config["data_folder"],
            reports_csv=config["reports_csv"],
            labels_csv=config["labels_csv"],
            section="test"
        )

    elif name == "ct_rate":
        return CTRATEDataset(
            data_folder=config["data_folder"],
            reports_csv=config["reports_csv"],
            labels_csv=config["labels_csv"],
            section="test"
        )
        
    elif name == "radchest":
        return RadChestCTDataset(
            data_folder=config["data_folder"],
            labels_csv=config["labels_csv"]
        )
    
    else:
        raise ValueError(f"Unknown dataset: {name}")


def run_inference(config):
    """
    Main inference function.
    config: dict with parameters
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Dataset & DataLoader
    dataset = get_dataset(config)
    dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=False, 
                            num_workers=config["num_workers"], pin_memory=True)
    
    print(f"Dataset {config['dataset']} loaded. Size: {len(dataset)}")

    # 2. Tokenizer
    bert_name = config.get("bert_name", "microsoft/BiomedVLP-CXR-BERT-specialized")
    try:
        tokenizer = BertTokenizer.from_pretrained(bert_name, do_lower_case=True, local_files_only=False)
    except Exception:
        # Fallback to online if local fails
        print("Local BERT load failed, trying online...")
        tokenizer = BertTokenizer.from_pretrained(bert_name, do_lower_case=True)

    # 3. Model Construction
    image_encoder = None
    is_mri = config["dataset"] in ["ah_knee", "ah_spine"]
    
    if is_mri:
        sequences_count = 3 
        if hasattr(dataset, "SEQUENCES"):
            sequences_count = len(dataset.SEQUENCES)
        elif hasattr(dataset, "sequences"):
             sequences_count = len(dataset.sequences)
             
        print(f"Initializing MultiPlaneNet with {sequences_count} channels.")
        image_encoder = MultiPlaneNet(in_ch=sequences_count, pretrained=False)
    
    # Init CLIP
    clip_model = CTCLIP(
        text_encoder_name=bert_name,
        dim_image=512,
        dim_text=768,
        dim_latent=768,
        image_encoder=image_encoder
    )
    
    clip_model.to(device)

    # 4. Process Checkpoints
    model_path = config["model_path"]
    ckpt_files = []
    
    if os.path.isfile(model_path):
        ckpt_files.append(model_path)
    elif os.path.isdir(model_path):
        start = config.get("start_ckpt", 1000)
        end = config.get("end_ckpt", 50000)
        step = config.get("step_ckpt", 1000)
        
        for k in range(start, end + 1, step):
            path = os.path.join(model_path, f"CTClip.{k}.pt")
            if os.path.exists(path):
                ckpt_files.append((k, path))
    
    if not ckpt_files:
        print(f"No checkpoints found in {model_path}")
        return

    output_dir = config["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    max_length = config.get("max_length", 128)

    # 5. Inference Loop
    for item in ckpt_files:
        if isinstance(item, tuple):
            k, ckpt_path = item
            step_name = f"ckpt_{k}"
        else:
            ckpt_path = item
            k = "custom"
            step_name = os.path.splitext(os.path.basename(ckpt_path))[0]
            
        print(f"\nProcessing: {ckpt_path}")
        try:
            checkpoint = torch.load(ckpt_path, map_location=device)
            if "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            else:
                state_dict = checkpoint
            
            clip_model.load_state_dict(state_dict, strict=False)
            clip_model.eval()
            
            all_image_latents = []
            all_text_latents = []
            all_acc_names = []
            
            with torch.no_grad():
                for batch in tqdm(dataloader, desc=f"Infer {step_name}"):
                    image_input = None
                    text_input = []
                    acc_ids = []
                    
                    if is_mri and isinstance(batch, dict):
                         # MRI
                         batch_gpu = {
                             key: val.to(device, non_blocking=True) if torch.is_tensor(val) else val
                             for key, val in batch.items()
                         }
                         image_input = batch_gpu 
                         
                         reports = batch.get("report", "")
                         if isinstance(reports, str): reports = [reports]
                         text_input = reports
                         
                         ids = batch.get("case_id", [])
                         if isinstance(ids, str): ids = [ids]
                         elif torch.is_tensor(ids): ids = ids.tolist() 
                         acc_ids = ids
                         
                    elif isinstance(batch, (list, tuple)):
                        # CT
                        if len(batch) == 4:
                            video, txt, _, ac = batch
                            text_input = txt
                        elif len(batch) == 3:
                            video, _, ac = batch
                            text_input = [""] * len(video) 
                            
                        image_input = video.to(device, non_blocking=True)
                        acc_ids = ac
                    
                    else:
                        print("Unknown batch format, skipping.")
                        continue

                    # Text
                    has_text = any(t for t in text_input if t)
                    text_latents = None
                    
                    if has_text:
                        tokens = tokenizer(
                            text_input,
                            return_tensors="pt",
                            padding="max_length",
                            truncation=True,
                            max_length=max_length
                        ).to(device)
                        
                        res = clip_model(text_input=tokens, image_input=image_input, return_latents=True)
                        text_latents = res["text_latents"]
                        image_latents = res["image_latents"]
                    else:
                        image_latents = clip_model(image_input=image_input)
                        text_latents = torch.zeros((len(acc_ids), 768), device=device)

                    if image_latents is not None:
                        all_image_latents.append(image_latents.cpu().numpy())
                    if text_latents is not None:
                        all_text_latents.append(text_latents.cpu().numpy())
                    
                    if isinstance(acc_ids, tuple): acc_ids = list(acc_ids)
                    all_acc_names.extend(acc_ids)

            # Saving
            save_path = os.path.join(output_dir, step_name)
            os.makedirs(save_path, exist_ok=True)
            
            if all_image_latents:
                np.savez_compressed(os.path.join(save_path, "image_latents.npz"), data=np.concatenate(all_image_latents))
            if all_text_latents:
                np.savez_compressed(os.path.join(save_path, "text_latents.npz"), data=np.concatenate(all_text_latents))
            
            with open(os.path.join(save_path, "accessions.txt"), "w") as f:
                for acc in all_acc_names:
                    f.write(f"{str(acc)}\n")

        except Exception as e:
            print(f"Failed to process {ckpt_path}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    # Load configuration
    try:
        from LeanRad.config.config import INFERENCE_CONFIG
    except ImportError:
        # Fallback if package is not installed/discoverable
        print("Warning: Could not import INFERENCE_CONFIG from LeanRad.config.config")
        INFERENCE_CONFIG = {
             "dataset": "ah_chest",
             "data_folder": "./data/images",
             "reports_csv": "./data/reports.csv",
             "labels_csv": "./data/labels.csv",
             "model_path": "./checkpoints/CTClip.pt",
             "output_dir": "./features/ah_chest",
             "batch_size": 1,
             "num_workers": 4,
             "max_length": 128,
             "bert_name": "microsoft/BiomedVLP-CXR-BERT-specialized",
        }

    # You can override specific values here if needed
    # INFERENCE_CONFIG["dataset"] = "ah_chest"
    
    run_inference(INFERENCE_CONFIG)

