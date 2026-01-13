import os
import sys
import torch
import numpy as np
from torch.utils.data import DataLoader
from transformers import BertTokenizer
import argparse

# Path handling: Add LeanRad root to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, ".."))
if root_dir not in sys.path:
    sys.path.append(root_dir)

# Model Imports
from models.clip import CTCLIP
from models.mr_model import MultiPlaneNet
from inference.zero_shot import ZeroShotEvaluator

# Dataset Imports (Lazy loading might be better, but explicit here for clarity)
# Fallback logic for imports
try:
    from data.ah_knee import AHKneeDataset
    from data.ct_rate import CTRATEDataset
    from data.ah_chest import AHChestInferenceDataset
    from data.radchest import RadChestCTDataset
except ImportError:
    sys.path.append(os.path.join(root_dir, "data"))
    try:
        from ah_knee import AHKneeDataset
        from ct_rate import CTRATEDataset
        from ah_chest import AHChestInferenceDataset
        from radchest import RadChestCTDataset
    except ImportError:
        print("Warning: Could not import some datasets. Ensure python path is correct.")

# --- Configurations ---

# 1. CT-RATE / AH-Chest Pathologies (18 classes usually)
# Based on CSV header shared earlier (excluding VolumeName/Material/etc if needed)
# "Medical material" is typically excluded in some evaluations, but included in others. 
# The user's cls-no-cls script used 'dataset_multi_abnormality_labels_valid_predicted_labels.csv'.
# We extracted headers: Medical material, Arterial wall calcification...
CT_PATHOLOGIES = [
    "Medical material",
    "Arterial wall calcification",
    "Cardiomegaly",
    "Pericardial effusion",
    "Coronary artery wall calcification",
    "Hiatal hernia",
    "Lymphadenopathy",
    "Emphysema",
    "Atelectasis",
    "Lung nodule",
    "Lung opacity",
    "Pulmonary fibrotic sequela",
    "Pleural effusion",
    "Mosaic attenuation pattern",
    "Peribronchial thickening",
    "Consolidation",
    "Bronchiectasis",
    "Interlobular septal thickening"
]

# 2. MRI Knee Pathologies (14 classes)
MRI_KNEE_PATHOLOGIES = [
    "Medial Meniscus Injury", "Lateral Meniscus Injury", "ACL Injury", "PCL Injury", 
    "MCL Injury", "LCL Injury", "Cartilage Defect", "Bone Marrow Edema", "Fracture", 
    "Joint Effusion", "Synovitis", "Post-operative Changes", "Osteoarthritis", "Normal"
]

# 3. RadChest Pathologies
# Requires verifying map. Using CT_PATHOLOGIES as placeholder if they overlap or empty list
RADCHEST_PATHOLOGIES = CT_PATHOLOGIES # Assumption: similar labels? Often it's subset.

DATASET_CONFIGS = {
    "ah_knee": {
        "dataset_cls": "AHKneeDataset",
        "pathologies": MRI_KNEE_PATHOLOGIES,
        "model_type": "mr",
        "checkpoint_dir": "./checkpoints/mr_align",
        "init_kwargs": {
            "img_dir": "./data/knee/valid_images",
            "label_csv": "./data/knee/val_labels.csv",
            "report_xlsx": "./data/knee/reports.xlsx",
            "train_mode": False
        }
    },
    "ct_rate": {
        "dataset_cls": "CTRATEDataset",
        "pathologies": CT_PATHOLOGIES,
        "model_type": "ct",
        "checkpoint_dir": "./checkpoints/ct_align", # Example path
        "init_kwargs": {
            "data_folder": "./data/ct_rate/valid_images",
            "reports_csv": "./data/ct_rate/val_reports.csv",
            "labels_csv": "./data/ct_rate/val_labels.csv",
            "section": "valid"
        }
    },
    "ah_chest": {
        "dataset_cls": "AHChestInferenceDataset",
        "pathologies": CT_PATHOLOGIES,
        "model_type": "ct",
        "checkpoint_dir": "./checkpoints/ct_align",
        "init_kwargs": {
            "data_folder": "./data/ah_chest/validation",
            "labels_csv": "./data/ah_chest/val_labels.csv",
            "reports_csv": "./data/ah_chest/reports.csv"
        }
    },
    "radchest": {
        "dataset_cls": "RadChestCTDataset",
        "pathologies": RADCHEST_PATHOLOGIES,
        "model_type": "ct",
        # Use a relative, portable default; override via CLI --ckpt or config if needed
        "checkpoint_dir": "./checkpoints/ct_align",
        "init_kwargs": {
            "data_folder": "./data/radchest/images",
            "labels_csv": "./data/radchest/labels.csv",
            "target_shape": (192, 192, 96)
        }
    }
}

def get_dataset(name, kwargs):
    if name == "ah_knee":
        return AHKneeDataset(**kwargs)
    elif name == "ct_rate":
        return CTRATEDataset(**kwargs)
    elif name == "ah_chest":
         # Check import success
        return AHChestInferenceDataset(**kwargs)
    elif name == "radchest":
        return RadChestCTDataset(**kwargs)
    else:
        raise ValueError(f"Unknown dataset: {name}")

def run_zero_shot(dataset_name="ah_knee", 
                  prompt_id="p3", 
                  force_checkpoint=None, 
                  batch_size=16):
    
    config = DATASET_CONFIGS.get(dataset_name)
    if not config:
        print(f"Error: Dataset {dataset_name} not found in configs.")
        return

    print(f"--- Running Zero-Shot for {dataset_name} ---")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # 1. Model Setup
    pathologies = config['pathologies']
    model_type = config['model_type']
    
    print(f"Initializing Model ({model_type})...")
    
    image_encoder = None
    if model_type == 'mr':
        # Default MR Backbone
        image_encoder = MultiPlaneNet(in_ch=3, pretrained=True)
    elif model_type == 'ct':
        # Default CT Backbone (None triggers r3d_18 inside CTCLIP)
        image_encoder = None 
        
    model = CTCLIP(
        dim_image=512,
        dim_text=768,
        dim_latent=768,
        num_classes=len(pathologies),
        image_encoder=image_encoder,
        text_encoder_name="microsoft/BiomedVLP-CXR-BERT-specialized"
    ).to(device)
    
    tokenizer = BertTokenizer.from_pretrained("microsoft/BiomedVLP-CXR-BERT-specialized")
    
    # 2. Data Setup
    print("Initializing Data...")
    try:
        val_dataset = get_dataset(dataset_name, config['init_kwargs'])
    except Exception as e:
        print(f"Error loading {dataset_name}: {e}")
        return

    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4, 
        pin_memory=True,
        # collate_fn might be needed for some datasets handling text/lists?
        # Standard default_collate usually works if data are tensors/dicts
    )
    
    # 3. Checkpoints
    ckpt_list = []
    if force_checkpoint:
        ckpt_list = [force_checkpoint]
    else:
        cp_dir = config['checkpoint_dir']
        if os.path.exists(cp_dir):
            files = os.listdir(cp_dir)
            ckpt_list = sorted([os.path.join(cp_dir, f) for f in files if f.endswith('.pt')])
    
    if not ckpt_list:
        print("No checkpoints found. Testing with initialized weights.")
        ckpt_list = [None]
        
    # 4. Loop
    for ckpt_path in ckpt_list:
        if ckpt_path:
            print(f"\n>> Loading Checkpoint: {os.path.basename(ckpt_path)}")
            try:
                checkpoint = torch.load(ckpt_path, map_location=device)
                state_dict = checkpoint.get("model_state_dict", checkpoint)
                # Fix keys
                new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
                
                # Strict=False to allow for missing keys (like if model was trained with fewer classes)
                # But careful about checking size mismatches
                msg = model.load_state_dict(new_state_dict, strict=False)
                print(f"Load Msg: {msg}")
            except Exception as e:
                print(f"Failed to load checkpoint {ckpt_path}: {e}")
                continue
        
        # Evaluate
        evaluator = ZeroShotEvaluator(
            model=model,
            dataloader=val_loader,
            device=device,
            pathologies=pathologies,
            tokenizer=tokenizer
        )
        
        try:
            auc = evaluator.evaluate(prompt_id=prompt_id)
            print(f"[{dataset_name}] Mean AUC: {auc:.4f}")
        except Exception as e:
            print(f"Evaluation failed: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="ah_knee", 
                        choices=["ah_knee", "ct_rate", "ah_chest", "radchest"],
                        help="Dataset to evaluate on")
    parser.add_argument("--prompt", type=str, default="p3", help="Prompt ID (p1-p5)")
    parser.add_argument("--ckpt", type=str, default=None, help="Specific checkpoint path")
    parser.add_argument("--bs", type=int, default=1, help="Batch Size")
    
    args = parser.parse_args()
    
    run_zero_shot(
        dataset_name=args.dataset,
        prompt_id=args.prompt,
        force_checkpoint=args.ckpt,
        batch_size=args.bs
    )
