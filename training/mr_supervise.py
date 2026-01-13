
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
import numpy as np

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.mr_model import MultiPlaneNet, MultiPlaneClassifier

def multi_label_loss(logits, targets, mask):
    """
    Masked BCE loss.
    targets: [B, K] (-1 indicates missing)
    mask: [B, K] (1 present, 0 missing)
    """
    # Simply use BCEWithLogitsLoss(reduction='none') and mask
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    masked_loss = bce * mask
    
    # Avoid div by zero
    valid_elements = mask.sum()
    if valid_elements > 0:
        return masked_loss.sum() / valid_elements
    else:
        return masked_loss.sum() * 0.0

def run_mr_supervised_training(
    train_loader,
    val_loader,
    config,
):
    """
    Stage 1: MR Supervised Pre-training.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    save_dir = config.get("save_dir", "./checkpoints/mr_sup")
    os.makedirs(save_dir, exist_ok=True)
    
    wandb.init(project=config.get("project_name", "LeanRad_MR_Sup"), config=config)

    # 1. Model
    in_ch = config.get("in_ch", 3) # default 3 sequences
    num_classes = config.get("num_classes", 13) # knee=13
    
    backbone = MultiPlaneNet(in_ch=in_ch, pretrained=True)
    model = MultiPlaneClassifier(backbone, num_classes).to(device)
    
    # 2. Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.get("lr", 1e-4), weight_decay=config.get("wd", 1e-4))
    
    epochs = config.get("epochs", 20)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * len(train_loader))
    
    print(f"Start MR Training on {device}...")
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        
        iter_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in iter_bar:
            # Move images
            # Batch contains 'img_sag', 'img_cor', 'img_tra' inside
            batch_gpu = {k: v.to(device) for k, v in batch.items() if k.startswith("img_")}
            
            labels = batch['labels'].float().to(device)
            mask = batch['label_mask'].float().to(device)
            
            # Forward
            logits = model(batch_gpu)
            
            # Loss
            loss = multi_label_loss(logits, labels, mask)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            epoch_loss += loss.item()
            wandb.log({"train_loss": loss.item()})
            iter_bar.set_postfix({"loss": f"{loss.item():.4f}"})
            
        # Save
        print(f"Epoch {epoch+1} done. Avg Loss: {epoch_loss/len(train_loader):.4f}")
        torch.save(model.state_dict(), os.path.join(save_dir, f"epoch_{epoch+1}.pt"))

if __name__ == "__main__":
    # Example Config for MR Supervised Training
    config = {
        "project_name": "LeanRad_MR_Sup_Debug",
        "save_dir": "./checkpoints/mr_sup_debug",
        "in_ch": 3,
        "num_classes": 13, # Knee
        "lr": 1e-4,
        "epochs": 5,
        "batch_size": 2,
        
        "dataset": "ah_knee",
        # NOTE: Update these paths to real data
        "img_dir": "/data/yingtaili/AH-Musculo-skeletal/knee-image-preprocessed-nii/train",
        "label_csv": "/data/yingtaili/AH-Musculo-skeletal/Labels/train_knee_doubao_2507.csv", 
    }
    
    try:
        from data.ah_knee import AHKneeDataset
    except ImportError:
        try:
            from LeanRad.data import AHKneeDataset
        except ImportError:
            print("Could not import AHKneeDataset. Ensure python path is correct.")
            exit(1)
        
    print("Setting up datasets...")
    # MRDataset expects img_dir, label_csv. (No report needed for purely supervised if labels are enough, but let's check init)
    # MRDataset __init__ loads reports if report_xlsx is provided, otherwise empty.
    
    train_dataset = AHKneeDataset(
        img_dir=config["img_dir"],
        label_csv=config["label_csv"],
        train_mode=True
        # anatomy="knee" is default in AHKneeDataset
    )
    
    val_dataset = AHKneeDataset(
        img_dir=config["img_dir"],
        label_csv=config["label_csv"],
        train_mode=False
    )
    
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)
    
    run_mr_supervised_training(train_loader, val_loader, config)

