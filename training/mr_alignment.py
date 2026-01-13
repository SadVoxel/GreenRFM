
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
from models.mr_model import MultiPlaneNet
from models.clip import CTCLIP 
from transformers import BertModel, BertTokenizer

def contrastive_loss(text_latents, image_latents, logit_scale=None):
    """
    Computes Contrastive Loss WITHOUT L2 Normalization.
    Shapes: [B, D]
    CTCLIPTrainer5 behavior: Remove logit_scale scaling implies temp=1.0.
    """
    sim_i2t = torch.matmul(image_latents, text_latents.t()) 
    sim_t2i = sim_i2t.t()
    
    labels = torch.arange(len(image_latents), device=image_latents.device)
    loss_i2t = F.cross_entropy(sim_i2t, labels)
    loss_t2i = F.cross_entropy(sim_t2i, labels)
    return (loss_i2t + loss_t2i) / 2

def run_mr_alignment_training(
    train_loader,
    val_loader,
    config
):
    """
    Stage 2: MR Alignment Training.
    Ubiquitous Supervision: Loss = L_contrastive + lambda * (L_cls_img + L_cls_text)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    save_dir = config.get("save_dir", "./checkpoints/mr_align")
    os.makedirs(save_dir, exist_ok=True)
    
    wandb.init(project=config.get("project_name", "LeanRad_MR_Align"), config=config)
    
    # 1. Initialize Model using CTCLIP with MultiPlaneNet backbone
    in_ch = config.get("in_ch", 3)
    # Default knee 13, spine 8 etc.
    num_classes = config.get("num_classes", 13)
    bert_name = config.get("bert_name", "microsoft/BiomedVLP-CXR-BERT-specialized")
    
    print(f"Initializing MultiPlaneNet(in_ch={in_ch}) injected into CTCLIP...")
    mr_backbone = MultiPlaneNet(in_ch=in_ch, pretrained=True)
    
    # Feature dim of MR backbone is 512 (after mean pooling)
    model = CTCLIP(
        text_encoder_name=bert_name,
        dim_image=512,       
        dim_text=768,
        dim_latent=768,
        num_classes=num_classes,
        use_shared_classifier=True,
        image_encoder=mr_backbone 
    ).to(device)
    
    # 2. Optimizer & Scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config.get("lr", 1e-5), 
        weight_decay=config.get("wd", 0.0)
    )
    epochs = config.get("epochs", 20)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs * len(train_loader)
    )
    
    # Masked BCE for multi-label
    cls_loss_fn = nn.BCEWithLogitsLoss(reduction='none')

    # Tokenizer
    tokenizer = BertTokenizer.from_pretrained(bert_name)

    print("Start MR Alignment Training...")
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        
        iter_bar = tqdm(train_loader, desc=f"Align Epoch {epoch+1}/{epochs}")
        
        for batch in iter_bar:
            # 1. Prepare Data
            # Handling MR specific dict inputs
            # Only keep img_ keys for image_input dict
            img_input = {k: v.to(device, non_blocking=True) for k, v in batch.items() if k.startswith("img_")}
            
            # Text is typically "report" key string list
            raw_text = batch.get("report", [])
            if isinstance(raw_text, str): raw_text = [raw_text]
            
            # Tokenize on fly
            text_tokens = tokenizer(
                list(raw_text), 
                return_tensors='pt', 
                padding="max_length",
                truncation=True, 
                max_length=config.get("max_length", 256)
            ).to(device)
            
            labels = batch['labels'].float().to(device)
            mask = batch['label_mask'].float().to(device)
            
            # 2. Forward
            # CTCLIP expects image_input to be passed to image_encoder.
            # MultiPlaneNet expects dict. Perfect match.
            outputs = model(
                text_input=text_tokens, 
                image_input=img_input, 
                return_latents=True
            )
            
            # 3. Loss Calculation
            # Contrastive
            loss_clip = contrastive_loss(
                outputs['text_latents'], 
                outputs['image_latents'], 
                outputs['logit_scale']
            )
            
            # Classification (Ubiquitous Supervision)
            # Masked BCE
            l_img_raw = cls_loss_fn(outputs['image_cls_logits'], labels) * mask
            l_txt_raw = cls_loss_fn(outputs['text_cls_logits'], labels) * mask
            
            valid_elements = mask.sum().clamp(min=1.0)
            l_img = l_img_raw.sum() / valid_elements
            l_txt = l_txt_raw.sum() / valid_elements
            
            loss = loss_clip + (l_img + l_txt) / 2.0
            
            # 4. Backward
            optimizer.zero_grad()
            loss.backward()
            
            # Max Gradient Clip
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            
            optimizer.step()
            scheduler.step()
            
            # 5. Logging
            epoch_loss += loss.item()
            wandb.log({
                "loss_total": loss.item(),
                "loss_clip": loss_clip.item(),
                "loss_cls": (l_img + l_txt).item()
            })
            iter_bar.set_postfix({"loss": f"{loss.item():.4f}"})
            
        print(f"Epoch {epoch+1} done. Avg Loss: {epoch_loss/len(train_loader):.4f}")
        
        # Save Checkpoint
        state = {
            "model_state_dict": model.state_dict(),
            "config": config,
            "epoch": epoch
        }
        torch.save(state, os.path.join(save_dir, f"epoch_{epoch+1}.pt"))

if __name__ == "__main__":
    # Example Configuration
    config = {
        "project_name": "LeanRad_MR_Align_Local",
        "save_dir": "./checkpoints/mr_align_debug",
        "in_ch": 3,
        "num_classes": 14, # Knee
        "bert_name": "microsoft/BiomedVLP-CXR-BERT-specialized",
        "lr": 1e-5,
        "wd": 0.1,
        "epochs": 5,
        "batch_size": 8,
        "num_workers": 0, # Debug
        "max_length": 256,
        
        # Dataset Config
        "dataset": "ah_knee",
        "img_dir": "./data/knee/images", # Example path
        "label_csv": "./data/knee/labels.csv",
        "report_xlsx": "./data/knee/reports.xlsx",
    }
    
    # Import Datasets locally to avoid circular dependencies at top level if any
    try:
        from data.ah_knee import AHKneeDataset
    except ImportError:
        try:
            from LeanRad.data import AHKneeDataset
        except ImportError:
            print("Could not import AHKneeDataset. python path issue.")
            exit(1)
        
    print("Initializing Datasets...")
    # Using 'valid' data for both train/val just for demonstration/debugging
    train_dataset = AHKneeDataset(
        img_dir=config["img_dir"],
        label_csv=config["label_csv"],
        report_xlsx=config["report_xlsx"],
        train_mode=True,
        # anatomy="knee"
    )
    
    val_dataset = AHKneeDataset(
        img_dir=config["img_dir"],
        label_csv=config["label_csv"],
        report_xlsx=config["report_xlsx"],
        train_mode=False,
        # anatomy="knee"
    )
    
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=config["num_workers"])
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=config["num_workers"])
    
    run_mr_alignment_training(train_loader, val_loader, config)
