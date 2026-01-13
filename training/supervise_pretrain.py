
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
import torchvision.models.video as models
from transformers import BertModel
import numpy as np

def run_supervised_training(
    mode, # "image" or "text"
    train_loader,
    val_loader,
    config,
):
    """
    Stage 1: Independent Supervised Pre-training (Legacy Single-Card Style).
    Refactored from `sup_pretrain_image.py`.
    """
    # 1. Setup Device & Logging
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    save_dir = config.get("save_dir", "./checkpoints")
    os.makedirs(save_dir, exist_ok=True)
    
    wandb.init(project=f"LeanRad_Supervised_{mode}", config=config)

    # 2. Initialize Model
    num_classes = config.get("num_classes", 18)
    lr = config.get("lr", 1e-4) # Legacy `sup_pretrain_image.py` used 1e-4
    
    if mode == "image":
        print(f"Initializing Image Encoder (3D ResNet-18)... Lite={config.get('lite_version', False)}")
        encoder = models.r3d_18(pretrained=True)
        # Task-Aligned Consistency: Stride modification
        if config.get("lite_version", False):
             encoder.stem[0] = nn.Conv3d(1, 64, kernel_size=(4, 4, 4), stride=(4, 4, 4), padding=(0, 0, 0), bias=False)
        else:
             encoder.stem[0] = nn.Conv3d(1, 64, kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 0, 0), bias=False)
        
        # Legacy behavior: avgpool identity, manual pooling in loop
        # image_encoder.avgpool = nn.Identity()
        # image_encoder.fc = nn.Identity()
        # VideoResNet forward will flatten the output of avgpool. If avgpool is Identity, it flattens the spatial/temporal dimensions.
        # We need [B, 512], so we use AdaptiveAvgPool3d(1) before flattening.
        encoder.avgpool = nn.AdaptiveAvgPool3d(1)
        encoder.fc = nn.Identity()
        embed_dim = 512
        
    elif mode == "text":
        print("Initializing Text Encoder (CXR-BERT)...")
        encoder = BertModel.from_pretrained("microsoft/BiomedVLP-CXR-BERT-specialized", trust_remote_code=True)
        embed_dim = 768
    
    encoder = encoder.to(device)
    
    # 3. Classifier (Ubiquitous Supervision)
    # Legacy code used `classifier_related` and `classifier_unrelated` separately.
    # Here we assume single multi-label classifier for simplicity unless specified.
    classifier = nn.Linear(embed_dim, num_classes).to(device)
    
    # Optimizer
    # Legacy: list(image_encoder.parameters()) + list(classifier_related.parameters())...
    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(classifier.parameters()), 
        lr=lr,
        weight_decay=config.get("wd", 1e-4)
    )
    
    # Scheduler
    # Legacy: CosineAnnealingLR(..., T_max=epochs * num_batches)
    num_batches = len(train_loader)
    epochs = config.get("epochs", 20)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs * num_batches, eta_min=1e-6
    )
    
    criterion = nn.BCEWithLogitsLoss()
    
    print(f"Start training for {epochs} epochs on {device}...")

    # ================= Training Loop (Legacy Style) =================
    for epoch in range(epochs):
        encoder.train()
        classifier.train()
        
        # Tqdm exactly like legacy
        iter_bar = tqdm(enumerate(train_loader), total=num_batches, desc=f"Epoch {epoch+1}/{epochs} [{mode}]")
        
        for i, batch in iter_bar:
            # Data unpacking depends on dataset, assuming simplified standardized dict here
            # Legacy: inputs, _, labels, acc_no = batch
            
            if mode == "image":
                 # Check if batch is dict or tuple
                 if isinstance(batch, dict):
                     inputs = batch['image'].to(device)
                     labels = batch['labels'].float().to(device)
                 else:
                     inputs, _, labels, _ = batch # Legacy unpacking
                     inputs = inputs.to(device)
                     labels = labels.float().to(device)
                     
                 # Forward
                 # [B, 512, D, H, W]
                 map_feats = encoder(inputs)
                 
                 # Representation Primacy: Global Average Pooling (GAP)
                 # Legacy: image_features.reshape(..., 3, 6, 6) -> permute -> mean(1,2,3)
                 # Here we can simplify if we trust adaptive_avg_pool3d
                 if map_feats.dim() == 5:
                    feats = F.adaptive_avg_pool3d(map_feats, (1, 1, 1)).flatten(1)
                 else:
                    feats = map_feats.flatten(1)
                    
            else:
                 # Text
                 if isinstance(batch, dict):
                     text_inputs = {k: v.to(device) for k, v in batch['text_input'].items()}
                     labels = batch['labels'].float().to(device)
                 else:
                     # Assume text dataset returns input_ids etc
                     raise NotImplementedError("Text dataset tuple unpacking needs specific implementation")
                     
                 outputs = encoder(**text_inputs)
                 feats = outputs.last_hidden_state[:, 0, :] # [CLS]
            
            # Legacy: l2_norm check
            # if config.get("l2_norm", True): ...
            # Principle: No L2 Norm for alignment, but for supervised classification L2 norm is sometimes used before FC.
            # But recent paper suggests removing it for alignment.
            # We follow paper guidance: No L2 norm.
            
            logits = classifier(feats)
            loss = criterion(logits, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            # Logging
            wandb.log({"train_loss": loss.item()})
            iter_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        # ================= Save Checkpoint =================
        # Legacy: save_every 1000 iter, here we save per epoch
        path = os.path.join(save_dir, f"checkpoint_epoch_{epoch+1}.pt")
        state = {
            'encoder': encoder.state_dict(),
            'classifier': classifier.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch
        }
        torch.save(state, path)
        print(f"Epoch {epoch+1} finished. Saved: {path}")
            
    print("Training finished.")

if __name__ == "__main__":
    # Example Configuration for Supervised Pretraining
    config = {
        "mode": "image", # "image" or "text"
        "save_dir": "./checkpoints/sup_pretrain_ctrate",
        "num_classes": 18,
        "lr": 1e-4,
        "epochs": 5,
        "batch_size": 10,
        "lite_version": False, # Use standard ResNet3D stem 
        
        # Dataset Config
        "dataset": "ct_rate",
        "data_folder": "./data/ct_rate/images",
        "reports_csv": "./data/ct_rate/reports.csv",
        "labels_csv": "./data/ct_rate/labels.csv",
    }
    
    try:
        import sys
        import os
        # Ensure we can import from parent directory if running directly
        sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
        from data.ct_rate import CTRATEDataset
    except ImportError:
        print("LeanRad package not found. Run with python -m LeanRad.training.supervise_pretrain")
        exit(1)
        
    print(f"Running Supervised Training in mode: {config['mode']} on CT-RATE")
    
    # Dataset Loader Logic (Example for CT-RATE)
    full_dataset = CTRATEDataset(
        data_folder=config["data_folder"],
        reports_csv=config["reports_csv"],
        labels_csv=config["labels_csv"],
        section="train"
    )
    
    # Update num_classes dynamically based on dataset
    if len(full_dataset) > 0:
        sample = full_dataset[0]
        config["num_classes"] = len(sample['labels'])
        print(f"Detected Number of Classes: {config['num_classes']}")
    
    # Simple split
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_subset, val_subset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_subset, batch_size=config["batch_size"], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_subset, batch_size=config["batch_size"], shuffle=False, num_workers=4)
    
    run_supervised_training(
        mode=config["mode"],
        train_loader=train_loader,
        val_loader=val_loader,
        config=config
    )
