
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
import numpy as np

def contrastive_loss(text_latents, image_latents, logit_scale=None):
    """
    Computes Contrastive Loss WITHOUT L2 Normalization.
    Principle: Task-Aligned Consistency.
    Match CTCLIPTrainer5 behavior: Remove logit_scale scaling implies temp=1.0.
    """
    # Dot product similarity (No F.normalize)
    # Shapes: [B, D]
    sim_i2t = torch.matmul(image_latents, text_latents.t()) 
    
    # If using standard CLIP, we would multiply by logit_scale.
    # But to match CTCLIPTrainer5 (which has faster convergence without L2 norm), we ignore it.
    # if logit_scale is not None:
    #     sim_i2t = sim_i2t * logit_scale

    sim_t2i = sim_i2t.t()
    
    labels = torch.arange(len(image_latents), device=image_latents.device)
    loss_i2t = F.cross_entropy(sim_i2t, labels)
    loss_t2i = F.cross_entropy(sim_t2i, labels)
    return (loss_i2t + loss_t2i) / 2


def run_alignment_training(
    model,
    train_loader,
    val_loader,
    optimizer,
    config
):
    """
    Stage 2: Alignment Training (Legacy Single-Card Style).
    Ubiquitous Supervision: Loss = L_contrastive + lambda * (L_cls_img + L_cls_text)
    """
    # 1. Setup Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    save_dir = config.get("save_dir", "./checkpoints/alignment")
    os.makedirs(save_dir, exist_ok=True)
    
    wandb.init(project=config.get("project_name", "LeanRad_Alignment"), config=config)
        
    model = model.to(device)
    
    cls_loss_fn = nn.BCEWithLogitsLoss()
    scheduler = config.get("scheduler", None)
    epochs = config.get("epochs", 20)
    
    print(f"Start Alignment Training for {epochs} epochs...")
    
    # ================= Training Loop =================
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        
        num_batches = len(train_loader)
        iter_bar = tqdm(enumerate(train_loader), total=num_batches, desc=f"Align Epoch {epoch+1}/{epochs}")
        
        for i, batch in iter_bar:
            # Data unpacking
            if isinstance(batch, dict):
                images = batch['image'].to(device)
                # Handle text input dict
                if isinstance(batch.get('text_input'), dict):
                     text_input = {k: v.to(device) for k, v in batch['text_input'].items()}
                else:
                     text_input = batch['text_input'].to(device)
                labels = batch['labels'].float().to(device)
            else:
                # Fallback for simpler datasets / old legacy returning tuple
                # Assuming: images, texts, labels = batch or similar
                # This needs strict dataset agreement
                raise NotImplementedError("Batch unpacking requires dict format currently.")
                
            # --- Forward ---
            # Return dict with latents and logits
            outputs = model(text_input, images, return_latents=True)
            
            # --- Loss Calculation ---
            # 1. Contrastive Loss (No L2 Norm)
            loss_clip = contrastive_loss(
                outputs['text_latents'], 
                outputs['image_latents'], 
                outputs['logit_scale']
            )
            
            # 2. Shared Classifier Loss (Ubiquitous Supervision)
            loss_cls_img = cls_loss_fn(outputs['image_cls_logits'], labels)
            loss_cls_txt = cls_loss_fn(outputs['text_cls_logits'], labels)
            
            # Total loss
            loss = loss_clip + (loss_cls_img + loss_cls_txt) / 2.0
            
            # --- Backward ---
            optimizer.zero_grad()
            loss.backward()
            
            # Max Gradient Clip
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)

            optimizer.step()
            
            if scheduler:
                scheduler.step()
            
            # Logging
            epoch_loss += loss.item()
            wandb.log({
                "loss_total": loss.item(),
                "loss_clip": loss_clip.item(),
                "loss_cls_img": loss_cls_img.item(),
                "loss_cls_txt": loss_cls_txt.item()
            })
            iter_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        # ================= Save Checkpoint =================
        print(f"Epoch {epoch+1} finished. Avg Loss: {epoch_loss / num_batches:.4f}")
        path = os.path.join(save_dir, f"checkpoint_epoch_{epoch+1}.pt")
        
        state = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch
        }
        torch.save(state, path)
        print(f"Saved: {path}")

if __name__ == "__main__":
    # Example Configuration for CT-RATE Alignment (aligned to train_clip setup)
    config = {
        "project_name": "LeanRad_CT_Align_CTRATE",
        "save_dir": "./checkpoints/ct_align_ctrate",
        "num_classes": 18, 
        "bert_name": "microsoft/BiomedVLP-CXR-BERT-specialized",
        "lr": 1e-5,
        "epochs": 5,
        "batch_size": 10, # CT-CLIP is memory hungry
        "wd": 0.0,
        
        # Dataset Config
        "dataset": "ct_rate",
        "train_data_folder": "./data/ct_rate/train_images",
        "val_data_folder": "./data/ct_rate/val_images",
        "train_reports_csv": "./data/ct_rate/train_reports.csv",
        "val_reports_csv": "./data/ct_rate/val_reports.csv",
        "train_labels_csv": "./data/ct_rate/train_labels.csv",
        "val_labels_csv": "./data/ct_rate/val_labels.csv",
        
    }
    
    try:
        import sys
        import os
        from transformers import BertTokenizer, BertModel, BertConfig
        sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
        from models.clip import CTCLIP
        from data.ct_rate import CTRATEDataset
    except ImportError:
        print("Could not import modules. Ensure python path is correct.")
        exit(1)

    print("Initializing Datasets...")
    train_dataset = CTRATEDataset(
        data_folder=config["train_data_folder"],
        reports_csv=config["train_reports_csv"],
        labels_csv=config["train_labels_csv"],
        section="train"
    )

    val_dataset = CTRATEDataset(
        data_folder=config["val_data_folder"],
        reports_csv=config["val_reports_csv"],
        labels_csv=config["val_labels_csv"],
        section="train"
    )

    if len(train_dataset) > 0:
        sample = train_dataset[0]
        config["num_classes"] = len(sample['labels'])
        print(f"Detected Num Classes: {config['num_classes']}")
    
    # Text Tokenizer and Collate Function
    tokenizer = BertTokenizer.from_pretrained(config["bert_name"], do_lower_case=True, local_files_only=False)
    
    def collate_fn(batch):
        # Default collate only works if fields are tensors. Text is string.
        # Custom collate to tokenize text on the fly.
        images = [item['image'] for item in batch]
        images = torch.stack(images, dim=0)
        
        labels = [item['labels'] for item in batch]
        labels = torch.stack(labels, dim=0)
        
        texts = [item['text'] for item in batch]
        text_inputs = tokenizer(
            texts, 
            padding="max_length",
            truncation=True, 
            max_length=128, 
            return_tensors="pt"
        )
        
        return {
            "image": images,
            "labels": labels,
            "text_input": text_inputs
        }

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=4, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=4, collate_fn=collate_fn)
    
    print("Initializing Model...")
    model = CTCLIP(
        num_classes=config["num_classes"],
        text_encoder_name=config["bert_name"],
        dim_image=512,
        dim_text=768,
        dim_latent=768
    )

    # Optimizer defines (align to train_clip: AdamW with decoupled weight decay excluding LN/bias)
    optimizer = torch.optim.AdamW(
        [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in ['bias', 'LayerNorm.weight'])], 'weight_decay': config.get("wd", 0.0)},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in ['bias', 'LayerNorm.weight'])], 'weight_decay': 0.0}
        ],
        lr=config.get("lr", 1e-5)
    )

    
    run_alignment_training(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        config=config
    )

