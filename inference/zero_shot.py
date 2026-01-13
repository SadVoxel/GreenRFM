
import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score

class ZeroShotEvaluator:
    def __init__(self, model, dataloader, device, pathologies, tokenizer):
        self.model = model
        self.dataloader = dataloader
        self.device = device
        self.pathologies = pathologies
        self.tokenizer = tokenizer
        # Prompts from Paper
        self.prompts = {
            "p1": lambda p: (f"{p}.", f"not {p}."),
            "p2": lambda p: (f"{p}.", ""),
            "p3": lambda p: (f"There is {p}.", f"There is no {p}."),
            "p4": lambda p: (f"{p} is present.", f"{p} is not present."),
            "p5": lambda p: (f"Findings are compatible with {p}.", f"Findings are not compatible with {p}.")
        }
        
    def evaluate(self, prompt_id="p1"):
        self.model.eval()
        
        # 1. Precompute Text Embeddings for all pathologies
        # Shape: [Num_Pathologies, 2, Dim] -> [Pos, Neg]
        
        text_embeddings = [] # List of tuples (pos_emb, neg_emb)
        
        print("Encoding Prompts...")
        with torch.no_grad():
            for p in self.pathologies:
                pos_txt, neg_txt = self.prompts[prompt_id](p)
                texts = [pos_txt, neg_txt]
                # Tokenizer: Expects list of strings
                # Ensure truncation/padding
                inputs = self.tokenizer(
                    texts, 
                    return_tensors='pt', 
                    padding=True, 
                    truncation=True, 
                    max_length=128
                ).to(self.device)
                
                # Forward Text Encoder (via model wrapper to get projection)
                # We use the model's forward or manual calls
                # CTCLIP.forward(text_input=..., return_latents=True) -> {'text_latents': ...}
                
                outputs = self.model(text_input=inputs, return_latents=True)
                latents = outputs['text_latents'] # [2, Dim]
                
                # NO L2 Norm (as per training)
                text_embeddings.append(latents)
                
                # Stack: [num_classes, 2, dim]
        text_embeddings = torch.stack(text_embeddings) # On GPU
        
        # 2. Inference Loop
        all_preds = []
        all_labels = []
        
        print("Inference on Images...")
        with torch.no_grad():
            for batch in tqdm(self.dataloader):
                # Handle Image Input (Tensor vs Dict)
                if isinstance(batch, dict):
                    # Check if it's MR (keys like img_sag) or just single 'image' key
                    if 'image' in batch:
                        img_input = batch['image'].to(self.device)
                    else:
                        # Assumes MR dict with img_* keys
                        img_input = {k: v.to(self.device) for k, v in batch.items() if k.startswith("img_")}
                elif isinstance(batch, (list, tuple)):
                    # Assume index 0 is image
                    img_input = batch[0].to(self.device)
                else:
                    # Generic tensor
                    img_input = batch.to(self.device)
                
                # Check for list/tuple which happens if DataLoader yields [img, label]
                if isinstance(img_input, list):
                     img_input = [x.to(self.device) for x in img_input] if isinstance(img_input[0], torch.Tensor) else img_input

                # Labels
                # Batch might have 'labels' key (standard in MRDataset)
                if isinstance(batch, dict) and 'labels' in batch:
                    # Move to CPU numpy
                    labels = batch['labels'].numpy()
                elif isinstance(batch, (list, tuple)) and len(batch) > 2:
                     # e.g. [img, text, label, id] or [img, label, ...]
                     # AHChest returns [img, text, labels, id] -> index 2 is labels
                     # But some might be [img, label]. Need to be careful or generic.
                     # AHChestBaseVolumeDataset __getitem__: return video_tensor, text, labels, subject_id
                     # So labels is index 2.
                     
                     # Check type of index 1
                     if isinstance(batch[1], (torch.Tensor, np.ndarray)) and batch[1].shape[-1] == len(self.pathologies):
                          labels = batch[1].numpy()
                     elif len(batch) > 2 and isinstance(batch[2], (torch.Tensor, np.ndarray)):
                          labels = batch[2].numpy()
                     else:
                          # Fallback/Guess
                          labels = np.zeros((len(img_input) if hasattr(img_input, '__len__') else 1, len(self.pathologies)))
                elif isinstance(batch, (list, tuple)) and len(batch) == 2:
                      labels = batch[1].numpy()
                else:
                    # Provide dummy or handle differently? 
                    # Assuming we are running eval on labeled data
                    labels = np.zeros((len(batch) if not isinstance(batch, dict) else len(list(batch.values())[0]), len(self.pathologies))) # Fallback
                
                # Image Forward
                outputs = self.model(image_input=img_input, return_latents=True)
                img_latents = outputs['image_latents'] # [B, Dim]

                
                # Score: Dot Product
                # img_latents: [B, D]
                # text_embeddings: [C, 2, D]
                
                # Broadcast dot product
                # Result: [B, C, 2] -> Sim with Pos, Sim with Neg
                
                # Einsum: b=batch, c=class, t=type(pos/neg), d=dim
                sims = torch.einsum('bd,ctd->bct', img_latents, text_embeddings) 
                
                # Softmax over t dimension (Pos vs Neg)
                probs = F.softmax(sims, dim=2) # [B, C, 2]
                pos_probs = probs[:, :, 0] # Prob of Positive
                
                all_preds.append(pos_probs.cpu().numpy())
                all_labels.append(labels)
                
        all_preds = np.concatenate(all_preds, axis=0) # [N, C]
        all_labels = np.concatenate(all_labels, axis=0)
        
        # 3. Compute Metrics
        aucs = []
        for i, p in enumerate(self.pathologies):
            try:
                auc = roc_auc_score(all_labels[:, i], all_preds[:, i])
            except:
                auc = 0.5
            aucs.append(auc)
            print(f"{p}: {auc:.4f}")
            
        print(f"Mean AUC: {np.mean(aucs):.4f}")
        return np.mean(aucs)

