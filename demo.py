import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
import warnings
import nibabel as nib
import numpy as np

# Add models path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from models.clip import CTCLIP
except ImportError:
    print("Warning: Could not import CTCLIP from models.clip. Please ensure you are running from the GreenRFM root directory.")
    sys.exit(1)

# Suppress some noisy warnings for cleaner demo output
warnings.filterwarnings("ignore")

class DemoDataset(Dataset):
    def __init__(self, data_path, num_samples=1):
        self.data_path = data_path
        self.num_samples = num_samples # Duplicate the single file for batch testing
        self.pathologies = [
            "Medical material", "Arterial wall calcification", "Cardiomegaly", 
            "Pericardial effusion", "Coronary artery wall calcification", 
            "Hiatal hernia", "Lymphadenopathy", "Emphysema", "Atelectasis", 
            "Lung nodule", "Lung opacity", "Pulmonary fibrotic sequela", 
            "Pleural effusion", "Mosaic attenuation pattern", 
            "Peribronchial thickening", "Consolidation", "Bronchiectasis", 
            "Interlobular septal thickening"
        ]
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Load NIfTI file
        image = None
        try:
            if os.path.exists(self.data_path):
                nii_img = nib.load(self.data_path)
                data = nii_img.get_fdata()
                # Assuming data is [192, 192, 96] -> [W, H, D]
                # PyTorch 3D Convention: [C, D, H, W]
                # Transpose to [D, H, W] -> [96, 192, 192]
                tensor_data = torch.from_numpy(data).float()
                
                # Permute: (2, 1, 0)
                tensor_data = tensor_data.permute(2, 1, 0)
                
                image = tensor_data.unsqueeze(0) # [1, 96, 192, 192]
            else:
                raise FileNotFoundError(f"{self.data_path} not found")
            
        except Exception as e:
            # Fallback to random noise if file load fails
            image = torch.randn(1, 96, 192, 192)
        
        # Dummy Label (Multi-label binary vector)
        label = torch.randint(0, 2, (len(self.pathologies),)).float()
        
        # Dummy Report Text
        report = "Small consolidation in the right lower lobe. No pleural effusion. Heart size is normal."
        
        return {
            "image": image,
            "label": label,
            "report_text": report
        }

def run_demo():
    print("============================================================")
    print("   GreenRFM Demo: Zero-Shot Classification on Synthetic Data")
    print("============================================================")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Initialize Model
    print("\n[1/4] Initializing CT-CLIP Model (Lite Version)...")
    # Using 'bert-base-uncased' as a fallback if cxr-bert is large/slow, but keeping default if possible.
    # We'll stick to default but catch errors if download fails.
    try:
        model = CTCLIP(
            dim_text=768,
            dim_image=512,
            dim_latent=768,
            num_classes=18,
            text_encoder_name="bert-base-uncased", # Faster download for demo
            lite_version=True # Use lite version for speed
        ).to(device)
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    model.eval()
    print("      Model initialized successfully.")

    # 2. Prepare Data
    print("\n[2/4] Preparing Dataset...")
    
    # Path to sample data
    demo_file = os.path.join(os.path.dirname(__file__), "demo_data", "sample_ct.nii.gz")
    
    # Create Demo Dataset
    dataset = DemoDataset(data_path=demo_file, num_samples=2)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    print(f"      Loading data from: {demo_file}")
    if not os.path.exists(demo_file):
         print("      [Warning] Demo file not found. Falling back to synthetic noise inside Dataset.")

    # 3. Define Pathologies & Prompts
    pathologies = dataset.pathologies
    prompts = [f"Findings consistent with {p}." for p in pathologies]
    neg_prompts = [f"No evidence of {p}." for p in pathologies]
    
    print("\n[3/4] Encoding Text Prompts (Zero-Shot Prototypes) using model.forward()...")
    
    # Store positive and negative embeddings
    pos_embeddings = []
    neg_embeddings = []
    
    with torch.no_grad():
        # Encode Positive Prompts
        inputs = tokenizer(prompts, return_tensors='pt', padding=True, truncation=True).to(device)
        # Use model forward for text
        # Returns dict with 'text_latents' [B, Dim]
        out_pos = model(text_input=inputs, return_latents=True)
        pos_emb = out_pos['text_latents']
        # Magnitude is preserved (No L2 Norm)
        
        # Encode Negative Prompts
        inputs_neg = tokenizer(neg_prompts, return_tensors='pt', padding=True, truncation=True).to(device)
        out_neg = model(text_input=inputs_neg, return_latents=True)
        neg_emb = out_neg['text_latents']
        # Magnitude is preserved (No L2 Norm)
            
    # Stack [Num_Classes, Dim]
    pos_emb = pos_emb # [18, Dim]
    neg_emb = neg_emb # [18, Dim]
    
    print(f"      Encoded {len(pathologies)} pathology prompts.")

    # 4. Run Inference
    print("\n[4/4] Running Inference Loop...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            images = batch['image'].to(device)
            # report_texts = batch['report_text'] # Not used for Zero-Shot image inference
            
            # Forward Image via model.forward
            # Returns dict with 'image_latents' [B, Dim]
            img_out = model(image_input=images, return_latents=True)
            img_feat = img_out['image_latents']
            # Magnitude is preserved (No L2 Norm)
            
            # Zero-Shot Calculation
            # For each pathology, we compare image against (Pos, Neg) text prototypes
            # Similarity: Cosine similarity
            
            # Sim Positive: [B, Dim] @ [P, Dim].T -> [B, P]
            sim_pos = img_feat @ pos_emb.T
            # Sim Negative: [B, Dim] @ [P, Dim].T -> [B, P]
            sim_neg = img_feat @ neg_emb.T
            
            # Stack to [B, P, 2] -> [Pos, Neg]
            logits = torch.stack([sim_pos, sim_neg], dim=-1) # [B, P, 2]
            
            # Softmax over the last dimension (Pos vs Neg)
            # This represents P(Positive | Image) vs P(Negative | Image)
            probs = torch.softmax(logits * model.logit_scale.exp(), dim=-1)[:, :, 0] # [B, P]
            
            print(f"      Batch {batch_idx+1}: Processed {len(images)} images.")
            print(f"      Sample Output (First Image in Batch):")
            for i in range(3): # Show first 3 pathologies
                p_name = pathologies[i]
                p_prob = probs[0, i].item()
                print(f"        - {p_name}: {p_prob:.4f}")
            print("      ...")

    print("\n============================================================")
    print("   Demo Completed Successfully!")
    print("============================================================")

if __name__ == "__main__":
    run_demo()
