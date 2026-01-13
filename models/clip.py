
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models.video as models
from transformers import BertModel
import numpy as np

class CTCLIP(nn.Module):
    """
    CT-CLIP Model Structure.
    Follows 'Representation Primacy' and 'Ubiquitous Supervision'.
    """
    def __init__(
        self,
        dim_text=768,
        dim_image=512,
        dim_latent=768,
        num_classes=18,
        text_encoder_name="microsoft/BiomedVLP-CXR-BERT-specialized",
        lite_version=False, 
        use_shared_classifier=True,
        image_encoder=None
    ):
        super().__init__()
        
        self.dim_text = dim_text
        self.dim_image = dim_image
        self.dim_latent = dim_latent
        self.use_shared_classifier = use_shared_classifier
        
        # 1. Text Encoder (CXR-BERT)
        # We assume local files or cache available, otherwise it downloads
        # Allow downloading weights if not cached for release portability
        self.text_encoder = BertModel.from_pretrained(text_encoder_name, local_files_only=False)
        
        # 2. Vision Encoder
        if image_encoder is not None:
            self.image_encoder = image_encoder
        else:
            # Default: 3D ResNet-18
            self.image_encoder = models.r3d_18(pretrained=True)
            
            # Modify Stem based on 'Task-Aligned Consistency' (Lite vs Standard)
            if lite_version:
                # Lite: Stride 4
                self.image_encoder.stem[0] = nn.Conv3d(
                    1, 64, kernel_size=(4, 4, 4), stride=(4, 4, 4), padding=(0, 0, 0), bias=False
                )
            else:
                # Standard: Stride 2 (Match legacy 2x2x2 config)
                self.image_encoder.stem[0] = nn.Conv3d(
                    1, 64, kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 0, 0), bias=False
                )
            
            # Replace Head and Pooling
            # VideoResNet forward calls flatten(1) always which flattens spatial dims if pool is Identity
            # We use AdaptiveAvgPool3d(1) to ensure we get [B, 512] feature vector
            self.image_encoder.avgpool = nn.AdaptiveAvgPool3d(1)
            self.image_encoder.fc = nn.Identity()
        
        # 3. Projections
        # Map both to shared latent space
        self.to_text_latent = nn.Linear(dim_text, dim_latent, bias=False)
        self.to_visual_latent = nn.Linear(dim_image, dim_latent, bias=False)
        
        # 4. Shared Classifier (Ubiquitous Supervision)
        if self.use_shared_classifier:
            self.classifier = nn.Linear(dim_latent, num_classes)
            
        # Temperature for contrastive loss
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))


    def forward(self, text_input=None, image_input=None, return_latents=False):
        """
        Forward pass handling image, text, or both.
        """
        img_latents = None
        txt_latents = None
        img_logits = None
        txt_logits = None
        
        # --- Image Forward ---
        if image_input is not None:
             # [B, 512, D, H, W] due to avgpool=Identity
            features_map = self.image_encoder(image_input)
            
            # Application of Global Average Pooling (GAP)
            # This is crucial for 'Representation Primacy' appropriately aggregating 3D info
            # [B, 512, D, H, W] -> [B, 512]
            if features_map.dim() == 5:
                features = F.adaptive_avg_pool3d(features_map, (1, 1, 1)).flatten(1)
            else:
                features = features_map.flatten(1)
                
            img_latents = self.to_visual_latent(features)
            
            if self.use_shared_classifier:
                img_logits = self.classifier(img_latents)

        # --- Text Forward ---
        if text_input is not None:
            # Expects dict with input_ids, attention_mask
            output = self.text_encoder(**text_input)
            # Use [CLS] token
            cls_token = output.last_hidden_state[:, 0, :]
            txt_latents = self.to_text_latent(cls_token)
            
            if self.use_shared_classifier:
                txt_logits = self.classifier(txt_latents)
                
        if return_latents:
            return {
                "image_latents": img_latents,
                "text_latents": txt_latents,
                "image_cls_logits": img_logits,
                "text_cls_logits": txt_logits,
                "logit_scale": self.logit_scale.exp()
            }
            
        # If not returning full dict, return whatever is computed (legacy behavior support)
        if img_latents is not None and txt_latents is not None:
             return img_latents, txt_latents
        elif img_latents is not None:
             return img_latents
        else:
             return txt_latents

if __name__ == "__main__":
    import torch
    
    # 1. Initialize Model
    print("--- Testing CTCLIP ---")
    model = CTCLIP(
        dim_text=768, 
        dim_image=512, 
        dim_latent=512, 
        num_classes=18, 
        lite_version=True
    ).cuda()
    
    # 2. Dummy Inputs
    B = 2
    # Image: [B, 1, D, H, W] -> 1 channel 3D volume
    dummy_image = torch.randn(B, 1, 96, 192, 192).cuda()
    
    # Text: dict mimicking BERT tokenizer output
    # Input IDs: [B, L]
    dummy_text = {
        "input_ids": torch.randint(0, 30000, (B, 128)).cuda(),
        "attention_mask": torch.ones(B, 128).cuda(),
        "token_type_ids": torch.zeros(B, 128).long().cuda()
    }
    
    # 3. Forward Pass
    try:
        outputs = model(text_input=dummy_text, image_input=dummy_image, return_latents=True)
        
        print("Model forward successful.")
        print(f"Image Latents: {outputs['image_latents'].shape}")
        print(f"Text Latents: {outputs['text_latents'].shape}")
        print(f"Image Logits: {outputs['image_cls_logits'].shape}")
        print(f"Text Logits: {outputs['text_cls_logits'].shape}")
        print(f"Logit Scale: {outputs['logit_scale']}")
        
    except Exception as e:
        print(f"Error during CTCLIP forward: {e}")
        import traceback
        traceback.print_exc()

