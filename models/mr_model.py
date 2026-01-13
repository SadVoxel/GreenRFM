
import torch
import torch.nn as nn
import torchvision.models.video as models

class MultiPlaneNet(nn.Module):
    """
    MR-Specific Backbone.
    Aggregates features from 3 orthogonal planes (Sagittal, Coronal, Axial).
    Used for 'Representation Primacy' in MR tasks.
    """
    def __init__(self, in_ch: int, pretrained: bool = True):
        super().__init__()

        def _make_backbone():
            # 3D ResNet-18
            # Weights: KINETICS400_V1 is standard for video/3D
            net = (models.r3d_18(weights="KINETICS400_V1") if pretrained else models.r3d_18())
            
            # Conv3d Customization for input channels
            # Legacy code specific: kernel=(1,2,2), stride=(1,2,2). 
            # This is anisotropic, likely due to thick slices in Z direction?
            net.stem[0] = nn.Conv3d(
                in_ch, 64, 
                kernel_size=(1, 2, 2),
                stride=(1, 2, 2), 
                padding=0, 
                bias=False
            )
            
            # Replace pooling/head with Identity to extract features manually
            net.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
            net.fc = nn.Identity()
            return net

        self.planes = ["sag", "cor", "tra"]
        self.backbones = nn.ModuleDict({p: _make_backbone() for p in self.planes})

    def forward(self, batch):
        """
        Expects batch to be a dict with keys 'img_sag', 'img_cor', 'img_tra'.
        Returns pooled feature vector [B, 512].
        """
        # Feature extraction per plane
        plane_feats = []
        for p in self.planes:
            img = batch[f"img_{p}"] # [B, C, D, H, W]
            feat = self.backbones[p](img) # [B, 512] (due to AdaptiveAvgPool being inside _make_backbone)
            
            # Wait, `net.avgpool` is AdaptiveAvgPool3d((1,1,1)). 
            # Output of net before fc is [B, 512, 1, 1, 1].
            # Flatten it.
            feat = feat.flatten(1)
            plane_feats.append(feat)
            
        # Fusion Strategy: Mean (as per legacy)
        # Legacy: torch.stack(..., 0).mean(0)
        feats = torch.stack(plane_feats, dim=0).mean(dim=0) # [B, 512]
        
        return feats

class MultiPlaneClassifier(nn.Module):
    """
    Wrapper for Supervised Training of MR.
    """
    def __init__(self, backbone: MultiPlaneNet, num_classes: int):
        super().__init__()
        self.backbone = backbone
        # Ubiquitous Supervision
        self.classifier = nn.Linear(512, num_classes)
        
    def forward(self, batch):
        feats = self.backbone(batch)
        logits = self.classifier(feats)
        return logits

if __name__ == "__main__":
    import torch
    
    print("--- Testing MultiPlaneNet / MultiPlaneClassifier ---")
    
    # 1. Initialize Backbone
    backbone = MultiPlaneNet(in_ch=3, pretrained=False).cuda()
    
    # 2. Initialize Classifier
    model = MultiPlaneClassifier(backbone, num_classes=14).cuda()
    
    # 3. Dummy Inputs (Sequence Dict)
    # Assumes inputs are [B, 3, D, H, W] (e.g. 3 slices/channels per plane or stack)
    B = 2
    dummy_input = {
        "img_sag": torch.randn(B, 3, 20, 192, 192).cuda(),
        "img_cor": torch.randn(B, 3, 20, 192, 192).cuda(),
        "img_tra": torch.randn(B, 3, 20, 192, 192).cuda()
    }
    
    try:
        # 4. Forward Backbone
        feats = backbone(dummy_input)
        print(f"Backbone Features: {feats.shape}") # Should be [B, 512]
        
        # 5. Forward Classifier
        logits = model(dummy_input)
        print(f"Classifier Logits: {logits.shape}") # Should be [B, 14]
        
    except Exception as e:
        print(f"Error during MR Model forward: {e}")
        import traceback
        traceback.print_exc()
