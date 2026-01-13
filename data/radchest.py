import os
import glob
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from functools import partial
import monai.data as data
from typing import Optional, List, Tuple

class RadChestCTDataset(data.PersistentDataset):
    def __init__(self, 
                 data_folder: str, 
                 labels_csv: str, 
                 min_slices=20, 
                 resize_dim=500, 
                 target_shape=(192,192,96)):
        self.data_folder = data_folder
        self.labels_csv = labels_csv
        self.min_slices = min_slices
        self.target_shape = target_shape

        self.transform = transforms.Compose([
            transforms.Resize((resize_dim,resize_dim)),
            transforms.ToTensor()
        ])
        
        self.samples = self.prepare_samples()
        
    def prepare_samples(self):
        samples = []
        accession_folder = glob.glob(os.path.join(self.data_folder, '*')) # Files
        
        # Read labels
        if not os.path.exists(self.labels_csv):
            print(f"Warning: Label CSV {self.labels_csv} not found.")
            return []
            
        test_df = pd.read_csv(self.labels_csv)
        # Assuming format from legacy: NoteAcc_DEID, label1, label2...
        if 'NoteAcc_DEID' not in test_df.columns:
            # Fallback or check columns
            pass
            
        test_label_cols = list(test_df.columns[1:])
        # Pre-compute labels map for speed
        # test_df['one_hot_labels'] = list(test_df[test_label_cols].values) 
        # Making a dict is faster than filtering DF in loop
        
        label_map = {}
        for _, row in test_df.iterrows():
            acc = str(row['NoteAcc_DEID'])
            labels = row[test_label_cols].values.astype(np.float32)
            label_map[acc] = labels

        for nii_file in accession_folder:
            # Parse accession from filename
            # filename format assumed: .../Accession.npz or .nii.gz?
            # Legacy used `nii_file.split("/")[-1].split(".")[0]`
            basename = os.path.basename(nii_file)
            accession_number = basename.split(".")[0]
            
            if accession_number in label_map:
                samples.append((nii_file, label_map[accession_number]))
        
        return samples

    def __len__(self):
        return len(self.samples)

    def nii_img_to_tensor(self, path):
        # Legacy loaded .npz with 'arr_0'. 
        # Checks if file is npz or nii.
        if path.endswith('.npz'):
            img_data = np.load(path)['arr_0']
        else:
            # Fallback for NifTI if needed, but legacy was specific about npz in one place
            # but variable name was `nii_file`.
            # We assume npz based on legacy `np.load(path)['arr_0']`.
            # If it fails, we try nib.
            import nibabel as nib
            img_data = nib.load(path).get_fdata()

        # Legacy Transpose: (1, 2, 0) -> loaded as [D, H, W] maybe? 
        # Typically npz save might be [D, H, W].
        # Transpose to [H, W, D]? 
        # Legacy: `np.transpose(img_data, (1, 2, 0))` 
        
        if img_data.ndim == 3:
             img_data = np.transpose(img_data, (1, 2, 0))

        # Value normalization
        # Legacy: img_data * 1000 ? Then clip -1000, 200.
        # This implies stored data was divided by 1000? Or just normalization step?
        # "去掉下面4行,就对应[-1000,1000]" comment suggests raw is not HU?
        # Let's follow the code:
        img_data = img_data * 1000
        hu_min, hu_max = -1000, 200 
        img_data = np.clip(img_data, hu_min, hu_max)
        img_data = (((img_data+400 ) / 600)).astype(np.float32)

        tensor = torch.tensor(img_data) # [H, W, D]
        
        # Crop/Pad
        target_shape = self.target_shape # (192,192,96)
        h, w, d = tensor.shape
        dh, dw, dd = target_shape
        
        h_start = max((h - dh) // 2, 0)
        h_end = min(h_start + dh, h)
        w_start = max((w - dw) // 2, 0)
        w_end = min(w_start + dw, w)
        d_start = max((d - dd) // 2, 0)
        d_end = min(d_start + dd, d)

        tensor = tensor[h_start:h_end, w_start:w_end, d_start:d_end]

        pad_h = dh - tensor.size(0)
        pad_w = dw - tensor.size(1)
        pad_d = dd - tensor.size(2)
        
        # symmetric padding
        pad_h_before = pad_h // 2
        pad_h_after = pad_h - pad_h_before
        pad_w_before = pad_w // 2
        pad_w_after = pad_w - pad_w_before
        pad_d_before = pad_d // 2
        pad_d_after = pad_d - pad_d_before

        tensor = torch.nn.functional.pad(
            tensor, 
            (pad_d_before, pad_d_after, pad_w_before, pad_w_after, pad_h_before, pad_h_after), 
            value=-1
        )

        tensor = tensor.permute(2, 0, 1) # [D, H, W] -> (C, D, H, W) is typical for Monai?
        # Legacy: `tensor.permute(2, 0, 1)` -> [D, H, W]
        # Then `tensor.unsqueeze(0)` -> [1, D, H, W]
        
        tensor = tensor.unsqueeze(0)
        return tensor

    def __getitem__(self, index):
        nii_file, labels = self.samples[index]
        video_tensor = self.nii_img_to_tensor(nii_file)
        basename = os.path.basename(nii_file).split(".")[0]
        
        return video_tensor, labels, basename

if __name__ == "__main__":
    from torch.utils.data import DataLoader
    
    # Define paths
    data_folder = "./data/radchest/images"
    labels_csv = "./data/radchest/labels.csv"
    
    print(f"Checking data folder: {data_folder}")
    if not os.path.exists(data_folder):
        print("Data folder not found!")
        
    print(f"Checking labels csv: {labels_csv}")
    if not os.path.exists(labels_csv):
        print("Labels CSV not found!")

    # Initialize dataset
    try:
        print("Initializing dataset...")
        dataset = RadChestCTDataset(data_folder, labels_csv)
        print(f"Dataset length: {len(dataset)}")
        
        if len(dataset) > 0:
            # Create dataloader
            dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
            
            # Test one batch
            print("Testing one batch...")
            for batch in dataloader:
                # __getitem__ returns: video_tensor, labels, basename
                images, labels, accessions = batch
                print("Image shape:", images.shape)
                print("Labels shape:", labels.shape)
                print("Accessions:", accessions)
                break
        else:
            print("Dataset is empty. Check paths and matching logic.")
            
    except Exception as e:
        print(f"An error occurred: {e}")
