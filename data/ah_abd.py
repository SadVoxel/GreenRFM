"""Dataset implementation for Abdominal CT (AH-Abd) experiments."""

from __future__ import annotations

import glob
import math
import os
import random
from typing import Dict, List, Optional, Tuple, Union

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import monai.transforms as transforms
from torch.utils.data import Dataset

__all__ = ["AHAbdDataset"]


class AHAbdDataset(Dataset):
    """
    Abdominal CT Dataset (AH-Abd).
    Combines logic for training and inference based on 'split'.
    """

    def __init__(
        self,
        data_folder: str,
        reports_csv: str,
        labels_csv: str,
        split: str = "train",
        cache_dir: Optional[str] = None,
    ) -> None:
        """
        Args:
            data_folder: Root directory containing subject folders with .nii.gz files.
            reports_csv: Path to CSV/Excel with report text.
            labels_csv: Path to CSV with classification labels.
            split: 'train', 'val', or 'test'. Defaults to 'train'.
        """
        super().__init__()
        self.data_folder = data_folder
        self.labels_csv = labels_csv
        self.split = split.strip().lower()
        self.accession_meta = self._load_accession_text(reports_csv)
        self.samples = self._prepare_samples()
        
        # Consistent shuffle for reproducibility (optional) if needed, 
        # but normally Dataset just provides access. Legacy code shuffled.
        if self.split == "train":
            random.shuffle(self.samples)

        # Transforms configuration based on legacy code
        if self.split == "train":
            self.transform = transforms.Compose([
                transforms.ResizeWithPadOrCrop((280, 280, 180), mode="constant", constant_values=-1.0),
                transforms.RandSpatialCrop(roi_size=(224, 224, 144), random_size=False),
                transforms.ToTensor(),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ResizeWithPadOrCrop((280, 280, 180), mode="constant", constant_values=-1.0),
                transforms.CenterSpatialCrop(roi_size=(224, 224, 144)),
                transforms.ToTensor(),
            ])

    def _load_accession_text(self, csv_file: str) -> Dict[str, Dict[str, str]]:
        if csv_file.lower().endswith((".xlsx", ".xls")):
            df = pd.read_excel(csv_file)
        elif csv_file.lower().endswith(".csv"):
            df = pd.read_csv(csv_file, encoding="utf-8")
        else:
            raise ValueError("Unsupported file format for reports_csv")
            
        accession_meta = {}
        for _, row in df.iterrows():
            study_id = row.get("subject id") or row.get("subject_id")
            if not isinstance(study_id, str):
                continue
            accession_meta[study_id] = {
                "subject_id": study_id,
                "findings": str(row.get("findings_translated") or ""),
                "impression": str(row.get("impression_translated") or ""),
            }
        return accession_meta

    def _prepare_samples(self) -> List[Tuple[str, str, np.ndarray, str]]:
        samples = []
        df = pd.read_csv(self.labels_csv)
        
        # Normalize subject_id column
        if "subject id" in df.columns and "subject_id" not in df.columns:
            df = df.rename(columns={"subject id": "subject_id"})
            
        label_cols = [c for c in df.columns if c != "subject_id"]
        
        # Ensure numeric labels
        df[label_cols] = (
            df[label_cols]
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0.0)
            .astype(np.float32)
        )
        # Store as list for easy access
        df["one_hot_labels"] = df[label_cols].values.tolist()
        
        # Labels dataframe for lookup
        labels_df = df

        # Logic for split: Last 20% of sorted IDs is TEST. 
        # (Legacy code logic: intersection of accession_meta and labels_df keys)
        all_ids = set(self.accession_meta.keys()) & set(labels_df["subject_id"].astype(str).tolist())
        all_accessions = sorted(list(all_ids))
        
        total = len(all_accessions)
        test_accessions = set()
        if total > 0:
            n_test = max(1, int(math.ceil(total * 0.2)))
            test_accessions = set(all_accessions[-n_test:])

        # Find NIfTI files
        # Legacy looks recursively: os.path.join(self.data_folder, "**", "*.nii.gz")
        nii_files = glob.glob(os.path.join(self.data_folder, "**", "*.nii.gz"), recursive=True)
        
        # Debug info
        print(f"[AHAbdDataset] Total .nii.gz files found: {len(nii_files)}")
        print(f"[AHAbdDataset] Total labelled subjects: {len(labels_df)}")
        print(f"[AHAbdDataset] Intersection of identifiers: {len(all_accessions)}")
        print(f"[AHAbdDataset] Test set size (last 20%): {len(test_accessions)}")

        for nii_file in nii_files:
            # Determining accession number from path
            # Legacy: parent directory name is accession number?
            # "parent = os.path.basename(os.path.dirname(nii_file))"
            parent = os.path.basename(os.path.dirname(nii_file))
            accession_number = parent
            
            meta = self.accession_meta.get(accession_number)
            if not meta:
                continue

            # Split filtering
            is_test_sample = accession_number in test_accessions
            
            if self.split == "test":
                if not is_test_sample:
                    continue
            elif self.split == "train":
                if is_test_sample:
                    continue
            
            # Additional safety for val or other splits if needed, currently 'val' includes everything if used.
            # If user asks for 'val', we ideally should split train further, but current logic mimics legacy.
            
            findings = meta.get("findings", "")
            impressions = meta.get("impression", "")
            text_final = self._sanitize_text(f"{findings}{impressions}")

            match = labels_df[labels_df["subject_id"] == accession_number]
            if match.empty:
                continue
            
            one_hot = np.array(match["one_hot_labels"].iloc[0], dtype=np.float32)

            if len(one_hot) > 0:
                samples.append((nii_file, text_final, one_hot, accession_number))
        
        print(f"[AHAbdDataset] Samples selected for split='{self.split}': {len(samples)}")
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def _hu_window_normalize(self, vol_np: np.ndarray) -> np.ndarray:
        # Legacy: clip(-1000, 1000) / 1000.0
        vol_np = np.clip(vol_np, -1000, 1000)
        vol_np = (vol_np / 1000).astype(np.float32)
        return vol_np

    def _sanitize_text(self, s: str) -> str:
        return s.replace('"', '').replace("'", '').replace('(', '').replace(')', '')

    def _nii_img_to_tensor(self, path: str) -> torch.Tensor:
        try:
            img_data = nib.load(path).get_fdata()
            # Legacy expected layout: (C, H, W, D) or similar? 
            # Legacy says: np.transpose(img_data, (1, 2, 0)) -> (H, W, D) if loaded from .npy?
            # nib.load().get_fdata() usually returns (H, W, D).
            # Legacy code explanation: "img_data = np.transpose(img_data, (1, 2, 0))" 
            # implies input might have been different or they want to swap axes.
            # Standard nibabel 3D is (X, Y, Z). 
            # Let's trust legacy code's transpose: (1, 2, 0)
            
            if len(img_data.shape) == 3:
                # Assuming (X, Y, Z) from nibabel
                # Legacy transpose: (Y, Z, X) ? No, (1,2,0).
                img_data = np.transpose(img_data, (1, 2, 0))
            
            img_data = self._hu_window_normalize(img_data)
            
            # Add channel dim: (1, H, W, D)
            img_data = np.expand_dims(img_data, axis=0) 
            
            # Apply transforms (monai expects channel first usually)
            # transform returns tensor.squeeze(0) -> (H, W, D) in legacy?
            # Wait, legacy: 
            # tensor = self.transform(img_data).squeeze(0)
            # tensor = tensor.permute(2, 0, 1).unsqueeze(0)
            # Seems legacy wants (1, D, H, W) output.
            
            tensor = self.transform(img_data)
            if isinstance(tensor, torch.Tensor):
                tensor = tensor.squeeze(0) # (H,W,D)
            # Legacy manual permute: (2, 0, 1) -> (D, H, W)
            tensor = tensor.permute(2, 0, 1).unsqueeze(0) # (1, D, H, W)
            return tensor
            
        except Exception as e:
            print(f"Error loading {path}: {e}")
            # Return zero tensor or handle error
            return torch.zeros((1, 144, 224, 224), dtype=torch.float32) # approximate shape

    def __getitem__(self, index: int):
        nii_file, input_text, one_hot_labels, name_acc = self.samples[index]
        video_tensor = self._nii_img_to_tensor(nii_file)
        return video_tensor, input_text, one_hot_labels, name_acc


if __name__ == "__main__":
    import torch
    data_folder = "./data/abd/images"
    reports_csv = "./data/abd/reports.csv"
    labels_csv = "./data/abd/labels.csv"
    
    print("Testing AHAbdDataset (TRAIN)...")
    ds_train = AHAbdDataset(data_folder, reports_csv, labels_csv, split="train")
    print(f"Train Dataset length: {len(ds_train)}")

    print("\nTesting AHAbdDataset (TEST)...")
    ds_test = AHAbdDataset(data_folder, reports_csv, labels_csv, split="test")
    print(f"Test Dataset length: {len(ds_test)}")

    print(f"\nTotal Dataset Coverage: {len(ds_train) + len(ds_test)}")
    
    if len(ds_train) > 0:
        item = ds_train[0]
        # video_tensor, input_text, one_hot_labels, name_acc
        print(f"\nExample Item:")
        print(f"Tensor shape: {item[0].shape}")
        print(f"Text: {item[1][:50]}...")
        print(f"Labels: {item[2]}")
        print(f"Accession: {item[3]}")
