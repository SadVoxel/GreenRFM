"""Dataset helpers for AH-Chest experiments."""

from __future__ import annotations

import glob
import math
import os
from typing import Dict, List, Tuple, Optional

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from monai import transforms as monai_transforms
from torch.utils.data import Dataset
from nibabel.filebasedimages import ImageFileError

__all__ = [
    "AHChestBaseVolumeDataset",
    "AHChestInferenceDataset",
    "AHChestImageDataset",
    "load_label_map",
    "load_report_map",
]


def _sanitize_report_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value).replace('"', "").replace("'", "").replace("(", "").replace(")", "")


def load_report_map(csv_path: str) -> Dict[str, str]:
    df = pd.read_csv(csv_path)
    mapping: Dict[str, str] = {}
    for _, row in df.iterrows():
        subject = str(row.get("subject_id", "")).strip()
        if not subject:
            continue
        mapping[subject] = _sanitize_report_text(row.get("report"))
    return mapping


def load_label_map(csv_path: str) -> Dict[str, np.ndarray]:
    df = pd.read_csv(csv_path)
    label_cols = [c for c in df.columns if c != "subject_id"]
    if not label_cols:
        raise ValueError(f"No label columns found in {csv_path}.")
    df[label_cols] = (
        df[label_cols]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
        .astype(np.float32)
    )
    labels: Dict[str, np.ndarray] = {}
    for _, row in df.iterrows():
        subject = str(row.get("subject_id", "")).strip()
        if not subject:
            continue
        labels[subject] = row[label_cols].to_numpy(dtype=np.float32)
    return labels


def _normalize_hu(volume: np.ndarray) -> np.ndarray:
    volume = np.clip(volume, -1000.0, 200.0)
    return ((volume + 400.0) / 600.0).astype(np.float32)


class AHChestBaseVolumeDataset(Dataset):
    """Shared functionality for AH-Chest 3D volumes."""

    def __init__(
        self,
        data_folder: str,
        reports_csv: str,
        labels_csv: str,
        transform: monai_transforms.Compose,
    ) -> None:
        self.data_folder = data_folder
        self.reports_map = load_report_map(reports_csv)
        self.labels_map = load_label_map(labels_csv)
        self.transform = transform
        self._bad_samples: set[str] = set()
        self.samples = self._build_samples()
        if not self.samples:
            # Warning only to allow execution even if empty initially
            print(f"Warning: No overlapping AHChest samples found in {data_folder}")

    def _build_samples(self) -> List[Tuple[str, str]]:
        samples: List[Tuple[str, str]] = []
        subject_dirs = [
            path for path in glob.glob(os.path.join(self.data_folder, "*")) if os.path.isdir(path)
        ]
        for subject_dir in subject_dirs:
            subject_id = os.path.basename(subject_dir)
            if subject_id not in self.reports_map or subject_id not in self.labels_map:
                continue
            nii_files = sorted(glob.glob(os.path.join(subject_dir, "*.nii.gz")))
            for nii_path in nii_files:
                samples.append((nii_path, subject_id))
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def _volume_to_tensor(self, path: str) -> torch.Tensor:
        img_data = nib.load(path).get_fdata()
        img_data = np.transpose(img_data, (1, 2, 0)) # Standardize axis
        img_data = _normalize_hu(img_data)
        img_data = np.expand_dims(img_data, axis=0)
        tensor = self.transform(img_data).squeeze(0)
        tensor = tensor.permute(2, 0, 1).unsqueeze(0) # (1, D, H, W)
        return tensor

    def __getitem__(self, index: int):
        dataset_len = len(self.samples)
        for offset in range(dataset_len):
            real_index = (index + offset) % dataset_len
            nii_path, subject_id = self.samples[real_index]
            if nii_path in self._bad_samples:
                continue
            try:
                video_tensor = self._volume_to_tensor(nii_path)
            except (OSError, EOFError, ImageFileError, ValueError) as exc:
                self._bad_samples.add(nii_path)
                print(f"[WARN] Skipping corrupted AHChest volume {nii_path}: {exc}")
                continue
            text = self.reports_map[subject_id]
            labels = self.labels_map[subject_id].copy()
            return video_tensor, text, labels, subject_id
        raise RuntimeError("All AHChest samples appear to be corrupted or unreadable.")

    def get_accession_for_index(self, index: int) -> str:
        return self.samples[index][1]


class AHChestImageDataset(AHChestBaseVolumeDataset):
    def __init__(self, data_folder: str, reports_csv: str, labels_csv: str) -> None:
        # Standard training transform
        transform = monai_transforms.Compose([
            monai_transforms.ResizeWithPadOrCrop((240, 240, 120)),
            monai_transforms.RandSpatialCrop(roi_size=(192, 192, 96), random_size=False),
            monai_transforms.ToTensor(),
        ])
        super().__init__(data_folder, reports_csv, labels_csv, transform)


class AHChestInferenceDataset(AHChestBaseVolumeDataset):
    def __init__(self, data_folder: str, reports_csv: str, labels_csv: str) -> None:
        # Standard inference transform
        transform = monai_transforms.Compose([
            monai_transforms.ResizeWithPadOrCrop((240, 240, 120)),
            monai_transforms.CenterSpatialCrop(roi_size=(192, 192, 96)),
            monai_transforms.ToTensor(),
        ])
        super().__init__(data_folder, reports_csv, labels_csv, transform)


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    
    # Define paths
    base_dir = "./data/ah_chest"
    data_folder = os.path.join(base_dir, "images")
    reports_csv = os.path.join(base_dir, "reports.csv")
    labels_csv = os.path.join(base_dir, "labels.csv")
    
    print("Checking paths...")
    print(f"Data: {data_folder} Exists: {os.path.exists(data_folder)}")
    print(f"Reports: {reports_csv} Exists: {os.path.exists(reports_csv)}")
    print(f"Labels: {labels_csv} Exists: {os.path.exists(labels_csv)}")

    try:
        print("Initializing AHChestImageDataset...")
        dataset = AHChestImageDataset(data_folder, reports_csv, labels_csv)
        print(f"Dataset length: {len(dataset)}")
        
        if len(dataset) > 0:
            dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
            print("Testing one batch...")
            for batch in dataloader:
                # __getitem__ returns: video_tensor, text, labels, subject_id
                video_tensors, texts, labels, subject_ids = batch
                print("Video tensor shape:", video_tensors.shape)
                print("Labels shape:", labels.shape)
                print("Subject IDs:", subject_ids)
                print("Sample text:", texts[0][:100] + "...")
                break
        else:
            print("Dataset is empty.")
            
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc() 