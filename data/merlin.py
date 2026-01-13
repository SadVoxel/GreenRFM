
import os
import glob
import torch
import pandas as pd
import numpy as np
import tqdm
import monai.transforms as transforms
import monai.data as data
import random
import nibabel as nib

class MerlinDataset(data.PersistentDataset):
    def __init__(self, data_folder, reports_csv, labels_csv, cache_dir=None, section="train"):
        self.data_folder = data_folder
        self.labels_csv = labels_csv
        self.reports_csv = reports_csv
        self.section = section
        
        # Load Metadatas
        self.accession_to_text = self._load_accession_text(reports_csv)
        self.samples = self._prepare_samples()

        if section == "train":
            random.shuffle(self.samples)

        # Transformation pipeline
        # Merlin: [-1000, 1000]
        # Crop 224x224x144 for training (from Paper)
        if section == "train":
             self.transform = transforms.Compose([
                transforms.ResizeWithPadOrCrop((280, 280, 180)),
                transforms.RandSpatialCrop(roi_size=(224, 224, 144), random_size=False),
                transforms.ToTensor(),
            ])
        else:
             self.transform = transforms.Compose([
                transforms.ResizeWithPadOrCrop((224, 224, 144)), # Center crop logic
                transforms.ToTensor(),
            ])
        
        super().__init__(data=self.samples, transform=self.transform, cache_dir=cache_dir)


    def _load_accession_text(self, csv_file):
        if csv_file.endswith('.xlsx') or csv_file.endswith('.xls'):
            df = pd.read_excel(csv_file)
        else:
            df = pd.read_csv(csv_file)
        accession_to_text = {}
        # Try to find study id column
        id_col = "study id" if "study id" in df.columns else "VolumeName"
        # If still not found, print warning or error? Assumed "study id" based on valid data.
        
        for _, row in df.iterrows():
            if id_col in row:
                key = str(row[id_col])
                findings = row.get("Findings", row.get("Findings_EN", ""))
                impressions = row.get("Impression", row.get("Impressions_EN", ""))
                accession_to_text[key] = (findings, impressions)
        return accession_to_text

    def _prepare_samples(self):
        samples = []
        # Support flat or recursive directory structure for .nii.gz
        nii_files = glob.glob(os.path.join(self.data_folder, "**", "*.nii.gz"), recursive=True)

        df = pd.read_csv(self.labels_csv)
        # Handle study id column
        id_col = "study id" if "study id" in df.columns else "VolumeName"
        
        label_cols = [c for c in df.columns if c != id_col]
        df[label_cols] = df[label_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        label_dict = df.set_index(id_col)[label_cols].to_dict('index')

        for nii_file in tqdm.tqdm(nii_files):
            # Filename is like ACxxxxx.nii.gz. Key is ACxxxxx.
            basename = os.path.basename(nii_file)
            accession_number = basename.replace(".nii.gz", "").replace(".npz", "")
            
            if accession_number not in self.accession_to_text:
                continue
            if accession_number not in label_dict:
                continue
                
            findings, impressions = self.accession_to_text[accession_number]
            # Handle NaN or float values in text
            if pd.isna(findings): findings = ""
            if pd.isna(impressions): impressions = ""
            
            text_final = f"{findings} {impressions}".strip()
            
            labels = np.array(list(label_dict[accession_number].values()), dtype=np.float32)
            
            samples.append({
                "image": nii_file,
                "text": text_final,
                "labels": labels,
                "accession": accession_number
            })
        return samples

    def _hu_window_normalize(self, vol_np):
        # Merlin specific window: [-1000, 1000] -> [-1, 1]
        vol_np = np.clip(vol_np, -1000, 1000)
        vol_np = vol_np / 1000.0
        return vol_np.astype(np.float32)

    def __getitem__(self, index):
        sample = self.samples[index]
        
        try:
            # Modified to load .nii.gz using nibabel
            img_data = nib.load(sample['image']).get_fdata()
            # Legacy transpose from previous implementation
            # Assumes input (H, W, D) -> (W, D, H)? Or trying to match monai transform expectations?
            # Keeping transpose for consistency with previous file logic, although it might depend on source orientation.
            if len(img_data.shape) == 3:
                img_data = np.transpose(img_data, (1, 2, 0))
            
            img_data = self._hu_window_normalize(img_data)
            img_data = np.expand_dims(img_data, axis=0) # CHWD
            
            img_tensor = self.transform(img_data)
            if isinstance(img_tensor, list):
                img_tensor = img_tensor[0]
                
            return {
                "image": img_tensor,
                "text": sample['text'],
                "labels": torch.tensor(sample['labels']),
                "accession": sample['accession']
            }
        except Exception as e:
            print(f"Error loading {sample['image']}: {e}")
            return self.__getitem__(random.randint(0, len(self)-1))

    def __len__(self):
        return len(self.samples)


if __name__ == "__main__":
    import torch
    
    data_folder = "./data/merlin/images"
    reports_csv = "./data/merlin/reports.xlsx"
    labels_csv = "./data/merlin/labels.csv"
    
    print("Testing MerlinDataset...")
    ds = MerlinDataset(data_folder, reports_csv, labels_csv, section="train")
    print(f"Dataset length: {len(ds)}")
    
    if len(ds) > 0:
        item = ds[0]
        print(f"Image shape: {item['image'].shape}")
        print(f"Text: {item['text'][:50]}...")
        print(f"Labels: {item['labels']}")
        print(f"Accession: {item['accession']}")
