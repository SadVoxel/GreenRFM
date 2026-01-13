
import os
import glob
import torch
import pandas as pd
import numpy as np
import tqdm
import monai.transforms as transforms
import monai.data as data
import random
import math

class CTRATEDataset(data.PersistentDataset):
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
        # CT-RATE: [-1000, 200]
        # Crop 192x192x96 for training
        if section == "train":
             self.transform = transforms.Compose([
                transforms.ResizeWithPadOrCrop((240, 240, 120)),
                transforms.RandSpatialCrop(roi_size=(192, 192, 96), random_size=False),
                transforms.ToTensor(),
            ])
        else:
             self.transform = transforms.Compose([
                transforms.CenterSpatialCrop((192, 192, 96)), # Center crop logic approximation
                transforms.ToTensor(),
            ])
        
        super().__init__(data=self.samples, transform=self.transform, cache_dir=cache_dir)


    def _load_accession_text(self, csv_file):
        df = pd.read_csv(csv_file)
        accession_to_text = {}
        # Assuming typical columns. Adjust if needed based on legacy code.
        # Legacy: VolumeName, Findings_EN, Impressions_EN
        for _, row in df.iterrows():
            accession_to_text[row['VolumeName']] = (row.get("Findings_EN", ""), row.get("Impressions_EN", ""))
        return accession_to_text

    def _prepare_samples(self):
        samples = []
        patient_folders = glob.glob(os.path.join(self.data_folder, '*'))

        df = pd.read_csv(self.labels_csv)
        # Handle labels, convert -1 to 0 or logic as per paper (Paper says handle uncertainty explicitly?)
        # For training, usually -1 is ignored or treated as 0 or 1.
        # Legacy code: fillna(0.0). So -1 might be present.
        
        label_cols = [c for c in df.columns if c != "VolumeName"]
        # Ensure numeric
        df[label_cols] = df[label_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        
        # Create a dictionary for fast lookup
        label_dict = df.set_index("VolumeName")[label_cols].to_dict('index')

        for patient_folder in tqdm.tqdm(patient_folders):
            accession_folders = glob.glob(os.path.join(patient_folder, '*'))
            for accession_folder in accession_folders:
                # Assuming .npz files or .nii.gz
                # Legacy code looked for .npz
                nii_files = glob.glob(os.path.join(accession_folder, '*.npz'))
                
                for nii_file in nii_files:
                    # Logic to extract accession number might vary
                    # Legacy: os.path.basename(nii_file).replace(".npz", ".nii.gz")
                    accession_number = os.path.basename(nii_file).replace(".npz", ".nii.gz")
                    
                    if accession_number not in self.accession_to_text:
                        continue
                        
                    if accession_number not in label_dict:
                        continue
                        
                    findings, impressions = self.accession_to_text[accession_number]
                    text_final = f"{findings or ''} {impressions or ''}".strip()
                    
                    labels = np.array(list(label_dict[accession_number].values()), dtype=np.float32)
                    
                    samples.append({
                        "image": nii_file,
                        "text": text_final,
                        "labels": labels,
                        "accession": accession_number
                    })
        return samples

    def _hu_window_normalize(self, vol_np):
        # CT-RATE specific window
        # Legacy code: vol * 1000? Wait.
        # Legacy: vol_np = vol_np * 1000.0  <- This implies input was already normalized small?
        # Typically CT is int16. 
        # If input is .npz, it might be raw HU or already preprocessed.
        # Legacy code says: img_data = np.load(path)['arr_0']; img_data = np.transpose(img_data, (1, 2, 0))
        # Then _hu_window_normalize.
        
        # If raw HU: [-1000, 200] -> [0, 1] normalization typically.
        # Legacy: clip(-1000, 200). (val + 400) / 600.
        # -1000 -> -600/600 = -1. 
        # 200 -> 600/600 = 1.
        # So range is [-1, 1].
        vol_np = vol_np * 1000.0
        vol_np = np.clip(vol_np, -1000, 200)
        vol_np = (vol_np + 400) / 600.0
        return vol_np.astype(np.float32)

    def __getitem__(self, index):
        sample = self.samples[index]
        
        try:
             # Load image
            img_data = np.load(sample['image'])['arr_0']
            img_data = np.transpose(img_data, (1, 2, 0)) # HWD
            img_data = self._hu_window_normalize(img_data)
            img_data = np.expand_dims(img_data, axis=0) # CHWD
            
            # Apply transform
            # Transform expects channel first
            img_tensor = self.transform(img_data).squeeze(0) # HWD
            img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0) # 1DHW



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
            # Return a dummy or fail?
            # Better to handle gracefully in DataLoader collate_fn or skip
            # Retrying random sample
            return self.__getitem__(random.randint(0, len(self)-1))

    def __len__(self):
        return len(self.samples)


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    
    # Define paths
    data_folder = "./data/ct_rate/images"
    reports_csv = "./data/ct_rate/reports.csv"
    labels_csv = "./data/ct_rate/labels.csv"
    
    # Initialize dataset
    print("Initializing dataset...")
    try:
        dataset = CTRATEDataset(data_folder, reports_csv, labels_csv, section="train")
        print(f"Dataset length: {len(dataset)}")
        
        # Create dataloader
        dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)
        
        # Test one batch
        for batch in dataloader:
            print("Image shape:", batch["image"].shape)
            print("Labels shape:", batch["labels"].shape)
            print("Text:", batch["text"])
            print("Accession:", batch["accession"])
            break
            
    except Exception as e:
        print(f"An error occurred: {e}") 