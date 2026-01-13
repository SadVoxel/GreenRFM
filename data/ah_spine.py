from .mr_dataset import MRDataset

class AHSpineDataset(MRDataset):
    def __init__(self, **kwargs):
        super().__init__(anatomy="spine", **kwargs)

if __name__ == "__main__":
    import os
    from torch.utils.data import DataLoader

    # Configure paths
    base_dir = "./data/spine"
    img_dir = os.path.join(base_dir, "images")
    label_csv = os.path.join(base_dir, "labels.csv")
    report_xlsx = os.path.join(base_dir, "reports.xlsx")

    print(f"--- Testing AHSpineDataset ---")
    print(f"Img Dir: {img_dir}")
    print(f"Label CSV: {label_csv}")

    if os.path.exists(img_dir) and os.path.exists(label_csv):
        try:
            # Initialize dataset
            dataset = AHSpineDataset(
                img_dir=img_dir,
                label_csv=label_csv,
                report_xlsx=report_xlsx,
                train_mode=True
            )
            print(f"Dataset length: {len(dataset)}")

            if len(dataset) > 0:
                loader = DataLoader(dataset, batch_size=2, shuffle=True)
                print("Loading one batch...")
                for batch in loader:
                    print(f"Batch keys: {list(batch.keys())}")
                    for k in ['img_sag', 'img_cor', 'img_tra']:
                        if k in batch:
                            print(f"{k} shape: {batch[k].shape}")
                    
                    if 'labels' in batch:
                        print(f"Labels shape: {batch['labels'].shape}")
                        print(f"Sample labels: {batch['labels'][0]}")
                    
                    if 'case_id' in batch:
                        print(f"Case IDs: {batch['case_id']}")
                    break
            else:
                print("Dataset is empty.")

        except Exception as e:
            print(f"Error during Spine dataset test: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("Required files not found.")
