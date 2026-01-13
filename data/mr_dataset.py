
import os
import glob
import random
import math
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import nibabel as nib
import torch
from torch.utils.data import Dataset
import monai.transforms as transforms

class MRDataset(Dataset):
    """
    Universal MRI Dataset reader, compatible with different anatomies (e.g. knee / spine).
    Returns for each sample:
        img_{plane} : [1, D, H, W] or [C, D, H, W] - 3D tensor
        report      : original report text
        labels      : shape=[max_labels], padded with -1
        label_mask  : same shape, 0 for padding, 1 for valid
        anatomy     : 'knee' / 'spine'
        case_id     : ID
    """
    
    # Support multi-sequence multi-plane
    SEQUENCES = ["pd", "t1", "t2"]
    PLANES    = ["sag", "cor", "tra"]

    def __init__(
        self,
        img_dir: str,
        label_csv: str,
        report_xlsx: Optional[str] = None,
        anatomy: str = "knee",
        resize_shape: Tuple[int, int, int] = (240, 240, 24),
        crop_shape:   Tuple[int, int, int] = (192, 192, 20),
        max_labels:   int = None,
        positive_ratio: float = None,
        train_mode: bool = True,
    ):
        super().__init__()
        self.img_dir   = img_dir
        self.anatomy   = anatomy
        self.train_mode = train_mode

        # ---------- 1) Labels ----------
        self.case2label = self._load_labels(label_csv)
        self.num_labels = max(len(v) for v in self.case2label.values())
        if max_labels:
            self.num_labels = max(self.num_labels, max_labels)

        # ---------- 2) Text ----------
        self.case2txt = self._load_reports(report_xlsx) if report_xlsx else {}

        # ---------- 3) Paths ----------
        self.samples = self._gather_samples()
        if positive_ratio is not None and train_mode:
            self.samples = self._oversample_positive(self.samples, positive_ratio)

        # ---------- 4) 3D Transforms ----------
        t_list: List = [
            transforms.ResizeWithPadOrCrop(resize_shape),
        ]
        if train_mode:
            t_list.append(transforms.RandSpatialCrop(roi_size=crop_shape, random_size=False))
        else:
            t_list.append(transforms.CenterSpatialCrop(roi_size=crop_shape))
        t_list.append(transforms.ToTensor())
        self.transform = transforms.Compose(t_list)

    # ==================== helpers ==================== #
    def _load_labels(self, csv_file: str) -> Dict[str, np.ndarray]:
        """
        Select columns ending with *_code but not exam_item_code.
        Returns dict: case_id -> np.ndarray[label_dim]
        """
        df = pd.read_csv(csv_file)

        label_cols = [c for c in df.columns
                      if c.endswith("_code") and c != "exam_item_code"]
        if not label_cols:
            # Fallback or allow empty
            pass

        self.label_names = label_cols
        case2label = {
            str(row["subject_id"]): row[label_cols].values.astype(np.float32)
            for _, row in df.iterrows()
        }
        return case2label

    def _load_reports(self, xlsx_file: str) -> Dict[str, str]:
        df = pd.read_excel(xlsx_file)

        # Auto-detect case ID column
        case_col = None
        for c in df.columns:
            lc = str(c).lower()
            if lc in {"case_id", "subject_id", "study_id", "exam_id"}:
                case_col = c
                break
        if case_col is None:
            case_col = df.columns[0]

        # Auto-detect report column
        report_col = None
        for c in df.columns:
            if str(c).lower() in {"report", "report_txt", "report_text", "text"}:
                report_col = c
                break
        if report_col is None and len(df.columns) >= 2:
            report_col = df.columns[1]

        case2txt = {}
        for _, r in df.iterrows():
            case_id = str(r[case_col])
            val = r[report_col] if report_col in r else ""
            case2txt[case_id] = "" if pd.isna(val) else str(val)
        return case2txt

    def _gather_samples(self):
        """
        Each case (ZMR-xxxx) is 1 sample.
        Returns list of (p2s, txt_content, label_vec, case_id)
        p2s is dict: plane -> {seq: path or None}
        """
        samples = []
        candidate_dirs = glob.glob(os.path.join(self.img_dir, "*"))
        
        for case_dir in candidate_dirs:
            if not os.path.isdir(case_dir): 
                continue
                
            case_id = os.path.basename(case_dir)
            if case_id not in self.case2label:
                continue

            # Init dict
            p2s = {
                p: {seq: None for seq in self.SEQUENCES}
                for p in self.PLANES
            }
            
            # Find files
            for nii in glob.glob(os.path.join(case_dir, "*.nii.gz")):
                name = os.path.basename(nii).split(".")[0].lower()
                tokens = name.split("_")
                if len(tokens) < 2:
                    continue
                seq, plane = tokens[0], tokens[1]
                # Filter valid seqs/planes
                if seq in self.SEQUENCES and plane in self.PLANES:
                    p2s[plane][seq] = nii

            samples.append((p2s,
                            self.case2txt.get(case_id, ""),
                            self.case2label[case_id],
                            case_id))
        
        if not self.train_mode:
            samples.sort(key=lambda x: x[3])
        else:
            random.shuffle(samples)
            
        return samples

    def _load_plane(self, seq_paths: Dict[str, str]):
        """
        seq_paths: {seq_name: path or None}
        Merges sequences into channels.
        Handles missing sequences by padding with zeros.
        """
        raw_imgs: Dict[str, np.ndarray] = {}
        seq_mask: List[float] = []

        # 1) Read existing sequences, record max shape
        max_shape = [0, 0, 0]   # H,W,D
        for seq in self.SEQUENCES:
            path = seq_paths[seq]
            if path is not None and os.path.exists(path):
                img = nib.load(path).get_fdata().astype(np.float32)
                
                # Z-score normalization per volume
                p1, p99 = np.percentile(img, (1, 99))
                img = np.clip(img, p1, p99)
                std_val = img.std()
                if std_val < 1e-6: std_val = 1e-6
                img = (img - img.mean()) / std_val
                
                raw_imgs[seq] = img
                seq_mask.append(1.0)
                for i, s in enumerate(img.shape):
                    max_shape[i] = max(max_shape[i], s)
            else:
                raw_imgs[seq] = None
                seq_mask.append(0.0)

        # Fallback if all missing
        if max_shape == [0, 0, 0]:
            max_shape = [1, 1, 1] # dummy

        # 2) Pad/Crop to max_shape
        def pad_crop_center(arr: np.ndarray, tgt: Tuple[int, int, int]):
            out = np.zeros(tgt, dtype=arr.dtype)
            h, w, d = arr.shape
            th, tw, td = tgt
            
            # Start indices
            hs, ws, ds = max((h - th) // 2, 0), max((w - tw)//2, 0), max((d - td)//2, 0)
            hd, wd, dd = max((th - h) // 2, 0), max((tw - w)//2, 0), max((td - d)//2, 0)
            lh, lw, ld = min(h, th), min(w, tw), min(d, td)
            
            out[hd:hd+lh, wd:wd+lw, dd:dd+ld] = arr[hs:hs+lh, ws:ws+lw, ds:ds+ld]
            return out

        img_list = []
        for seq in self.SEQUENCES:
            if raw_imgs[seq] is None:
                img_list.append(np.zeros(max_shape, dtype=np.float32))
            else:
                img_list.append(pad_crop_center(raw_imgs[seq], tuple(max_shape)))

        # 3) Stack -> (C, H, W, D) -> Transform
        arr = np.stack(img_list, 0)                  # C = len(SEQUENCES)
        
        arr_t = self.transform(arr)  # Returns Tensor (C, H', W', D')
        arr_t = arr_t.permute(0, 3, 1, 2) 
        
        mask = torch.tensor(seq_mask, dtype=torch.float32)

        return arr_t, mask


    def _oversample_positive(
        self, samples: List, desired_ratio: float = 0.15, max_expand: float = 100.0
    ):
        if not samples: return samples
        num_cls = len(samples[0][2])
        pos, neg = [0] * num_cls, [0] * num_cls
        pool     = [[] for _ in range(num_cls)]

        for idx, (_, _, lb, _) in enumerate(samples):
            for c, v in enumerate(lb):
                (pos if v else neg)[c] += 1
                if v:
                    pool[c].append(idx)

        need = [
            max(0, math.ceil(desired_ratio * neg[c] - pos[c])) for c in range(num_cls)
        ]
        rep = [1] * len(samples)
        while max(need) > 0 and sum(rep) < len(samples) * max_expand:
            c_star = int(np.argmax(need))
            if not pool[c_star]:
                need[c_star] = 0
                continue
            idx    = random.choice(pool[c_star])
            rep[idx] += 1
            _, _, lb, _ = samples[idx]
            for c, v in enumerate(lb):
                (pos if v else neg)[c] += 1
                need[c] = max(0, math.ceil(desired_ratio * neg[c] - pos[c]))
        new_samples = []
        for i, k in enumerate(rep):
            new_samples.extend([samples[i]] * k)
        random.shuffle(new_samples)
        return new_samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        p2s, txt, raw_lbl, case_id = self.samples[idx]

        img_dict  = {}
        mask_dict = {}
        for plane in self.PLANES:
            # Load and process plane
            imgs, msk = self._load_plane(p2s[plane]) 
            img_dict[f"img_{plane}"]   = imgs
            mask_dict[f"mask_{plane}"] = msk

        # Padded Labels
        padded = np.full((self.num_labels,), -1, dtype=np.float32)
        lbl_m  = np.zeros_like(padded,      dtype=np.float32)
        if len(raw_lbl) > 0:
            eff_len = min(len(raw_lbl), self.num_labels)
            padded[: eff_len] = raw_lbl[: eff_len]
            lbl_m [: eff_len] = 1.0

        sample = {
            **img_dict,
            **mask_dict,
            "labels":      torch.tensor(padded, dtype=torch.float32),
            "label_mask":  torch.tensor(lbl_m,  dtype=torch.float32),
            "anatomy":     self.anatomy,
            "case_id":     case_id,
            "report":      txt,
        }
        return sample
