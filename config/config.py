"""
Centralized configuration for LeanRad project.
"""
import os

# --- Global Defaults ---
BERT_NAME = "microsoft/BiomedVLP-CXR-BERT-specialized"

# --- Feature Extraction Configuration ---
# You can import this dictionary and modify specific keys in your script
INFERENCE_CONFIG = {
    "dataset": "ah_abd",   # Options: ah_knee, ah_spine, ah_abd, ah_chest, merlin, ct_rate, radchest
    
    # Paths (Default placeholders, please update before running)
    "data_folder": "./data/images",
    "reports_csv": "./data/reports.csv",
    "labels_csv": "./data/labels.csv",
    "model_path": "./checkpoints/model.pt", # File or Directory
    "output_dir": "./features/output",
    
    # Runtime Parameters
    "batch_size": 1,
    "num_workers": 4,
    "max_length": 128,     # 256 for MRI often
    "bert_name": BERT_NAME,
    
    # Checkpoint Traversal (if model_path is a directory)
    "start_ckpt": 1000,
    "end_ckpt": 50000,
    "step_ckpt": 1000,
}

# --- Known Label Sets ---
# 30 Classes (Abdominal / Merlin)
LABEL_COLUMNS_30 = [
    "submucosal_edema", "renal_hypodensities", "aortic_valve_calcification", "coronary_calcification",
    "thrombosis", "metastatic_disease", "pancreatic_atrophy", "renal_cyst", "osteopenia",
    "surgically_absent_gallbladder", "atelectasis", "abdominal_aortic_aneurysm", "anasarca",
    "hiatal_hernia", "lymphadenopathy", "prostatomegaly", "biliary_ductal_dilation", "cardiomegaly",
    "splenomegaly", "hepatomegaly", "atherosclerosis", "ascites", "pleural_effusion",
    "hepatic_steatosis", "appendicitis", "gallstones", "hydronephrosis", "bowel_obstruction",
    "free_air", "fracture",
]

# 18 Classes (Chest / CT-RATE)
LABEL_COLUMNS_18 = [
    "medical_material", "arterial_wall_calcification", "cardiomegaly", "pericardial_effusion",
    "coronary_artery_wall_calcification", "hiatal_hernia", "lymphadenopathy", "emphysema",
    "atelectasis", "lung_nodule", "lung_opacity", "pulmonary_fibrotic_sequela", "pleural_effusion",
    "pneumothorax", "consolidation", "mass", "nodule", "fracture" 
]
