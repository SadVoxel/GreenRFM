# GreenRFM

GreenRFM is a resource-efficient pre-training framework that achieves state-of-the-art performance in radiology foundation models (RFMs). The framework ensures robust generalization across diverse patient populations and imaging protocols, reducing computational requirements by orders of magnitude while surpassing complex, parameter-heavy models.

These capabilities stem from **principled supervision design** that aims to maximally utilize supervisory signals via four key principles (MUST):
- **M**ore distilled supervision: Harnessing LLMs to distill diverse reports into structured labels.
- **U**biquitous supervision: Injecting guidance into every component (vision encoder, text encoder, and alignment space).
- **S**emantic-enforcing supervision: Pre-training unimodal encoders before alignment to ensure semantic discriminability.
- **T**ask-aligning supervision: Aligning architectural design and objectives strictly with clinical protocols.

We offer two configurations:
1.  **GreenRFM**: Establishes new SOTA on a single 24GB GPU within 24 hours.
2.  **GreenRFM-L (Lightweight)**: Matches existing benchmarks with 6GB VRAM in 4 hours (trainable on a laptop).

## 1. System Requirements

### Operating System
- **Linux** (Tested on Ubuntu 20.04/22.04)
- Windows/Mac (Not primarily tested but potential support via Python environment)

### Software Dependencies
- **Python**: 3.8 or higher (Recommend 3.9/3.10)
- **CUDA**: 12.1 or higher (Tested with CUDA 12.1)
- **PyTorch**: 2.1.0+ (Matches CUDA version)

### Hardware Requirements
- **GPU**: 
  - **Standard**: 1 NVIDIA GPU with 24GB VRAM (e.g., RTX 3090, RTX 4090) for full SOTA training.
  - **Lightweight**: 1 NVIDIA GPU with 6GB VRAM (e.g., consumer laptop GPUs) for competitive performance.
- **CPU**: Multi-core processor (8+ cores recommended for data loading).
- **RAM**: 16GB+.

### Key Python Packages
See `requirements.txt` for full list. Major dependencies:
- `torch`
- `torchvision`
- `transformers`
- `monai`
- `nibabel`
- `pandas`
- `numpy`

## 2. Installation Guide

### Instructions
1.  **Clone the repository** (if applicable):
    ```bash
    git clone <repo_url>
    cd GreenRFM
    ```

2.  **Create a Conda Environment** (Recommended):
    ```bash
    conda create -n greenrfm python=3.9 -y
    conda activate greenrfm
    ```

3.  **Install Dependencies**:
    Estimated time: 5-10 minutes on a standard broadband connection.
    ```bash
    pip install -r requirements.txt
    ```
    *Note: Ensure you have the correct PyTorch version compatible with your CUDA driver (see [pytorch.org](https://pytorch.org/)).*

## 3. Demo

We provide a standalone demo script (`demo.py`) that runs zero-shot classification on a sample 3D CT volume. We include a synthetic sample file in `demo_data/sample_ct.nii.gz` to demonstrate the data loading and inference pipeline without requiring large external downloads.

### Instructions to Run
Run the following command in the root directory:

```bash
python demo.py
```

### Expected Output
The script will:
1.  Initialize the `CTCLIP` model (using `bert-base-uncased` and `r3d_18`).
2.  **Load the sample 3D volume** from `demo_data/sample_ct.nii.gz`.
3.  Encode text prompts for 18 common pathologies.
4.  Run inference and output probability scores.

**Sample Output:**
```text
============================================================
   GreenRFM Demo: Zero-Shot Classification on Synthetic Data
============================================================
Using device: cuda

[1/4] Initializing CT-CLIP Model (Lite Version)...
      Model initialized successfully.

[2/4] Preparing Dataset...
      Loading data from: .../GreenRFM/demo_data/sample_ct.nii.gz

[3/4] Encoding Text Prompts (Zero-Shot Prototypes) using model.forward()...
      Encoded 18 pathology prompts.

[4/4] Running Inference Loop...
      Batch 1: Processed 1 images.
      Sample Output (First Image in Batch):
        - Medical material: 0.5123
        - Arterial wall calcification: 0.4891
        - Cardiomegaly: 0.5012
      ...

============================================================
   Demo Completed Successfully!
============================================================
```
*(Note: Probabilities will be random/near 0.5 since the model is initialized with random weights and data is noise.)*

### Expected Run Time
- **< 1 minute** on a desktop with a modern GPU (RTX 3060 or better).
- **~2-3 minutes** on CPU (depending on available RAM and cores).

## 4. Instructions for Use

### Inference on Real Data (Zero-Shot)
To run inference on your own dataset (e.g., CT volumes in NIfTI format):

1.  **Prepare Data CSV**: Create a CSV file with columns `Image Path` (absolute path to `.nii.gz`) and `Report` (ground truth report, optional).
2.  **Configure Dataset**: Update `config/` or `inference/run_zero_shot.py` to point to your CSV.
3.  **Run Inference**:
    ```bash
    python inference/run_zero_shot.py
    ```

### Training (Alignment)
To train the model on image-text pairs:
1. Check `training/alignment.py`.
2. Ensure you have a dataset class defined in `data/` that yields `(image, text)` pairs.
3. Run:
    ```bash
    python training/alignment.py --batch_size 10 --epochs 5
    ```

## 5. Reproduction Instructions

We encourage you to reproduce the quantitative results presented in the manuscript. Our framework demonstrates superior performance on chest and abdominal CT datasets (CT-RATE, Merlin) and transfers to MRI modalities (Internal Knee/Spine).

To reproduce the main results:

1.  **Prepare Data**:
    - Download **CT-RATE** and **Merlin** public datasets.
    - **Chest CT**: Resample to 1.5x1.5x3.0mm spacing. Clip HU to [-1000, 200]. Train/Eval crop size: 192x192x96.
    - **Abdominal CT**: Resample to 1.5x1.5x3.0mm spacing. Clip HU to [-1000, 1000]. Train/Eval crop size: 224x224x144.
    - **Label Extraction**: Use the provided `llm_pipeline/` scripts to extract structured diagnostic labels from reports if labels are not provided.

2.  **Training (Two-Stage Strategy)**:
    - **Stage 1 (Unimodal Supervised Pre-training)**: Train the vision encoder (3D ResNet-18) and text encoder (CXR-BERT) independently using the extracted labels.
      ```bash
      python training/supervise_pretrain.py --mode vision --epochs 5
      python training/supervise_pretrain.py --mode text --epochs 5
      ```
    - **Stage 2 (Alignment)**: Align the pre-trained encoders with a shared classifier constraint.
      ```bash
      python training/alignment.py --pretrained_vision_path [path] --pretrained_text_path [path] --epochs 5
      ```

3.  **Inference (Zero-Shot)**:
    - Run the zero-shot script on the test sets using the aligned checkpoint.
      ```bash
      python inference/run_zero_shot.py --checkpoint [path]
      ```

## Project Structure
- `demo.py`: Standalone demo script for quick verification.
- `llm_pipeline/`: Tools for distilling noisy reports into "Silver-Standard" diagnostic labels using LLMs.
- `training/`: Staged training scripts implementing Ubiquitous & Semantic-Enforcing supervision.
- `inference/`: Evaluation scripts for Zero-Shot Diagnosis and Report-to-Volume Retrieval.
- `models/`: Streamlined architectures (3D ResNet-18, CXR-BERT, Alignment Heads).
- `data/`: Loaders for CT-RATE, Merlin, RAD-Chest, and internal MRI datasets.

