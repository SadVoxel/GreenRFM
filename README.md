# GreenRFM

GreenRFM is a lightweight and efficient framework for radiology report understanding, multi-modal alignment (image-text), and feature extraction. It supports both CT and MRI modalities (Chest, Abdomen, Knee, Spine).

## Project Structure

- `llm_pipeline/`:  Tools for extracting structured labels from radiology reports using LLMs (e.g., OpenAI / Compatible APIs).
- `training/`:      Scripts for training image-text alignment models (CLIP-style) and supervised classification models.
- `inference/`:     Inference scripts for zero-shot classification and feature extraction.
- `models/`:        Model architectures (3D ResNet, CLIP, Text Encoders).
- `data/`:          Dataset loaders and processing utilities for various datasets (CT-RATE, Merlin, AH-Knee, etc.).
- `config/`:        Centralized configuration files.

## Installation

Ensure you have Python 3.8+ installed.

```bash
pip install -r requirements.txt
```

*Dependencies include: `torch`, `torchvision`, `transformers`, `pandas`, `numpy`, `tqdm`, `wandb`, `openai`, `monai`, `nibabel`, `scikit-learn`.*

## Configuration

### API Keys
For the LLM Pipeline, you need to set your OpenAI (or compatible) API key and base URL as environment variables:

```bash
export OPENAI_API_KEY="your_api_key_here"
export OPENAI_BASE_URL="your_base_url_here"  # Optional, defaults to OpenAI
export OPENAI_MODEL="gpt-4"                  # Optional, defaults to gpt-4
```

### Data Paths
Edit `config/config.py` to point to your local dataset paths before running inference or training scripts.

## Usage

### 1. LLM Label Extraction
Extract structured labels from text reports.

```bash
python -m llm_pipeline.knee_mr
# or
python -m llm_pipeline.abdominal_ct
```

### 2. Training
Run supervised pre-training or alignment training.

```bash
# Supervised Pre-training
python -m training.supervise_pretrain

# Alignment Training (CLIP)
python -m training.alignment
```

### 3. Inference
Run zero-shot inference or feature extraction.

```bash
python -m inference.run_zero_shot
```

## Citation

If you use this code or framework in your research, please cite our manuscript:

> *[Citation Placeholder: Authors, "Title", Journal/Conference, Year]*

