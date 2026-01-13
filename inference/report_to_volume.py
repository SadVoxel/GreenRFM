import numpy as np
import torch
import tqdm
import wandb
import os
import argparse
import sys

# Path handling if needed
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, ".."))
if root_dir not in sys.path:
    sys.path.append(root_dir)

def compute_topk_streaming(image_data, text_data, list_ks, batch_size=256, force_cpu=False):
    """
    Streamed top-k hit computation to avoid full similarity matrix in memory.
    Calculates Text-to-Image Retrieval (R@K).
    Args:
        image_data: (N, D) numpy array
        text_data: (N, D) numpy array (assumes aligned pairs, i.e., text[i] matches image[i])
        list_ks: list of k values for R@K
    Returns:
        clip_hits: dict {k: accuracy}
        rand_hits: dict {k: accuracy} (baseline)
    """
    max_k = max(list_ks)
    device = torch.device("cpu") if force_cpu else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Ensure float32 for matmul stability
    image_tensor = torch.from_numpy(image_data).float()
    text_tensor = torch.from_numpy(text_data).float()
    
    # Normalize features if they aren't already (Cosine Similarity)
    # CT-CLIP usually outputs normalized, but let's be safe or assume dot product is sufficient if trained that way.
    # Standard CLIP eval uses cosine similarity.
    image_tensor = torch.nn.functional.normalize(image_tensor, p=2, dim=1)
    text_tensor = torch.nn.functional.normalize(text_tensor, p=2, dim=1)

    if device.type != "cpu":
        try:
            image_tensor = image_tensor.to(device)
        except RuntimeError:
            print("GPU OOM for Image Tensor, falling back to CPU.")
            device = torch.device("cpu")

    total = text_tensor.shape[0]
    clip_counts = {k: 0 for k in list_ks}
    rand_counts = {k: 0 for k in list_ks}

    # Iterate over text query batches
    for start in tqdm.tqdm(range(0, total, batch_size), desc="Evaluating Batches"):
        end = min(start + batch_size, total)
        text_batch = text_tensor[start:end].to(device)

        # Similarity Matrix [Batch, N_images]
        sims = text_batch @ image_tensor.T
        
        # Get Top-K indices
        # largest=True for cosine similarity
        topk_indices = torch.topk(sims, k=max_k, dim=1).indices
        
        # Ground Truth targets: indices [start...end]
        target = torch.arange(start, end, device=device).unsqueeze(1)
        
        for k in list_ks:
            # Check if target index is in top K
            clip_counts[k] += (topk_indices[:, :k] == target).any(dim=1).sum().item()

        # Random Baseline Simulation
        # To save compute, maybe skip or do less frequently? 
        # Or just compute once to show "chance" level (1/N).
        # Keeping it for consistency with reference code.
        # rand_sims = torch.rand_like(sims, device=device)
        # rand_topk_indices = torch.topk(rand_sims, k=max_k, dim=1).indices
        # for k in list_ks:
        #     rand_counts[k] += (rand_topk_indices[:, :k] == target).any(dim=1).sum().item()
        
        # Simple analytic random chance for R@input N is k/N 
        # Implementing the full random sim is expensive and noisy. 
        # I'll replace it with formula k/total later or keep counts 0.
        
        del text_batch, sims, topk_indices
        if device.type == "cuda":
            torch.cuda.empty_cache()

    clip_hits = {k: clip_counts[k] / total for k in list_ks}
    # rand_hits = {k: rand_counts[k] / total for k in list_ks}
    # Analytic random
    rand_hits = {k: k / total for k in list_ks}
    
    return clip_hits, rand_hits

def run_retrieval_eval(feature_dir, use_wandb=False, project_name="LeanRad_Retrieval"):
    """
    Scans feature directory for ckpt folders and evaluates retrieval.
    Expected structure: feature_dir/ckpt_X/image_latents.npz, text_latents.npz
    """
    
    # Init WandB global if needed, or per ckpt
    # Reference code inits per group.
    
    # 1. Find subdirectories
    if not os.path.exists(feature_dir):
        print(f"Feature directory not found: {feature_dir}")
        return

    subdirs = sorted([d for d in os.listdir(feature_dir) if os.path.isdir(os.path.join(feature_dir, d))])
    
    # Sort by ckpt number if possible
    def extract_step(name):
        if name.startswith("ckpt_"):
            try:
                return int(name.split("_")[1])
            except:
                return 0
        return 0
        
    subdirs.sort(key=extract_step)

    print(f"Found {len(subdirs)} checkpoints to evaluate in {feature_dir}")

    list_ks = [1, 5, 10, 50, 100]

    for subdir in subdirs:
        step = extract_step(subdir)
        ckpt_path = os.path.join(feature_dir, subdir)
        
        img_path = os.path.join(ckpt_path, "image_latents.npz")
        txt_path = os.path.join(ckpt_path, "text_latents.npz")
        
        if not (os.path.exists(img_path) and os.path.exists(txt_path)):
            # print(f"Skipping {subdir}: missing npz files.")
            continue
            
        print(f"\nEvaluating {subdir}...")
        
        # Load Data
        try:
            image_data = np.load(img_path)["data"]
            text_data = np.load(txt_path)["data"]
        except Exception as e:
            print(f"Error loading data: {e}")
            continue
            
        if image_data.shape[0] != text_data.shape[0]:
            print(f"Mismatch shapes: Img {image_data.shape}, Txt {text_data.shape}. Truncating to min.")
            min_len = min(image_data.shape[0], text_data.shape[0])
            image_data = image_data[:min_len]
            text_data = text_data[:min_len]

        print(f"N samples: {image_data.shape[0]}")
        
        # Compute Streaming TopK
        clip_hits, rand_hits = compute_topk_streaming(
            image_data, 
            text_data, 
            list_ks, 
            batch_size=256
        )
        
        # Log / Save
        results_txt = []
        md_table = "| K | R@K (CLIP) | Random |\n|---|---|---|\n"
        
        for k in list_ks:
            acc = clip_hits[k]
            rand = rand_hits[k]
            line = f"K={k}: Acc={acc:.4f} (Rand={rand:.4f})"
            results_txt.append(line)
            md_table += f"| {k} | {acc:.4f} | {rand:.4f} |\n"
            print(line)

        # Save results to file
        with open(os.path.join(ckpt_path, "retrieval_results.txt"), "w") as f:
            f.write("\n".join(results_txt))
            
        # WandB Logging
        if use_wandb:
            if wandb.run is None:
                wandb.init(project=project_name, name=os.path.basename(feature_dir), reinit=True)
            
            metrics = {}
            for k in list_ks:
                metrics[f"R@{k}"] = clip_hits[k]
            
            # Log with step if available
            wandb.log(metrics, step=step)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature_dir", type=str, required=False, 
                        help="Directory containing output from infer_feature.py (e.g. features/ah_knee/sup_model)")
    parser.add_argument("--wandb", action="store_true", help="Enable WandB logging")
    parser.add_argument("--project", type=str, default="LeanRad_Retrieval_Eval")
    
    args = parser.parse_args()

    # Default testing path if not provided
    if not args.feature_dir:
        # Check portable default under project root before creating dummy data
        default_path = "./feature_outputs/sup_for_retrieval"
        print(f"No feature_dir provided. Testing with default: {default_path}")
        if os.path.exists(default_path):
            args.feature_dir = default_path
        else:
            print("Default path does not exist. Creating dummy data for logic verification.")
            os.makedirs("dummy_features/ckpt_100", exist_ok=True)
            N = 100
            D = 768
            np.savez_compressed("dummy_features/ckpt_100/image_latents.npz", data=np.random.randn(N, D))
            np.savez_compressed("dummy_features/ckpt_100/text_latents.npz", data=np.random.randn(N, D))
            args.feature_dir = "dummy_features"

    if args.feature_dir:
        run_retrieval_eval(args.feature_dir, use_wandb=args.wandb, project_name=args.project)
