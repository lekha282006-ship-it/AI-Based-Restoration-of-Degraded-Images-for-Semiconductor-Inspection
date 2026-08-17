import argparse
import sys
from pathlib import Path

import numpy as np
import torch

from src.model import LightSRNet
from src.dataset import normalize_input_image

def find_model_weights(models_dir: str) -> str:
    """Finds the first .pth or .pt file in the models directory."""
    models_path = Path(models_dir)
    if not models_path.exists():
        raise FileNotFoundError(f"Models directory '{models_dir}' not found.")
    
    weights = list(models_path.glob("*.pth")) + list(models_path.glob("*.pt"))
    if not weights:
        raise FileNotFoundError(f"No .pth or .pt files found in '{models_dir}'.")
    return str(weights[0])

def main():
    parser = argparse.ArgumentParser(description="Run inference for repaired semiconductor images.")
    parser.add_argument("input_dir", type=str, help="Folder of noisy .npy images to restore.")
    parser.add_argument("output_dir", type=str, help="Output folder for restored .npy images.")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists() or not input_dir.is_dir():
        print(f"Error: Input directory '{input_dir}' does not exist or is not a directory.")
        sys.exit(1)

    # Automatically create <output-dir> if it does not exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set device dynamically to GPU if available, fallback to CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load pretrained weights from local models/ directory (OFFLINE)
    try:
        model_path = find_model_weights("models")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    print(f"Loading model weights from {model_path}")
    
    # Initialize model and load weights
    model = LightSRNet().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    # Handle both plain state_dict and full checkpoint saving formats
    state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state_dict)
    model.eval()

    # Read all .npy files from <input-dir>
    files = sorted(input_dir.glob("*.npy"))
    if not files:
        print(f"Warning: No .npy files found in '{input_dir}'.")
        sys.exit(0)

    print(f"Found {len(files)} files to process.")

    for file_path in files:
        try:
            # Load degraded image
            degraded = np.load(file_path).astype(np.float32)
            
            # Perform checks to guarantee NO NaN or Inf values exist
            if not np.isfinite(degraded).all():
                degraded = np.nan_to_num(degraded, nan=0.0, posinf=1.0, neginf=0.0)
            
            # Normalization as used during training
            degraded_norm = normalize_input_image(degraded)
            
            # Prepare tensor shape (1, 1, H, W)
            input_tensor = torch.from_numpy(degraded_norm).unsqueeze(0).unsqueeze(0).to(device)

            # Inference
            with torch.no_grad():
                restored = model(input_tensor)

            # Post-processing
            # Squeeze batch and channel dims assuming (1, 1, H, W) -> (H, W)
            restored_np = restored[0, 0].cpu().numpy()
            
            # Handle any potential NaNs or Infs from the network
            if not np.isfinite(restored_np).all():
                restored_np = np.nan_to_num(restored_np, nan=0.0, posinf=1.0, neginf=0.0)
                
            # Clip values strictly within [0.0, 1.0]
            restored_np = np.clip(restored_np, 0.0, 1.0)
            
            # Generate one restored .npy file per input file with identical filenames
            out_path = output_dir / file_path.name
            np.save(out_path, restored_np.astype(np.float32))
            
        except Exception as e:
            print(f"Error processing {file_path.name}: {e}")

    print("Inference completed successfully.")

if __name__ == "__main__":
    main()
