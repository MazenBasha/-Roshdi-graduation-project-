"""
Inference script for Egyptian Currency Classification Model.

Run prediction on single images or directories.
Supports both PyTorch checkpoint (.pth) and TorchScript Lite (.ptl) models.

Usage:
    python infer.py --image path/to/image.jpg
    python infer.py --image-dir path/to/images/
    python infer.py --image path/to/image.jpg --model outputs/model.ptl
    python infer.py --image path/to/image.jpg --top-k 3
"""

import argparse
import os
import time
from typing import List, Tuple

import torch
from PIL import Image

import config
from dataset import get_eval_transforms
from model import build_model


def parse_args():
    parser = argparse.ArgumentParser(description="Egyptian Currency Inference")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", type=str, help="Path to a single image")
    group.add_argument("--image-dir", type=str, help="Path to directory of images")
    parser.add_argument(
        "--model", type=str, default=config.BEST_MODEL_PATH,
        help="Path to model (.pth checkpoint or .ptl TorchScript Lite)",
    )
    parser.add_argument("--top-k", type=int, default=3, help="Number of top predictions to show")
    parser.add_argument("--threshold", type=float, default=0.1, help="Min confidence to display")
    return parser.parse_args()


def load_model(model_path: str, device: torch.device):
    """Load model from .pth checkpoint or .ptl TorchScript Lite file."""
    if model_path.endswith(".ptl") or model_path.endswith(".pt"):
        # TorchScript Lite model
        model = torch.jit.load(model_path, map_location=device)
        model.eval()
        print(f"Loaded TorchScript model from: {model_path}")
        return model
    else:
        # PyTorch checkpoint
        model = build_model()
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(device)
        model.eval()
        print(f"Loaded PyTorch checkpoint from: {model_path}")
        return model


def predict_image(
    model: torch.nn.Module,
    image_path: str,
    transform,
    device: torch.device,
    top_k: int = 3,
) -> List[Tuple[str, float]]:
    """
    Predict currency class for a single image.

    Returns:
        List of (class_name, confidence) tuples, sorted by confidence.
    """
    img = Image.open(image_path).convert("RGB")
    input_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        start = time.perf_counter()
        output = model(input_tensor)
        elapsed = (time.perf_counter() - start) * 1000  # ms

    probs = torch.softmax(output, dim=1)[0]
    top_probs, top_indices = probs.topk(min(top_k, len(config.CLASS_NAMES)))

    results = []
    for prob, idx in zip(top_probs.cpu(), top_indices.cpu()):
        results.append((config.CLASS_NAMES[idx.item()], prob.item()))

    return results, elapsed


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    if not os.path.exists(args.model):
        raise FileNotFoundError(
            f"Model not found: {args.model}\n"
            "Train the model first with: python train.py\n"
            "Or export with: python export_ptl.py"
        )

    model = load_model(args.model, device)
    transform = get_eval_transforms()

    # Collect images
    supported_ext = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    image_paths = []

    if args.image:
        if not os.path.exists(args.image):
            raise FileNotFoundError(f"Image not found: {args.image}")
        image_paths.append(args.image)
    elif args.image_dir:
        if not os.path.isdir(args.image_dir):
            raise FileNotFoundError(f"Directory not found: {args.image_dir}")
        for fname in sorted(os.listdir(args.image_dir)):
            if os.path.splitext(fname)[1].lower() in supported_ext:
                image_paths.append(os.path.join(args.image_dir, fname))
        if not image_paths:
            raise ValueError(f"No images found in: {args.image_dir}")

    # Run inference
    print(f"\nRunning inference on {len(image_paths)} image(s)...")
    print(f"Device: {device}")
    print(f"{'=' * 60}")

    total_time = 0.0
    for img_path in image_paths:
        results, elapsed = predict_image(model, img_path, transform, device, args.top_k)
        total_time += elapsed

        filename = os.path.basename(img_path)
        top_class, top_conf = results[0]

        print(f"\n  {filename}")
        print(f"  Prediction: {top_class} EGP ({top_conf:.1%})")
        print(f"  Inference time: {elapsed:.1f} ms")

        # Show top-k predictions above threshold
        if len(results) > 1:
            print(f"  Top-{args.top_k} predictions:")
            for rank, (cls_name, conf) in enumerate(results, 1):
                if conf >= args.threshold:
                    bar = "█" * int(conf * 30)
                    print(f"    {rank}. {cls_name:>10} EGP: {conf:.1%} {bar}")

    print(f"\n{'=' * 60}")
    print(f"Total images: {len(image_paths)}")
    print(f"Average inference time: {total_time / max(len(image_paths), 1):.1f} ms")


if __name__ == "__main__":
    main()
