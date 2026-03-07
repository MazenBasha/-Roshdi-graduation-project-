"""
Export trained model to TorchScript Lite (.ptl) for mobile deployment.

Exports the best trained checkpoint as an optimized TorchScript Lite file
compatible with PyTorch Mobile (Android/iOS).

Usage:
    python export_ptl.py
    python export_ptl.py --checkpoint outputs/best_model.pth
    python export_ptl.py --output outputs/model.ptl
    python export_ptl.py --optimize
"""

import argparse
import os
import time

import torch
from torch.utils.mobile_optimizer import optimize_for_mobile

import config
from model import build_model, count_parameters
from dataset import get_eval_transforms


def parse_args():
    parser = argparse.ArgumentParser(description="Export model to TorchScript Lite")
    parser.add_argument(
        "--checkpoint", type=str, default=config.BEST_MODEL_PATH,
        help="Path to trained model checkpoint",
    )
    parser.add_argument(
        "--output", type=str, default=config.EXPORT_PATH,
        help="Output path for .ptl file",
    )
    parser.add_argument(
        "--optimize", action="store_true", default=True,
        help="Apply mobile optimizations (default: True)",
    )
    parser.add_argument(
        "--no-optimize", action="store_true",
        help="Skip mobile optimizations",
    )
    parser.add_argument(
        "--verify", action="store_true", default=True,
        help="Verify exported model with dummy input",
    )
    return parser.parse_args()


def export_to_ptl(
    checkpoint_path: str,
    output_path: str,
    optimize: bool = True,
    verify: bool = True,
):
    """
    Export a trained PyTorch model to TorchScript Lite format.

    Steps:
    1. Load trained weights from checkpoint
    2. Trace the model with a dummy input
    3. Apply mobile optimizations (fuse ops, etc.)
    4. Save as TorchScript Lite (.ptl)
    5. Verify the exported model produces matching outputs
    """
    print(f"{'=' * 60}")
    print("Exporting model to TorchScript Lite (.ptl)")
    print(f"{'=' * 60}")

    # 1. Load model with trained weights
    print(f"\n[1/5] Loading checkpoint: {checkpoint_path}")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            "Train the model first with: python train.py"
        )

    model = build_model()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    epoch = checkpoint.get("epoch", "?")
    metrics = checkpoint.get("metrics", {})
    acc = metrics.get("accuracy", "?")
    print(f"  Epoch: {epoch}, Val Accuracy: {acc}")
    print(f"  Parameters: {count_parameters(model):,}")

    # 2. Trace the model
    print(f"\n[2/5] Tracing model with dummy input...")
    dummy_input = torch.randn(1, config.IMG_CHANNELS, config.IMG_SIZE, config.IMG_SIZE)

    with torch.no_grad():
        original_output = model(dummy_input)
        traced_model = torch.jit.trace(model, dummy_input)

    print(f"  Input shape:  {list(dummy_input.shape)}")
    print(f"  Output shape: {list(original_output.shape)}")

    # 3. Apply mobile optimizations
    if optimize:
        print(f"\n[3/5] Applying mobile optimizations...")
        try:
            traced_model = optimize_for_mobile(traced_model)
            print("  Optimizations applied: conv-bn fusion, dropout removal, etc.")
        except Exception as e:
            print(f"  Warning: Mobile optimization failed ({e}), using unoptimized model")
    else:
        print(f"\n[3/5] Skipping mobile optimizations")

    # 4. Save as TorchScript Lite
    print(f"\n[4/5] Saving TorchScript Lite model...")
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    traced_model._save_for_lite_interpreter(output_path)

    file_size = os.path.getsize(output_path)
    print(f"  Saved to: {output_path}")
    print(f"  File size: {file_size / 1024 / 1024:.2f} MB")

    # 5. Verify exported model
    if verify:
        print(f"\n[5/5] Verifying exported model...")
        loaded_model = torch.jit.load(output_path, map_location="cpu")
        loaded_model.eval()

        with torch.no_grad():
            exported_output = loaded_model(dummy_input)

        # Check output shapes match
        assert original_output.shape == exported_output.shape, (
            f"Shape mismatch: {original_output.shape} vs {exported_output.shape}"
        )

        # Check outputs are close
        max_diff = (original_output - exported_output).abs().max().item()
        print(f"  Output shape: {list(exported_output.shape)} ✓")
        print(f"  Max output difference: {max_diff:.8f}")

        if max_diff < 1e-4:
            print("  Verification: PASSED ✓")
        else:
            print(f"  Verification: WARNING - max diff {max_diff:.6f} (may be due to optimizations)")

        # Benchmark inference speed
        print(f"\n  Benchmarking inference speed (CPU)...")
        times = []
        for _ in range(20):
            start = time.perf_counter()
            with torch.no_grad():
                _ = loaded_model(dummy_input)
            times.append((time.perf_counter() - start) * 1000)
        avg_time = sum(times[5:]) / len(times[5:])  # Skip warmup
        print(f"  Average inference time: {avg_time:.1f} ms (CPU)")
    else:
        print(f"\n[5/5] Skipping verification")

    # Summary
    print(f"\n{'=' * 60}")
    print(f"Export complete!")
    print(f"  Model file: {output_path}")
    print(f"  File size:  {file_size / 1024 / 1024:.2f} MB")
    print(f"  Classes:    {config.NUM_CLASSES} ({', '.join(config.CLASS_NAMES)})")
    print(f"  Input size: 1x{config.IMG_CHANNELS}x{config.IMG_SIZE}x{config.IMG_SIZE}")
    print(f"{'=' * 60}")

    return output_path


def main():
    args = parse_args()
    do_optimize = not args.no_optimize
    export_to_ptl(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        optimize=do_optimize,
        verify=args.verify,
    )


if __name__ == "__main__":
    main()
