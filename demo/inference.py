"""
Demo inference script for MPLNet & MCPNet.

This script demonstrates loading a trained model and running semantic segmentation
on a sample input image. It is designed for quick verification and portfolio
demonstration only — not for production or benchmarking.

Requirements:
    pip install torch torchvision pillow numpy

Usage:
    python demo/inference.py \
        --model mcpnet \
        --image sample_input/scene_001.jpg \
        --output sample_output/result.png
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

# Ensure the project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Model builders (inline to keep the demo self-contained)
# ---------------------------------------------------------------------------

def build_mcpnet(num_classes: int = 22, pretrained: bool = False):
    """Build MCPNet from local model code."""
    from models import get_mcpnet
    return get_mcpnet(num_classes=num_classes, pretrained=pretrained)


def build_mplnet(num_classes: int = 22, pretrained: bool = False):
    """Build MPLNet from local model code."""
    from models import get_mplnet
    return get_mplnet(num_classes=num_classes, pretrained=pretrained)


MODEL_BUILDERS = {
    "mcpnet": build_mcpnet,
    "mplnet": build_mplnet,
}

# ---------------------------------------------------------------------------
# Pre-processing
# ---------------------------------------------------------------------------

TV_RSI_413_MEAN = (0.485, 0.456, 0.406)
TV_RSI_413_STD = (0.229, 0.224, 0.225)


def load_and_preprocess(image_path: str, img_size: int = 512):
    """
    Load an image from disk, resize, convert to tensor, and normalise.
    Returns (tensor[B=1, 3, H, W], original_pil).
    """
    img = Image.open(image_path).convert("RGB")
    orig = img.copy()
    img = img.resize((img_size, img_size), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - np.array(TV_RSI_413_MEAN)) / np.array(TV_RSI_413_STD)
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # (1,3,H,W)
    return tensor, orig


# ---------------------------------------------------------------------------
# Colormap (22-class palette — illustrative)
# ---------------------------------------------------------------------------

CLASS_COLORMAP = np.array([
    [0,   0,   0],    # 0  background
    [128, 0,   0],    # 1  building
    [128, 64,  0],    # 2  road
    [0,   128, 128],  # 3  water
    [0,   128, 0],    # 4  vegetation
    [128, 128, 0],    # 5  farmland
    [128, 128, 128],  # 6  bare_soil
    [64,  0,   0],    # 7  temple
    [192, 0,   0],    # 8  ancestral_hall
    [64,  128, 0],    # 9  well
    [0,   64,  128],  # 10 pond
    [128, 0,   128],  # 11 fence
    [128, 128, 64],   # 12 bridge
    [0,   0,   128],  # 13 traditional_structure
    [0,   128, 64],   # 14 modern_building
    [192, 128, 128],  # 15 mountain
    [64,  0,   128],  # 16 forest
    [128, 64,  128],  # 17 grass
    [64,  0,   64],   # 18 cultivated_land
    [192, 192, 192],  # 19 construction_land
    [255, 215, 0],    # 20 cultural_heritage_site
    [192, 192, 0],    # 21 other
], dtype=np.uint8)


def pred_to_color(pred: np.ndarray, colormap: np.ndarray = CLASS_COLORMAP):
    """Convert a class-index map [H,W] to an RGB visualisation."""
    H, W = pred.shape
    color = np.zeros((H, W, 3), dtype=np.uint8)
    for c in range(colormap.shape[0]):
        color[pred == c] = colormap[c]
    return color


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_inference(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    device: torch.device,
):
    """Run forward pass and return argmax prediction."""
    model.eval()
    image_tensor = image_tensor.to(device)
    output = model(image_tensor)

    if isinstance(output, tuple):
        seg_logits = output[0]
    else:
        seg_logits = output

    # Upsample to input resolution if needed
    seg_logits = F.interpolate(
        seg_logits,
        size=image_tensor.shape[2:],
        mode="bilinear",
        align_corners=False,
    )
    pred = seg_logits.argmax(dim=1).squeeze(0).cpu().numpy()
    return pred


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="MPLNet / MCPNet demo inference"
    )
    parser.add_argument(
        "--model", type=str, default="mcpnet",
        choices=["mcpnet", "mplnet"],
        help="Model name",
    )
    parser.add_argument(
        "--image", type=str, required=True,
        help="Path to input image (jpg/png)",
    )
    parser.add_argument(
        "--output", type=str, default="sample_output/result.png",
        help="Path to save the output visualisation",
    )
    parser.add_argument(
        "--num_classes", type=int, default=22,
    )
    parser.add_argument(
        "--img_size", type=int, default=512,
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help=(
            "Path to a model checkpoint (.pth). "
            "If not provided, the model will use ImageNet-pretrained weights "
            "and produce illustrative (not calibrated) predictions."
        ),
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device override (cuda / cpu). Auto-detected by default.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Device
    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)
    print(f"[demo] Using device: {device}")

    # Model
    builder = MODEL_BUILDERS[args.model]
    model = builder(num_classes=args.num_classes, pretrained=False)
    model = model.to(device)

    # -- Weights -----------------------------------------------------------------
    if args.checkpoint and os.path.isfile(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        state = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state, strict=False)
        print(f"[demo] Loaded checkpoint: {args.checkpoint}")
    else:
        print(
            "[demo] WARNING — No checkpoint provided. "
            "Running with random ImageNet backbone only. "
            "Output is illustrative, NOT calibrated for heritage mapping."
        )
        print("[demo] Trained weights available upon reasonable academic request.")

    # Input image
    image_tensor, orig_pil = load_and_preprocess(args.image, args.img_size)

    # Inference
    pred = run_inference(model, image_tensor, device)

    # Upsample prediction to original size
    pred_img = Image.fromarray(pred.astype(np.uint8))
    pred_img = pred_img.resize(orig_pil.size, Image.NEAREST)

    # Visualisation: overlay colour on original
    pred_color = pred_to_color(np.array(pred_img), CLASS_COLORMAP)

    # Side-by-side: original | prediction
    result = Image.new("RGB", (orig_pil.width * 2, orig_pil.height))
    result.paste(orig_pil, (0, 0))
    result.paste(Image.fromarray(pred_color), (orig_pil.width, 0))

    # Save
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    result.save(args.output)
    print(f"[demo] Result saved to: {args.output}")


if __name__ == "__main__":
    main()