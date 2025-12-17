import sys
import os
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

# Import model architecture and edge detection from model.py
from model import UNet, detect_edges

IMAGE_SIZE = 512


def resize_with_padding(img, size=512):
    """Resize image to size x size maintaining aspect ratio"""
    img.thumbnail((size, size), Image.Resampling.LANCZOS)
    new_img = Image.new(
        "RGB" if img.mode == "RGB" else "L",
        (size, size),
        (0, 0, 0) if img.mode == "RGB" else 0,
    )
    paste_x = (size - img.width) // 2
    paste_y = (size - img.height) // 2
    new_img.paste(img, (paste_x, paste_y))
    return new_img


def to_tensor(img):
    """Convert PIL image to tensor normalized to [0, 1]"""
    arr = np.array(img).astype(np.float32) / 255.0
    if len(arr.shape) == 2:  # Grayscale
        arr = arr[np.newaxis, :, :]  # Add channel dimension
    else:
        arr = arr.transpose(2, 0, 1)  # HWC to CHW
    return torch.from_numpy(arr)


def process_image(model, image_path, device):
    """Process a single image through the model"""
    # Load and preprocess
    img = Image.open(image_path).convert("RGB")
    img = resize_with_padding(img, IMAGE_SIZE)

    # Apply edge detection
    edge_img = detect_edges(img)

    # Convert to tensor
    input_tensor = to_tensor(edge_img).unsqueeze(0).to(device)  # Add batch dimension

    # Run inference
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)

    # Convert output to image
    output_np = (
        output.squeeze(0).squeeze(0).cpu().numpy()
    )  # Remove batch and channel dims
    output_np = np.clip(output_np * 255, 0, 255).astype(np.uint8)
    output_img = Image.fromarray(output_np, mode="L")

    return output_img


def main():
    parser = argparse.ArgumentParser(description="Run inference on images")
    parser.add_argument("input", help="Input image path or directory")
    parser.add_argument("output", help="Output image path or directory")
    parser.add_argument(
        "--model",
        default="best_model.pth",
        help="Path to model checkpoint (default: best_model.pth)",
    )
    args = parser.parse_args()

    # Check if model exists
    if not os.path.exists(args.model):
        print(f"Error: Model file not found: {args.model}")
        print("Please train the model first using train.py")
        sys.exit(1)

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    print(f"Loading model from {args.model}...")
    model = UNet(in_channels=3, out_channels=1).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    print("Model loaded successfully")

    # Determine if input is file or directory
    input_path = Path(args.input)
    output_path = Path(args.output)

    if input_path.is_file():
        # Single image
        print(f"Processing {input_path}...")
        output_img = process_image(model, input_path, device)

        # Create output directory if needed
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_img.save(output_path)
        print(f"Saved output to {output_path}")

    elif input_path.is_dir():
        # Directory of images
        print(f"Processing directory: {input_path}")

        # Create output directory
        output_path.mkdir(parents=True, exist_ok=True)

        # Find all image files
        image_extensions = [".png", ".jpg", ".jpeg"]
        image_files = []
        for ext in image_extensions:
            image_files.extend(input_path.glob(f"*{ext}"))
            image_files.extend(input_path.glob(f"*{ext.upper()}"))

        if not image_files:
            print(f"Error: No image files found in {input_path}")
            sys.exit(1)

        print(f"Found {len(image_files)} images")

        # Process each image
        for img_path in sorted(image_files):
            try:
                print(f"  Processing {img_path.name}...")
                output_img = process_image(model, img_path, device)

                # Save with same name
                output_file = output_path / img_path.name
                output_img.save(output_file)
            except Exception as e:
                print(f"  Error processing {img_path.name}: {e}")
                continue

        print(f"\nProcessed {len(image_files)} images. Output saved to {output_path}")

    else:
        print(f"Error: Input path does not exist: {input_path}")
        sys.exit(1)


if __name__ == "__main__":
    main()
