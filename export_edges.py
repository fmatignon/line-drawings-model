import os
from pathlib import Path
from PIL import Image
from model import detect_edges

# Directories
originals_dir = Path("dataset/originals")
edges_dir = Path("dataset/edges")

# Create output directory
edges_dir.mkdir(parents=True, exist_ok=True)

# Find all image files
image_extensions = [".png", ".jpg", ".jpeg"]
image_files = []
for ext in image_extensions:
    image_files.extend(originals_dir.glob(f"*{ext}"))
    image_files.extend(originals_dir.glob(f"*{ext.upper()}"))

print(f"Found {len(image_files)} images to process")

# Process each image
for img_path in sorted(image_files):
    try:
        print(f"Processing {img_path.name}...")
        
        # Load image
        img = Image.open(img_path).convert("RGB")
        
        # Apply edge detection
        edge_img = detect_edges(img)
        
        # Save with same filename
        output_path = edges_dir / img_path.name
        edge_img.save(output_path)
        
    except Exception as e:
        print(f"  Error processing {img_path.name}: {e}")
        continue

print(f"\nDone! Exported {len(image_files)} edge-detected images to {edges_dir}")

