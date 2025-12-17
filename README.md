# Image-to-Image Line Drawing Model

Minimal repository to train and run inference for a CNN-based image-to-image model that converts cabinet door photos (with runtime edge detection) into clean architectural line drawings.

## Installation

```bash
pip install -r requirements.txt
```

## Dataset Structure

```
dataset/
├── originals/      # Input cabinet door photos (*.png, *.jpg)
└── line_drawings/  # Target line drawings (*.png, *.jpg)
```

Filenames must match exactly between the two directories (e.g., `door001.png` in originals matches `door001.png` in line_drawings).

## Training

Train the model on the paired dataset:

```bash
python train.py
```

**Test Mode** (to check for overfitting on 10 images):

```bash
python train.py --test
```

This will:
- Load images from `dataset/originals/` and `dataset/line_drawings/`
- Apply edge detection to originals at runtime (not saved to disk)
- Train a U-Net model with L1 + SSIM loss
- Save best model to `best_model.pth` and final model to `final_model.pth`
- Use early stopping if validation loss doesn't improve for 10 epochs

**Training hyperparameters** (can be modified in `train.py`):
- Image size: 512×512
- Batch size: 4
- Learning rate: 1e-4
- Epochs: 200 (with early stopping)

## Inference

Convert a single image:

```bash
python infer.py <input_image> <output_image>
```

Example:
```bash
python infer.py my_cabinet.jpg output_line_drawing.png
```

Convert all images in a directory:

```bash
python infer.py <input_directory> <output_directory>
```

Example:
```bash
python infer.py dataset/originals/ results/
```

The script will:
- Load the trained model (default: `best_model.pth`)
- Apply edge detection to input images
- Run inference through the model
- Save line drawing outputs

**Specify a different model:**
```bash
python infer.py input.jpg output.png --model final_model.pth
```

## Model Architecture

- **U-Net encoder-decoder** with skip connections
- **Input**: 3-channel RGB edge-detected images (512×512)
- **Output**: 1-channel grayscale line drawings (512×512)
- **Parameters**: ~15-20M (fits in 12GB VRAM)

## Requirements

- Training: Windows/Linux with CUDA-capable GPU (12GB+ VRAM recommended)
- Inference: Any system with PyTorch support (CPU or GPU)

## Notes

- Edge detection is applied at runtime during both training and inference (not saved to disk)
- The model uses L1 + SSIM loss for better structural preservation
- Augmentations (flip, rotation, brightness/contrast) are applied only to training inputs
- With ~140 samples, the model may overfit; use early stopping and test mode to verify learning

