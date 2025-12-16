# Line Drawings Model

Minimal image-to-image model that converts cabinet door photos into clean black-and-white line drawings.

## Installation

```bash
pip install -r requirements.txt
```

## Training

Train the model on the paired dataset:

```bash
python train.py
```

This will:
- Load images from `dataset/originals/` and `dataset/line_drawings/`
- Train a LoRA adapter on Stable Diffusion 1.5
- Save LoRA weights to `lora_weights/`

Training typically takes 1-2 hours on an RTX 3060 (12GB VRAM).

## Inference

Convert a new image to a line drawing:

```bash
python infer.py <input_image> <output_image>
```

Example:
```bash
python infer.py my_cabinet.jpg output_line_drawing.png
```

The script will:
- Load the trained LoRA weights
- Preprocess the input image
- Run inference through the model
- Post-process to clean black and white
- Save the result

## Requirements

- Training: Windows/Linux with CUDA-capable GPU (12GB+ VRAM recommended)
- Inference: macOS (Apple Silicon) or any system with PyTorch support
