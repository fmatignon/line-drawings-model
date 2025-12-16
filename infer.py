import sys
import torch
from pathlib import Path
from PIL import Image
import numpy as np
from diffusers import StableDiffusionImg2ImgPipeline

# Hardcoded hyperparameters
IMAGE_SIZE = 512
MODEL_ID = "runwayml/stable-diffusion-v1-5"
LORA_WEIGHTS_DIR = "lora_weights"
STRENGTH = 0.8
GUIDANCE_SCALE = 1.0
NUM_INFERENCE_STEPS = 20

def preprocess_image(image_path, size=512):
    """Load and preprocess image for inference"""
    img = Image.open(image_path).convert("RGB")
    
    # Resize maintaining aspect ratio with padding
    img.thumbnail((size, size), Image.Resampling.LANCZOS)
    new_img = Image.new("RGB", (size, size), (0, 0, 0))
    paste_x = (size - img.width) // 2
    paste_y = (size - img.height) // 2
    new_img.paste(img, (paste_x, paste_y))
    
    return new_img

def otsu_threshold(gray):
    """Simple Otsu's thresholding implementation"""
    hist, bins = np.histogram(gray.flatten(), 256, [0, 256])
    hist = hist.astype(float)
    
    # Normalize histogram
    hist /= hist.sum()
    
    # Calculate cumulative sums and means
    cumsum = np.cumsum(hist)
    cummean = np.cumsum(hist * np.arange(256))
    
    # Calculate between-class variance for all thresholds
    between_class_variance = np.zeros(256)
    for t in range(256):
        w0 = cumsum[t]
        w1 = 1.0 - w0
        if w0 == 0 or w1 == 0:
            continue
        m0 = cummean[t] / w0 if w0 > 0 else 0
        m1 = (cummean[255] - cummean[t]) / w1 if w1 > 0 else 0
        between_class_variance[t] = w0 * w1 * (m0 - m1) ** 2
    
    # Find threshold with maximum variance
    threshold = np.argmax(between_class_variance)
    return threshold

def postprocess_to_bw(image_tensor):
    """Convert model output to clean black and white line drawing"""
    # Convert tensor to numpy array
    if isinstance(image_tensor, torch.Tensor):
        # Denormalize from [-1, 1] to [0, 255]
        arr = image_tensor.cpu().numpy()
        arr = (arr + 1.0) / 2.0  # [-1, 1] -> [0, 1]
        arr = np.clip(arr, 0, 1)
        arr = (arr * 255).astype(np.uint8)
        
        # Handle CHW format
        if arr.shape[0] == 3:
            arr = arr.transpose(1, 2, 0)  # CHW to HWC
    else:
        arr = np.array(image_tensor)
        if arr.max() <= 1.0:
            arr = (arr * 255).astype(np.uint8)
    
    # Convert to grayscale (RGB to grayscale)
    if len(arr.shape) == 3 and arr.shape[2] == 3:
        # Use standard RGB to grayscale weights
        gray = (0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]).astype(np.uint8)
    else:
        gray = arr
    
    # Apply Otsu's threshold for clean black/white
    threshold = otsu_threshold(gray)
    binary = (gray > threshold).astype(np.uint8) * 255
    
    # Invert if needed (ensure black lines on white background)
    # Check if most pixels are dark (should be inverted)
    if np.mean(binary) < 128:
        binary = 255 - binary
    
    return Image.fromarray(binary, mode="L").convert("RGB")

def main():
    if len(sys.argv) != 3:
        print("Usage: python infer.py <input_image> <output_image>")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    
    # Check if input exists
    if not Path(input_path).exists():
        print(f"Error: Input image not found: {input_path}")
        sys.exit(1)
    
    # Check if LoRA weights exist
    if not Path(LORA_WEIGHTS_DIR).exists():
        print(f"Error: LoRA weights directory not found: {LORA_WEIGHTS_DIR}")
        print("Please train the model first using train.py")
        sys.exit(1)
    
    # Determine device
    if torch.backends.mps.is_available():
        device = "mps"
        dtype = torch.float32  # MPS doesn't fully support float16
        print("Using MPS (Apple Silicon)")
    elif torch.cuda.is_available():
        device = "cuda"
        dtype = torch.float16
        print("Using CUDA")
    else:
        device = "cpu"
        dtype = torch.float32
        print("Using CPU")
    
    # Load pipeline
    print("Loading Stable Diffusion 1.5...")
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype
    )
    pipe = pipe.to(device)
    
    # Load LoRA weights using PEFT (matches training setup)
    print("Loading LoRA weights...")
    from peft import PeftModel
    pipe.unet = PeftModel.from_pretrained(pipe.unet, LORA_WEIGHTS_DIR)
    
    # Preprocess input image
    print(f"Processing {input_path}...")
    input_image = preprocess_image(input_path, size=IMAGE_SIZE)
    
    # Run inference
    print("Running inference...")
    # Use empty prompt for deterministic output
    result = pipe(
        prompt="",
        image=input_image,
        strength=STRENGTH,
        guidance_scale=GUIDANCE_SCALE,
        num_inference_steps=NUM_INFERENCE_STEPS,
    ).images[0]
    
    # Post-process to clean black and white
    print("Post-processing to black and white...")
    bw_result = postprocess_to_bw(result)
    
    # Save output
    bw_result.save(output_path)
    print(f"Saved output to {output_path}")

if __name__ == "__main__":
    main()

