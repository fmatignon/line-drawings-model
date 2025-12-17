import sys
import torch
import torch.nn as nn
from pathlib import Path
from PIL import Image
import numpy as np
from diffusers import StableDiffusionImg2ImgPipeline
from peft import PeftModel

# Hardcoded hyperparameters
IMAGE_SIZE = 512
MODEL_ID = "runwayml/stable-diffusion-v1-5"
LORA_WEIGHTS_DIR = "lora_weights"
STRENGTH = 0.9  # Higher strength for more transformation
GUIDANCE_SCALE = 7.5  # Higher guidance for better adherence to learned style
NUM_INFERENCE_STEPS = 50  # More steps for better quality


def detect_edges(img):
    """Convert image to edge-detected version using Sobel filters"""
    # Convert PIL to numpy array
    arr = np.array(img).astype(np.float32) / 255.0

    # Convert to grayscale if RGB
    if len(arr.shape) == 3:
        gray = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
    else:
        gray = arr

    # Sobel kernels for edge detection
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)

    # Apply Sobel filters using numpy vectorized operations
    def convolve2d(image, kernel):
        """Fast 2D convolution using numpy"""
        h, w = image.shape
        kh, kw = kernel.shape
        pad_h, pad_w = kh // 2, kw // 2
        padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode="edge")

        # Use numpy's sliding window view for efficient convolution
        from numpy.lib.stride_tricks import sliding_window_view

        windows = sliding_window_view(padded, (kh, kw))
        result = np.sum(windows * kernel, axis=(2, 3))
        return result

    grad_x = convolve2d(gray, sobel_x)
    grad_y = convolve2d(gray, sobel_y)

    # Compute edge magnitude
    edge_magnitude = np.sqrt(grad_x**2 + grad_y**2)

    # Normalize to [0, 1] and invert (edges are white on black background)
    edge_magnitude = np.clip(edge_magnitude, 0, 1)
    edge_magnitude = 1.0 - edge_magnitude  # Invert: black edges on white background

    # Convert back to RGB (3 channels)
    edge_rgb = np.stack([edge_magnitude, edge_magnitude, edge_magnitude], axis=2)

    return Image.fromarray((edge_rgb * 255).astype(np.uint8))


def preprocess_image(image_path, size=512):
    """Load and preprocess image for inference"""
    img = Image.open(image_path).convert("RGB")

    # Resize maintaining aspect ratio with padding
    img.thumbnail((size, size), Image.Resampling.LANCZOS)
    new_img = Image.new("RGB", (size, size), (0, 0, 0))
    paste_x = (size - img.width) // 2
    paste_y = (size - img.height) // 2
    new_img.paste(img, (paste_x, paste_y))

    # Convert to edge-detected image (runtime preprocessing)
    new_img = detect_edges(new_img)

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


def modify_unet_for_img2img(unet):
    """Modify UNet to accept 8 channels (4 for original + 4 for noisy target)"""
    # Get original conv_in layer properties
    in_channels = 8  # 4 (original) + 4 (noisy target)
    out_channels = unet.conv_in.out_channels
    kernel_size = unet.conv_in.kernel_size
    stride = unet.conv_in.stride
    padding = unet.conv_in.padding

    # Update config
    unet.register_to_config(in_channels=in_channels)

    # Create new conv_in layer with 8 input channels
    with torch.no_grad():
        new_conv_in = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        # Initialize new channels to zero, copy original weights to first 4 channels
        new_conv_in.weight.zero_()
        new_conv_in.weight[:, :4, :, :].copy_(unet.conv_in.weight)
        if unet.conv_in.bias is not None:
            new_conv_in.bias = nn.Parameter(unet.conv_in.bias.clone())
        else:
            new_conv_in.bias = None

    # Replace the layer
    unet.conv_in = new_conv_in
    return unet


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
        gray = (
            0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
        ).astype(np.uint8)
    else:
        gray = arr

    # Less aggressive post-processing: preserve grayscale values
    # Only apply light thresholding to enhance contrast, not pure binarization
    # The targets have grayscale (42 unique values), so we should preserve that

    # Apply gentle contrast enhancement instead of hard thresholding
    # Normalize to ensure good contrast without losing grayscale detail
    gray_min, gray_max = gray.min(), gray.max()
    if gray_max > gray_min:
        gray_normalized = ((gray - gray_min) / (gray_max - gray_min) * 255).astype(
            np.uint8
        )
    else:
        gray_normalized = gray

    # Optional: light thresholding to clean up but preserve grayscale
    # Use a softer threshold (keep more grayscale values)
    threshold = otsu_threshold(gray_normalized)
    # Apply threshold but keep some grayscale - use a softer approach
    # Instead of pure binary, use threshold as a guide for enhancement
    enhanced = np.clip(
        (gray_normalized.astype(float) - threshold) * 2 + 128, 0, 255
    ).astype(np.uint8)

    # Invert if needed (ensure black lines on white background)
    if np.mean(enhanced) < 128:
        enhanced = 255 - enhanced

    return Image.fromarray(enhanced, mode="L").convert("RGB")


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
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(MODEL_ID, torch_dtype=dtype)
    pipe = pipe.to(device)

    # Modify UNet to accept 8 channels (same as training)
    print("Modifying UNet for img2img conditioning (8 channels)...")
    pipe.unet = modify_unet_for_img2img(pipe.unet)

    # Load LoRA weights using PEFT
    print("Loading LoRA weights...")
    import os

    lora_path = os.path.abspath(LORA_WEIGHTS_DIR)

    # Load PEFT adapter
    pipe.unet = PeftModel.from_pretrained(pipe.unet, lora_path, local_files_only=True)
    print("LoRA weights loaded successfully.")

    # Preprocess input image
    print(f"Processing {input_path}...")
    input_image = preprocess_image(input_path, size=IMAGE_SIZE)

    # Convert input image to tensor (normalize to [-1, 1])
    import numpy as np

    arr = np.array(input_image).astype(np.float32) / 255.0
    arr = arr * 2.0 - 1.0  # Normalize to [-1, 1]
    arr = arr.transpose(2, 0, 1)  # HWC to CHW
    input_tensor = torch.from_numpy(arr).unsqueeze(0).to(device).to(dtype)

    # Encode input image to latent space
    print("Encoding input image...")
    with torch.no_grad():
        input_latents = pipe.vae.encode(input_tensor).latent_dist.sample()
        input_latents = input_latents * pipe.vae.config.scaling_factor

    # Prepare scheduler
    pipe.scheduler.set_timesteps(NUM_INFERENCE_STEPS, device=device)
    timesteps = pipe.scheduler.timesteps

    # Calculate initial timestep based on strength
    init_timestep = int(NUM_INFERENCE_STEPS * STRENGTH)
    t_start = max(NUM_INFERENCE_STEPS - init_timestep, 0)
    timesteps = timesteps[t_start:]

    # Start from input latents with noise
    noise = torch.randn_like(input_latents)
    latents = pipe.scheduler.add_noise(input_latents, noise, timesteps[0:1])

    # Prepare text embeddings (empty prompt)
    print("Running inference...")
    prompt = ""
    text_inputs = pipe.tokenizer(
        prompt,
        padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = pipe.text_encoder(text_inputs.input_ids.to(device))[0]

    # Custom denoising loop with 8-channel conditioning
    for i, t in enumerate(timesteps):
        # Concatenate original input latents with current noisy latents
        conditioned_latents = torch.cat([input_latents, latents], dim=1)

        # Predict noise
        noise_pred = pipe.unet(
            conditioned_latents,
            t,
            encoder_hidden_states=text_embeddings,
        ).sample

        # Denoise step
        latents = pipe.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

        if (i + 1) % 10 == 0:
            print(f"  Step {i + 1}/{len(timesteps)}")

    # Decode latents to image
    print("Decoding result...")
    with torch.no_grad():
        latents = 1 / pipe.vae.config.scaling_factor * latents
        result = pipe.vae.decode(latents).sample
        result = (result / 2 + 0.5).clamp(0, 1)
        result = result.cpu().permute(0, 2, 3, 1).numpy()
        result = (result * 255).astype(np.uint8)
        result = Image.fromarray(result[0])

    # Post-process to clean black and white
    print("Post-processing to black and white...")
    bw_result = postprocess_to_bw(result)

    # Save output - create directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"Created output directory: {output_dir}")

    bw_result.save(output_path)
    print(f"Saved output to {output_path}")


if __name__ == "__main__":
    main()
