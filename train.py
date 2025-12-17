import os
import random
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from PIL import Image
import torchvision.transforms as transforms
from pytorch_msssim import ssim

# Import model and edge detection from separate module
from model import UNet, detect_edges


class ConditionalUNet(nn.Module):
    """
    Conditional UNet for diffusion: conditions on edge-detected input image.

    WHY DIFFUSION FOR LINE DRAWINGS:
    - Regression models (pixel-wise loss) struggle with sparse line drawings
    - Diffusion models excel at generating complete structures through iterative denoising
    - The denoising process naturally encourages shape closure and topology completion
    - Missing lines are penalized at every denoising step, not just final output

    CONDITIONING APPROACH:
    - Concatenate edge image (3 channels) with noisy target (1 channel) = 4 input channels
    - UNet learns to predict noise that, when removed, produces line drawing matching edge structure
    - This ensures generated lines follow the edge-detected geometry
    """

    def __init__(self, base_unet):
        super(ConditionalUNet, self).__init__()
        self.base_unet = base_unet

        # Modify first layer to accept 4 channels (3 edge + 1 noisy target)
        # Get original conv_in properties
        original_conv = base_unet.enc1[0]
        in_channels = 4  # 3 (edge) + 1 (noisy target)
        out_channels = original_conv.out_channels
        kernel_size = original_conv.kernel_size
        stride = original_conv.stride
        padding = original_conv.padding

        # Create new first conv layer
        new_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)

        # Initialize: copy weights for first 3 channels (edge), zero for 4th (noisy target)
        with torch.no_grad():
            new_conv.weight[:, :3, :, :].copy_(original_conv.weight[:, :3, :, :])
            new_conv.weight[:, 3:, :, :].zero_()
            if original_conv.bias is not None:
                new_conv.bias = nn.Parameter(original_conv.bias.clone())

        # Replace first conv in enc1
        base_unet.enc1[0] = new_conv

    def forward(self, noisy_target, edge_image, timestep):
        """
        Forward pass for conditional diffusion.

        Args:
            noisy_target: [B, 1, H, W] - noisy line drawing at timestep t
            edge_image: [B, 3, H, W] - edge-detected input image (conditioning)
            timestep: [B] - diffusion timestep (0 to NUM_DIFFUSION_TIMESTEPS-1)
                     Currently unused but kept for future timestep embedding

        Returns:
            predicted_noise: [B, 1, H, W] - predicted noise to remove
        """
        # Concatenate edge image with noisy target
        # Shape: [B, 4, H, W] = [B, 3 (edge) + 1 (noisy), H, W]
        conditioned_input = torch.cat([edge_image, noisy_target], dim=1)

        # For now, ignore timestep embedding (can add later if needed)
        # The UNet will learn to denoise based on the conditioning alone
        # The base_unet's sigmoid was already removed in create_conditional_unet
        predicted_noise = self.base_unet(conditioned_input)

        return predicted_noise


class DDPMScheduler:
    """
    Simple DDPM noise scheduler for forward diffusion.

    HOW DIFFUSION WORKS:
    1. Forward process: gradually add noise to clean line drawing
       x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon
    2. Training: model learns to predict epsilon (noise) given noisy x_t and conditioning
    3. Inference: iteratively denoise from pure noise, conditioned on edge image
       - Start with random noise
       - At each step, predict and remove noise
       - After T steps, recover clean line drawing

    This iterative process naturally encourages complete structures and shape closure.
    """

    def __init__(self, num_timesteps=1000, beta_start=0.0001, beta_end=0.02):
        self.num_timesteps = num_timesteps

        # Linear noise schedule
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)

        # Precompute for efficiency
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

    def add_noise(self, x_0, noise, timesteps):
        """
        Add noise to clean image x_0 at given timesteps.

        Args:
            x_0: [B, C, H, W] - clean image
            noise: [B, C, H, W] - noise to add
            timesteps: [B] - timestep indices

        Returns:
            x_t: [B, C, H, W] - noisy image
        """
        # Move to same device as inputs
        sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(x_0.device)
        sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(
            x_0.device
        )

        # Get alpha values for each sample in batch
        sqrt_alphas_cumprod_t = sqrt_alphas_cumprod[timesteps].view(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod[timesteps].view(
            -1, 1, 1, 1
        )

        # Forward diffusion: x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon
        x_t = sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise

        return x_t

    def sample_timesteps(self, batch_size, device):
        """Sample random timesteps for training."""
        return torch.randint(0, self.num_timesteps, (batch_size,), device=device)


# Hyperparameters
IMAGE_SIZE = 512
BATCH_SIZE = 2  # Reduced for diffusion (more memory intensive)
LEARNING_RATE = 1e-4
NUM_EPOCHS = 200
EARLY_STOPPING_PATIENCE = 10

# Diffusion hyperparameters
NUM_DIFFUSION_TIMESTEPS = 1000  # Standard DDPM timesteps
BETA_START = 0.0001  # Noise schedule start
BETA_END = 0.02  # Noise schedule end


class PairedImageDataset(Dataset):
    def __init__(self, originals_dir, targets_dir, size=512, is_training=True):
        self.size = size
        self.is_training = is_training
        self.originals_dir = Path(originals_dir)
        self.targets_dir = Path(targets_dir)

        # Find all matching pairs (handle .png and .jpg extensions)
        originals = {}
        for ext in [".png", ".jpg", ".jpeg"]:
            for f in self.originals_dir.glob(f"*{ext}"):
                base_name = f.stem
                originals[base_name] = f

        targets = {}
        for ext in [".png", ".jpg", ".jpeg"]:
            for f in self.targets_dir.glob(f"*{ext}"):
                base_name = f.stem
                targets[base_name] = f

        # Find matching pairs
        self.pairs = []
        for name in sorted(originals.keys()):
            if name in targets:
                self.pairs.append((originals[name], targets[name]))

        print(f"Found {len(self.pairs)} paired images")

        # Augmentations for training
        if is_training:
            self.augment = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomRotation(degrees=5),
                    transforms.ColorJitter(brightness=0.1, contrast=0.1),
                ]
            )
        else:
            self.augment = None

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        original_path, target_path = self.pairs[idx]

        # Load images
        original = Image.open(original_path).convert("RGB")
        target = Image.open(target_path).convert("L")  # Grayscale

        # Resize maintaining aspect ratio with padding
        original = self._resize_with_padding(original)
        target = self._resize_with_padding(target)

        # Apply edge detection to original (runtime preprocessing)
        original = detect_edges(original)

        # Apply augmentations to edge-detected input (training only)
        if self.is_training and self.augment:
            original = self.augment(original)

        # Convert to tensors and normalize
        original_tensor = self._to_tensor(original)  # [3, 512, 512], [0, 1]
        target_tensor = self._to_tensor(target)  # [1, 512, 512], [0, 1]

        # Binarize target to make it more binary-like for line drawings
        # This ensures the model learns crisp lines, not blurry grayscale
        target_tensor = self._binarize_target(target_tensor, threshold=0.5)

        return original_tensor, target_tensor

    def _resize_with_padding(self, img):
        """Resize image to self.size x self.size maintaining aspect ratio"""
        img.thumbnail((self.size, self.size), Image.Resampling.LANCZOS)
        new_img = Image.new(
            "RGB" if img.mode == "RGB" else "L",
            (self.size, self.size),
            (0, 0, 0) if img.mode == "RGB" else 0,
        )
        paste_x = (self.size - img.width) // 2
        paste_y = (self.size - img.height) // 2
        new_img.paste(img, (paste_x, paste_y))
        return new_img

    def _to_tensor(self, img):
        """Convert PIL image to tensor normalized to [0, 1]"""
        arr = np.array(img).astype(np.float32) / 255.0
        if len(arr.shape) == 2:  # Grayscale
            arr = arr[np.newaxis, :, :]  # Add channel dimension
        else:
            arr = arr.transpose(2, 0, 1)  # HWC to CHW
        return torch.from_numpy(arr)

    def _binarize_target(self, target_tensor, threshold=0.5):
        """
        Binarize target to make it more binary-like for line drawings.
        Line drawings should be treated as binary masks (lines vs background),
        not continuous grayscale. This helps the model learn crisp boundaries.
        """
        # Threshold: values above threshold become 1, below become 0
        # Use a relatively low threshold (0.5) to preserve most line pixels
        # but still create a clear binary distinction
        return (target_tensor > threshold).float()


def create_conditional_unet(base_unet):
    """
    Create conditional UNet that accepts edge image as conditioning.
    Modifies the base UNet to accept 4 channels (3 edge + 1 noisy target).
    """
    # Create a copy of base UNet structure but modify first layer
    # We'll modify it in-place to accept 4 channels
    conditional_unet = ConditionalUNet(base_unet)
    return conditional_unet


def compute_edge_loss(pred, target):
    """
    Edge-aware loss: penalize missing edges more than extra pixels.
    Uses Sobel filters to detect edges in both pred and target,
    then computes loss on the edge maps. This ensures missing structural
    lines (which are edges) are heavily penalized.
    """
    # Sobel kernels for edge detection
    sobel_x = torch.tensor(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=pred.dtype, device=pred.device
    ).view(1, 1, 3, 3)
    sobel_y = torch.tensor(
        [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=pred.dtype, device=pred.device
    ).view(1, 1, 3, 3)

    # Compute edge maps for pred and target
    def get_edges(img):
        # img: [B, C, H, W]
        grad_x = F.conv2d(img, sobel_x, padding=1)
        grad_y = F.conv2d(img, sobel_y, padding=1)
        edge_mag = torch.sqrt(grad_x**2 + grad_y**2 + 1e-6)
        return edge_mag

    pred_edges = get_edges(pred)
    target_edges = get_edges(target)

    # L1 loss on edge maps - missing edges are heavily penalized
    edge_loss = F.l1_loss(pred_edges, target_edges)
    return edge_loss


def compute_line_presence_loss(pred, target, threshold=0.5):
    """
    Line Presence / Coverage Loss: explicitly penalizes missing line pixels.

    WHY THIS EXISTS:
    - Pixel loss treats all pixels equally, so missing sparse line pixels
      has minimal impact on average error
    - This loss converts both pred and target to binary masks and directly
      penalizes missing predicted line pixels relative to target
    - Strongly punishes missing strokes, forcing the model to produce complete lines

    Implementation:
    - Binarize both pred and target using fixed threshold
    - Compute L1 loss on binary masks
    - This gives equal weight to each line pixel, regardless of background density
    """
    # Convert to binary line masks
    pred_binary = (pred > threshold).float()
    target_binary = (target > threshold).float()

    # L1 loss on binary masks - missing line pixels are directly penalized
    line_presence_loss = F.l1_loss(pred_binary, target_binary)

    return line_presence_loss


def compute_thickness_consistency_loss(pred, target, blur_kernel_size=3):
    """
    Thickness Consistency Loss: encourages stable line thickness.

    WHY THIS EXISTS:
    - Ultra-thin, broken, or noisy lines are visually poor even if pixel-accurate
    - Slight blurring smooths out thickness variations
    - Loss on blurred maps discourages the model from producing fragile line structures

    Implementation:
    - Apply box blur (simple averaging) to both pred and target
    - Compute L1 loss on blurred maps
    - This encourages consistent line thickness and reduces noise
    """
    # Create box blur kernel (simple averaging filter)
    # This approximates Gaussian blur and is faster
    blur_kernel = torch.ones(
        1, 1, blur_kernel_size, blur_kernel_size, dtype=pred.dtype, device=pred.device
    )
    blur_kernel = blur_kernel / (blur_kernel_size * blur_kernel_size)

    # Apply blur (convolution with padding to maintain size)
    def blur_image(img):
        # img: [B, C, H, W]
        # Apply blur to each channel separately
        blurred_channels = []
        for c in range(img.shape[1]):
            channel = img[:, c : c + 1, :, :]
            blurred = F.conv2d(channel, blur_kernel, padding=blur_kernel_size // 2)
            blurred_channels.append(blurred)
        blurred = torch.cat(blurred_channels, dim=1)
        return blurred

    pred_blurred = blur_image(pred)
    target_blurred = blur_image(target)

    # L1 loss on blurred maps
    thickness_loss = F.l1_loss(pred_blurred, target_blurred)

    return thickness_loss


def save_validation_samples(samples, epoch, device):
    """
    Save validation samples for visual debugging.
    Creates side-by-side comparison: edge input | ground truth | model output
    """
    edge_images, targets, outputs = samples

    # Create output directory
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    # Process first sample in batch
    edge_img = edge_images[0]
    target_img = targets[0]
    output_img = outputs[0]

    # Convert to numpy and denormalize
    def tensor_to_image(tensor):
        # tensor: [C, H, W] in [0, 1]
        arr = tensor.cpu().numpy()
        if arr.shape[0] == 3:  # RGB
            arr = arr.transpose(1, 2, 0)
        else:  # Grayscale
            arr = arr.squeeze(0)
        arr = np.clip(arr * 255, 0, 255).astype(np.uint8)
        if len(arr.shape) == 2:
            return Image.fromarray(arr, mode="L")
        else:
            return Image.fromarray(arr, mode="RGB")

    edge_pil = tensor_to_image(edge_img)
    target_pil = tensor_to_image(target_img)
    output_pil = tensor_to_image(output_img)

    # Create side-by-side comparison
    # Width: 3 images * 512, Height: 512
    comparison = Image.new("RGB", (512 * 3, 512), (255, 255, 255))
    comparison.paste(edge_pil.convert("RGB"), (0, 0))
    comparison.paste(target_pil.convert("RGB"), (512, 0))
    comparison.paste(output_pil.convert("RGB"), (1024, 0))

    # Save
    output_path = output_dir / f"epoch_{epoch:03d}_comparison.png"
    comparison.save(output_path)
    print(f"  -> Saved validation sample to {output_path}")


def combined_loss(pred, target, return_components=False):
    """
    Structure-aware loss function that heavily penalizes missing lines.

    WHY PIXEL LOSS ALONE IS INSUFFICIENT:
    - Line drawings are SPARSE: ~1-5% of pixels are lines, 95-99% are white background
    - Pixel-wise losses (L1/MSE) compute average error across ALL pixels
    - Missing a line pixel contributes minimally to average (1 pixel out of 512*512)
    - Model can "cheat" by predicting mostly white with faint lines, achieving low pixel loss
    - Result: model learns to minimize average error, not maximize line presence

    HOW NEW LOSSES ADDRESS THIS:
    1. Pixel Loss (0.2): Downweighted stabilizer - prevents extreme outputs but not main driver
    2. Edge Loss (0.4): Computes loss on edge maps - missing structural lines (edges) heavily penalized
    3. Line Presence Loss (0.3): Binary mask loss - directly penalizes missing line pixels
    4. Thickness Loss (0.1): Blurred map loss - encourages stable, consistent line thickness

    The combination ensures missing lines cause large gradient signals, forcing complete output.
    """
    # Component 1: Pixel loss (downweighted - stabilizer only)
    pixel_loss = F.l1_loss(pred, target)

    # Component 2: Edge-aware loss (high weight - correct geometry/contour placement)
    edge_loss = compute_edge_loss(pred, target)

    # Component 3: Line presence loss (high weight - strongly punishes missing strokes)
    line_presence_loss = compute_line_presence_loss(pred, target)

    # Component 4: Thickness consistency loss (low weight - discourages ultra-thin/broken lines)
    thickness_loss = compute_thickness_consistency_loss(pred, target)

    # Combined loss with specified weights
    total_loss = (
        LOSS_WEIGHT_PIXEL * pixel_loss
        + LOSS_WEIGHT_EDGE * edge_loss
        + LOSS_WEIGHT_LINE_PRESENCE * line_presence_loss
        + LOSS_WEIGHT_THICKNESS * thickness_loss
    )

    if return_components:
        return total_loss, {
            "pixel": pixel_loss.item(),
            "edge": edge_loss.item(),
            "line_presence": line_presence_loss.item(),
            "thickness": thickness_loss.item(),
        }
    return total_loss


def compute_line_coverage(pred, target, threshold=0.5):
    """
    Compute line coverage metric: what % of target line pixels are present in pred.
    This metric directly measures if the model is producing complete lines.
    Returns value in [0, 1] where 1.0 = perfect coverage.
    """
    # Binarize both
    pred_binary = (pred > threshold).float()
    target_binary = (target > threshold).float()

    # Count line pixels in target
    target_lines = target_binary.sum()

    if target_lines == 0:
        return 1.0  # No lines to cover

    # Count overlapping line pixels
    overlap = (pred_binary * target_binary).sum()

    # Coverage = overlap / target_lines
    coverage = overlap / target_lines

    return coverage.item()


def main():
    parser = argparse.ArgumentParser(
        description="Train image-to-image model for line drawings"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test mode: train on only 10 images to check for overfitting",
    )
    parser.add_argument(
        "--no-stop",
        action="store_true",
        help="Disable early stopping - keep training and saving best models",
    )
    args = parser.parse_args()

    # Set random seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load dataset
    full_dataset = PairedImageDataset(
        "dataset/originals", "dataset/line_drawings", size=IMAGE_SIZE, is_training=True
    )

    # Split dataset
    if args.test:
        print("TEST MODE: Training on 10 images to check for overfitting")
        total_size = min(10, len(full_dataset))
        val_size = 2
        train_size = total_size - val_size

        train_indices = list(range(train_size))
        val_indices = list(range(train_size, total_size))

        train_dataset = Subset(full_dataset, train_indices)
        val_dataset = Subset(full_dataset, val_indices)

        print(
            f"TEST MODE - Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}"
        )
        print(
            "  (If model overfits, train loss should drop to < 0.01 while val loss stays reasonable)"
        )

        # Log the training images
        print("\nTraining images (saving to test_training_images/):")
        test_output_dir = Path("test_training_images")
        test_output_dir.mkdir(exist_ok=True)

        for i, idx in enumerate(train_indices):
            original_path, target_path = full_dataset.pairs[idx]
            print(f"  [{i + 1}] {original_path.name}")

            # Load and save original
            original = Image.open(original_path).convert("RGB")
            original_resized = full_dataset._resize_with_padding(original)
            original_resized.save(
                test_output_dir / f"train_{i + 1:02d}_original_{original_path.name}"
            )

            # Apply edge detection and save
            edge_img = detect_edges(original_resized)
            edge_img.save(
                test_output_dir / f"train_{i + 1:02d}_edge_{original_path.name}"
            )

            # Save target
            target = Image.open(target_path).convert("L")
            target_resized = full_dataset._resize_with_padding(target)
            target_resized.save(
                test_output_dir / f"train_{i + 1:02d}_target_{target_path.name}"
            )

        print(f"\nSaved training images to {test_output_dir}/")
        print(
            "  (original_* = input photos, edge_* = edge-detected inputs, target_* = line drawing targets)"
        )
    else:
        total_size = len(full_dataset)
        val_size = min(10, total_size)
        train_size = total_size - val_size

        train_indices = list(range(train_size))
        val_indices = list(range(train_size, total_size))

        train_dataset = Subset(full_dataset, train_indices)
        val_dataset = Subset(full_dataset, val_indices)

        print(
            f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}"
        )

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2
    )

    # Initialize base UNet (will be wrapped for conditioning)
    # For diffusion, we need raw output (no sigmoid) to predict noise
    base_unet = UNet(in_channels=3, out_channels=1)

    # Remove sigmoid from final layer for noise prediction
    # Noise can be negative, so we need unbounded output
    base_unet.sigmoid = nn.Identity()  # Replace sigmoid with identity

    # Wrap in conditional UNet that accepts edge image as conditioning
    model = create_conditional_unet(base_unet).to(device)

    print("Using conditional diffusion model (DDPM)")
    print(f"  - Conditioning: edge-detected input image (3 channels)")
    print(f"  - Target: line drawing (1 channel)")
    print(f"  - Model predicts noise to remove from noisy target")

    # Initialize weights
    def init_weights(m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    model.apply(init_weights)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")

    # Diffusion scheduler
    diffusion_scheduler = DDPMScheduler(
        num_timesteps=NUM_DIFFUSION_TIMESTEPS, beta_start=BETA_START, beta_end=BETA_END
    )

    # Loss and optimizer
    # For diffusion, we use simple MSE loss on predicted noise
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.999), weight_decay=1e-5
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    # Training loop
    best_val_loss = float("inf")
    best_composite_metric = float(
        "inf"
    )  # For model selection based on structural quality
    patience_counter = 0

    print(f"\nStarting training for {NUM_EPOCHS} epochs...")
    print(f"Batch size: {BATCH_SIZE}, Learning rate: {LEARNING_RATE}")

    for epoch in range(NUM_EPOCHS):
        # Training phase
        model.train()
        train_loss = 0.0
        num_train_batches = 0

        for batch_idx, (edge_images, targets) in enumerate(train_loader):
            # inputs are edge-detected images (3 channels), targets are line drawings (1 channel)
            edge_images = edge_images.to(
                device
            )  # [B, 3, H, W] - edge-detected conditioning
            targets = targets.to(device)  # [B, 1, H, W] - clean line drawings

            # DIFFUSION TRAINING:
            # 1. Sample random timesteps
            timesteps = diffusion_scheduler.sample_timesteps(
                batch_size=edge_images.shape[0], device=device
            )

            # 2. Sample noise to add
            noise = torch.randn_like(targets)  # [B, 1, H, W]

            # 3. Add noise to clean targets (forward diffusion)
            noisy_targets = diffusion_scheduler.add_noise(targets, noise, timesteps)

            # 4. Predict noise conditioned on edge image
            optimizer.zero_grad()
            predicted_noise = model(noisy_targets, edge_images, timesteps)

            # 5. MSE loss on predicted noise (standard DDPM loss)
            loss = F.mse_loss(predicted_noise, noise)

            # Backward pass
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            num_train_batches += 1

            if (batch_idx + 1) % 10 == 0:
                print(
                    f"Epoch {epoch + 1}/{NUM_EPOCHS}, Batch {batch_idx + 1}/{len(train_loader)}, Loss: {loss.item():.6f}"
                )

        avg_train_loss = (
            train_loss / num_train_batches if num_train_batches > 0 else 0.0
        )

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_coverage = 0.0
        num_val_batches = 0
        val_samples_to_save = []  # Store samples for visual debugging

        with torch.no_grad():
            for batch_idx, (edge_images, targets) in enumerate(val_loader):
                edge_images = edge_images.to(device)  # [B, 3, H, W]
                targets = targets.to(device)  # [B, 1, H, W]

                # Validation: sample timesteps and compute diffusion loss
                timesteps = diffusion_scheduler.sample_timesteps(
                    batch_size=edge_images.shape[0], device=device
                )
                noise = torch.randn_like(targets)
                noisy_targets = diffusion_scheduler.add_noise(targets, noise, timesteps)

                predicted_noise = model(noisy_targets, edge_images, timesteps)
                loss = F.mse_loss(predicted_noise, noise)

                val_loss += loss.item()

                # For coverage metric, we need to denoise a sample
                # Use a simple single-step denoising approximation for validation
                # (Full denoising would require running full diffusion process)
                # Approximate: remove predicted noise from noisy target
                with torch.no_grad():
                    # Simple approximation: denoise using predicted noise
                    # This is not the full diffusion process, just for metric
                    denoised_approx = noisy_targets - predicted_noise
                    denoised_approx = torch.clamp(denoised_approx, 0, 1)

                    batch_coverage = 0.0
                    for i in range(denoised_approx.shape[0]):
                        batch_coverage += compute_line_coverage(
                            denoised_approx[i : i + 1], targets[i : i + 1]
                        )
                    val_coverage += batch_coverage / denoised_approx.shape[0]

                num_val_batches += 1

                # Save first batch of validation samples for visual debugging
                if batch_idx == 0 and epoch % 5 == 0:  # Every 5 epochs
                    val_samples_to_save = (
                        edge_images.cpu(),
                        targets.cpu(),
                        denoised_approx.cpu(),
                    )

        avg_val_loss = val_loss / num_val_batches if num_val_batches > 0 else 0.0
        avg_val_coverage = (
            val_coverage / num_val_batches if num_val_batches > 0 else 0.0
        )

        # For diffusion, use validation loss as the metric (MSE on noise prediction)
        composite_metric = avg_val_loss

        # Visual debug output: save side-by-side comparison
        if val_samples_to_save and epoch % 5 == 0:
            save_validation_samples(val_samples_to_save, epoch, device)

        # Learning rate scheduling
        old_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(avg_val_loss)
        new_lr = optimizer.param_groups[0]["lr"]
        if new_lr < old_lr:
            print(f"  -> Learning rate reduced: {old_lr:.2e} -> {new_lr:.2e}")

        # Print epoch summary
        print(
            f"Epoch {epoch + 1}/{NUM_EPOCHS} - Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}"
        )
        print(
            f"  Val Coverage: {avg_val_coverage:.3f} (MSE on noise prediction: {avg_val_loss:.6f})"
        )

        # Explain loss values
        if epoch == 0:
            print("  NOTE: Diffusion model trains on noise prediction (MSE loss).")
            print(
                "  Lower loss = better noise prediction = better line reconstruction."
            )
            print(
                "  Diffusion naturally encourages complete structures through iterative denoising."
            )
            print("  Watch 'Val Coverage' (target: >0.7) to track line completeness.")

        # Save best model based on composite metric (edge + line_presence)
        # This ensures we select models with better structural output, not just low pixel loss
        if composite_metric < best_composite_metric:
            best_composite_metric = composite_metric
            best_val_loss = avg_val_loss  # Track for logging
            patience_counter = 0
            torch.save(model.state_dict(), "best_model.pth")
            print(
                f"  -> Saved best model (composite: {composite_metric:.6f}, val loss: {avg_val_loss:.6f})"
            )
        else:
            patience_counter += 1

        # Early stopping (unless disabled)
        if not args.no_stop and patience_counter >= EARLY_STOPPING_PATIENCE:
            print(f"\nEarly stopping triggered after {epoch + 1} epochs")
            print(
                f"Best composite metric: {best_composite_metric:.6f} (val loss: {best_val_loss:.6f})"
            )
            break

    # Save final model
    torch.save(model.state_dict(), "final_model.pth")
    print(
        f"\nTraining complete! Final composite metric: {composite_metric:.6f}, Val loss: {avg_val_loss:.6f}"
    )
    print(
        f"Best model (composite: {best_composite_metric:.6f}) saved to: best_model.pth"
    )
    print(f"Final model saved to: final_model.pth")


if __name__ == "__main__":
    main()
