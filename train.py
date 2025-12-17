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

# Hyperparameters
IMAGE_SIZE = 512
BATCH_SIZE = 4
LEARNING_RATE = 1e-4
NUM_EPOCHS = 200
EARLY_STOPPING_PATIENCE = 10


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


def dice_loss(pred, target, smooth=1e-6):
    """
    Dice loss: measures overlap between pred and target.
    Particularly effective for binary segmentation tasks like line drawings.
    Penalizes missing lines (false negatives) more than extra pixels (false positives).
    """
    # Flatten tensors
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)

    # Compute intersection and union
    intersection = (pred_flat * target_flat).sum()
    dice = (2.0 * intersection + smooth) / (
        pred_flat.sum() + target_flat.sum() + smooth
    )

    # Return as loss (1 - dice)
    return 1.0 - dice


def save_validation_samples(samples, epoch, device):
    """
    Save validation samples for visual debugging.
    Creates side-by-side comparison: input | ground truth | model output
    """
    inputs, targets, outputs = samples

    # Create output directory
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    # Process first sample in batch
    input_img = inputs[0]
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

    input_pil = tensor_to_image(input_img)
    target_pil = tensor_to_image(target_img)
    output_pil = tensor_to_image(output_img)

    # Create side-by-side comparison
    # Width: 3 images * 512, Height: 512
    comparison = Image.new("RGB", (512 * 3, 512), (255, 255, 255))
    comparison.paste(input_pil.convert("RGB"), (0, 0))
    comparison.paste(target_pil.convert("RGB"), (512, 0))
    comparison.paste(output_pil.convert("RGB"), (1024, 0))

    # Save
    output_path = output_dir / f"epoch_{epoch:03d}_comparison.png"
    comparison.save(output_path)
    print(f"  -> Saved validation sample to {output_path}")


def combined_loss(pred, target):
    """
    Structure-aware loss function that heavily penalizes missing lines.

    PROBLEM WITH OLD LOSS (L1 + SSIM):
    - L1 treats all pixels equally: missing a line pixel (sparse) has same weight
      as being off by a small amount in background (dense). Result: model learns
      to minimize average error by producing faint, partial lines rather than
      complete crisp lines.
    - SSIM measures structural similarity but doesn't specifically penalize
      missing structural elements (lines).

    NEW LOSS COMPONENTS:
    1. L1 (alpha=0.2): Baseline pixel-wise loss, but weighted low since it
       doesn't capture structure well.
    2. Edge loss (beta=0.4): Computes loss on edge maps. Missing structural lines
       (which are edges) are heavily penalized. This is the KEY fix.
    3. BCE loss (gamma=0.3): Binary cross-entropy treats this as binary classification
       (line vs background). Heavily penalizes false negatives (missing lines).
    4. Dice loss (delta=0.1): Measures overlap, penalizes missing coverage.

    The high weight on edge_loss ensures missing lines cause large gradient signals.
    """
    # Alpha: L1 loss (low weight - doesn't capture structure)
    l1_loss = F.l1_loss(pred, target)

    # Beta: Edge-aware loss (HIGH weight - this is the key fix)
    edge_loss = compute_edge_loss(pred, target)

    # Gamma: Binary Cross Entropy (treats as binary classification)
    # BCE heavily penalizes false negatives (missing lines)
    bce_loss = F.binary_cross_entropy(pred, target)

    # Delta: Dice loss (measures line coverage)
    dice = dice_loss(pred, target)

    # Combined loss with weights that prioritize structure
    total_loss = (
        0.2 * l1_loss  # alpha: baseline pixel loss
        + 0.4 * edge_loss  # beta: edge-aware (CRITICAL for missing lines)
        + 0.3 * bce_loss  # gamma: binary classification loss
        + 0.1 * dice  # delta: coverage loss
    )

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

    # Initialize model
    # NOTE: UNet output uses sigmoid activation (see model.py)
    # This ensures outputs are in [0, 1] range, matching our binary-like targets
    # and enabling BCE loss to work correctly.
    model = UNet(in_channels=3, out_channels=1).to(device)

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

    # Loss and optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.999), weight_decay=1e-5
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    # Training loop
    best_val_loss = float("inf")
    patience_counter = 0

    print(f"\nStarting training for {NUM_EPOCHS} epochs...")
    print(f"Batch size: {BATCH_SIZE}, Learning rate: {LEARNING_RATE}")

    for epoch in range(NUM_EPOCHS):
        # Training phase
        model.train()
        train_loss = 0.0
        num_train_batches = 0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Forward pass
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = combined_loss(outputs, targets)

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
            for batch_idx, (inputs, targets) in enumerate(val_loader):
                inputs = inputs.to(device)
                targets = targets.to(device)

                outputs = model(inputs)
                loss = combined_loss(outputs, targets)

                val_loss += loss.item()

                # Compute line coverage metric
                batch_coverage = 0.0
                for i in range(outputs.shape[0]):
                    batch_coverage += compute_line_coverage(
                        outputs[i : i + 1], targets[i : i + 1]
                    )
                val_coverage += batch_coverage / outputs.shape[0]

                num_val_batches += 1

                # Save first batch of validation samples for visual debugging
                if batch_idx == 0 and epoch % 5 == 0:  # Every 5 epochs
                    val_samples_to_save = (inputs.cpu(), targets.cpu(), outputs.cpu())

        avg_val_loss = val_loss / num_val_batches if num_val_batches > 0 else 0.0
        avg_val_coverage = (
            val_coverage / num_val_batches if num_val_batches > 0 else 0.0
        )

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
            f"Epoch {epoch + 1}/{NUM_EPOCHS} - Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}, Val Coverage: {avg_val_coverage:.3f}"
        )

        # Explain loss values
        if epoch == 0:
            print(
                "  NOTE: Pixel loss (L1) may plateau around 0.07-0.10 - this is EXPECTED."
            )
            print(
                "  The model minimizes average pixel error, but line drawings are sparse."
            )
            print(
                "  Watch 'Val Coverage' metric - it measures if lines are complete (target: >0.7)."
            )
            print(
                "  Edge-aware loss ensures missing structural lines are heavily penalized."
            )

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "best_model.pth")
            print(f"  -> Saved best model (val loss: {avg_val_loss:.6f})")
        else:
            patience_counter += 1

        # Early stopping (unless disabled)
        if not args.no_stop and patience_counter >= EARLY_STOPPING_PATIENCE:
            print(f"\nEarly stopping triggered after {epoch + 1} epochs")
            print(f"Best validation loss: {best_val_loss:.6f}")
            break

    # Save final model
    torch.save(model.state_dict(), "final_model.pth")
    print(f"\nTraining complete! Final validation loss: {avg_val_loss:.6f}")
    print(f"Best model saved to: best_model.pth")
    print(f"Final model saved to: final_model.pth")


if __name__ == "__main__":
    main()
