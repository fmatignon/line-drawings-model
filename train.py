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

# Hyperparameters
IMAGE_SIZE = 512
BATCH_SIZE = 4
LEARNING_RATE = 1e-4
NUM_EPOCHS = 200
EARLY_STOPPING_PATIENCE = 10


class UNet(nn.Module):
    """U-Net encoder-decoder architecture for image-to-image translation"""

    def __init__(self, in_channels=3, out_channels=1):
        super(UNet, self).__init__()

        # Encoder (downsampling path)
        self.enc1 = self._conv_block(in_channels, 64)
        self.enc2 = self._conv_block(64, 128)
        self.enc3 = self._conv_block(128, 256)
        self.enc4 = self._conv_block(256, 512)

        # Bottleneck
        self.bottleneck = self._conv_block(512, 512)

        # Decoder (upsampling path) with skip connections
        self.dec4 = self._conv_block(
            512 + 512, 256
        )  # 512 from skip + 512 from bottleneck
        self.dec3 = self._conv_block(256 + 256, 128)  # 256 from skip + 256 from dec4
        self.dec2 = self._conv_block(128 + 128, 64)  # 128 from skip + 128 from dec3
        self.dec1 = self._conv_block(64 + 64, 64)  # 64 from skip + 64 from dec2

        # Final output layer
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

        # Pooling and upsampling
        self.pool = nn.MaxPool2d(2, 2)
        self.upsample4 = nn.ConvTranspose2d(512, 512, 2, stride=2)
        self.upsample3 = nn.ConvTranspose2d(256, 256, 2, stride=2)
        self.upsample2 = nn.ConvTranspose2d(128, 128, 2, stride=2)
        self.upsample1 = nn.ConvTranspose2d(64, 64, 2, stride=2)

    def _conv_block(self, in_channels, out_channels):
        """Convolutional block: Conv2d -> BatchNorm -> ReLU"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # Encoder path
        enc1 = self.enc1(x)  # 512x512 -> 512x512
        enc1_pooled = self.pool(enc1)  # 512x512 -> 256x256

        enc2 = self.enc2(enc1_pooled)  # 256x256 -> 256x256
        enc2_pooled = self.pool(enc2)  # 256x256 -> 128x128

        enc3 = self.enc3(enc2_pooled)  # 128x128 -> 128x128
        enc3_pooled = self.pool(enc3)  # 128x128 -> 64x64

        enc4 = self.enc4(enc3_pooled)  # 64x64 -> 64x64
        enc4_pooled = self.pool(enc4)  # 64x64 -> 32x32

        # Bottleneck
        bottleneck = self.bottleneck(enc4_pooled)  # 32x32 -> 32x32

        # Decoder path with skip connections
        dec4 = self.upsample4(bottleneck)  # 32x32 -> 64x64
        dec4 = torch.cat([dec4, enc4], dim=1)  # Concatenate skip connection
        dec4 = self.dec4(dec4)  # 64x64 -> 64x64

        dec3 = self.upsample3(dec4)  # 64x64 -> 128x128
        dec3 = torch.cat([dec3, enc3], dim=1)  # Concatenate skip connection
        dec3 = self.dec3(dec3)  # 128x128 -> 128x128

        dec2 = self.upsample2(dec3)  # 128x128 -> 256x256
        dec2 = torch.cat([dec2, enc2], dim=1)  # Concatenate skip connection
        dec2 = self.dec2(dec2)  # 256x256 -> 256x256

        dec1 = self.upsample1(dec2)  # 256x256 -> 512x512
        dec1 = torch.cat([dec1, enc1], dim=1)  # Concatenate skip connection
        dec1 = self.dec1(dec1)  # 512x512 -> 512x512

        # Final output
        output = self.final_conv(dec1)
        output = self.sigmoid(output)

        return output


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


def combined_loss(pred, target):
    """Combined L1 + SSIM loss"""
    l1_loss = F.l1_loss(pred, target)

    # SSIM expects [B, C, H, W] format, values in [0, 1]
    # pytorch_msssim returns value in [-1, 1], we need to convert
    ssim_value = ssim(pred, target, data_range=1.0)
    ssim_loss = 1.0 - ssim_value  # Convert to loss (lower is better)

    return 0.5 * l1_loss + 0.5 * ssim_loss


def main():
    parser = argparse.ArgumentParser(
        description="Train image-to-image model for line drawings"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test mode: train on only 10 images to check for overfitting",
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
        num_val_batches = 0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)

                outputs = model(inputs)
                loss = combined_loss(outputs, targets)

                val_loss += loss.item()
                num_val_batches += 1

        avg_val_loss = val_loss / num_val_batches if num_val_batches > 0 else 0.0

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

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "best_model.pth")
            print(f"  -> Saved best model (val loss: {avg_val_loss:.6f})")
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= EARLY_STOPPING_PATIENCE:
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
