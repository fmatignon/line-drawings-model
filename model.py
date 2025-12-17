import numpy as np
import torch
import torch.nn as nn
from PIL import Image


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

