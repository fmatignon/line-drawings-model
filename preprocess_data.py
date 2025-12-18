import os
import cv2
import numpy as np


def resize_with_padding(img, size=512, interpolation=cv2.INTER_AREA):
    """
    Resize image to size x size maintaining aspect ratio with padding.
    Matches the preprocessing used in training.
    
    Args:
        img: Input image (numpy array)
        size: Target size (width and height)
        interpolation: OpenCV interpolation method (INTER_AREA for photos, INTER_NEAREST for line drawings)
    """
    h, w = img.shape[:2]

    # Calculate scaling factor to fit within size x size
    scale = min(size / w, size / h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    # Resize maintaining aspect ratio
    resized = cv2.resize(img, (new_w, new_h), interpolation=interpolation)

    # Create square canvas with black padding
    if len(img.shape) == 3:
        canvas = np.zeros((size, size, 3), dtype=img.dtype)
    else:
        canvas = np.zeros((size, size), dtype=img.dtype)

    # Center the resized image
    paste_x = (size - new_w) // 2
    paste_y = (size - new_h) // 2

    if len(img.shape) == 3:
        canvas[paste_y : paste_y + new_h, paste_x : paste_x + new_w] = resized
    else:
        canvas[paste_y : paste_y + new_h, paste_x : paste_x + new_w] = resized

    return canvas


def align_images(photo_dir, drawing_dir, output_dir, size=512):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Get filenames from photo directory
    photos = sorted(
        [
            f
            for f in os.listdir(photo_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
    )

    count = 0
    for filename in photos:
        photo_path = os.path.join(photo_dir, filename)
        drawing_path = os.path.join(drawing_dir, filename)

        if not os.path.exists(drawing_path):
            print(f"Skipping {filename}: No matching line drawing found.")
            continue

        # Read images
        img_a = cv2.imread(photo_path)
        img_b = cv2.imread(drawing_path)

        if img_a is None or img_b is None:
            print(f"Skipping {filename}: Failed to read image(s).")
            continue

        # Resize maintaining aspect ratio with padding (matches training preprocessing)
        # Use INTER_AREA for photos (A), INTER_NEAREST for line drawings (B) to avoid anti-aliasing
        img_a = resize_with_padding(img_a, size, interpolation=cv2.INTER_AREA)
        img_b = resize_with_padding(img_b, size, interpolation=cv2.INTER_NEAREST)

        # Combine side-by-side (A is Photo, B is Drawing)
        combined = np.concatenate([img_a, img_b], axis=1)

        # Save to output folder
        cv2.imwrite(os.path.join(output_dir, filename), combined)
        count += 1

    print(f"Done! Processed {count} pairs into {output_dir}")


if __name__ == "__main__":
    align_images(
        photo_dir="dataset/originals",
        drawing_dir="dataset/line_drawings",
        output_dir="dataset/aligned",
    )
