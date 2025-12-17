import os
import cv2
import numpy as np


def align_images(photo_dir, drawing_dir, output_dir, size=(512, 512)):
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

        # Resize to square (standardizing for the model)
        img_a = cv2.resize(img_a, size, interpolation=cv2.INTER_AREA)
        img_b = cv2.resize(img_b, size, interpolation=cv2.INTER_AREA)

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
