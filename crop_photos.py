import os
import sys
import numpy as np
from PIL import Image
import argparse

def crop_image(image_path, bg_color=(99, 100, 102), tolerance=5):
    try:
        with Image.open(image_path) as img:
            if img.mode != 'RGB':
                img = img.convert('RGB')

            data = np.array(img)
            height, width, _ = data.shape

            # Banner is on the left, fixed size relative to resolution
            # Based on 399/1920 ratio
            banner_width = int(width * (399 / 1920))

            # We look for the photo to the right of the banner
            search_area = data[:, banner_width:]

            # Detect foreground (pixels that are not the background color within tolerance)
            # We use max absolute difference across channels
            diff = np.abs(search_area.astype(np.int16) - np.array(bg_color, dtype=np.int16))
            mask = np.any(diff > tolerance, axis=-1)

            if not np.any(mask):
                print(f"Warning: No foreground found in {image_path} after banner area. Skipping.")
                return False

            # Find bounding box of foreground in the search area
            coords = np.argwhere(mask)
            y0, x0 = coords.min(axis=0)
            y1, x1 = coords.max(axis=0) + 1

            # x0, x1 are relative to search_area, so add banner_width
            x0 += banner_width
            x1 += banner_width

            # Crop the original image
            cropped_img = img.crop((x0, y0, x1, y1))

            # Save back to the same path
            cropped_img.save(image_path, quality=95)
            print(f"Successfully cropped {image_path} to {cropped_img.size[0]}x{cropped_img.size[1]}")
            return True

    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Crop photos to remove banner and grey background.")
    parser.add_argument("folder", help="Folder containing images to process")
    parser.add_argument("--tolerance", type=int, default=5, help="Tolerance for background color matching (default 5)")

    args = parser.parse_args()

    if not os.path.isdir(args.folder):
        print(f"Error: {args.folder} is not a directory.")
        sys.exit(1)

    extensions = ('.jpg', '.jpeg', '.png', '.webp')
    files = [f for f in os.listdir(args.folder) if f.lower().endswith(extensions)]

    if not files:
        print(f"No image files found in {args.folder}")
        return

    print(f"Processing {len(files)} images in {args.folder}...")

    count = 0
    for filename in files:
        filepath = os.path.join(args.folder, filename)
        if crop_image(filepath, tolerance=args.tolerance):
            count += 1

    print(f"Done. Processed {count} images.")

if __name__ == "__main__":
    main()
