import os
import sys
import subprocess
import pytesseract
from PIL import Image, ImageOps

# Mapping of prize to XMP Rating
PRIZE_MAPPING = {
    "diamond": 5,
    "platinum": 4,
    "gold": 3,
    "silver": 2,
    "bronze": 1
}

def get_rating_from_text(text):
    text = text.lower()
    # Check for prizes in descending order of value to handle potential multi-match if it ever happens
    # though user says there is only 1.
    for prize in ["diamond", "platinum", "gold", "silver", "bronze"]:
        if prize in text:
            return PRIZE_MAPPING[prize]
    return None

def update_xmp_rating(filepath, rating):
    try:
        # Use exiftool to update XMP:Rating
        # -overwrite_original avoids creating _original files
        subprocess.run(
            ["exiftool", f"-XMP:Rating={rating}", "-overwrite_original", filepath],
            check=True,
            capture_output=True,
            text=True
        )
        print(f"Updated {filepath} with Rating={rating}")
    except subprocess.CalledProcessError as e:
        print(f"Error updating {filepath}: {e.stderr}")

def preprocess_image(img):
    """Preprocess image to improve OCR accuracy."""
    # Convert to grayscale
    img = img.convert('L')

    # Crop to the left side (left 40% as specified by user "left hand size")
    width, height = img.size
    img = img.crop((0, 0, int(width * 0.4), height))

    # Upscale to ensure text is large enough for Tesseract
    img = img.resize((img.width * 2, img.height * 2), resample=Image.LANCZOS)

    # Auto contrast to make text pop
    img = ImageOps.autocontrast(img)

    return img

def process_directory(directory):
    if not os.path.isdir(directory):
        print(f"Error: {directory} is not a valid directory.")
        return

    # Get absolute path for directory
    directory = os.path.abspath(directory)
    print(f"Processing directory: {directory}")

    # Process common image formats
    extensions = (".jpg", ".jpeg", ".png", ".tiff", ".webp")
    files = [f for f in os.listdir(directory) if f.lower().endswith(extensions)]
    if not files:
        print(f"No image files found in {directory}.")
        return

    for filename in files:
        filepath = os.path.join(directory, filename)
        try:
            # Open image
            with Image.open(filepath) as img:
                # Preprocess for better OCR
                processed_img = preprocess_image(img)

                # Try multiple PSM modes and combine results
                # PSM 6 is good for uniform blocks (like the prize text in some layouts)
                # PSM 11 is good for sparse text
                all_text = ""
                for psm in [6, 11]:
                    config = f'--oem 3 --psm {psm}'
                    all_text += " " + pytesseract.image_to_string(processed_img, config=config)

                rating = get_rating_from_text(all_text)

                # Fallback: try original image if preprocessing was too aggressive
                if rating is None:
                    all_text_fallback = ""
                    for psm in [6, 11]:
                        config = f'--oem 3 --psm {psm}'
                        all_text_fallback += " " + pytesseract.image_to_string(img, config=config)
                    rating = get_rating_from_text(all_text_fallback)

                if rating is not None:
                    update_xmp_rating(filepath, rating)
                else:
                    print(f"Skipping {filepath}: No prize detected in OCR text.")
                    # print(f"DEBUG text: {all_text}") # Uncomment for debugging
        except Exception as e:
            print(f"Error processing {filepath}: {e}")

if __name__ == "__main__":
    target_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    process_directory(target_dir)
