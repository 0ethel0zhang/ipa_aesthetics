import os
import sys
import subprocess
import pytesseract
from PIL import Image

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

def process_directory(directory):
    if not os.path.isdir(directory):
        print(f"Error: {directory} is not a valid directory.")
        return

    # Get absolute path for directory
    directory = os.path.abspath(directory)
    print(f"Processing directory: {directory}")

    files = [f for f in os.listdir(directory) if f.lower().endswith((".jpg", ".jpeg"))]
    if not files:
        print("No JPEG files found.")
        return

    for filename in files:
        filepath = os.path.join(directory, filename)
        try:
            # Open image and perform OCR
            img = Image.open(filepath)
            text = pytesseract.image_to_string(img)

            rating = get_rating_from_text(text)
            if rating is not None:
                update_xmp_rating(filepath, rating)
            else:
                print(f"Skipping {filepath}: No prize detected in OCR text.")
        except Exception as e:
            print(f"Error processing {filepath}: {e}")

if __name__ == "__main__":
    target_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    process_directory(target_dir)
