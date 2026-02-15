import argparse
from pathlib import Path
from PIL import Image

def convert_webp_to_png(source_dir, output_dir):
    source_path = Path(source_dir).resolve()
    output_path = Path(output_dir).resolve()

    if not source_path.exists():
        print(f"Error: Source directory {source_path} does not exist.")
        return

    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)

    # Find all .webp files recursively
    webp_files = list(source_path.rglob("*.webp"))

    # Filter out files that are already inside the output directory to avoid infinite loops or re-converting
    # if output_dir is inside source_dir
    webp_files = [f for f in webp_files if not f.is_relative_to(output_path)]

    if not webp_files:
        print(f"No .webp files found in {source_path} (excluding {output_path})")
        return

    print(f"Found {len(webp_files)} .webp files. Starting conversion...")

    for webp_file in webp_files:
        # Determine relative path from source root
        rel_path = webp_file.relative_to(source_path)

        # Construct target path
        target_file = output_path / rel_path.with_suffix(".png")

        # Create target subfolders if necessary
        target_file.parent.mkdir(parents=True, exist_ok=True)

        try:
            with Image.open(webp_file) as img:
                img.save(target_file, "PNG")
            print(f"Converted: {rel_path} -> {target_file.relative_to(output_path.parent)}")
        except Exception as e:
            print(f"Failed to convert {webp_file}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert WebP images to PNG while preserving folder structure.")
    parser.add_argument("source", nargs="?", default=".", help="Source directory containing .webp files (default: current directory)")
    parser.add_argument("--output", default="png", help="Output directory for PNG files (default: 'png')")

    args = parser.parse_args()

    convert_webp_to_png(args.source, args.output)
