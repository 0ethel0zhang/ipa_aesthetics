#!/usr/bin/env python3
import subprocess
import json
import os
import argparse
import sys
import shutil

def check_exiftool():
    """Checks if exiftool is installed and accessible."""
    if shutil.which("exiftool") is None:
        print("Error: 'exiftool' not found in PATH. Please install it to use this script.", file=sys.stderr)
        sys.exit(1)

def get_ratings_dict(directory, extensions=None):
    """
    Extracts XMP ratings from images in a directory recursively.
    Returns a dictionary mapping filename to rating (int 0-5).
    Assumes filenames are unique across subdirectories.
    """
    if extensions is None:
        extensions = ['jpg', 'jpeg', 'JPG', 'JPEG']

    ratings_dict = {}
    try:
        cmd = ["exiftool", "-Rating", "-json", "-r"]
        for ext in extensions:
            cmd.extend(["-ext", ext])
        cmd.append(directory)

        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        # Handle empty output if no files found
        if not result.stdout.strip():
            return {}

        data = json.loads(result.stdout)

        for item in data:
            filepath = item.get("SourceFile")
            if not filepath:
                continue
            filename = os.path.basename(filepath)
            # Default to 0 if Rating is missing
            rating = item.get("Rating", 0)
            try:
                val = int(rating)
                val = max(0, min(5, val))
            except (ValueError, TypeError):
                val = 0
            ratings_dict[filename] = val

        return ratings_dict
    except subprocess.CalledProcessError as e:
        # If exiftool returns 1 it might just mean no files were found
        if e.returncode == 1 and "No matching files" in e.stderr:
            return {}
        print(f"Error running exiftool on {directory}: {e.stderr}", file=sys.stderr)
        return {}
    except json.JSONDecodeError:
        print(f"Error parsing exiftool output for {directory}", file=sys.stderr)
        return {}
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        return {}

def main():
    parser = argparse.ArgumentParser(description="Compare XMP ratings between a golden folder and a target folder.")
    parser.add_argument("golden_dir", help="Directory containing the golden (reference) photos.")
    parser.add_argument("target_dir", help="Directory containing the photos to be checked.")

    args = parser.parse_args()

    check_exiftool()

    if not os.path.isdir(args.golden_dir):
        print(f"Error: Golden directory '{args.golden_dir}' not found.")
        sys.exit(1)
    if not os.path.isdir(args.target_dir):
        print(f"Error: Target directory '{args.target_dir}' not found.")
        sys.exit(1)

    # Use .jpg and .jpeg as requested
    extensions = ['jpg', 'jpeg', 'JPG', 'JPEG']

    print(f"Reading golden ratings from: {args.golden_dir}")
    golden_ratings = get_ratings_dict(args.golden_dir, extensions)
    print(f"Found {len(golden_ratings)} images in golden directory.")

    print(f"Reading target ratings from: {args.target_dir}")
    target_ratings = get_ratings_dict(args.target_dir, extensions)
    print(f"Found {len(target_ratings)} images in target directory.")

    # Match filenames exactly
    common_files = set(golden_ratings.keys()) & set(target_ratings.keys())

    if not common_files:
        print("\nNo matching filenames found between the two directories.")
        return

    matches = 0
    for filename in common_files:
        if golden_ratings[filename] == target_ratings[filename]:
            matches += 1

    accuracy = (matches / len(common_files)) * 100

    print("-" * 30)
    print(f"Comparison Results:")
    print(f"Files in both: {len(common_files)}")
    print(f"Matches:       {matches}")
    print(f"Overall Accuracy: {accuracy:.2f}%")
    print("-" * 30)

if __name__ == "__main__":
    main()
