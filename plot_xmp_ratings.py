#!/usr/bin/env python3
import subprocess
import json
import os
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import argparse
import sys
import shutil

def check_exiftool():
    """Checks if exiftool is installed and accessible."""
    if shutil.which("exiftool") is None:
        print("Error: 'exiftool' not found in PATH. Please install it to use this script.", file=sys.stderr)
        sys.exit(1)

def get_ratings(directory, recursive=True, extensions=None):
    """
    Extracts XMP ratings from all images in a directory using exiftool.
    Returns a list of ratings (integers 0-5).
    """
    print(f"Processing directory: {directory}")
    try:
        # -Rating: Extracts the Rating field (often XMP:Rating)
        # -json: Output as JSON for easy parsing
        # -r: Recursive search
        cmd = ["exiftool", "-Rating", "-json"]
        if recursive:
            cmd.append("-r")

        if extensions:
            for ext in extensions:
                cmd.extend(["-ext", ext])

        cmd.append(directory)
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)

        ratings = []
        for item in data:
            # Handle missing Rating field by defaulting to 0 as requested
            rating = item.get("Rating", 0)
            try:
                # Ensure it's an integer
                val = int(rating)
                # Clamp to 0-5 just in case
                val = max(0, min(5, val))
                ratings.append(val)
            except (ValueError, TypeError):
                ratings.append(0)

        print(f"Found {len(ratings)} images with ratings in {directory}")
        return ratings
    except subprocess.CalledProcessError as e:
        print(f"Error running exiftool on {directory}: {e.stderr}", file=sys.stderr)
        return []
    except json.JSONDecodeError:
        print(f"Error parsing exiftool output for {directory}", file=sys.stderr)
        return []
    except Exception as e:
        print(f"An unexpected error occurred while processing {directory}: {e}", file=sys.stderr)
        return []

def plot_ratings(data_dict, output_file="rating_distribution.png"):
    """
    Plots the distribution of ratings as a bar chart.
    data_dict: Dictionary where keys are folder labels and values are lists of ratings.
    """
    # Academic styling
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman', 'Liberation Serif', 'DejaVu Serif', 'serif']
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12

    fig, ax = plt.subplots(figsize=(10, 7))

    ratings_range = np.arange(6)  # 0, 1, 2, 3, 4, 5
    num_series = len(data_dict)
    bar_width = 0.8 / num_series

    # Use a professional color palette
    colors = plt.cm.viridis(np.linspace(0, 0.8, num_series))

    for i, (label, ratings) in enumerate(data_dict.items()):
        counts = Counter(ratings)
        y_values = [counts.get(r, 0) for r in ratings_range]

        # Calculate offset for grouped bars
        offset = (i - (num_series - 1) / 2) * bar_width

        bars = ax.bar(ratings_range + offset, y_values, bar_width,
                      label=label, color=colors[i], edgecolor='black', linewidth=1)

        # Optionally add counts on top of bars
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.annotate(f'{int(height)}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=10)

    ax.set_ylabel('Number of Samples')
    ax.set_title('Distribution of XMP Ratings across Photo Repositories')
    ax.set_xticks(ratings_range)
    ax.set_xticklabels([f'{r}' for r in ratings_range])

    # Label the X-axis with a note about 0
    ax.set_xlabel('XMP Rating (0-5 Stars, 0 includes No Rating)')

    if num_series > 1:
        ax.legend(title="Repositories")

    # Academic style grid
    ax.yaxis.grid(True, linestyle='--', which='major', color='grey', alpha=0.5)
    ax.set_axisbelow(True) # Put grid behind bars

    # Remove top and right spines for a cleaner look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"Successfully saved plot to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Plot XMP rating distribution from image folders.")
    parser.add_argument("directories", nargs="+", help="One or more directories containing images.")
    parser.add_argument("--output", "-o", default="rating_distribution.png", help="Output PNG filename.")
    parser.add_argument("--no-recursive", action="store_false", dest="recursive", help="Do not search subdirectories.")
    parser.add_argument("--extensions", "-e", nargs="+", help="Only process files with these extensions (e.g., jpg png).")

    args = parser.parse_args()

    check_exiftool()

    data_dict = {}
    for directory in args.directories:
        if not os.path.isdir(directory):
            print(f"Warning: '{directory}' is not a valid directory. Skipping.")
            continue

        # Use folder path as label to avoid collisions
        label = directory

        ratings = get_ratings(directory, recursive=args.recursive, extensions=args.extensions)

        if ratings:
            data_dict[label] = ratings
        else:
            print(f"No ratings found in {directory}.")

    if not data_dict:
        print("No data to plot. Exiting.")
        sys.exit(1)

    plot_ratings(data_dict, args.output)

if __name__ == "__main__":
    main()
