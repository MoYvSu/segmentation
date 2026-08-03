# -*- coding: utf-8 -*-
"""
Instance ID map color visualization
====================================
Read _inst.png (uint8 instance IDs) and _class.json (ID->class mapping),
render each instance with a distinct color:
  - Ferrite (class=1): warm hues (red/orange/yellow, HSV H: 0-60)
  - Pearlite (class=0): cool hues (cyan/blue/purple, HSV H: 180-300)
  - Background: black

Usage:
  # Batch: process all _inst.png in a directory
  python visualize_instances.py --input_dir outputs/inference

  # Single file
  python visualize_instances.py --inst outputs/inference/sample_inst.png
"""

import argparse
import glob
import json
import os

import cv2
import numpy as np


def generate_instance_colors(class_map: dict, seed: int = 42) -> dict:
    """
    Generate a distinct color for each instance ID.

    Ferrite instances use warm hues (H: 0-60), pearlite instances use
    cool hues (H: 180-300). Saturation and value are kept high for
    vivid colors. A fixed seed ensures reproducibility.

    Args:
        class_map: {str(id): int(class)} mapping
        seed: random seed for reproducible colors

    Returns:
        {int(id): (B, G, R)} color mapping (uint8, OpenCV BGR)
    """
    rng = np.random.RandomState(seed)
    colors = {}

    # Separate IDs by class
    ferrite_ids = [int(k) for k, v in class_map.items() if v == 1]
    pearlite_ids = [int(k) for k, v in class_map.items() if v == 0]

    # Generate warm hues for ferrite (H: 0-60 degrees -> OpenCV H: 0-30)
    for idx, inst_id in enumerate(sorted(ferrite_ids)):
        hue = rng.uniform(0, 30)  # OpenCV H range: 0-180
        sat = rng.uniform(180, 255)
        val = rng.uniform(200, 255)
        hsv = np.array([[[hue, sat, val]]], dtype=np.uint8)
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
        colors[inst_id] = tuple(int(c) for c in bgr)

    # Generate cool hues for pearlite (H: 180-300 degrees -> OpenCV H: 90-150)
    for idx, inst_id in enumerate(sorted(pearlite_ids)):
        hue = rng.uniform(90, 150)  # OpenCV H range: 0-180
        sat = rng.uniform(180, 255)
        val = rng.uniform(200, 255)
        hsv = np.array([[[hue, sat, val]]], dtype=np.uint8)
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
        colors[inst_id] = tuple(int(c) for c in bgr)

    return colors


def visualize_instance_map(
    inst_map: np.ndarray,
    class_map: dict,
    seed: int = 42,
) -> np.ndarray:
    """
    Render an instance ID map as a color image.

    Args:
        inst_map: [H, W] uint8 array, pixel value = instance ID (0=background)
        class_map: {str(id): int(class)} mapping (1=ferrite, 0=pearlite)
        seed: random seed for color generation

    Returns:
        color_image: [H, W, 3] uint8 BGR image
    """
    h, w = inst_map.shape[:2]
    color_image = np.zeros((h, w, 3), dtype=np.uint8)

    # Generate colors for all instances
    colors = generate_instance_colors(class_map, seed=seed)

    # Fill colors
    for inst_id_str, cls in class_map.items():
        inst_id = int(inst_id_str)
        mask = inst_map == inst_id
        if mask.any():
            color_image[mask] = colors[inst_id]

    return color_image


def process_file(
    inst_path: str,
    class_json_path: str = None,
    output_path: str = None,
    seed: int = 42,
) -> str:
    """
    Process a single _inst.png file and save the color visualization.

    Args:
        inst_path: path to _inst.png
        class_json_path: path to _class.json (auto-inferred if None)
        output_path: output path (auto-inferred if None)
        seed: random seed

    Returns:
        output_path: path to saved color image
    """
    basename = os.path.basename(inst_path)
    # Auto-infer class_json path
    if class_json_path is None:
        # Replace _inst.png with _class.json
        base = inst_path.replace("_inst.png", "_class.json")
        if os.path.exists(base):
            class_json_path = base
        else:
            # Try replacing extension
            class_json_path = os.path.splitext(inst_path)[0].replace("_inst", "_class") + ".json"
            if not os.path.exists(class_json_path):
                raise FileNotFoundError(f"Cannot find class JSON for {inst_path}")

    # Auto-infer output path
    if output_path is None:
        base = inst_path.replace("_inst.png", "_inst_color.png")
        output_path = base

    # Read instance map
    inst_map = cv2.imread(inst_path, cv2.IMREAD_GRAYSCALE)
    if inst_map is None:
        raise FileNotFoundError(f"Cannot read instance map: {inst_path}")

    # Read class mapping
    with open(class_json_path, "r", encoding="utf-8") as f:
        class_map = json.load(f)

    # Render
    color_image = visualize_instance_map(inst_map, class_map, seed=seed)

    # Save
    cv2.imwrite(output_path, color_image)

    # Print summary
    n_ferrite = sum(1 for v in class_map.values() if v == 1)
    n_pearlite = sum(1 for v in class_map.values() if v == 0)
    print(f"  {basename}: {len(class_map)} instances "
          f"(ferrite={n_ferrite}, pearlite={n_pearlite}) -> {output_path}")

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Visualize instance ID maps with distinct colors per instance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  # Batch: process all _inst.png in a directory
  python visualize_instances.py --input_dir outputs/inference

  # Single file with explicit paths
  python visualize_instances.py \\
      --inst outputs/inference/sample_inst.png \\
      --class_json outputs/inference/sample_class.json

  # Custom output path
  python visualize_instances.py --inst outputs/inference/sample_inst.png \\
      --output outputs/inference/sample_color.png

Color scheme:
  Ferrite (class=1):  warm hues (red/orange/yellow)
  Pearlite (class=0): cool hues (cyan/blue/purple)
  Background:         black
""",
    )
    parser.add_argument(
        "--input_dir", type=str, default=None,
        help="Directory containing _inst.png and _class.json files (batch mode)",
    )
    parser.add_argument(
        "--inst", type=str, default=None,
        help="Path to a single _inst.png file",
    )
    parser.add_argument(
        "--class_json", type=str, default=None,
        help="Path to _class.json (auto-inferred if not specified)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output path for color image (auto-inferred if not specified)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for color generation (default: 42)",
    )
    args = parser.parse_args()

    if args.input_dir is not None:
        # Batch mode
        inst_files = sorted(glob.glob(os.path.join(args.input_dir, "*_inst.png")))
        if len(inst_files) == 0:
            print(f"No _inst.png files found in {args.input_dir}")
            return

        print(f"Found {len(inst_files)} instance maps in {args.input_dir}")
        for inst_path in inst_files:
            try:
                process_file(inst_path, seed=args.seed)
            except Exception as e:
                print(f"  ERROR: {os.path.basename(inst_path)}: {e}")

        print(f"\nDone! {len(inst_files)} files processed.")

    elif args.inst is not None:
        # Single file mode
        output_path = process_file(
            args.inst,
            class_json_path=args.class_json,
            output_path=args.output,
            seed=args.seed,
        )
        print(f"\nDone! Saved to {output_path}")

    else:
        parser.error("Either --input_dir or --inst must be specified")


if __name__ == "__main__":
    main()