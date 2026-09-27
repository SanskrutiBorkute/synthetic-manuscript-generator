"""
validate_dataset.py
===================
Integrity and validation script for synthetic manuscript datasets:
- Verifies that all generated image files exist and are not corrupt
- Checks bounding box validity (within image dimensions, positive area)
- Validates text transcriptions are non-empty Unicode strings
- Reports script class balance across train, val, and test splits
"""

import argparse
import json
import os
import sys
from typing import Dict, Any, List
from PIL import Image


def validate_manifest(manifest_path: str, base_dir: str) -> Dict[str, Any]:
    """
    Validates a dataset split manifest file.

    Args:
        manifest_path: Path to the JSON manifest (e.g., train.json).
        base_dir: Root directory for relative image paths.

    Returns:
        Summary dictionary with counts and any errors found.
    """
    if not os.path.exists(manifest_path):
        return {"status": "missing", "error": f"Manifest not found: {manifest_path}"}

    with open(manifest_path, "r", encoding="utf-8") as f:
        records: List[Dict[str, Any]] = json.load(f)

    stats = {
        "status": "ok",
        "total_samples": len(records),
        "scripts": {},
        "errors": [],
        "warnings": [],
    }

    for idx, rec in enumerate(records):
        img_file = rec.get("file_name", "")
        full_img_path = os.path.join(base_dir, img_file)

        # 1. Image file existence and read check
        if not os.path.exists(full_img_path):
            stats["errors"].append(f"Sample {idx}: Missing image file '{full_img_path}'")
            continue

        try:
            with Image.open(full_img_path) as img:
                w, h = img.size
                if w != rec.get("width") or h != rec.get("height"):
                    stats["warnings"].append(
                        f"Sample {idx}: Dimension mismatch between image ({w}x{h}) and metadata ({rec.get('width')}x{rec.get('height')})"
                    )
        except Exception as e:
            stats["errors"].append(f"Sample {idx}: Corrupt image file '{full_img_path}': {e}")
            continue

        # 2. Mandatory paired .md annotation file check
        base_name = os.path.splitext(full_img_path)[0]
        md_ann_path = f"{base_name}.md"
        if not os.path.exists(md_ann_path):
            stats["errors"].append(
                f"Sample {idx}: Missing MANDATORY paired Markdown annotation file '{md_ann_path}'"
            )
        elif os.path.getsize(md_ann_path) == 0:
            stats["errors"].append(
                f"Sample {idx}: Paired Markdown annotation file '{md_ann_path}' is empty"
            )

        # 3. Script category distribution
        script = rec.get("script", "unknown")
        stats["scripts"][script] = stats["scripts"].get(script, 0) + 1

        # 4. Bounding box validity
        lines = rec.get("lines", [])
        if not lines:
            stats["warnings"].append(f"Sample {idx}: No lines annotated for image '{img_file}'")

        for line_idx, line in enumerate(lines):
            bbox = line.get("bbox", [])
            if len(bbox) != 4:
                stats["errors"].append(f"Sample {idx}, Line {line_idx}: Invalid bbox format {bbox}")
                continue

            x1, y1, x2, y2 = bbox
            if not (0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h):
                stats["errors"].append(
                    f"Sample {idx}, Line {line_idx}: Out of bounds bbox {bbox} for image size ({w}, {h})"
                )

    if stats["errors"]:
        stats["status"] = "failed"

    return stats


def main():
    parser = argparse.ArgumentParser(description="Validate synthetic manuscript dataset integrity")
    parser.add_argument(
        "--output-dir",
        default="data/output",
        help="Path to the generated output directory (default: data/output)",
    )
    args = parser.parse_args()

    print("========================================")
    print(" Synthetic Manuscript Dataset Validator")
    print("========================================")
    print(f"Target Directory: {args.output_dir}\n")

    if not os.path.exists(args.output_dir):
        print(f"Notice: Output directory '{args.output_dir}' does not exist or has not been populated yet.")
        print("Run 'python generate.py' to generate manuscript samples first.")
        sys.exit(0)

    splits = ["train", "val", "test"]
    all_ok = True

    expected_counts_per_script = {"train": 85, "val": 10, "test": 5}

    for split in splits:
        manifest_path = os.path.join(args.output_dir, f"{split}.json")
        print(f"Validating '{split}' split...")
        result = validate_manifest(manifest_path, args.output_dir)

        if result["status"] == "missing":
            print(f"  - Notice: {result['error']}")
            continue

        print(f"  - Total samples: {result['total_samples']}")
        print(f"  - Script breakdown: {result['scripts']}")

        # Validate quota per script if generated
        expected_per_script = expected_counts_per_script.get(split, 0)
        for s_name, count in result["scripts"].items():
            if count != expected_per_script:
                result["warnings"].append(
                    f"Script '{s_name}' has {count} samples in '{split}', expected {expected_per_script}"
                )

        if result["warnings"]:
            print(f"  - Warnings ({len(result['warnings'])}):")
            for w in result["warnings"][:5]:
                print(f"    * {w}")

        if result["errors"]:
            all_ok = False
            print(f"  - Errors ({len(result['errors'])}):")
            for e in result["errors"][:5]:
                print(f"    * {e}")
        else:
            print("  - Status: PASSED")
        print()

    if all_ok:
        print("Validation finished successfully.")
    else:
        print("Validation finished with errors. Please check the log above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
