#!/usr/bin/env python
"""
Test that COCO export produces valid polygons within bounding boxes.
Validates fix for GitHub Issue #2847.

Usage: python test_fix_2847.py
"""

import json
from pathlib import Path

import fiftyone as fo
import fiftyone.zoo as foz


EXPORT_DIR = Path("test_export_coco")


def check_coco_validity(labels_path):
    """
    Check if all polygon vertices are within their bounding boxes.
    Returns (num_violations, max_deviation_px).
    """
    data = json.loads(labels_path.read_text())

    violations = 0
    max_dev = 0.0

    for ann in data.get("annotations", []):
        bbox = ann.get("bbox")
        seg = ann.get("segmentation")

        if not bbox or not seg or not isinstance(seg, list):
            continue

        x, y, w, h = bbox
        x2, y2 = x + w, y + h

        for poly in seg:
            if not isinstance(poly, list):
                continue
            # COCO polygon format: [x1, y1, x2, y2, ...]
            for i in range(0, len(poly) - 1, 2):
                px, py = poly[i], poly[i + 1]

                dx = max(x - px, px - x2, 0)
                dy = max(y - py, py - y2, 0)
                dev = max(dx, dy)

                if dev > 0:
                    violations += 1
                    max_dev = max(max_dev, dev)

    return violations, max_dev


def main():
    print("Testing COCO export for Issue #2847")
    print("=" * 40)

    # Load test data
    ds_name = "test-2847"
    if ds_name in fo.list_datasets():
        fo.delete_dataset(ds_name)

    print("\nLoading Open Images samples...")
    dataset = foz.load_zoo_dataset(
        "open-images-v7",
        split="validation",
        max_samples=15,
        label_types=["segmentations"],
        dataset_name=ds_name,
    )
    print(f"Loaded {dataset.count()} samples")

    # Export to COCO
    EXPORT_DIR.mkdir(exist_ok=True)
    labels_path = EXPORT_DIR / "labels.json"
    if labels_path.exists():
        labels_path.unlink()

    print("\nExporting to COCO format...")
    dataset.export(
        export_dir=str(EXPORT_DIR),
        dataset_type=fo.types.COCODetectionDataset,
        label_field="ground_truth",
    )

    # Check validity
    violations, max_dev = check_coco_validity(labels_path)

    print(f"\nResults:")
    print(f"  Violations: {violations}")
    print(f"  Max deviation: {max_dev:.2f}px")

    # Cleanup
    fo.delete_dataset(ds_name)

    if violations == 0:
        print("\nPASS: All polygons within bounding boxes")
        return 0
    else:
        print(f"\nFAIL: {violations} vertices outside bbox")
        return 1


if __name__ == "__main__":
    exit(main())
