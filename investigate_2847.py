#!/usr/bin/env python
"""
Debug script for GitHub Issue #2847 - segmentation vertices outside bbox

Investigating why polygon vertices exceed bounding box boundaries when
exporting instance masks to COCO format.

Usage: python investigate_2847.py
"""

import numpy as np
from skimage import measure

import fiftyone as fo
import fiftyone.zoo as foz


def test_find_contours_subpixel():
    """
    Demonstrate that find_contours(mask, 0.5) produces sub-pixel coords.
    This is the core of the issue.
    """
    print("\n--- Testing find_contours behavior ---")

    # Simple 3x3 True region inside a 5x5 mask
    mask = np.zeros((5, 5), dtype=bool)
    mask[1:4, 1:4] = True

    print("Mask (3x3 True region at [1:4, 1:4]):")
    print(mask.astype(int))
    print("Bounding box should be: x=1, y=1, w=3, h=3 -> max coords (4, 4)")

    # FiftyOne pads the mask before finding contours
    padded = np.pad(mask, 1, mode="constant", constant_values=0)
    contours = measure.find_contours(padded, 0.5)
    contours = [c - 1 for c in contours]  # undo padding offset

    for c in contours:
        # FiftyOne flips to (x, y) order
        c = np.flip(c, axis=1)
        print(f"\nContour range: x=[{c[:, 0].min():.1f}, {c[:, 0].max():.1f}], "
              f"y=[{c[:, 1].min():.1f}, {c[:, 1].max():.1f}]")

        # Check: does 0.5 appear? That's outside the integer bbox
        if c.min() < 1 or c.max() > 3:
            print("^ Sub-pixel coords detected - this causes the bbox overflow")


def test_with_real_data():
    """
    Test with actual Open Images data to see real-world deviation.
    """
    print("\n--- Testing with Open Images V7 ---")

    ds_name = "debug-2847"
    if ds_name in fo.list_datasets():
        fo.delete_dataset(ds_name)

    try:
        dataset = foz.load_zoo_dataset(
            "open-images-v7",
            split="validation",
            max_samples=5,
            label_types=["segmentations"],
            dataset_name=ds_name,
        )
    except Exception as e:
        print(f"Couldn't load dataset: {e}")
        return

    # Find a sample with a mask
    for sample in dataset:
        if not sample.ground_truth:
            continue
        for det in sample.ground_truth.detections:
            if det.mask is None:
                continue

            # Get image size
            if sample.metadata:
                w, h = sample.metadata.width, sample.metadata.height
            else:
                from PIL import Image
                with Image.open(sample.filepath) as img:
                    w, h = img.size

            # Render full mask (same as FiftyOne export does)
            import fiftyone.utils.eta as foue
            import eta.core.image as etai

            dobj = foue.to_detected_object(det, extra_attrs=False)
            full_mask = etai.render_instance_image(
                dobj.mask, dobj.bounding_box, (w, h))

            # Find contours
            padded = np.pad(full_mask, 1, constant_values=0)
            contours = measure.find_contours(padded, 0.5)
            contours = [c - 1 for c in contours]

            # Get bbox in pixels
            bx, by, bw, bh = det.bounding_box
            bbox_xmin, bbox_ymin = bx * w, by * h
            bbox_xmax, bbox_ymax = bbox_xmin + bw * w, bbox_ymin + bh * h

            print(f"\nSample: {det.label}")
            print(
                f"BBox: [{bbox_xmin:.1f}, {bbox_ymin:.1f}] to [{bbox_xmax:.1f}, {bbox_ymax:.1f}]")

            for contour in contours:
                contour = measure.approximate_polygon(contour, tolerance=2)
                if len(contour) < 3:
                    continue
                contour = np.flip(contour, axis=1)

                xmin, xmax = contour[:, 0].min(), contour[:, 0].max()
                ymin, ymax = contour[:, 1].min(), contour[:, 1].max()

                # How much does contour exceed bbox?
                over_left = bbox_xmin - xmin
                over_right = xmax - bbox_xmax
                over_top = bbox_ymin - ymin
                over_bottom = ymax - bbox_ymax

                max_over = max(over_left, over_right, over_top, over_bottom)
                if max_over > 0:
                    print(f"Contour exceeds bbox by {max_over:.2f}px")
                else:
                    print("Contour within bbox")

            fo.delete_dataset(ds_name)
            return

    print("No masks found in samples")
    fo.delete_dataset(ds_name)


def explain_issue():
    """Print explanation of why this happens."""
    print(
        """
      === Root Cause ===

      find_contours(mask, 0.5) traces the boundary where pixel values equal 0.5.
      For a binary mask (0 and 1), this boundary lies BETWEEN pixels - at sub-pixel
      positions like x=10.5 (halfway between pixel 10 and 11).

      The detection's bounding box is stored in float coords (e.g., x * width),
      but the mask gets placed at integer pixel positions by ETA's render functions.

      This mismatch means the contour can exceed the bbox by ~0.5-1.5 pixels.
      """
    )


if __name__ == "__main__":
    print("Issue #2847 Investigation")
    print("=" * 40)

    test_find_contours_subpixel()
    test_with_real_data()
    explain_issue()
