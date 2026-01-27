import json
import os
from pathlib import Path

import numpy as np
import fiftyone as fo
import fiftyone.zoo as foz


EXPORT_DIR = Path("export_oiv7_coco").resolve()
LABELS_JSON = EXPORT_DIR / "labels.json"


def _iter_coco_segmentation_points(seg):
    """
    Yields (x, y) points from COCO polygon segmentation lists.
    COCO polygon format: segmentation = [[x1,y1,x2,y2,...], [ ... ], ...]
    """
    if not isinstance(seg, list):
        return
    for poly in seg:
        if not isinstance(poly, list):
            continue
        # poly is [x1,y1,x2,y2,...]
        for i in range(0, len(poly) - 1, 2):
            yield float(poly[i]), float(poly[i + 1])


def main():
    print("PY:", os.sys.executable)
    print("Python:", os.sys.version.split()[0])
    print("NumPy:", np.__version__)
    print("FiftyOne:", getattr(fo, "__version__", "unknown"))
    print("FO file:", getattr(fo, "__file__", None))

    # --- Load a small dataset slice with segmentations ---
    # IMPORTANT: give it a deterministic name so reruns are predictable
    ds_name = "open-images-v7-validation-15"

    if ds_name in fo.list_datasets():
        # delete to ensure we always start clean
        fo.delete_dataset(ds_name)

    dataset = foz.load_zoo_dataset(
        "open-images-v7",
        split="validation",
        max_samples=15,
        shuffle=True,
        label_types=["segmentations"],  # force segs
        dataset_name=ds_name,
    )

    print("Loaded:", dataset.name)
    print("Count:", dataset.count())

    # --- Export to COCO ---
    if EXPORT_DIR.exists():
        # don't blow away manually, but keep it simple for you:
        # remove old labels.json so we always read the fresh one
        try:
            (EXPORT_DIR / "labels.json").unlink()
        except FileNotFoundError:
            pass
    else:
        EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    dataset.export(
        export_dir=str(EXPORT_DIR),
        dataset_type=fo.types.COCODetectionDataset,
        label_field="ground_truth",
    )

    print("COCO export dir:", str(EXPORT_DIR))

    if not LABELS_JSON.exists():
        raise RuntimeError(f"Expected labels.json not found at: {LABELS_JSON}")

    data = json.loads(LABELS_JSON.read_text(encoding="utf-8"))

    # --- Check for segmentation vertices outside bbox ---
    annotations = data.get("annotations", [])
    oob_hits = 0
    max_outside = 0.0

    for ann in annotations:
        bbox = ann.get("bbox", None)
        seg = ann.get("segmentation", None)
        if bbox is None or seg is None:
            continue
        if not isinstance(seg, list):
            # RLE etc. skip
            continue

        x, y, w, h = bbox
        x2 = x + w
        y2 = y + h

        for px, py in _iter_coco_segmentation_points(seg):
            dx = 0.0
            dy = 0.0
            if px < x:
                dx = x - px
            elif px > x2:
                dx = px - x2
            if py < y:
                dy = y - py
            elif py > y2:
                dy = py - y2

            outside = max(dx, dy)
            if outside > 0:
                oob_hits += 1
                if outside > max_outside:
                    max_outside = outside

    print("Total OOB vertex hits:", oob_hits)
    print("Max outside (px):", max_outside)

    if oob_hits > 0:
        print("\n⚠️ Reproduced: segmentation vertices outside bbox. See max outside above.")
    else:
        print("\n✅ Not reproduced: no segmentation vertices outside bbox in this export.")


if __name__ == "__main__":
    main()
