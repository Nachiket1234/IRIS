"""Generate pseudo segmentation masks for the ISIC skin cancer dataset.

Since the Kaggle classification dataset does not ship pixel-level masks, we
approximate lesion regions with adaptive thresholding and morphological
post-processing so that IRIS can be trained in a segmentation regime. Each
pixel that belongs to the detected lesion is annotated with the class ID of
the originating folder (1..9), while background remains 0.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
from PIL import Image, ImageOps
from scipy import ndimage as ndi


CLASS_NAMES: Tuple[str, ...] = (
    "actinic keratosis",
    "basal cell carcinoma",
    "dermatofibroma",
    "melanoma",
    "nevus",
    "pigmented benign keratosis",
    "seborrheic keratosis",
    "squamous cell carcinoma",
    "vascular lesion",
)
CLASS_TO_ID = {name: idx + 1 for idx, name in enumerate(CLASS_NAMES)}


def _generate_mask_array(image: Image.Image) -> np.ndarray:
    """Approximate a lesion mask using intensity & texture cues."""
    gray = ImageOps.grayscale(image)
    gray = ImageOps.equalize(gray)
    arr = np.asarray(gray, dtype=np.float32) / 255.0

    blurred = ndi.gaussian_filter(arr, sigma=1.2)
    low = np.percentile(blurred, 35.0)
    high = np.percentile(blurred, 65.0)
    thresh = 0.5 * (low + high)
    lesion = blurred <= thresh

    # Combine with a highlight-based heuristic to handle bright lesions.
    bright_mask = blurred >= np.percentile(blurred, 92.0)
    lesion = np.logical_or(lesion, bright_mask)

    lesion = ndi.binary_closing(lesion, structure=np.ones((5, 5)), iterations=2)
    lesion = ndi.binary_opening(lesion, structure=np.ones((3, 3)))
    lesion = ndi.binary_fill_holes(lesion)

    labeled, num_objects = ndi.label(lesion)
    if num_objects > 0:
        object_sizes = ndi.sum(lesion, labeled, index=range(1, num_objects + 1))
        largest_label = int(np.argmax(object_sizes)) + 1
        lesion = labeled == largest_label
    else:
        lesion = np.zeros_like(lesion, dtype=bool)

    return lesion.astype(np.uint8)


def generate_mask(image_path: Path, class_id: int, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.open(image_path).convert("RGB")
    lesion = _generate_mask_array(image)
    mask = np.zeros_like(lesion, dtype=np.uint8)
    mask[lesion > 0] = class_id
    Image.fromarray(mask, mode="L").save(output_path)


def iter_image_paths(root: Path) -> Iterable[Tuple[Path, int]]:
    for split in ("Train", "Test"):
        for class_name, class_id in CLASS_TO_ID.items():
            class_dir = root / split / class_name
            if not class_dir.exists():
                continue
            for file in sorted(class_dir.glob("*.jpg")):
                yield file, class_id


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate pseudo ISIC masks.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("datasets/isic/Skin cancer ISIC The International Skin Imaging Collaboration"),
        help="Path to the extracted ISIC dataset.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("datasets/isic/masks"),
        help="Directory where generated masks will be stored.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Optional cap on the number of images to process (for debugging).",
    )
    args = parser.parse_args()

    if not args.dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {args.dataset_root}")

    processed = 0
    for image_path, class_id in iter_image_paths(args.dataset_root):
        relative = image_path.relative_to(args.dataset_root)
        output_path = args.output_root / relative.with_suffix(".png")
        if output_path.exists():
            processed += 1
            continue
        generate_mask(image_path, class_id, output_path)
        processed += 1
        if processed % 200 == 0:
            print(f"[INFO] Generated {processed} masks...")
        if args.max_images is not None and processed >= args.max_images:
            break

    print(f"[DONE] Total masks processed: {processed}")


if __name__ == "__main__":
    main()


