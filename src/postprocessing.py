"""
Postprocessing routines for binary mowing prediction maps.

Three levels of postprocessing, designed to be applied in sequence:

  1. median_filter        — removes salt-and-pepper single-pixel noise
  2. morphological_clean  — binary open/close + minimum area filter
  3. slic_majority_vote   — object-based smoothing using SLIC superpixels

The main entry point is postprocess_prediction(), which accepts a method name
or list of method names so notebooks can run clean comparisons.
"""

import numpy as np
from scipy.ndimage import (
    median_filter as scipy_median_filter,
    binary_opening,
    binary_closing,
    label as ndimage_label,
)

# skimage is only needed for SLIC — import lazily so the module loads without it
try:
    from skimage.segmentation import slic
    from skimage.morphology import disk
    _SKIMAGE_AVAILABLE = True
except ImportError:
    _SKIMAGE_AVAILABLE = False

try:
    from skimage.morphology import disk as skdisk
    _DISK_AVAILABLE = True
except ImportError:
    _DISK_AVAILABLE = False


# ---------------------------------------------------------------------------
# Individual operators
# ---------------------------------------------------------------------------

def apply_median_filter(pred_map: np.ndarray, size: int = 3) -> np.ndarray:
    """
    Majority/median filter over a square neighbourhood.
    Removes isolated single-pixel positive or negative blobs (salt-and-pepper).

    Parameters
    ----------
    pred_map : 2-D int8/uint8 array with values {-1, 0, 1}
               (-1 = no-data / outside mask)
    size     : filter window side length (default 3 → 3×3 neighbourhood)
    """
    nodata_mask = pred_map == -1
    result = scipy_median_filter(np.where(nodata_mask, 0, pred_map).astype(np.float32),
                                 size=size)
    result = (result >= 0.5).astype(np.int8)
    result[nodata_mask] = -1
    return result


def apply_morphological_clean(pred_map: np.ndarray,
                               disk_radius: int = 2,
                               min_area_pixels: int = 4) -> np.ndarray:
    """
    Binary morphological opening (removes small positive blobs) followed by
    closing (fills small holes), then a minimum area filter that discards
    connected components smaller than min_area_pixels.

    Parameters
    ----------
    pred_map         : 2-D int8/uint8 array with values {-1, 0, 1}
    disk_radius      : radius for the structuring disk element
    min_area_pixels  : connected components with fewer pixels are discarded
                       (default 4 pixels = 400 m² at 10 m resolution,
                       mirrors the ground-truth cleaning threshold from PA2)
    """
    nodata_mask = pred_map == -1
    binary = (pred_map == 1)

    if _DISK_AVAILABLE:
        struct = skdisk(disk_radius)
    else:
        # Fallback: square structuring element
        r = disk_radius
        struct = np.ones((2 * r + 1, 2 * r + 1), dtype=bool)

    opened = binary_opening(binary, structure=struct)
    closed = binary_closing(opened, structure=struct)

    # Minimum area filter
    labeled, n_components = ndimage_label(closed)
    for comp_id in range(1, n_components + 1):
        if np.sum(labeled == comp_id) < min_area_pixels:
            closed[labeled == comp_id] = False

    result = closed.astype(np.int8)
    result[nodata_mask] = -1
    return result


def apply_slic_majority_vote(pred_map: np.ndarray,
                              scene_image: np.ndarray,
                              n_segments: int = 300,
                              compactness: float = 0.05,
                              threshold: float = 0.5) -> np.ndarray:
    """
    Object-based smoothing using SLIC superpixel segmentation.
    Segments are derived from the spectral content of the satellite scene;
    within each segment, pixels are assigned the majority prediction label.

    Parameters
    ----------
    pred_map     : 2-D int8 array {-1, 0, 1}
    scene_image  : 3-D float array (H, W, C) — the multi-band AFTER image
                   normalised to [0, 1].  Typically RGB or all 5 S2 bands.
    n_segments   : approximate number of superpixels (default 300)
    compactness  : SLIC compactness.  Low value (0.05) forces spectral
                   over spatial clustering — good for irregular field shapes.
    threshold    : fraction of positive pixels in a segment required to label
                   the whole segment as mowed (default 0.5 = majority vote)

    Requires scikit-image.
    """
    if not _SKIMAGE_AVAILABLE:
        raise ImportError(
            "scikit-image is required for SLIC postprocessing. "
            "Install it with: pip install scikit-image"
        )

    nodata_mask = pred_map == -1
    binary = (pred_map == 1).astype(np.int8)

    segments = slic(scene_image, n_segments=n_segments, compactness=compactness,
                    start_label=0, channel_axis=-1)

    result = np.zeros_like(binary)
    for seg_id in np.unique(segments):
        seg_pixels = segments == seg_id
        valid = seg_pixels & ~nodata_mask
        if valid.sum() == 0:
            continue
        mowed_fraction = binary[valid].mean()
        result[seg_pixels] = 1 if mowed_fraction >= threshold else 0

    result[nodata_mask] = -1
    return result


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def postprocess_prediction(pred_map: np.ndarray,
                            method: str = 'morphological',
                            scene_image: np.ndarray = None,
                            median_size: int = 3,
                            disk_radius: int = 2,
                            min_area_pixels: int = 4,
                            n_segments: int = 300,
                            slic_compactness: float = 0.05) -> np.ndarray:
    """
    Apply one or more postprocessing steps to a binary prediction map.

    Parameters
    ----------
    pred_map     : 2-D int8 array with values {-1 (nodata), 0, 1}
    method       : one of:
                     'median'           — median filter only
                     'morphological'    — morphological open/close + area filter
                     'slic'             — SLIC majority vote (requires scene_image)
                     'median+morphological'   — both in sequence
                     'morphological+slic'     — both in sequence
                     'full'             — all three in sequence
    scene_image  : required for methods containing 'slic'; shape (H, W, C),
                   values normalised to [0, 1]

    Returns
    -------
    Postprocessed 2-D int8 array, same shape as pred_map.
    """
    result = pred_map.copy()

    steps = method.split('+')

    for step in steps:
        step = step.strip().lower()
        if step == 'median':
            result = apply_median_filter(result, size=median_size)
        elif step == 'morphological':
            result = apply_morphological_clean(result, disk_radius=disk_radius,
                                               min_area_pixels=min_area_pixels)
        elif step == 'slic':
            if scene_image is None:
                raise ValueError("scene_image must be provided for SLIC postprocessing.")
            result = apply_slic_majority_vote(result, scene_image,
                                              n_segments=n_segments,
                                              compactness=slic_compactness)
        elif step == 'full':
            result = apply_median_filter(result, size=median_size)
            result = apply_morphological_clean(result, disk_radius=disk_radius,
                                               min_area_pixels=min_area_pixels)
            if scene_image is not None:
                result = apply_slic_majority_vote(result, scene_image,
                                                  n_segments=n_segments,
                                                  compactness=slic_compactness)
        else:
            raise ValueError(
                f"Unknown postprocessing step: '{step}'. "
                "Valid steps: median, morphological, slic, full "
                "(combine with '+', e.g. 'median+morphological')."
            )

    return result
