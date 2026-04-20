"""
SAR feature computation for Sentinel-1 GRD data.

Operates on local GeoTIFFs (already exported from GEE in dB scale).
No GEE dependency — pure numpy + rasterio.

Expected input: 2-band GeoTIFF with Band 1 = VV (dB), Band 2 = VH (dB).

Six features per pixel:
  State (after):   vv_after, vh_after, cr_after
  Change (after − before_near): vv_diff, vh_diff, cr_diff

Cross-ratio (CR) in dB = VH − VV.
  High CR  → canopy volume scattering dominates (tall grass)
  Low CR   → surface scattering dominates (bare/cut grass)
  CR drop  → strong mowing signal
"""

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling

# Pixels with dB values below this floor are treated as nodata
# (ocean, shadow, or missing data produce unrealistically low values)
NODATA_FLOOR_DB = -35.0


def load_s1_bands(filepath: str):
    """
    Load VV and VH bands from a 2-band S1 GeoTIFF.

    Returns
    -------
    vv : 2-D float32 array (dB), nodata → NaN
    vh : 2-D float32 array (dB), nodata → NaN
    profile : rasterio dataset profile (crs, transform, shape)
    """
    with rasterio.open(filepath) as src:
        vv = src.read(1).astype(np.float32)
        vh = src.read(2).astype(np.float32)
        profile = src.profile.copy()

    # Replace nodata and below-floor values with NaN
    nodata_val = profile.get('nodata', None)
    for arr in (vv, vh):
        if nodata_val is not None:
            arr[arr == nodata_val] = np.nan
        arr[arr < NODATA_FLOOR_DB] = np.nan

    return vv, vh, profile


def compute_cr(vv_db: np.ndarray, vh_db: np.ndarray) -> np.ndarray:
    """
    Cross-ratio in dB: CR = VH − VV.

    Positive CR values indicate vegetation canopy (volume scattering).
    After mowing, CR drops as the canopy collapses.
    NaN propagates from either input.
    """
    return vh_db - vv_db


def compute_sar_features(before_vv: np.ndarray,
                          before_vh: np.ndarray,
                          after_vv: np.ndarray,
                          after_vh: np.ndarray) -> dict:
    """
    Compute the 6 SAR features for a before/after image pair.

    All inputs must be 2-D float arrays in dB with the same shape.
    NaN pixels (nodata or below floor) propagate to all output features.

    Parameters
    ----------
    before_vv, before_vh : arrays from the BEFORE_NEAR S1 image
    after_vv,  after_vh  : arrays from the AFTER S1 image

    Returns
    -------
    dict with keys: vv_after, vh_after, cr_after, vv_diff, vh_diff, cr_diff
    All values are 2-D float32 arrays of the same shape as the inputs.
    """
    cr_before = compute_cr(before_vv, before_vh)
    cr_after  = compute_cr(after_vv,  after_vh)

    return {
        'vv_after': after_vv.astype(np.float32),
        'vh_after': after_vh.astype(np.float32),
        'cr_after': cr_after.astype(np.float32),
        'vv_diff':  (after_vv  - before_vv).astype(np.float32),
        'vh_diff':  (after_vh  - before_vh).astype(np.float32),
        'cr_diff':  (cr_after  - cr_before).astype(np.float32),
    }


def align_s1_to_reference(s1_path: str, ref_path: str) -> tuple:
    """
    Reproject an S1 GeoTIFF to exactly match the grid of a reference S2 tile.

    Uses bilinear resampling (appropriate for continuous dB values).
    Returns reprojected (vv, vh) arrays aligned to the reference grid.

    Parameters
    ----------
    s1_path  : path to S1 GeoTIFF (2 bands: VV, VH)
    ref_path : path to any S2 GeoTIFF whose CRS/transform/shape to match

    Returns
    -------
    vv, vh : float32 arrays with the same shape and transform as ref_path
    ref_profile : profile of the reference file
    """
    with rasterio.open(ref_path) as ref:
        ref_profile = ref.profile.copy()
        dst_crs       = ref.crs
        dst_transform = ref.transform
        dst_shape     = (ref.height, ref.width)

    with rasterio.open(s1_path) as src:
        src_vv = src.read(1).astype(np.float32)
        src_vh = src.read(2).astype(np.float32)
        src_crs       = src.crs
        src_transform = src.transform

    vv_out = np.full(dst_shape, np.nan, dtype=np.float32)
    vh_out = np.full(dst_shape, np.nan, dtype=np.float32)

    for src_arr, dst_arr in ((src_vv, vv_out), (src_vh, vh_out)):
        reproject(
            source=src_arr,
            destination=dst_arr,
            src_transform=src_transform,
            src_crs=src_crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.bilinear,
            src_nodata=np.nan,
            dst_nodata=np.nan,
        )

    # Apply nodata floor after reprojection
    vv_out[vv_out < NODATA_FLOOR_DB] = np.nan
    vh_out[vh_out < NODATA_FLOOR_DB] = np.nan

    return vv_out, vh_out, ref_profile


def save_s1_features(feature_dict: dict, ref_profile: dict, output_path: str):
    """
    Save the 6 SAR feature arrays as a 6-band GeoTIFF.

    Band order: vv_after, vh_after, cr_after, vv_diff, vh_diff, cr_diff

    Parameters
    ----------
    feature_dict : output of compute_sar_features()
    ref_profile  : rasterio profile to use for CRS/transform/shape
    output_path  : destination .tif path
    """
    band_order = ['vv_after', 'vh_after', 'cr_after', 'vv_diff', 'vh_diff', 'cr_diff']

    profile = ref_profile.copy()
    profile.update(
        count=6,
        dtype='float32',
        nodata=np.nan,
        compress='lzw',
        driver='GTiff',
    )

    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with rasterio.open(output_path, 'w', **profile) as dst:
        for i, name in enumerate(band_order, start=1):
            dst.write(feature_dict[name], i)
            dst.update_tags(i, name=name)
