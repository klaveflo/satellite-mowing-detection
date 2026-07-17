"""
Mowing-detection pipeline logic (shared, UI-agnostic).

This module holds the pure data/ML logic extracted from
`prototype_application/apply_mowing_detection.ipynb`, so that both the notebook
and the Streamlit dashboard can use the exact same implementation without
duplicating code.

Band layout (1-based) expected for the combined Sentinel-2 GeoTIFFs produced by
`convert_copernicus_folder` (and accepted directly as "pre-combined" uploads):

    1 = Blue (B02)   2 = Green (B03)   3 = Red (B04)
    4 = NIR  (B08)   5 = SWIR (B11)    6 = Cloud probability (%)
"""

from __future__ import annotations

from pathlib import Path
import re

import geopandas as gpd
import joblib
import numpy as np
import pandas as pd
from pyproj import CRS as _ProjCRS, Geod as _Geod
import rasterio
from rasterio.features import rasterize as rio_rasterize
from rasterio.warp import Resampling, reproject
from scipy.ndimage import binary_closing, binary_opening
from scipy.ndimage import label as ndimage_label

try:
    from skimage.morphology import disk as skdisk
    _DISK_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    _DISK_AVAILABLE = False


# --------------------------------------------------------------------------- #
# Band layout
# --------------------------------------------------------------------------- #

# Default 1-based band indices for the combined 6-band Sentinel-2 format.
DEFAULT_BAND_LAYOUT = {
    "blue": 1,
    "green": 2,
    "red": 3,
    "nir": 4,
    "swir": 5,
    "cloud": 6,
}


# --------------------------------------------------------------------------- #
# Vegetation indices
# --------------------------------------------------------------------------- #

def calculate_gndvi(nir, green):
    with np.errstate(divide="ignore", invalid="ignore"):
        r = (nir - green) / (nir + green)
    r[np.isinf(r)] = np.nan
    return r


def calculate_ndii(nir, swir):
    with np.errstate(divide="ignore", invalid="ignore"):
        r = (nir - swir) / (nir + swir)
    r[np.isinf(r)] = np.nan
    return r


# --------------------------------------------------------------------------- #
# Postprocessing
# --------------------------------------------------------------------------- #

def apply_median_filter(pred_map, size=3):
    """Median pre-filter to remove salt-and-pepper noise. Preserves nodata pixels."""
    from scipy.ndimage import median_filter as scipy_median_filter

    nodata_mask = (pred_map == -1)
    filtered = scipy_median_filter(
        np.where(nodata_mask, 0, pred_map).astype(np.float32), size=size)
    result = (filtered >= 0.5).astype(np.int8)
    result[nodata_mask] = -1
    return result


def apply_morphological_clean(pred_map, disk_radius=2, min_area_pixels=4):
    """
    Binary opening + closing then minimum-area filter.

    pred_map: 2-D int8 array with values {-1 (nodata), 0, 1}
    """
    nodata_mask = (pred_map == -1)
    binary = (pred_map == 1)
    struct = skdisk(disk_radius) if _DISK_AVAILABLE else np.ones(
        (2 * disk_radius + 1, 2 * disk_radius + 1), dtype=bool)
    opened = binary_opening(binary, structure=struct)
    closed = binary_closing(opened, structure=struct)
    labeled, n_comp = ndimage_label(closed)
    for cid in range(1, n_comp + 1):
        if np.sum(labeled == cid) < min_area_pixels:
            closed[labeled == cid] = False
    result = closed.astype(np.int8)
    result[nodata_mask] = -1
    return result


# --------------------------------------------------------------------------- #
# Spatial masking from vector files
# --------------------------------------------------------------------------- #

def build_vector_mask(vector_path, layer, target_crs, transform, H, W,
                      filter_column=None, filter_value=None, log=print):
    """
    Read a vector file, optionally filter rows by a column value, reproject to
    target_crs, and rasterize to a flat boolean mask on the prediction grid.

    Returns a 1-D boolean array of length H*W.
    """
    gdf = gpd.read_file(str(vector_path), layer=layer)
    log(f"Loaded {len(gdf)} features from '{Path(vector_path).name}' (CRS: {gdf.crs})")
    gdf = gdf.to_crs(target_crs)
    if filter_column is not None and filter_value is not None:
        gdf = gdf[gdf[filter_column] == filter_value]
        log(f"After filter '{filter_column}' == '{filter_value}': {len(gdf)} features")
    if len(gdf) == 0:
        log("WARNING: no features remain after filtering, mask will be all-False.")
        return np.zeros(H * W, dtype=bool)
    mask_2d = rio_rasterize(
        shapes=((geom, 1) for geom in gdf.geometry if geom is not None),
        out_shape=(H, W),
        transform=transform,
        fill=0,
        dtype="uint8",
        all_touched=False,
    )
    return mask_2d.astype(bool).flatten()


# --------------------------------------------------------------------------- #
# Copernicus Browser folder -> combined 6-band GeoTIFF
# --------------------------------------------------------------------------- #

def convert_copernicus_folder(folder_path, out_dir, label="", log=print):
    """
    Convert a Copernicus Browser L2A download folder (one .tif/.jp2 per band)
    into a combined 6-band GeoTIFF and write it to `out_dir`.

    Returns the path to the written GeoTIFF.
    """
    folder = Path(folder_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    band_keywords = {
        "B02": "_B02_", "B03": "_B03_", "B04": "_B04_",
        "B08": "_B08_", "B11": "_B11_", "SCL": "_Scene_classification_map_",
    }

    # Gather .tif/.tiff/.jp2 regardless of case, recursively.
    all_files = []
    for ext in ["*.tif", "*.tiff", "*.jp2", "*.TIF", "*.TIFF", "*.JP2"]:
        all_files.extend(list(folder.rglob(ext)))

    found_bands = {}
    for band_key, keyword in band_keywords.items():
        matches = [f for f in all_files if keyword in f.name]
        if matches:
            found_bands[band_key] = matches[0]

    missing = [k for k in band_keywords if k not in found_bands]
    if missing:
        raise FileNotFoundError(
            f"Missing required band(s) in '{folder}': {missing}. "
            f"Found: {sorted(found_bands.keys())}. "
            "Make sure the download includes bands B02, B03, B04, B08, B11 and "
            "the Scene classification map (SCL)."
        )

    # Extract acquisition date from the B02 filename.
    dashed_match = re.search(r"(\d{4}-\d{2}-\d{2})", found_bands["B02"].name)
    if dashed_match:
        date_str = dashed_match.group(1)
    else:
        fallback_match = re.search(r"(\d{8})", found_bands["B02"].name)
        if fallback_match:
            raw = fallback_match.group(1)
            date_str = f"{raw[:4]}-{raw[4:6]}-{raw[6:8]}"
        else:
            raise ValueError(
                f"Could not extract an acquisition date from filename: "
                f"{found_bands['B02'].name}"
            )

    # B02 is the 10 m reference grid.
    with rasterio.open(str(found_bands["B02"])) as ref_src:
        ref_crs = ref_src.crs
        ref_transform = ref_src.transform
        ref_h, ref_w = ref_src.height, ref_src.width

    def _read_resampled(path, method):
        with rasterio.open(str(path)) as src:
            if src.height == ref_h and src.width == ref_w:
                return src.read(1).astype(np.float32)
            out = np.empty((ref_h, ref_w), dtype=np.float32)
            reproject(
                source=src.read(1).astype(np.float32),
                destination=out,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=ref_transform,
                dst_crs=ref_crs,
                resampling=method,
            )
            return out

    with rasterio.open(str(found_bands["B02"])) as s:
        b02 = s.read(1).astype(np.float32)
    with rasterio.open(str(found_bands["B03"])) as s:
        b03 = s.read(1).astype(np.float32)
    with rasterio.open(str(found_bands["B04"])) as s:
        b04 = s.read(1).astype(np.float32)
    with rasterio.open(str(found_bands["B08"])) as s:
        b08 = s.read(1).astype(np.float32)
    b11 = _read_resampled(found_bands["B11"], Resampling.bilinear)
    scl = _read_resampled(found_bands["SCL"], Resampling.nearest)

    # SCL cloud classes: 8=cloud medium, 9=cloud high, 10=thin cirrus -> 100 %.
    cloud_prob = np.where(
        np.isin(scl.astype(np.int16), [8, 9, 10]), 100, 0).astype(np.float32)

    # Band order matches DEFAULT_BAND_LAYOUT.
    stack = np.stack([b02, b03, b04, b08, b11, cloud_prob], axis=0)

    out_path = out_dir / f"{date_str}.tif"
    with rasterio.open(str(out_path), "w", driver="GTiff", dtype="float32",
                       width=ref_w, height=ref_h, count=6,
                       crs=ref_crs, transform=ref_transform, compress="lzw") as dst:
        dst.write(stack)

    log(f"Converted {label or folder.name} -> {out_path.name} "
        f"(6 bands, {ref_w}x{ref_h}, CRS: {ref_crs})")
    return out_path


# --------------------------------------------------------------------------- #
# Feature computation
# --------------------------------------------------------------------------- #

def compute_features(before_path, after_path, band_layout=None, cloud_threshold=30):
    """
    Load a before/after Sentinel-2 pair and compute the features required by the
    s2_only models. Verifies both images share the same CRS.

    Returns
    -------
    feature_df : DataFrame with all model feature columns (NaN for invalid pixels)
    valid_mask : 1-D bool array (H*W) - non-cloudy, non-NaN pixels
    cloud_mask : 1-D bool array (H*W) - True where cloud prob exceeds threshold
    meta       : rasterio metadata dict (CRS, transform, shape)
    H, W       : scene height and width
    """
    bl = band_layout or DEFAULT_BAND_LAYOUT

    with rasterio.open(str(before_path)) as src_b, \
         rasterio.open(str(after_path)) as src_a:

        if src_b.crs != src_a.crs:
            raise ValueError(
                f"CRS mismatch between before ({src_b.crs}) and after "
                f"({src_a.crs}) images. Reproject them to the same CRS first."
            )
        if src_b.width != src_a.width or src_b.height != src_a.height:
            raise ValueError(
                f"Image size mismatch: before is {src_b.width}x{src_b.height}, "
                f"after is {src_a.width}x{src_a.height}. Both images must cover "
                "the same area on the same grid."
            )

        meta = src_b.meta.copy()
        H, W = meta["height"], meta["width"]

        def _rd(src, key, scale=10000.0):
            return src.read(bl[key]).astype(float) / scale

        green_b = _rd(src_b, "green")
        nir_b = _rd(src_b, "nir")
        swir_b = _rd(src_b, "swir")
        cloud_b = src_b.read(bl["cloud"]).astype(float)

        green_a = _rd(src_a, "green")
        nir_a = _rd(src_a, "nir")
        swir_a = _rd(src_a, "swir")
        cloud_a = src_a.read(bl["cloud"]).astype(float)

    ndii_b = calculate_ndii(nir_b, swir_b)
    ndii_a = calculate_ndii(nir_a, swir_a)
    gndvi_b = calculate_gndvi(nir_b, green_b)
    gndvi_a = calculate_gndvi(nir_a, green_a)

    feature_df = pd.DataFrame({
        "ndii_diff": (ndii_a - ndii_b).flatten(),
        "gndvi_diff": (gndvi_a - gndvi_b).flatten(),
        "swir_diff": (swir_a - swir_b).flatten(),
        "ndii_after": ndii_a.flatten(),
        "swir_after": swir_a.flatten(),
    })

    cloud_mask = ((cloud_b > cloud_threshold) | (cloud_a > cloud_threshold)).flatten()
    nan_mask = np.isnan(feature_df.values).any(axis=1)
    valid_mask = ~cloud_mask & ~nan_mask

    return feature_df, valid_mask, cloud_mask, meta, H, W


# --------------------------------------------------------------------------- #
# Model + RGB helpers
# --------------------------------------------------------------------------- #

def load_model(model_path):
    """Load a bundled model dict: {model, features, threshold}."""
    obj = joblib.load(model_path)
    return obj["model"], obj["features"], obj["threshold"]


def read_rgb(path, band_layout=None):
    """Read R/G/B bands, normalise to 98th-percentile ceiling -> (H,W,3) in [0,1]."""
    bl = band_layout or DEFAULT_BAND_LAYOUT
    with rasterio.open(str(path)) as src:
        rgb = src.read([bl["red"], bl["green"], bl["blue"]]).astype(np.float32)
    rgb = np.moveaxis(rgb, 0, -1)
    p98 = np.percentile(rgb, 98)
    return np.clip(rgb, 0, max(p98, 1e-9)) / max(p98, 1e-9)


def pixel_dimensions_m(meta):
    """Return (pixel_x_m, pixel_y_m, pixel_area_m2) accounting for geographic CRS."""
    crs = meta["crs"]
    transform = meta["transform"]
    H, W = meta["height"], meta["width"]
    crs_obj = _ProjCRS.from_user_input(crs)
    if crs_obj.is_geographic:
        geod = _Geod(ellps="WGS84")
        cx = transform.c + transform.a * W / 2
        cy = transform.f + transform.e * H / 2
        _, _, dx = geod.inv(cx, cy, cx + transform.a, cy)
        _, _, dy = geod.inv(cx, cy, cx, cy + transform.e)
        px_x, px_y = abs(dx), abs(dy)
    else:
        px_x, px_y = abs(transform.a), abs(transform.e)
    return px_x, px_y, px_x * px_y


# --------------------------------------------------------------------------- #
# Prediction
# --------------------------------------------------------------------------- #

def run_prediction(feature_df, valid_mask, model, features, threshold, H, W,
                   apply_postprocessing=True, pp_median_size=3,
                   pp_disk_radius=1, pp_min_area_pixels=4):
    """
    Run the model over valid pixels and (optionally) postprocess.

    Returns (pred_map_pp, pred_map_raw, proba_flat) where pred maps are 2-D int8
    arrays with values {-1 nodata, 0 not mowed, 1 mowed}.
    """
    proba_flat = np.full(H * W, np.nan, dtype=np.float32)
    proba_flat[valid_mask] = model.predict_proba(
        feature_df.loc[valid_mask, features])[:, 1]

    pred_flat = np.full(H * W, -1, dtype=np.int8)
    pred_flat[valid_mask] = (proba_flat[valid_mask] >= threshold).astype(np.int8)
    pred_map_raw = pred_flat.reshape(H, W)

    if apply_postprocessing:
        pred_map_pp = pred_map_raw.copy()
        if pp_median_size and pp_median_size > 0:
            pred_map_pp = apply_median_filter(pred_map_pp, size=pp_median_size)
        pred_map_pp = apply_morphological_clean(
            pred_map_pp, disk_radius=pp_disk_radius,
            min_area_pixels=pp_min_area_pixels)
    else:
        pred_map_pp = pred_map_raw.copy()

    return pred_map_pp, pred_map_raw, proba_flat


def save_prediction_geotiff(pred_map_pp, meta, out_path, tags=None):
    """Write prediction as a single-band uint8 GeoTIFF (1=mowed,0=not,255=nodata)."""
    out_meta = meta.copy()
    out_meta.update({"count": 1, "dtype": "uint8", "nodata": 255, "compress": "lzw"})
    out_arr = pred_map_pp.astype(np.uint8)
    out_arr[pred_map_pp == -1] = 255
    with rasterio.open(str(out_path), "w", **out_meta) as dst:
        dst.write(out_arr, 1)
        if tags:
            dst.update_tags(**{k: str(v) for k, v in tags.items()})
    return out_path
