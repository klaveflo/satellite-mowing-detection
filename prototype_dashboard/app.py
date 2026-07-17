"""
Mowing Detection: Prototype Dashboard
======================================

This is a Streamlit front-end for the standalone mowing-detection pipeline. Users upload a
before/after Sentinel-2 pair (pre-combined GeoTIFFs or raw Copernicus Browser
downloads), optionally add AOI / grassland / ground-truth files, pick a model,
run detection, view results and download the outputs.

Run locally with:   streamlit run app.py
"""

from __future__ import annotations

import io
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
from rasterio.features import rasterize as rio_rasterize
from sklearn.metrics import f1_score, precision_score, recall_score

import mowing_pipeline as mp

# --------------------------------------------------------------------------- #
# Page config & constants
# --------------------------------------------------------------------------- #

st.set_page_config(
    page_title="Mowing Detection Dashboard",
    page_icon="🌱",
    layout="wide",
)

APP_DIR = Path(__file__).resolve().parent

# Where to look for the bundled .joblib models.
MODEL_SEARCH_DIRS = [
    APP_DIR / "standalone_models"
]

MODEL_DESCRIPTIONS = {
    "s2_only_diff_svm.joblib": "SVM · diff features · recommended",
    "s2_only_diff_rf.joblib": "Random Forest · diff features",
    "s2_only_best_svm.joblib": "SVM · extended features",
    "s2_only_best_rf.joblib": "Random Forest · extended features",
}
RECOMMENDED_MODEL = "s2_only_diff_svm.joblib"


def find_model_dir() -> Path | None:
    for d in MODEL_SEARCH_DIRS:
        if d.is_dir() and any(d.glob("*.joblib")):
            return d
    return None


@st.cache_resource(show_spinner=False)
def load_detection_model(model_path: str):
    """Cache the bundled model object so reruns do not reload joblib files."""
    return mp.load_model(Path(model_path))


# --------------------------------------------------------------------------- #
# Session-scoped temp workspace (auto-discarded when the session/server ends)
# --------------------------------------------------------------------------- #

def get_workspace() -> Path:
    if "workspace" not in st.session_state:
        st.session_state.workspace = tempfile.mkdtemp(prefix="mowing_dash_")
    return Path(st.session_state.workspace)


def save_upload(uploaded_file, dest_dir: Path) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    out = dest_dir / uploaded_file.name
    out.write_bytes(uploaded_file.getbuffer())
    return out


def extract_to_dir(uploaded_zip, dest_dir: Path) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(uploaded_zip.getbuffer())) as zf:
        zf.extractall(dest_dir)
    return dest_dir


# --------------------------------------------------------------------------- #
# Header & instructions
# --------------------------------------------------------------------------- #

st.title("Mowing Detection Dashboard")
st.caption(
    "Detect grassland mowing events between two Sentinel-2 acquisitions using a "
    "pre-trained model. This is a prototype: Uploads and results are temporary and are "
    "discarded when the session ends."
)

with st.expander("ℹ️  How to use this dashboard", expanded=True):
    st.markdown(
        """
**What you need:** two Sentinel-2 acquisitions of the same exact area: one **before**
and one **after** a suspected mowing event (a few days apart, both cloud-free).

**Two ways to provide the images** (choose in the sidebar):

1. **Pre-combined GeoTIFFs:** one multi-band `.tif` per date, in the 6-band
   layout `Blue, Green, Red, NIR, SWIR, Cloud Probability` (e.g. exported from this pipeline).
   In case you use different band layouts, you can adjust the indices in the **Advanced settings** in the sidebar.
2. **Raw Copernicus Browser download:** for each date, upload the download
   `.zip` (or multi-select the individual band files). The pipeline combines the
   bands for you. Requires bands **B02, B03, B04, B08, B11** and the
   **Scene classification map**.

> Don't have images yet? Download two cloud-free scenes from the
> [Copernicus Browser](https://browser.dataspace.copernicus.eu).
> 
> Download steps:
> - Open the download window and switch to the **Analytical** tab.
> - Set **Image format** to **TIFF (16-bit)**.
> - Set **Image resolution** to **HIGH**.
> - Make sure **Clip extra bands** is turned **off**.
> - Under the band selection, tick the five raw bands **B02, B03, B04, B08, B11** and the **Scene classification map**.
> - Once the selection is correct, click **Download**.

**Optional inputs** (all safe to leave empty):
- **Study area (AOI):** restrict analysis to a boundary polygon.
- **Grassland mask:** restrict analysis to grass-covered areas.
- **Ground truth:** known mowing polygons, to score accuracy (precision / recall / F1).

**Then:** pick a model → click **Run detection** → view the maps and statistics →
download the prediction GeoTIFF and the summary CSV.

*Vector files:* GeoPackage (`.gpkg`) is easiest (single file). For Shapefiles,
upload a `.zip` containing the `.shp/.shx/.dbf/.prj` set.
        """
    )

# --------------------------------------------------------------------------- #
# Sidebar – inputs
# --------------------------------------------------------------------------- #

workspace = get_workspace()
sb = st.sidebar
sb.header("1. Sentinel-2 images")

input_mode = sb.radio(
    "Input type",
    ["Pre-combined GeoTIFFs", "Raw Copernicus download"],
    help="How your before/after images are provided.",
)

if input_mode == "Pre-combined GeoTIFFs":
    up_before = sb.file_uploader("Before image (.tif)", type=["tif", "tiff"],
                                 key="before_tif")
    up_after = sb.file_uploader("After image (.tif)", type=["tif", "tiff"],
                                key="after_tif")
    cop_before = cop_after = None
else:
    cop_before = sb.file_uploader(
        "Before Copernicus .zip or band files",
        type=["zip", "tif", "tiff", "jp2"], accept_multiple_files=True,
        key="before_cop")
    cop_after = sb.file_uploader(
        "After Copernicus .zip or band files",
        type=["zip", "tif", "tiff", "jp2"], accept_multiple_files=True,
        key="after_cop")
    up_before = up_after = None

# --- Model ---------------------------------------------------------------- #
sb.header("2. Model")
model_dir = find_model_dir()
if model_dir is None:
    sb.error("No models found. Expected .joblib files in a 'standalone_models' folder.")
    model_choice = None
else:
    available = sorted(p.name for p in model_dir.glob("*.joblib"))
    default_idx = available.index(RECOMMENDED_MODEL) if RECOMMENDED_MODEL in available else 0
    model_choice = sb.selectbox(
        "Detection model",
        available,
        index=default_idx,
        format_func=lambda n: f"{n}  ({MODEL_DESCRIPTIONS.get(n, 'model')})",
    )
    _, _, model_threshold_default = load_detection_model(str(model_dir / model_choice))

# --- Optional inputs ------------------------------------------------------ #
sb.header("3. Optional inputs")

with sb.expander("Study area (AOI)"):
    up_aoi = st.file_uploader("AOI polygon (.gpkg / zipped .shp)",
                              type=["gpkg", "zip"], key="aoi")
    aoi_layer = st.text_input("Layer name (optional)", key="aoi_layer") or None
    aoi_col = st.text_input("Filter column (optional)", key="aoi_col") or None
    aoi_val = st.text_input("Keep value (optional)", key="aoi_val") or None

with sb.expander("Grassland mask"):
    up_grass = st.file_uploader("Grassland polygons (.gpkg / zipped .shp)",
                                type=["gpkg", "zip"], key="grass")
    grass_layer = st.text_input("Layer name (optional)", key="grass_layer") or None
    grass_col = st.text_input("Filter column (optional)", key="grass_col") or None
    grass_val = st.text_input("Keep value (optional)", key="grass_val") or None

with sb.expander("Ground truth mask"):
    up_gt = st.file_uploader("Ground-truth polygons (.gpkg / zipped .shp)",
                             type=["gpkg", "zip"], key="gt")
    gt_layer = st.text_input("Layer name (optional)", key="gt_layer") or None
    gt_date_col = st.text_input(
        "Date column (optional)", key="gt_date_col") or None

# --- Advanced ------------------------------------------------------------- #
with sb.expander("Advanced settings"):
    cloud_threshold = st.slider("Cloud probability threshold (%)", 0, 100, 30)
    apply_pp = st.checkbox("Apply postprocessing", value=True)
    if model_choice is not None:
        decision_threshold = st.number_input(
            "Decision threshold",
            min_value=0.10,
            max_value=0.95,
            value=float(np.clip(model_threshold_default, 0.10, 0.95)),
            step=0.01,
            format="%.2f",
            help="Default is the threshold stored in the selected model. Change it only if you want to experiment.",
        )
    else:
        decision_threshold = 0.90
    pp_median = st.select_slider("Median filter window", [0, 3, 5, 7], value=3)
    pp_disk = st.number_input("Morphology disk radius (px)", 0, 10, 1)
    pp_min_area = st.number_input("Min. patch area (px)", 0, 100, 4)
    st.caption("Band layout (1-based) of the combined images:")
    b_blue = st.number_input("Blue (B02)", 1, 20, 1)
    b_green = st.number_input("Green (B03)", 1, 20, 2)
    b_red = st.number_input("Red (B04)", 1, 20, 3)
    b_nir = st.number_input("NIR (B08)", 1, 20, 4)
    b_swir = st.number_input("SWIR (B11)", 1, 20, 5)
    b_cloud = st.number_input("Cloud Probability", 1, 20, 6)

band_layout = {"blue": b_blue, "green": b_green, "red": b_red,
               "nir": b_nir, "swir": b_swir, "cloud": b_cloud}

run = sb.button("Run detection", type="primary", width="stretch")


# --------------------------------------------------------------------------- #
# Helpers to resolve uploads into on-disk paths
# --------------------------------------------------------------------------- #

def resolve_combined_image(uploaded, cop_uploads, label, ws) -> Path:
    """Return a path to a combined GeoTIFF for one date, from either input mode."""
    if input_mode == "Pre-combined GeoTIFFs":
        if uploaded is None:
            raise ValueError(f"No {label} image uploaded.")
        return save_upload(uploaded, ws / label)

    # Copernicus mode: files may be a zip or a set of band files.
    if not cop_uploads:
        raise ValueError(f"No {label} Copernicus files uploaded.")
    folder = ws / f"cop_{label}"
    folder.mkdir(parents=True, exist_ok=True)
    for uf in cop_uploads:
        if uf.name.lower().endswith(".zip"):
            extract_to_dir(uf, folder)
        else:
            save_upload(uf, folder)
    return mp.convert_copernicus_folder(folder, ws / "combined", label=label, log=log_msg)


def resolve_vector(uploaded, ws, label) -> Path | None:
    """Save an uploaded vector file (or unzip a shapefile) and return its path."""
    if uploaded is None:
        return None
    dest = ws / f"vec_{label}"
    if uploaded.name.lower().endswith(".zip"):
        extract_to_dir(uploaded, dest)
        shp = list(dest.rglob("*.shp"))
        if not shp:
            raise ValueError(f"No .shp found in the uploaded {label} zip.")
        return shp[0]
    return save_upload(uploaded, dest)


# --------------------------------------------------------------------------- #
# Logging sink shown in an expander
# --------------------------------------------------------------------------- #

_log_lines: list[str] = []


def log_msg(msg: str):
    _log_lines.append(str(msg))


def parse_required_scene_date(stem: str, label: str):
    """Parse scene dates from filenames; require YYYY-MM-DD in the stem."""
    try:
        return datetime.strptime(stem, "%Y-%m-%d").date()
    except ValueError as exc:
        raise ValueError(
            f"Could not parse the {label} image date from '{stem}'. "
            "The dashboard expects a filename stem in YYYY-MM-DD format, for example '2019-08-27'."
        ) from exc


def parse_gt_dates(series: pd.Series, column_name: str):
    """Parse GT dates robustly and raise a clear error if values do not match."""
    parsed = pd.to_datetime(series, errors="coerce", cache=True)
    invalid_mask = series.notna() & parsed.isna()
    if invalid_mask.any():
        examples = series[invalid_mask].astype(str).head(5).tolist()
        raise ValueError(
            f"Could not parse some values in the '{column_name}' date column. "
            "Please use a date format that pandas can convert reliably, such as YYYY-MM-DD "
            "or an ISO datetime string (for example '2019-08-27' or '2019-08-27 00:00:00'). "
            f"Examples of unparseable values: {examples}"
        )
    return parsed.dt.date


# --------------------------------------------------------------------------- #
# Visualisation helpers
# --------------------------------------------------------------------------- #

def fig_rgb_comparison(before_path, after_path, valid_mask, cloud_mask, H, W,
                       scene_crs, scene_transform, has_spatial, gt_path, gt_layer):
    rgb_before = mp.read_rgb(before_path, band_layout)
    rgb_after = mp.read_rgb(after_path, band_layout)

    fig, axes = plt.subplots(1, 2, figsize=(6, 4))
    axes[0].imshow(rgb_before)
    axes[0].set_title(f"Before: {Path(before_path).stem}", fontsize=8)
    axes[0].axis("off")
    axes[1].imshow(rgb_after)
    axes[1].set_title(f"After: {Path(after_path).stem}", fontsize=8)
    axes[1].axis("off")
    fig.suptitle(f"Sentinel-2 RGB  |  {Path(before_path).stem}  →  {Path(after_path).stem}",
                 fontsize=9, fontweight="bold", y=1.0)
    fig.tight_layout(pad=0.7)
    return fig, rgb_after


def fig_optional_overlays(after_path, cloud_mask, H, W, scene_crs, scene_transform,
                          aoi_mask, grass_mask, gt_path, gt_layer, gt_date_col,
                          before_stem, after_stem):
    rgb_after = mp.read_rgb(after_path, band_layout)

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.imshow(rgb_after)

    handles = []

    if cloud_mask is not None and cloud_mask.any():
        cloud_2d = cloud_mask.reshape(H, W)
        cloud_rgba = np.zeros((H, W, 4), dtype=np.uint8)
        cloud_rgba[cloud_2d] = [242, 190, 25, 130]
        ax.imshow(cloud_rgba)
        handles.append(mpatches.Patch(color=(0.95, 0.75, 0.1, 0.5), label="Cloud-masked pixels"))

    if aoi_mask is not None and aoi_mask.any():
        aoi_2d = aoi_mask.reshape(H, W).astype(float)
        ax.contour(aoi_2d, levels=[0.5], colors="#FFD400", linewidths=1.2)
        handles.append(Line2D([0], [0], color="#FFD400", linewidth=1.2, label="AOI boundary"))

    if grass_mask is not None and grass_mask.any():
        grass_2d = grass_mask.reshape(H, W).astype(bool)
        grass_rgba = np.zeros((H, W, 4), dtype=np.uint8)
        grass_rgba[grass_2d] = [72, 181, 255, 60]
        ax.imshow(grass_rgba)
        handles.append(mpatches.Patch(color=(0.28, 0.71, 1.0, 0.28), label="Grassland area"))

    if gt_path is not None:
        gt_viz = gpd.read_file(str(gt_path), layer=gt_layer).to_crs(scene_crs)
        if gt_date_col is not None and gt_date_col in gt_viz.columns:
            d_before = parse_required_scene_date(before_stem, "before")
            d_after = parse_required_scene_date(after_stem, "after")
            gt_viz = gt_viz.assign(_parsed_date=parse_gt_dates(gt_viz[gt_date_col], gt_date_col))
            gt_viz = gt_viz[(gt_viz["_parsed_date"].notna()) &
                            (gt_viz["_parsed_date"] > d_before) &
                            (gt_viz["_parsed_date"] <= d_after)]
        gt_mask = rio_rasterize(
            shapes=((g, 1) for g in gt_viz.geometry if g is not None),
            out_shape=(H, W), transform=scene_transform, fill=0, dtype="uint8"
        ).astype(float)

        gt_rgba = np.zeros((H, W, 4), dtype=np.uint8)
        gt_rgba[gt_mask > 0] = [0, 229, 255, 90]  # bright cyan, semi-transparent
        ax.imshow(gt_rgba)

        ax.contour(gt_mask, levels=[0.5], colors="#00E5FF", linewidths=1.2)
        handles.append(mpatches.Patch(color=(0.0, 0.90, 1.0, 0.35), label="Ground truth mowing"))

    if handles:
        ax.legend(handles=handles, loc="lower right", fontsize=6, framealpha=0.6)
    ax.set_title(f"Optional overlays  |  {Path(after_path).stem}", fontsize=8, fontweight="bold")
    ax.axis("off")
    fig.tight_layout(pad=0.6)
    return fig


def fig_prediction(rgb_after, pred_map_pp, H, W, before_stem, after_stem,
                   model_name, threshold, pp_disk, pp_min_area, pp_median):
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.imshow(rgb_after)
    mowing_rgba = np.zeros((H, W, 4), dtype=np.uint8)
    mowing_rgba[pred_map_pp == 1] = [0, 204, 255, 180]
    ax.imshow(mowing_rgba)
    ax.set_title(
        f"Mowing Detection: {before_stem} → {after_stem}\n"
        f"Model: {model_name}  |  Threshold: {threshold:.3f}  |  "
        f"PP: disk_r={pp_disk}, min_area={pp_min_area}px, median={pp_median}",
        fontsize=8, pad=5)
    ax.axis("off")
    ax.legend(handles=[mpatches.Patch(color="#00CCFF", label="Detected mowing")],
              loc="lower right", fontsize=6)
    fig.tight_layout(pad=0.5)
    return fig


def evaluate_ground_truth(gt_path, gt_layer, gt_date_col, scene_crs, scene_transform,
                          H, W, valid_mask, pred_map_pp, rgb_after,
                          before_stem, after_stem):
    """Returns (metrics_dict, figure) or (None, None) if no polygons remain."""
    gt_gdf = gpd.read_file(str(gt_path), layer=gt_layer)
    log_msg(f"Loaded {len(gt_gdf)} ground-truth features (CRS: {gt_gdf.crs})")
    gt_gdf = gt_gdf.to_crs(scene_crs)

    if gt_date_col is not None and gt_date_col in gt_gdf.columns:
        try:
            d_before = parse_required_scene_date(before_stem, "before")
            d_after = parse_required_scene_date(after_stem, "after")
            gt_gdf = gt_gdf.assign(_parsed_date=parse_gt_dates(gt_gdf[gt_date_col], gt_date_col))
            n0 = len(gt_gdf)
            gt_gdf = gt_gdf[(gt_gdf["_parsed_date"].notna()) &
                            (gt_gdf["_parsed_date"] > d_before) &
                            (gt_gdf["_parsed_date"] <= d_after)]
            log_msg(f"Date filter {d_before} < date <= {d_after}: {len(gt_gdf)} of {n0} kept")
        except ValueError:
            raise

    if len(gt_gdf) == 0:
        return None, None

    gt_mask_2d = rio_rasterize(
        shapes=((g, 1) for g in gt_gdf.geometry if g is not None),
        out_shape=(H, W), transform=scene_transform, fill=0, dtype="uint8")
    gt_flat = gt_mask_2d.flatten().astype(bool)
    pred_flat = pred_map_pp.flatten()

    y_true = gt_flat[valid_mask].astype(int)
    y_pred = (pred_flat[valid_mask] == 1).astype(int)

    metrics = {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "gt_mowed_px": int(gt_flat[valid_mask].sum()),
        "pred_mowed_px": int(y_pred.sum()),
    }

    category = np.full(H * W, np.nan, dtype=float)
    tp = valid_mask & (pred_flat == 1) & gt_flat
    fp = valid_mask & (pred_flat == 1) & ~gt_flat
    fn = valid_mask & (pred_flat != 1) & gt_flat
    category[tp] = 1    # TP
    category[fp] = 2    # FP
    category[fn] = 3    # FN
    category_map = category.reshape(H, W)

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.imshow(rgb_after)
    ax.imshow(category_map, cmap=ListedColormap(["#00CCFF", "#FF9900", "#FF00FF"]),
              vmin=0.5, vmax=3.5, alpha=0.75)
    patches = [
        mpatches.Patch(color="#00CCFF", label=f"TP ({int((category==1).sum()):,})"),
        mpatches.Patch(color="#FF9900", label=f"FP ({int((category==2).sum()):,})"),
        mpatches.Patch(color="#FF00FF", label=f"FN ({int((category==3).sum()):,})"),
    ]
    ax.legend(handles=patches, loc="lower right", fontsize=6)
    ax.set_title(
        f"Ground-truth comparison  |  {before_stem} → {after_stem}\n"
        f"F1={metrics['f1']:.3f}  |  Precision={metrics['precision']:.3f}  |  "
        f"Recall={metrics['recall']:.3f}",
        fontsize=8, pad=5)
    ax.axis("off")
    fig.tight_layout(pad=0.6)
    return metrics, fig


# --------------------------------------------------------------------------- #
# Run pipeline
# --------------------------------------------------------------------------- #

if run:
    _log_lines.clear()
    try:
        with st.spinner("Running mowing detection…"):
            # 1. Resolve images
            before_path = resolve_combined_image(up_before, cop_before, "before", workspace)
            after_path = resolve_combined_image(up_after, cop_after, "after", workspace)

            # 2. Resolve optional vectors
            aoi_path = resolve_vector(up_aoi, workspace, "aoi")
            grass_path = resolve_vector(up_grass, workspace, "grass")
            gt_path = resolve_vector(up_gt, workspace, "gt")

            # 3. Model
            model, features, model_threshold = load_detection_model(str(model_dir / model_choice))
            threshold = float(decision_threshold)
            log_msg(
                f"Model {model_choice}: features={features}, model_threshold={model_threshold:.3f}, "
                f"decision_threshold={threshold:.3f}"
            )

            # 4. Features + cloud mask
            feature_df, valid_mask, cloud_mask, meta, H, W = mp.compute_features(
                before_path, after_path, band_layout=band_layout,
                cloud_threshold=cloud_threshold)
            scene_crs = meta["crs"]
            scene_transform = meta["transform"]
            n_total = H * W
            log_msg(f"Scene {H}x{W} ({n_total:,} px), CRS {scene_crs}; "
                    f"cloud-free {int(valid_mask.sum()):,} px")
            if valid_mask.sum() == 0:
                raise RuntimeError("No cloud-free pixels. Try raising the cloud threshold.")

            # 5. Spatial masks
            has_spatial = False
            aoi_mask = None
            if aoi_path is not None:
                sa = mp.build_vector_mask(aoi_path, aoi_layer, scene_crs,
                                          scene_transform, H, W,
                                          filter_column=aoi_col,
                                          filter_value=aoi_val, log=log_msg)
                aoi_mask = sa
                valid_mask = valid_mask & sa
                has_spatial = True
            grass_mask = None
            if grass_path is not None:
                gm = mp.build_vector_mask(grass_path, grass_layer, scene_crs,
                                          scene_transform, H, W,
                                          filter_column=grass_col,
                                          filter_value=grass_val, log=log_msg)
                grass_mask = gm
                valid_mask = valid_mask & gm
                has_spatial = True
            n_valid = int(valid_mask.sum())
            if n_valid == 0:
                raise RuntimeError("No valid pixels after spatial filtering. Check AOI/grassland inputs.")

            # 6. Predict
            pred_map_pp, pred_map_raw, _ = mp.run_prediction(
                feature_df, valid_mask, model, features, threshold, H, W,
                apply_postprocessing=apply_pp, pp_median_size=pp_median,
                pp_disk_radius=int(pp_disk), pp_min_area_pixels=int(pp_min_area))
            n_mowed = int((pred_map_pp == 1).sum())

            # 7. Metrics for area
            px_x, px_y, px_area = mp.pixel_dimensions_m(meta)
            mowed_ha = n_mowed * px_area / 10_000
            valid_ha = n_valid * px_area / 10_000
            scene_ha = n_total * px_area / 10_000

            # 8. Figures
            fig_rgb, rgb_after = fig_rgb_comparison(
                before_path, after_path, valid_mask, cloud_mask, H, W,
                scene_crs, scene_transform, has_spatial, gt_path, gt_layer)
            fig_optional = None
            if any(mask is not None for mask in (aoi_mask, grass_mask, gt_path)):
                fig_optional = fig_optional_overlays(
                    after_path, cloud_mask, H, W, scene_crs, scene_transform,
                    aoi_mask, grass_mask, gt_path, gt_layer, gt_date_col,
                    Path(before_path).stem, Path(after_path).stem)
            fig_pred = fig_prediction(
                rgb_after, pred_map_pp, H, W, Path(before_path).stem,
                Path(after_path).stem, Path(model_choice).stem, threshold,
                int(pp_disk), int(pp_min_area), pp_median)

            # 9. Ground truth
            gt_metrics, fig_gt = (None, None)
            if gt_path is not None:
                gt_metrics, fig_gt = evaluate_ground_truth(
                    gt_path, gt_layer, gt_date_col, scene_crs, scene_transform,
                    H, W, valid_mask, pred_map_pp, rgb_after,
                    Path(before_path).stem, Path(after_path).stem)

            # 10. Save GeoTIFF
            geotiff_path = workspace / f"mowing_detection_{Path(after_path).stem}.tif"
            mp.save_prediction_geotiff(
                pred_map_pp, meta, geotiff_path,
                tags={
                    "BEFORE_IMAGE": Path(before_path).name,
                    "AFTER_IMAGE": Path(after_path).name,
                    "MODEL": model_choice,
                    "THRESHOLD": round(threshold, 6),
                    "CREATED": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                })

        # ---- store results in session for display / download ---- #
        st.session_state.results = {
            "fig_rgb": fig_rgb,
            "fig_optional": fig_optional,
            "fig_pred": fig_pred,
            "fig_gt": fig_gt,
            "gt_metrics": gt_metrics,
            "geotiff_path": str(geotiff_path),
            "after_stem": Path(after_path).stem,
            "before_stem": Path(before_path).stem,
            "summary": {
                "before_image": Path(before_path).name,
                "after_image": Path(after_path).name,
                "model": model_choice,
                "threshold": round(threshold, 4),
                "crs": str(scene_crs),
                "pixel_area_m2": round(px_area, 1),
                "scene_area_ha": round(scene_ha, 2),
                "valid_area_ha": round(valid_ha, 2),
                "mowed_area_ha": round(mowed_ha, 2),
                "mowed_pixels": n_mowed,
                "valid_pixels": n_valid,
                "coverage_pct": round(100 * n_mowed / n_valid, 2),
            },
            "log": list(_log_lines),
        }
        st.success("Detection complete.")
    except Exception as exc:  # surface friendly errors to non-technical users
        st.error(f"❌ {exc}")
        if _log_lines:
            with st.expander("Processing log"):
                st.text("\n".join(_log_lines))
        st.stop()


# --------------------------------------------------------------------------- #
# Results display
# --------------------------------------------------------------------------- #

if "results" in st.session_state:
    r = st.session_state.results
    s = r["summary"]

    st.header("Results")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Detected mowing", f"{s['mowed_area_ha']:.2f} ha")
    m2.metric("Coverage of valid area", f"{s['coverage_pct']:.2f} %")
    m3.metric("Valid area", f"{s['valid_area_ha']:.1f} ha")
    m4.metric("Scene area", f"{s['scene_area_ha']:.1f} ha")

    if r["gt_metrics"]:
        g = r["gt_metrics"]
        st.subheader("Accuracy vs. ground truth")
        c1, c2, c3 = st.columns(3)
        c1.metric("Precision", f"{g['precision']:.3f}")
        c2.metric("Recall", f"{g['recall']:.3f}")
        c3.metric("F1 score", f"{g['f1']:.3f}")

    st.subheader("Before / After RGB")
    st.pyplot(r["fig_rgb"], width="content")

    if r.get("fig_optional") is not None:
        st.subheader("Optional overlays preview")
        st.pyplot(r["fig_optional"], width="content")

    st.subheader("Prediction map")
    st.pyplot(r["fig_pred"], width="content")

    if r["fig_gt"] is not None:
        st.subheader("Ground-truth comparison (TP / FP / FN)")
        st.pyplot(r["fig_gt"], width="content")

    # ---- Summary table + downloads ---- #
    st.subheader("Summary & downloads")
    summary_rows = {key: str(value) for key, value in s.items()}
    if r["gt_metrics"]:
        summary_rows.update({
            "precision": f"{r['gt_metrics']['precision']:.4f}",
            "recall": f"{r['gt_metrics']['recall']:.4f}",
            "f1_score": f"{r['gt_metrics']['f1']:.4f}",
            "gt_mowed_pixels": str(r["gt_metrics"]["gt_mowed_px"]),
        })
    summary_df = pd.DataFrame(summary_rows.items(), columns=["metric", "value"])
    st.dataframe(summary_df, width="stretch", hide_index=True)

    csv_bytes = summary_df.to_csv(index=False).encode("utf-8")
    with open(r["geotiff_path"], "rb") as fh:
        tif_bytes = fh.read()

    d1, d2 = st.columns(2)
    d1.download_button(
        "Download Prediction map (GeoTIFF)",
        data=tif_bytes,
        file_name=f"mowing_detection_{r['after_stem']}.tif",
        mime="image/tiff",
        width="stretch",
    )
    d2.download_button(
        "Download Summary statistics (CSV)",
        data=csv_bytes,
        file_name=f"mowing_summary_{r['after_stem']}.csv",
        mime="text/csv",
        width="stretch",
    )
    st.caption("GeoTIFF encoding: 1 = mowed, 0 = not mowed, 255 = no data.")

    with st.expander("Processing log"):
        st.text("\n".join(r["log"]))
else:
    st.info("Configure inputs in the sidebar and click **Run detection** to start.")
