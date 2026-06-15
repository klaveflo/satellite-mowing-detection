from pathlib import Path

# ---------------------------------------------------------------------------
# Repo root — resolved from this file's location so paths are always absolute
# regardless of the working directory the notebook is launched from.
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# Top-level directories
# ---------------------------------------------------------------------------
DATA_DIR    = REPO_ROOT / "data"
MODELS_DIR  = REPO_ROOT / "models"
DOCS_DIR    = REPO_ROOT / "docs" / "images"

# ---------------------------------------------------------------------------
# Input data directories
# ---------------------------------------------------------------------------
S2_RAW_DIR     = DATA_DIR / "Sentinel_S2_2019-2023_with_CloudCOVER" / "Flughafen"
S2_SCENES_DIR  = DATA_DIR / "Sentinel_CH"
S1_GRD_DIR     = DATA_DIR / "Sentinel_S1"
S1_SLC_DIR     = DATA_DIR / "Sentinel_S1_SLC"
GT_MASKS_DIR   = DATA_DIR / "ground_truth_masks"
AV_DATA_DIR    = DATA_DIR / "amtliche_vermessung_zh"

# ---------------------------------------------------------------------------
# Intermediate / feature directories
# ---------------------------------------------------------------------------
S2_FEATURES_DIR = DATA_DIR / "features_all_indices"
S1_FEATURES_DIR = DATA_DIR / "features_s1"
COH_DIR         = DATA_DIR / "features_coherence"
SNAP_TMP_DIR    = DATA_DIR / "_snap_tmp"

# ---------------------------------------------------------------------------
# Results directory
# ---------------------------------------------------------------------------
RESULTS_DIR = DATA_DIR / "operational_results"

# ---------------------------------------------------------------------------
# Static input files
# ---------------------------------------------------------------------------
GT_RAW_PATH        = DATA_DIR / "flughafen.gpkg"
GT_CLEANED_PATH    = DATA_DIR / "mowing_events_cleaned.gpkg"
AV_LANDUSE_PATH    = AV_DATA_DIR / "DM01AVZH24LV95.gpkg"
AV_FIREBRIGADE_PATH = AV_DATA_DIR / "Feuerwehr_-OGD.gpkg"

# ---------------------------------------------------------------------------
# Generated CSV / index files
# ---------------------------------------------------------------------------
MASKS_OVERVIEW_CSV      = GT_MASKS_DIR / "masks_overview.csv"
TEMPORAL_MATCHES_CSV    = DATA_DIR / "temporal_matches.csv"
CLOUD_STATS_CSV         = DATA_DIR / "cloud_masking_stats.csv"
S1_DATE_MATCHES_CSV     = DATA_DIR / "s1_date_matches.csv"
S1_EVENT_COVERAGE_CSV   = DATA_DIR / "s1_event_coverage.csv"
SLC_SCENE_INDEX_CSV     = DATA_DIR / "slc_scene_index.csv"
SLC_EVENT_COVERAGE_CSV  = DATA_DIR / "slc_event_coverage.csv"
SLC_CATALOGUE_CACHE     = DATA_DIR / "slc_catalogue_cache.json"

# ---------------------------------------------------------------------------
# Training sample CSVs
# ---------------------------------------------------------------------------
SAMPLES_S2_CSV          = DATA_DIR / "training_samples_all_indices.csv"
SAMPLES_FUSION_CSV      = DATA_DIR / "training_samples_fusion.csv"
SAMPLES_FUSION_COH_CSV  = DATA_DIR / "training_samples_fusion_coh.csv"

# ---------------------------------------------------------------------------
# S2 preprocessing parameters
# ---------------------------------------------------------------------------
TARGET_CRS        = "EPSG:2056"
CLOUD_BAND        = 13       # band index (1-based) for the cloud probability layer
CLOUD_THRESHOLD   = 30       # percent; pixels above this are masked
GT_MIN_AREA_M2    = 400      # minimum polygon area kept after cleaning
EROSION_PIXELS    = 1        # GT mask erosion buffer (pixels = 10 m)

# ---------------------------------------------------------------------------
# Temporal matching windows (days relative to mowing event date)
# ---------------------------------------------------------------------------
BEFORE_FAR_MIN  = 9
BEFORE_FAR_MAX  = 20
BEFORE_NEAR_MIN = 3
BEFORE_NEAR_MAX = 8
AFTER_MIN       = 1
AFTER_MAX       = 7
S1_GRD_TOLERANCE_DAYS = 3

# ---------------------------------------------------------------------------
# Pixel sampling
# ---------------------------------------------------------------------------
N_SAMPLES_PER_EVENT = 543   # balanced per-event sample count

# ---------------------------------------------------------------------------
# SNAP / coherence processing
# ---------------------------------------------------------------------------
SNAP_HOME      = Path("C:/Program Files/esa-snap")
SUBSWATH       = "IW2"
FIRST_BURST    = 1
LAST_BURST     = 9

# Coherence estimation window — matched to Tamm et al. (2016) R80/R160 setup
# ENL ≈ 46, ground footprint ≈ 71m × 69m
COH_WIN_RG     = 19
COH_WIN_AZ     = 5
OUTPUT_SPACING = 20.0        # metres; coherence output pixel size

# ---------------------------------------------------------------------------
# Postprocessing defaults (tuned on 2020-08-12 baseline; revisit in nb09)
# ---------------------------------------------------------------------------
PP_DISK_RADIUS = 2
PP_MIN_AREA_PX = 4
PP_N_SEGMENTS  = 300
PP_COMPACTNESS = 0.05
