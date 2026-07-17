# Automated Mowing Detection from Satellite Imagery

**Author:** Florian Klaver

Bachelor's thesis at ZHAW on automated detection of grassland mowing events using Sentinel-2 optical and Sentinel-1 SAR satellite data combined with supervised machine learning.

---

## Repository Structure

```text
satellite-mowing-detection/
│
├── notebooks/              # Analysis pipeline (run in order)
│   ├── 01_S2_preprocessing.ipynb         # Sentinel-2 data prep and cloud masking
│   ├── 02_S2_feature_engineering.ipynb   # Vegetation index computation
│   ├── 03_SAR_GRD_data_acquisition.ipynb # Sentinel-1 GRD download via GEE
│   ├── 04_SAR_GRD_preprocessing.ipynb    # SAR GRD preprocessing
│   ├── 05_SLC_data_acquisition.ipynb     # Sentinel-1 SLC acquisition
│   ├── 06_SLC_preprocessing.ipynb        # InSAR coherence processing
│   ├── 06b_snr_correction.ipynb          # SNR correction for coherence
│   ├── 07_feature_fusion.ipynb           # Merge S2 and S1 features
│   ├── 08_model_training.ipynb           # Train and evaluate classifiers
│   ├── 09_model_application.ipynb        # Apply models to test data
│   └── 10_full_application.ipynb         # End-to-end application run
│
├── src/                    # Shared Python modules used by the notebooks
│   ├── config.py
│   ├── evaluation.py
│   ├── postprocessing.py
│   ├── sar_features.py
│   ├── snr_correction.py
│   ├── temporal_matching.py
│   └── vegetation_indices.py
│
├── prototype_application/  # Self-contained prototype for end users (see below)
│   ├── apply_mowing_detection.ipynb      # Main prototype notebook
│   ├── standalone_models/                # Pre-trained model files (.joblib)
│   └── data/
│       ├── s2_images/                    # Place your input GeoTIFFs here
│       ├── masks/                        # Optional ground truth masks
│       └── results/                      # Output files written here
│
│
├── report/                 # Quarto source files for the thesis report (.qmd)
│
└── docs/                   # Rendered HTML report (served via GitHub Pages) + PDF
```

---

## Installation

Clone the repository and create the conda environment:

```bash
git clone https://github.com/klaveflo/satellite-mowing-detection.git
cd satellite-mowing-detection
conda env create -f environment.yml
conda activate satellite-mowing-detection
```


---

## Prototype Usage

The `prototype_application/` folder is a self-contained notebook that applies a pre-trained mowing detection model to any pair of Sentinel-2 images. No access to the full dataset or the research pipeline is required.

### What it does

Given two cloud-free Sentinel-2 scenes taken a few days apart (before and after a suspected mowing event), the notebook:

1. Loads both GeoTIFFs and computes vegetation index features (NDII, GNDVI, SWIR).
2. Applies one of the included pre-trained classifiers pixel-by-pixel.
3. Post-processes the result with morphological operations.
4. Saves a binary prediction raster (`1 = mowed`, `0 = not mowed`, `255 = no data`).
5. Produces visualisation maps (RGB comparison, prediction overlay, optional ground truth comparison).

### Files you need to provide

| File | Description |
|------|-------------|
| `before.tif` | Sentinel-2 L2A multi-band GeoTIFF taken **before** the mowing event |
| `after.tif` | Sentinel-2 L2A multi-band GeoTIFF taken **after** the mowing event |

Place both files in `prototype_application/data/s2_images/` (or point the config variables to their actual paths).

If you download images directly from [Copernicus Browser](https://browser.dataspace.copernicus.eu), the notebook includes an optional conversion step (Step 2 / "Data Preparation") that assembles individual band files into the required multi-band GeoTIFF format.

The GeoTIFFs must share the same CRS and cover the same spatial extent. Sentinel-2 L2A products in their native projection work out of the box.

### Files already included

The `standalone_models/` folder contains four pre-trained models (Sentinel-2 only, no radar required):

| File | Algorithm | Features |
|------|-----------|----------|
| `s2_only_best_rf.joblib` | Random Forest | ndii_diff, gndvi_diff, swir_diff, ndii_after, swir_after |
| `s2_only_best_svm.joblib` | SVM | same feature set, **recommended**|
| `s2_only_diff_rf.joblib` | Random Forest | ndii_diff, gndvi_diff, swir_diff |
| `s2_only_diff_svm.joblib` | SVM | same feature set |

### Step-by-step instructions

1. **Open** `prototype_application/apply_mowing_detection.ipynb` in JupyterLab or VS Code.
2. **Run Step 1** (Setup cell) — installs/checks required packages.
3. **Edit Step 2** (Configuration) — set the paths to your two input GeoTIFFs and choose a model:

   ```python
   INPUT_BEFORE = Path("data/s2_images/your_before_image.tif")
   INPUT_AFTER  = Path("data/s2_images/your_after_image.tif")
   MODEL_PATH   = Path("standalone_models/s2_only_best_rf.joblib")
   OUTPUT_DIR   = Path("data/results")
   ```

   Optional inputs (study area polygon, grassland mask, ground truth) can be left as `None`.
4. **Run all remaining cells** in order (Steps 3–11). Each step is clearly labelled.

### Expected outputs

All outputs are written to `OUTPUT_DIR` (default: `prototype_application/data/results/`):

| Output | Description |
|--------|-------------|
| `mowing_detection_<date>.tif` | Binary prediction raster (GeoTIFF, same CRS as inputs) |
| `rgb_comparison_<before>_<after>.png` | Side-by-side true-colour comparison of the image pair |
| `detection_<date>_prediction.png` | Prediction overlay on the after-image |
| `evaluation_<date>.png` | TP/FP/FN comparison map (only if ground truth is provided) |

---

## Report

The thesis report is available in two formats:

- **HTML (GitHub Pages):** [https://klaveflo.github.io/satellite-mowing-detection/](https://klaveflo.github.io/satellite-mowing-detection/)
- **PDF:** [`docs/Advanced-Earth-Observation-for-Grassland-Management.pdf`](docs/Advanced-Earth-Observation-for-Grassland-Management.pdf)

The Quarto source files for the report are in [`report/`](report/).

---

## Citation

Florian Klaver (July 2, 2026). Advanced Earth Observation for Grassland Management: Improving Automated Mowing Detection Using Multimodal Remote Sensing and Machine Learning. Zurich University of Applied Sciences, Departement Life Sciences and Facility Management, Institute for Computational Life Sciences.
