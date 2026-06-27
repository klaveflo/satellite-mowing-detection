"""
Model training helpers, evaluation metrics, scene feature preparation, and
prediction pipeline.  Lifted from ML_Model.ipynb and Model_Application.ipynb
(PA2 project) and refactored into reusable functions.
"""

import os
import numpy as np
import pandas as pd
import rasterio
import geopandas as gpd
import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from rasterio import features as rio_features
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc, precision_score, recall_score, f1_score,
    precision_recall_curve,
)
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier

from src.vegetation_indices import (
    calculate_ndvi, calculate_evi, calculate_savi, calculate_gndvi, calculate_ndii
)

# ---------------------------------------------------------------------------
# Constants (shared with all notebooks)
# ---------------------------------------------------------------------------

CLOUD_BAND = 13
CLOUD_THRESHOLD = 30  # percent

GRASSLAND_CLASS = 'humusiert.Acker_Wiese_Weide'

FEATURE_NAMES = [
    'ndvi_after', 'evi_after', 'savi_after', 'gndvi_after', 'ndii_after',
    'ndvi_diff', 'evi_diff', 'savi_diff', 'gndvi_diff', 'ndii_diff',
    'blue_diff', 'green_diff', 'red_diff', 'nir_diff', 'swir_diff',
    'blue_after', 'green_after', 'red_after', 'nir_after', 'swir_after',
]

IMG_DIR = "docs/images"


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def prepare_data(data, features_to_keep, random_state=42):
    """
    Select features and perform a group-aware train/test split.

    Returns
    -------
    X_train, y_train, groups_train, X_test, y_test
    """
    available_features = [f for f in features_to_keep if f in data.columns]
    if len(available_features) < len(features_to_keep):
        missing = set(features_to_keep) - set(available_features)
        print(f"Warning: Missing features {missing}. Continuing with available features.")

    X = data[available_features]
    y = data['label']
    groups = data['match_id']

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=random_state)
    train_idx, test_idx = next(gss.split(X, y, groups))

    X_train = X.iloc[train_idx]
    y_train = y.iloc[train_idx]
    groups_train = groups.iloc[train_idx]
    X_test = X.iloc[test_idx]
    y_test = y.iloc[test_idx]

    print(f"Training: {len(X_train)} samples ({groups.iloc[train_idx].nunique()} matches)")
    print(f"Testing:  {len(X_test)} samples ({groups.iloc[test_idx].nunique()} matches)")
    return X_train, y_train, groups_train, X_test, y_test


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_model(model, X_test, y_test, model_name="Model", color="darkorange", results_list=None):
    """
    Standard evaluation pipeline: report + confusion matrix + ROC curve.
    Optionally appends a metrics dict to results_list.
    """
    y_pred = model.predict(X_test)
    y_proba = (model.predict_proba(X_test)[:, 1]
               if hasattr(model, "predict_proba")
               else model.decision_function(X_test))

    score = model.score(X_test, y_test)
    class_labels = ["Not Mowed (0)", "Mowed (1)"]
    print(f"\n--- {model_name} ---")
    print(f"Accuracy: {score * 100:.2f}%")
    print(classification_report(y_test, y_pred, target_names=class_labels))

    if results_list is not None:
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        results_list.append({
            "Model": model_name,
            "Features": list(X_test.columns),
            "Num_Features": len(X_test.columns),
            "Accuracy": score,
            "Precision": precision_score(y_test, y_pred, average='weighted'),
            "Recall": recall_score(y_test, y_pred, average='weighted'),
            "F1_Score": f1_score(y_test, y_pred, average='weighted'),
            "ROC_AUC": round(auc(fpr, tpr), 3),
            "TP": tp, "TN": tn, "FP": fp, "FN": fn,
        })

    # Plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    ConfusionMatrixDisplay.from_estimator(
        model, X_test, y_test, display_labels=class_labels,
        cmap="Blues", xticks_rotation='horizontal', ax=ax1, colorbar=False
    )
    for t in ax1.texts:
        t.set_fontsize(11)
    ax1.grid(False)
    ax1.set_ylabel("True label", fontsize=12)
    ax1.set_xlabel("Predicted label", fontsize=12)

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    ax2.plot(fpr, tpr, color=color, lw=2, label=f"AUC = {roc_auc:.3f}")
    ax2.plot([0, 1], [0, 1], color="navy", linestyle="--", lw=1)
    ax2.set_xlabel("False Positive Rate", fontsize=12)
    ax2.set_ylabel("True Positive Rate", fontsize=12)
    # ax2.set_title(f"ROC — {model_name}")
    ax2.legend()
    plt.tight_layout()
    plt.savefig(f"../{IMG_DIR}/ml_training/eval_{model_name.replace(' ', '_')}.png", dpi=300, bbox_inches='tight')
    plt.show()


# ---------------------------------------------------------------------------
# Threshold calibration
# ---------------------------------------------------------------------------

def find_optimal_threshold(model, X_test, y_test):
    """
    Find the probability threshold that maximises F1 on a held-out test set.

    Sweeps all natural thresholds from the precision-recall curve and picks
    the one with the highest F1.  Use the returned threshold in run_prediction
    instead of the default 0.5 to correct for class-imbalance mismatch between
    balanced training data and the real-world scene distribution.

    Returns
    -------
    threshold : float  — optimal probability cutoff for predict_proba[:,1]
    f1_at_threshold : float  — F1 achieved at that threshold on the test set
    """
    proba = model.predict_proba(X_test)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_test, proba)
    # precision/recall arrays have length n+1; thresholds has length n
    f1_scores = (2 * precision[:-1] * recall[:-1]
                 / (precision[:-1] + recall[:-1] + 1e-8))
    best_idx = int(np.argmax(f1_scores))
    return float(thresholds[best_idx]), float(f1_scores[best_idx])


# ---------------------------------------------------------------------------
# Model training helpers
# ---------------------------------------------------------------------------

def train_tuned_svm(X_train, y_train, groups_train, X_test, y_test,
                    param_grid=None, results_list=None, model_alias="Tuned SVM"):
    if param_grid is None:
        param_grid = {'svm__C': [0.1, 1, 10], 'svm__gamma': ['scale', 'auto']}
    pipeline = Pipeline([('scaler', StandardScaler()),
                         ('svm', SVC(probability=True, random_state=42))])
    grid = GridSearchCV(pipeline, param_grid, cv=GroupKFold(n_splits=5),
                        scoring='f1', n_jobs=-1, verbose=1)
    grid.fit(X_train, y_train, groups=groups_train)
    print("Best params:", grid.best_params_)
    evaluate_model(grid.best_estimator_, X_test, y_test, model_name=model_alias,
                   color="purple", results_list=results_list)
    return grid.best_estimator_


def train_tuned_rf(X_train, y_train, groups_train, X_test, y_test,
                   param_grid=None, results_list=None, model_alias="Tuned RF"):
    if param_grid is None:
        param_grid = {'rf__n_estimators': [100, 200], 'rf__max_depth': [None, 10, 20]}
    pipeline = Pipeline([('rf', RandomForestClassifier(random_state=42, n_jobs=-1))])
    grid = GridSearchCV(pipeline, param_grid, cv=GroupKFold(n_splits=5),
                        scoring='f1', n_jobs=-1, verbose=1)
    grid.fit(X_train, y_train, groups=groups_train)
    print("Best params:", grid.best_params_)
    evaluate_model(grid.best_estimator_, X_test, y_test, model_name=model_alias,
                   color="darkorange", results_list=results_list)
    return grid.best_estimator_


def train_tuned_lgbm(X_train, y_train, groups_train, X_test, y_test,
                     param_grid=None, results_list=None, model_alias="Tuned LGBM"):
    if param_grid is None:
        param_grid = {'lgbm__num_leaves': [31, 63], 'lgbm__learning_rate': [0.05, 0.1]}
    pipeline = Pipeline([('lgbm', LGBMClassifier(random_state=42, n_jobs=-1, verbose=-1))])
    grid = GridSearchCV(pipeline, param_grid, cv=GroupKFold(n_splits=5),
                        scoring='f1', n_jobs=-1, verbose=1)
    grid.fit(X_train, y_train, groups=groups_train)
    print("Best params:", grid.best_params_)
    evaluate_model(grid.best_estimator_, X_test, y_test, model_name=model_alias,
                   color="green", results_list=results_list)
    return grid.best_estimator_


# ---------------------------------------------------------------------------
# Scene feature preparation
# ---------------------------------------------------------------------------

def prepare_scene_features(before_image_path, after_image_path,
                            ground_cover_path, fire_brigade_path,
                            features_to_keep):
    """
    Load a before/after Sentinel-2 scene pair, compute all 20 features,
    apply grassland + airport boundary masks, and return a flat DataFrame
    with only the requested features.

    Returns
    -------
    feature_df  : DataFrame (n_pixels, len(features_to_keep)); invalid pixels have NaN
    valid_mask  : boolean 1-D array (n_pixels,) — grassland + airport + non-cloud
    meta        : rasterio metadata dict (for writing output rasters)
    height, width : scene dimensions
    """
    print(f"Before: {os.path.basename(before_image_path)}")
    print(f"After:  {os.path.basename(after_image_path)}")

    with rasterio.open(before_image_path) as src_b, \
         rasterio.open(after_image_path) as src_a:

        meta = src_b.meta.copy()
        height, width = meta['height'], meta['width']
        transform = meta['transform']
        crs = meta['crs']

        blue_b  = src_b.read(1).astype(float)
        green_b = src_b.read(2).astype(float)
        red_b   = src_b.read(3).astype(float)
        nir_b   = src_b.read(7).astype(float)
        swir_b  = src_b.read(10).astype(float)
        cloud_b = src_b.read(CLOUD_BAND).astype(float)

        blue_a  = src_a.read(1).astype(float)
        green_a = src_a.read(2).astype(float)
        red_a   = src_a.read(3).astype(float)
        nir_a   = src_a.read(7).astype(float)
        swir_a  = src_a.read(10).astype(float)
        cloud_a = src_a.read(CLOUD_BAND).astype(float)

    ndvi_a = calculate_ndvi(nir_a, red_a)
    evi_a  = calculate_evi(nir_a, red_a, blue_a)
    savi_a = calculate_savi(nir_a, red_a)
    ndii_a = calculate_ndii(nir_a, swir_a)
    gndvi_a = calculate_gndvi(nir_a, green_a)

    ndvi_b = calculate_ndvi(nir_b, red_b)
    evi_b  = calculate_evi(nir_b, red_b, blue_b)
    savi_b = calculate_savi(nir_b, red_b)
    ndii_b = calculate_ndii(nir_b, swir_b)
    gndvi_b = calculate_gndvi(nir_b, green_b)

    all_features = {
        'ndvi_after': ndvi_a, 'evi_after': evi_a, 'savi_after': savi_a,
        'gndvi_after': gndvi_a, 'ndii_after': ndii_a,
        'ndvi_diff': ndvi_a - ndvi_b, 'evi_diff': evi_a - evi_b,
        'savi_diff': savi_a - savi_b, 'gndvi_diff': gndvi_a - gndvi_b,
        'ndii_diff': ndii_a - ndii_b,
        'blue_diff': blue_a - blue_b, 'green_diff': green_a - green_b,
        'red_diff': red_a - red_b, 'nir_diff': nir_a - nir_b,
        'swir_diff': swir_a - swir_b,
        'blue_after': blue_a, 'green_after': green_a, 'red_after': red_a,
        'nir_after': nir_a, 'swir_after': swir_a,
    }

    feature_stack = np.stack([all_features[n] for n in features_to_keep])
    feature_df = pd.DataFrame(
        feature_stack.reshape(len(features_to_keep), -1).T,
        columns=features_to_keep
    )

    # Grassland mask
    av_data = gpd.read_file(ground_cover_path, layer="Bodenbedeckung_BoFlaeche_Area").to_crs(crs)
    grassland = av_data[av_data['Art_TXT'] == GRASSLAND_CLASS]
    if len(grassland) > 0:
        grassland_mask = rio_features.rasterize(
            shapes=grassland.geometry, out_shape=(height, width),
            transform=transform, fill=0, default_value=1, dtype='uint8'
        ).astype(bool).flatten()
    else:
        grassland_mask = np.zeros(height * width, dtype=bool)

    # Airport boundary mask
    fb_data = gpd.read_file(fire_brigade_path, layer="FW_GEMEINDEN_F").to_crs(crs)
    airport_poly = fb_data[fb_data['GEMEINDENAME'] == 'Flughafen Zürich']
    if not airport_poly.empty:
        airport_mask = rio_features.rasterize(
            shapes=airport_poly.geometry, out_shape=(height, width),
            transform=transform, fill=0, default_value=1, dtype='uint8'
        ).astype(bool).flatten()
    else:
        airport_mask = np.ones(height * width, dtype=bool)

    cloud_mask = ((cloud_b > CLOUD_THRESHOLD) | (cloud_a > CLOUD_THRESHOLD)).flatten()
    nan_mask = np.isnan(feature_df.values).any(axis=1)

    valid_mask = grassland_mask & airport_mask & ~cloud_mask & ~nan_mask
    return feature_df, valid_mask, meta, height, width


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def run_prediction(model, feature_df, valid_mask, height, width, output_path, meta,
                   threshold=0.5):
    """Apply model to valid pixels and save prediction raster.

    Parameters
    ----------
    threshold : float
        Decision probability threshold applied to predict_proba[:,1].
        Use the value returned by find_optimal_threshold() to correct for
        class-imbalance mismatch between training and full-scene distribution.
        Defaults to 0.5 (standard predict behaviour).
    """
    print(f"Predicting on {valid_mask.sum():,} valid pixels (threshold={threshold:.3f})...")
    prediction_flat = np.full(height * width, -1, dtype=np.int8)
    if valid_mask.sum() > 0:
        if threshold != 0.5 and hasattr(model, 'predict_proba'):
            proba = model.predict_proba(feature_df[valid_mask])[:, 1]
            prediction_flat[valid_mask] = (proba >= threshold).astype(np.int8)
        else:
            prediction_flat[valid_mask] = model.predict(feature_df[valid_mask])
    else:
        print("Warning: No valid pixels.")

    prediction_map = prediction_flat.reshape(height, width)
    meta.update({'count': 1, 'dtype': 'int8', 'nodata': -1})
    with rasterio.open(output_path, 'w', **meta) as dst:
        dst.write(prediction_map, 1)
    print(f"Saved: {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# Evaluation + visualization
# ---------------------------------------------------------------------------

def evaluate_and_visualize(prediction_path, ground_truth_names, masks_dir,
                            background_image_path, output_dir=None):
    """
    Compute pixel-level metrics and plot a difference map (TP/FP/FN) on top
    of the RGB satellite image alongside a confusion matrix.

    Returns a dict of metrics: accuracy, precision_mowing, recall_mowing,
    f1_mowing, f1_macro.
    """
    base_name = os.path.basename(prediction_path).split('.')[0]

    with rasterio.open(prediction_path) as src:
        pred_map = src.read(1)
        height, width = pred_map.shape
    valid_mask = pred_map != -1

    gt_combined = np.zeros((height, width), dtype=np.uint8)
    for name in ground_truth_names:
        path = os.path.join(masks_dir, name)
        if os.path.exists(path):
            with rasterio.open(path) as src:
                gt_combined = np.maximum(gt_combined, src.read(1))

    y_pred = pred_map[valid_mask]
    y_true = gt_combined[valid_mask]

    if len(y_true) == 0:
        print("No valid pixels for evaluation.")
        return {}

    report = classification_report(
        y_true, y_pred, target_names=['No Mowing', 'Mowing'], output_dict=True
    )
    print(f"REPORT: {base_name}")
    print(classification_report(y_true, y_pred, target_names=['No Mowing', 'Mowing']))

    metrics = {
        'accuracy': report['accuracy'],
        'precision_mowing': report['Mowing']['precision'],
        'recall_mowing': report['Mowing']['recall'],
        'f1_mowing': report['Mowing']['f1-score'],
        'f1_macro': report['macro avg']['f1-score'],
    }

    # Difference map
    diff_map = np.full((height, width), np.nan)
    diff_map[(pred_map == 1) & (gt_combined == 1)] = 1  # TP
    diff_map[(pred_map == 1) & (gt_combined == 0)] = 2  # FP
    diff_map[(pred_map == 0) & (gt_combined == 1)] = 3  # FN

    with rasterio.open(background_image_path) as src:
        rgb = np.dstack([
            np.clip(src.read(3) / 2500, 0, 1),
            np.clip(src.read(2) / 2500, 0, 1),
            np.clip(src.read(1) / 2500, 0, 1),
        ])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

    ax1.imshow(rgb)
    cmap = ListedColormap(['#00CCFF', '#FF9900', '#FF00FF'])
    ax1.imshow(diff_map, cmap=cmap, vmin=0.5, vmax=3.5, alpha=0.7)
    legend_patches = [
        mpatches.Patch(color='#00CCFF', label='True Positive'),
        mpatches.Patch(color='#FF9900', label='False Positive'),
        mpatches.Patch(color='#FF00FF', label='False Negative'),
    ]
    ax1.legend(handles=legend_patches, loc='upper right', framealpha=0.9)
    # ax1.set_title(f"Spatial Analysis: {base_name}")
    ax1.axis('off')

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    ConfusionMatrixDisplay(cm, display_labels=['Not Mowed', 'Mowed']).plot(
        ax=ax2, cmap='Blues', colorbar=False, values_format='d'
    )
    # ax2.set_title("Confusion Matrix")
    ax2.grid(False)

    plt.tight_layout()
    if output_dir:
        save_path = os.path.join(output_dir, f"eval_map_{base_name}.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.show()

    return metrics


# ---------------------------------------------------------------------------
# Master pipeline
# ---------------------------------------------------------------------------

def run_full_assessment(model_path, model_features, before_img, after_img,
                        gt_masks, masks_dir, av_data_path, fb_data_path,
                        output_dir, background_img=None, postprocess_fn=None):
    """
    End-to-end pipeline: load model → prepare features → predict → evaluate.

    Parameters
    ----------
    postprocess_fn : optional callable(pred_map) → pred_map
        If provided, the prediction raster is postprocessed before evaluation.
    """
    if not os.path.exists(model_path):
        print(f"Skipping — not found: {model_path}")
        return None

    print(f"\n{'='*60}\nASSESSMENT: {os.path.basename(model_path)}\n{'='*60}")
    model = joblib.load(model_path)

    feature_df, valid_mask, meta, h, w = prepare_scene_features(
        before_img, after_img, av_data_path, fb_data_path, model_features
    )

    img_base   = os.path.basename(after_img).replace('.tif', '')
    model_base = os.path.basename(model_path).replace('.joblib', '')
    os.makedirs(output_dir, exist_ok=True)
    pred_path  = os.path.join(output_dir, f"pred_map_{img_base}_{model_base}.tif")

    run_prediction(model, feature_df, valid_mask, h, w, pred_path, meta)

    if postprocess_fn is not None:
        with rasterio.open(pred_path) as src:
            pred_map = src.read(1)
            pp_meta  = src.meta.copy()
        pred_map_pp = postprocess_fn(pred_map)
        pp_path = pred_path.replace('.tif', '_postprocessed.tif')
        with rasterio.open(pp_path, 'w', **pp_meta) as dst:
            dst.write(pred_map_pp, 1)
        pred_path = pp_path

    bg_img = background_img or after_img
    return evaluate_and_visualize(pred_path, gt_masks, masks_dir, bg_img, output_dir)
