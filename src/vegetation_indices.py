"""
Vegetation index calculations for Sentinel-2 imagery.
Lifted directly from Feature_Engineering.ipynb (PA2 project).
All functions guard against division-by-zero and return NaN for invalid pixels.
"""

import numpy as np


def calculate_ndvi(nir, red):
    """Normalized Difference Vegetation Index"""
    with np.errstate(divide='ignore', invalid='ignore'):
        ndvi = (nir - red) / (nir + red)
    ndvi[np.isinf(ndvi)] = np.nan
    return ndvi


def calculate_evi(nir, red, blue):
    """Enhanced Vegetation Index"""
    with np.errstate(divide='ignore', invalid='ignore'):
        evi = 2.5 * ((nir - red) / (nir + 6 * red - 7.5 * blue + 1))
    evi[np.isinf(evi)] = np.nan
    return evi


def calculate_savi(nir, red, L=0.5):
    """Soil Adjusted Vegetation Index (L=0.5 is standard)"""
    with np.errstate(divide='ignore', invalid='ignore'):
        savi = ((nir - red) / (nir + red + L)) * (1 + L)
    savi[np.isinf(savi)] = np.nan
    return savi


def calculate_gndvi(nir, green):
    """Green Normalized Difference Vegetation Index"""
    with np.errstate(divide='ignore', invalid='ignore'):
        gndvi = (nir - green) / (nir + green)
    gndvi[np.isinf(gndvi)] = np.nan
    return gndvi


def calculate_ndii(nir, swir):
    """Normalized Difference Infrared Index (sensitive to water content / canopy moisture)"""
    with np.errstate(divide='ignore', invalid='ignore'):
        ndii = (nir - swir) / (nir + swir)
    ndii[np.isinf(ndii)] = np.nan
    return ndii


def calculate_all(nir, red, green, blue, swir):
    """
    Compute all five indices at once.
    Returns a dict keyed by index name.
    """
    return {
        'ndvi': calculate_ndvi(nir, red),
        'evi':  calculate_evi(nir, red, blue),
        'savi': calculate_savi(nir, red),
        'gndvi': calculate_gndvi(nir, green),
        'ndii': calculate_ndii(nir, swir),
    }
