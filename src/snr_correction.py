"""
SNR decorrelation correction for Sentinel-1 interferometric coherence.

Implements Tamm et al. (2016) Equations 3 and 4:
  γ_total = γ_temporal × γ_SNR × (other terms)

The measured coherence is corrected to recover γ_temporal:
  γ_temporal = γ_measured / γ_SNR_pair

Reference: Tamm T., Zalite K., Voormansik K., Talgre L. (2016).
  Relating Sentinel-1 Interferometric Coherence to Mowing Events on Grasslands.
  Remote Sensing, 8(10), 802.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Default NESZ constants for Sentinel-1 IW2,  Values in dB.
# ---------------------------------------------------------------------------
NESZ_VV_DB: float = -25.0
NESZ_VH_DB: float = -27.0

# Minimum γ_SNR before applying the floor clamp (prevents blow-up on very
# noisy pixels where σ⁰ ≈ NESZ or σ⁰ < NESZ).
GAMMA_SNR_FLOOR: float = 0.10


# ---------------------------------------------------------------------------
# Unit helpers
# ---------------------------------------------------------------------------

def linear_from_db(db: np.ndarray) -> np.ndarray:
    """Convert dB to linear power (σ⁰_dB → σ⁰_linear)."""
    return np.power(10.0, np.asarray(db, dtype=np.float64) / 10.0)


def db_from_linear(lin: np.ndarray) -> np.ndarray:
    """Convert linear power to dB."""
    lin = np.asarray(lin, dtype=np.float64)
    with np.errstate(divide='ignore', invalid='ignore'):
        return 10.0 * np.log10(np.where(lin > 0, lin, np.nan))


# ---------------------------------------------------------------------------
# Core SNR decorrelation formulae (Tamm 2016)
# ---------------------------------------------------------------------------

def compute_gamma_snr_single(
    sigma0_linear: np.ndarray,
    nesz_linear: float,
) -> np.ndarray:
    """
    Per-pixel γ_SNR for a single SAR acquisition (Tamm Eq 3).

    γ_SNR_single = σ⁰ / (σ⁰ + NESZ)

    Parameters
    ----------
    sigma0_linear : array of σ⁰ values in linear power scale
    nesz_linear   : NESZ in linear power scale (scalar)

    Returns
    -------
    γ_SNR_single in [0, 1]; NaN propagates from NaN inputs.
    """
    s0 = np.asarray(sigma0_linear, dtype=np.float64)
    result = s0 / (s0 + nesz_linear)
    # Negative σ⁰ (below noise floor) → clamp output to 0
    result = np.where(s0 > 0, result, 0.0)
    return result.astype(np.float32)


def compute_gamma_snr_pair(
    sigma0_t1_lin: np.ndarray,
    sigma0_t2_lin: np.ndarray,
    nesz_lin: float,
) -> np.ndarray:
    """
    Per-pixel γ_SNR for an interferometric coherence pair (Tamm Eq 4).

    SNR_sat = (σ⁰_sat − NESZ) / NESZ
    γ_SNR_pair = 1 / sqrt((1 + 1/SNR_t1) × (1 + 1/SNR_t2))

    Pixels where σ⁰ ≤ NESZ (SNR ≤ 0) are handled by clamping SNR to a
    small positive minimum (SNR_MIN = 0.01), which produces a very small
    γ_SNR.  The apply_snr_correction floor then caps the correction factor.

    Parameters
    ----------
    sigma0_t1_lin : σ⁰ of acquisition t1 in linear power scale
    sigma0_t2_lin : σ⁰ of acquisition t2 in linear power scale
    nesz_lin      : NESZ in linear power scale (scalar, same for both)

    Returns
    -------
    γ_SNR_pair ∈ (0, 1]; NaN where either input is NaN.
    """
    SNR_MIN = 0.01
    s1 = np.asarray(sigma0_t1_lin, dtype=np.float64)
    s2 = np.asarray(sigma0_t2_lin, dtype=np.float64)

    snr1 = np.maximum((s1 - nesz_lin) / nesz_lin, SNR_MIN)
    snr2 = np.maximum((s2 - nesz_lin) / nesz_lin, SNR_MIN)

    factor1 = 1.0 + 1.0 / snr1
    factor2 = 1.0 + 1.0 / snr2

    gamma = 1.0 / np.sqrt(factor1 * factor2)

    # Propagate NaN from inputs
    nan_mask = np.isnan(s1) | np.isnan(s2)
    gamma = np.where(nan_mask, np.nan, gamma)

    return gamma.astype(np.float32)


# ---------------------------------------------------------------------------
# Correction application
# ---------------------------------------------------------------------------

def apply_snr_correction(
    coh_measured: np.ndarray,
    gamma_snr: np.ndarray,
    floor: float = GAMMA_SNR_FLOOR,
) -> np.ndarray:
    """
    Recover γ_temporal from γ_measured by dividing out γ_SNR.

    γ_temporal = γ_measured / max(γ_SNR, floor)

    The floor prevents division blow-up when γ_SNR is very small.
    The result is clipped to [0, 1] since coherence is a magnitude.

    Parameters
    ----------
    coh_measured : measured coherence array, values in [0, 1]
    gamma_snr    : per-pixel γ_SNR from compute_gamma_snr_pair()
    floor        : minimum γ_SNR before clamping (default 0.10)

    Returns
    -------
    γ_temporal clipped to [0, 1]; NaN where either input is NaN.
    """
    coh = np.asarray(coh_measured, dtype=np.float64)
    gsnr = np.asarray(gamma_snr, dtype=np.float64)

    gsnr_clamped = np.maximum(gsnr, floor)
    gamma_temporal = coh / gsnr_clamped

    # Coherence is a magnitude ∈ [0, 1]
    gamma_temporal = np.clip(gamma_temporal, 0.0, 1.0)

    # Propagate NaN from either input
    nan_mask = np.isnan(coh) | np.isnan(gamma_snr)
    gamma_temporal = np.where(nan_mask, np.nan, gamma_temporal)

    return gamma_temporal.astype(np.float32)
