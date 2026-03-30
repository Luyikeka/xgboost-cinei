"""
data_preprocessing.py
=====================
Data loading, regridding, and feature engineering for XGBoost-based
spatial downscaling of CINEI anthropogenic emission inventories.

Supports:
    - CINEI v2.1 emission data (0.25° resolution)
    - South American regional inventories (0.1° resolution)
    - TROPOMI / OMI satellite retrievals (3.5–7 km resolution)

Author : Yijuan Zhang
Project: CINEI – Coupled and Integrated National Emission Inventory
License: MIT
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import xarray as xr
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import uniform_filter

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SPECIES_MAP = {
    "CO": {"satellite_var": "carbonmonoxide_total_column", "unit": "mol/m²"},
    "NOx": {"satellite_var": "nitrogendioxide_tropospheric_column", "unit": "mol/m²"},
    "HCHO": {"satellite_var": "formaldehyde_tropospheric_vertical_column", "unit": "mol/m²"},
}

CINEI_RESOLUTION = 0.25  # degrees
SA_RESOLUTION = 0.10     # degrees


@dataclass
class GridSpec:
    """Specification for a regular lat/lon grid."""

    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float
    resolution: float

    @property
    def lats(self) -> np.ndarray:
        return np.arange(
            self.lat_min + self.resolution / 2,
            self.lat_max,
            self.resolution,
        )

    @property
    def lons(self) -> np.ndarray:
        return np.arange(
            self.lon_min + self.resolution / 2,
            self.lon_max,
            self.resolution,
        )

    @property
    def shape(self) -> tuple[int, int]:
        return (len(self.lats), len(self.lons))


@dataclass
class DownscalingDataset:
    """Container for matched coarse/fine resolution training pairs."""

    features: np.ndarray          # (N, n_features)
    targets: np.ndarray           # (N,)
    lats: np.ndarray              # (N,)
    lons: np.ndarray              # (N,)
    feature_names: list[str]      # length n_features
    species: str = ""
    coarse_resolution: float = 0.25
    fine_resolution: float = 0.05
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------
def load_cinei_emissions(
    filepath: str | Path,
    species: str,
    month: Optional[int] = None,
) -> xr.DataArray:
    """
    Load a CINEI v2.1 gridded emission field.

    Parameters
    ----------
    filepath : str or Path
        Path to the CINEI NetCDF file (e.g., ``CINEI_v2.1_CO_2019.nc``).
    species : str
        One of ``'CO'``, ``'NOx'``, ``'HCHO'``.
    month : int, optional
        If given, select a single month (1–12).

    Returns
    -------
    xr.DataArray
        Emission field with dimensions ``(lat, lon)`` or ``(time, lat, lon)``.
    """
    ds = xr.open_dataset(filepath)

    # Try common variable name conventions
    var_candidates = [
        species,
        f"emis_{species}",
        f"emission_{species}",
        species.lower(),
        f"emis_{species.lower()}",
    ]
    var_name = None
    for candidate in var_candidates:
        if candidate in ds.data_vars:
            var_name = candidate
            break
    if var_name is None:
        available = list(ds.data_vars)
        raise KeyError(
            f"Cannot find variable for '{species}' in {filepath}. "
            f"Available variables: {available}"
        )

    da = ds[var_name]
    if month is not None and "time" in da.dims:
        da = da.isel(time=month - 1)

    logger.info(
        "Loaded CINEI %s from %s  shape=%s  resolution=%.2f°",
        species, filepath, da.shape, CINEI_RESOLUTION,
    )
    return da


def load_satellite_l3(
    filepath: str | Path,
    species: str,
    qa_threshold: float = 0.5,
) -> xr.DataArray:
    """
    Load a Level-3 gridded satellite product (TROPOMI / OMI).

    Parameters
    ----------
    filepath : str or Path
        Path to the satellite NetCDF/HDF5 file.
    species : str
        One of ``'CO'``, ``'NOx'``, ``'HCHO'``.
    qa_threshold : float
        Minimum quality assurance value to retain pixels.

    Returns
    -------
    xr.DataArray
        Column density field at native satellite resolution.
    """
    var_name = SPECIES_MAP[species]["satellite_var"]
    ds = xr.open_dataset(filepath)

    if var_name not in ds.data_vars:
        available = list(ds.data_vars)
        raise KeyError(
            f"Variable '{var_name}' not found. Available: {available}"
        )

    da = ds[var_name]

    # Apply QA filtering if qa_value is available
    if "qa_value" in ds.data_vars:
        qa = ds["qa_value"]
        da = da.where(qa >= qa_threshold)
        n_filtered = int(np.isnan(da.values).sum())
        logger.info("QA filter (>= %.2f): masked %d pixels", qa_threshold, n_filtered)

    logger.info(
        "Loaded satellite %s from %s  shape=%s",
        species, filepath, da.shape,
    )
    return da


# ---------------------------------------------------------------------------
# Regridding
# ---------------------------------------------------------------------------
def conservative_regrid(
    source: xr.DataArray,
    target_lats: np.ndarray,
    target_lons: np.ndarray,
) -> xr.DataArray:
    """
    Area-weighted conservative regridding from fine to coarse grid.

    This is a simplified first-order conservative remapping suitable for
    regular lat/lon grids.  For production use on ICON's icosahedral grid
    consider CDO ``remapcon`` or ESMF regridding.

    Parameters
    ----------
    source : xr.DataArray
        High-resolution field with ``lat`` and ``lon`` coordinates.
    target_lats, target_lons : np.ndarray
        1-D arrays of target grid cell centres.

    Returns
    -------
    xr.DataArray
        Regridded field on the target grid.
    """
    src_lat = source.lat.values
    src_lon = source.lon.values
    src_dlat = np.abs(np.median(np.diff(src_lat)))
    src_dlon = np.abs(np.median(np.diff(src_lon)))

    tgt_dlat = np.abs(np.median(np.diff(target_lats)))
    tgt_dlon = np.abs(np.median(np.diff(target_lons)))

    result = np.full((len(target_lats), len(target_lons)), np.nan)

    for i, tlat in enumerate(target_lats):
        lat_lo = tlat - tgt_dlat / 2
        lat_hi = tlat + tgt_dlat / 2
        lat_mask = (src_lat >= lat_lo - src_dlat / 2) & (src_lat <= lat_hi + src_dlat / 2)

        for j, tlon in enumerate(target_lons):
            lon_lo = tlon - tgt_dlon / 2
            lon_hi = tlon + tgt_dlon / 2
            lon_mask = (src_lon >= lon_lo - src_dlon / 2) & (src_lon <= lon_hi + src_dlon / 2)

            sub = source.values[np.ix_(lat_mask, lon_mask)]
            if sub.size == 0:
                continue

            # Area weights (cos latitude)
            sub_lats = src_lat[lat_mask]
            weights = np.cos(np.deg2rad(sub_lats))[:, np.newaxis]
            weights = np.broadcast_to(weights, sub.shape)

            valid = ~np.isnan(sub)
            if valid.sum() == 0:
                continue

            result[i, j] = np.nansum(sub * weights) / np.nansum(weights[valid])

    return xr.DataArray(
        result,
        dims=["lat", "lon"],
        coords={"lat": target_lats, "lon": target_lons},
    )


def bilinear_interpolate(
    source: xr.DataArray,
    target_lats: np.ndarray,
    target_lons: np.ndarray,
) -> xr.DataArray:
    """
    Bilinear interpolation from coarse to fine grid.

    Parameters
    ----------
    source : xr.DataArray
        Coarse-resolution field.
    target_lats, target_lons : np.ndarray
        1-D arrays of fine-grid cell centres.

    Returns
    -------
    xr.DataArray
        Interpolated field on the fine grid.
    """
    interpolator = RegularGridInterpolator(
        (source.lat.values, source.lon.values),
        source.values,
        method="linear",
        bounds_error=False,
        fill_value=np.nan,
    )

    tgt_lat_2d, tgt_lon_2d = np.meshgrid(target_lats, target_lons, indexing="ij")
    pts = np.column_stack([tgt_lat_2d.ravel(), tgt_lon_2d.ravel()])
    result = interpolator(pts).reshape(len(target_lats), len(target_lons))

    return xr.DataArray(
        result,
        dims=["lat", "lon"],
        coords={"lat": target_lats, "lon": target_lons},
    )


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
def build_spatial_features(
    lats: np.ndarray,
    lons: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    """
    Generate spatial predictor features from coordinates.

    Features include normalised lat/lon, their products, trigonometric
    encodings (to capture periodicity), and distance from the equator.

    Returns
    -------
    features : np.ndarray, shape (N, n_feat)
    names : list[str]
    """
    lat_2d, lon_2d = np.meshgrid(lats, lons, indexing="ij")
    lat_flat = lat_2d.ravel()
    lon_flat = lon_2d.ravel()

    # Normalise to [-1, 1]
    lat_n = lat_flat / 90.0
    lon_n = lon_flat / 180.0

    features = np.column_stack([
        lat_n,
        lon_n,
        lat_n * lon_n,
        lat_n ** 2,
        lon_n ** 2,
        np.sin(np.deg2rad(lat_flat)),
        np.cos(np.deg2rad(lat_flat)),
        np.sin(np.deg2rad(lon_flat)),
        np.cos(np.deg2rad(lon_flat)),
        np.abs(lat_n),                    # distance from equator
    ])

    names = [
        "lat_norm", "lon_norm", "lat_lon_product",
        "lat_sq", "lon_sq",
        "sin_lat", "cos_lat", "sin_lon", "cos_lon",
        "abs_lat",
    ]
    return features, names


def compute_texture_features(
    field: np.ndarray,
    window_sizes: tuple[int, ...] = (3, 5, 9),
) -> tuple[np.ndarray, list[str]]:
    """
    Compute local texture / contextual features from a 2-D field.

    For each window size, calculates local mean, standard deviation,
    and gradient magnitude.

    Parameters
    ----------
    field : np.ndarray, shape (nlat, nlon)
    window_sizes : tuple of int

    Returns
    -------
    features : np.ndarray, shape (nlat*nlon, n_feat)
    names : list[str]
    """
    feats = []
    names = []

    for ws in window_sizes:
        local_mean = uniform_filter(field, size=ws, mode="nearest")
        local_sq_mean = uniform_filter(field ** 2, size=ws, mode="nearest")
        local_std = np.sqrt(np.maximum(local_sq_mean - local_mean ** 2, 0))

        # Sobel-like gradient magnitude
        grad_y, grad_x = np.gradient(field)
        grad_mag = np.sqrt(grad_y ** 2 + grad_x ** 2)

        feats.extend([
            local_mean.ravel(),
            local_std.ravel(),
            uniform_filter(grad_mag, size=ws, mode="nearest").ravel(),
        ])
        names.extend([
            f"mean_{ws}x{ws}",
            f"std_{ws}x{ws}",
            f"grad_mag_{ws}x{ws}",
        ])

    return np.column_stack(feats), names


def prepare_training_data(
    emission_coarse: xr.DataArray,
    satellite_fine: xr.DataArray,
    emission_fine: Optional[xr.DataArray] = None,
    species: str = "CO",
    month: int = 1,
) -> DownscalingDataset:
    """
    Build a matched training dataset for XGBoost downscaling.

    Workflow
    --------
    1.  Regrid satellite data to the coarse emission grid (conservative).
    2.  Build features at coarse resolution: satellite columns + spatial
        + texture features.
    3.  If a fine-resolution emission reference is provided (e.g. the
        South American 0.1° inventory), use it as the training target;
        otherwise fall back to the coarse emission field itself (useful
        for self-supervised pre-training).

    Parameters
    ----------
    emission_coarse : xr.DataArray
        CINEI emission field at 0.25°.
    satellite_fine : xr.DataArray
        Satellite column density at native resolution (~0.05°).
    emission_fine : xr.DataArray, optional
        Higher-resolution emission reference for supervised training.
    species : str
    month : int

    Returns
    -------
    DownscalingDataset
    """
    coarse_lats = emission_coarse.lat.values
    coarse_lons = emission_coarse.lon.values

    # --- Satellite → coarse grid (for feature alignment) ---
    sat_coarse = conservative_regrid(satellite_fine, coarse_lats, coarse_lons)

    # Fill NaN satellite pixels with regional median
    sat_vals = sat_coarse.values.copy()
    median_val = np.nanmedian(sat_vals)
    sat_vals[np.isnan(sat_vals)] = median_val

    # --- Spatial features ---
    spatial_feats, spatial_names = build_spatial_features(coarse_lats, coarse_lons)

    # --- Texture features on coarse satellite field ---
    texture_feats, texture_names = compute_texture_features(sat_vals)

    # --- Temporal features ---
    month_sin = np.full(spatial_feats.shape[0], np.sin(2 * np.pi * month / 12))
    month_cos = np.full(spatial_feats.shape[0], np.cos(2 * np.pi * month / 12))

    # --- Emission ratio features ---
    emission_vals = emission_coarse.values.ravel()
    emission_vals_safe = np.where(emission_vals == 0, np.nan, emission_vals)
    sat_to_emis_ratio = sat_vals.ravel() / np.nanmedian(emission_vals_safe)

    # --- Assemble features ---
    all_features = np.column_stack([
        sat_vals.ravel(),
        spatial_feats,
        texture_feats,
        month_sin,
        month_cos,
        sat_to_emis_ratio,
    ])

    feature_names = (
        [f"satellite_{species}"]
        + spatial_names
        + texture_names
        + ["month_sin", "month_cos", "sat_emis_ratio"]
    )

    # --- Target ---
    if emission_fine is not None:
        target_coarse = conservative_regrid(emission_fine, coarse_lats, coarse_lons)
        targets = target_coarse.values.ravel()
    else:
        targets = emission_vals

    # --- Coordinate arrays ---
    lat_2d, lon_2d = np.meshgrid(coarse_lats, coarse_lons, indexing="ij")

    # --- Remove NaN rows ---
    valid = ~(np.isnan(all_features).any(axis=1) | np.isnan(targets))
    logger.info(
        "Training samples: %d valid / %d total (%.1f%%)",
        valid.sum(), len(valid), 100 * valid.mean(),
    )

    return DownscalingDataset(
        features=all_features[valid],
        targets=targets[valid],
        lats=lat_2d.ravel()[valid],
        lons=lon_2d.ravel()[valid],
        feature_names=feature_names,
        species=species,
        coarse_resolution=CINEI_RESOLUTION,
        fine_resolution=float(np.abs(np.median(np.diff(satellite_fine.lat.values)))),
        metadata={
            "month": month,
            "n_features": len(feature_names),
            "satellite_median_fill": float(median_val),
        },
    )


def prepare_prediction_grid(
    satellite_fine: xr.DataArray,
    emission_coarse: xr.DataArray,
    species: str = "CO",
    month: int = 1,
) -> tuple[np.ndarray, list[str], np.ndarray, np.ndarray]:
    """
    Build the feature matrix on the fine (satellite) grid for prediction.

    The trained XGBoost model is applied to this matrix to produce
    high-resolution emission estimates.

    Returns
    -------
    features : np.ndarray, shape (N, n_features)
    feature_names : list[str]
    fine_lats : np.ndarray
    fine_lons : np.ndarray
    """
    fine_lats = satellite_fine.lat.values
    fine_lons = satellite_fine.lon.values

    # Satellite values on fine grid (fill NaN)
    sat_vals = satellite_fine.values.copy()
    sat_vals[np.isnan(sat_vals)] = np.nanmedian(sat_vals)

    # Interpolate coarse emissions to fine grid (as auxiliary feature)
    emis_interp = bilinear_interpolate(emission_coarse, fine_lats, fine_lons)
    emis_vals = emis_interp.values
    emis_safe = np.where(emis_vals == 0, np.nan, emis_vals)

    # Spatial features on fine grid
    spatial_feats, spatial_names = build_spatial_features(fine_lats, fine_lons)

    # Texture features on fine satellite field
    texture_feats, texture_names = compute_texture_features(sat_vals)

    # Temporal
    month_sin = np.full(spatial_feats.shape[0], np.sin(2 * np.pi * month / 12))
    month_cos = np.full(spatial_feats.shape[0], np.cos(2 * np.pi * month / 12))

    # Ratio
    sat_to_emis_ratio = sat_vals.ravel() / np.nanmedian(emis_safe)

    all_features = np.column_stack([
        sat_vals.ravel(),
        spatial_feats,
        texture_feats,
        month_sin,
        month_cos,
        sat_to_emis_ratio,
    ])

    feature_names = (
        [f"satellite_{species}"]
        + spatial_names
        + texture_names
        + ["month_sin", "month_cos", "sat_emis_ratio"]
    )

    return all_features, feature_names, fine_lats, fine_lons
