"""
evaluation.py
=============
Evaluation metrics and publication-quality visualisation for
XGBoost-downscaled CINEI emission fields.

Provides spatial comparison maps, scatter density plots, feature
importance charts, and statistical metric summaries suitable for
journal submission and conference presentations.

Author : Yijuan Zhang
Project: CINEI – Coupled and Integrated National Emission Inventory
License: MIT
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Publication-quality defaults
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


# ---------------------------------------------------------------------------
# Statistical metrics
# ---------------------------------------------------------------------------
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Compute a comprehensive set of evaluation metrics.

    Returns
    -------
    dict
        Keys: ``r2``, ``rmse``, ``mae``, ``nmb``, ``nme``,
        ``correlation``, ``index_of_agreement``, ``fractional_bias``.
    """
    valid = ~(np.isnan(y_true) | np.isnan(y_pred))
    yt = y_true[valid]
    yp = y_pred[valid]

    if len(yt) == 0:
        return {k: np.nan for k in [
            "r2", "rmse", "mae", "nmb", "nme",
            "correlation", "ioa", "fb",
        ]}

    # R²
    ss_res = np.sum((yt - yp) ** 2)
    ss_tot = np.sum((yt - yt.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # RMSE & MAE
    rmse = np.sqrt(np.mean((yt - yp) ** 2))
    mae = np.mean(np.abs(yt - yp))

    # NMB & NME (%)
    nmb = np.sum(yp - yt) / np.sum(yt) * 100
    nme = np.sum(np.abs(yp - yt)) / np.sum(yt) * 100

    # Pearson correlation
    corr = np.corrcoef(yt, yp)[0, 1] if len(yt) > 1 else np.nan

    # Index of Agreement (Willmott, 1981)
    diff_mean = np.abs(yp - yt.mean()) + np.abs(yt - yt.mean())
    ioa = 1 - ss_res / np.sum(diff_mean ** 2) if np.sum(diff_mean ** 2) > 0 else 0.0

    # Fractional bias
    fb = 2 * np.mean(yp - yt) / (np.mean(yp) + np.mean(yt)) * 100

    return {
        "r2": float(r2),
        "rmse": float(rmse),
        "mae": float(mae),
        "nmb": float(nmb),
        "nme": float(nme),
        "correlation": float(corr),
        "ioa": float(ioa),
        "fb": float(fb),
        "n_valid": int(len(yt)),
    }


# ---------------------------------------------------------------------------
# Spatial comparison panel
# ---------------------------------------------------------------------------
def plot_spatial_comparison(
    coarse: xr.DataArray,
    downscaled: xr.DataArray,
    satellite: Optional[xr.DataArray] = None,
    reference: Optional[xr.DataArray] = None,
    species: str = "CO",
    title: str = "",
    output_path: Optional[str | Path] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> mpl.figure.Figure:
    """
    Multi-panel spatial comparison of emission fields.

    Panels
    ------
    1. Original CINEI (coarse)
    2. XGBoost downscaled (fine)
    3. Satellite column density (fine)
    4. Reference inventory or difference map

    Parameters
    ----------
    coarse : xr.DataArray
        Original CINEI emission field.
    downscaled : xr.DataArray
        Downscaled emission field from XGBoost.
    satellite : xr.DataArray, optional
        Satellite column density.
    reference : xr.DataArray, optional
        Fine-resolution reference for validation.
    species : str
    title : str
    output_path : str or Path, optional
    vmin, vmax : float, optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    n_panels = 2 + (satellite is not None) + (reference is not None)
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 4.5))
    if n_panels == 1:
        axes = [axes]

    # Colour scale
    if vmin is None:
        vmin = 0
    if vmax is None:
        vmax = float(np.nanpercentile(downscaled.values, 98))

    cmap = plt.cm.YlOrRd
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    panel = 0

    # --- Panel 1: Coarse CINEI ---
    im = axes[panel].pcolormesh(
        coarse.lon, coarse.lat, coarse.values,
        cmap=cmap, norm=norm, shading="auto",
    )
    axes[panel].set_title(f"CINEI {species} (0.25°)")
    axes[panel].set_xlabel("Longitude")
    axes[panel].set_ylabel("Latitude")
    panel += 1

    # --- Panel 2: Downscaled ---
    axes[panel].pcolormesh(
        downscaled.lon, downscaled.lat, downscaled.values,
        cmap=cmap, norm=norm, shading="auto",
    )
    axes[panel].set_title(f"XGBoost Downscaled {species}")
    axes[panel].set_xlabel("Longitude")
    panel += 1

    # --- Panel 3: Satellite ---
    if satellite is not None:
        sat_vmax = float(np.nanpercentile(satellite.values, 98))
        axes[panel].pcolormesh(
            satellite.lon, satellite.lat, satellite.values,
            cmap=plt.cm.viridis, shading="auto",
            vmin=0, vmax=sat_vmax,
        )
        axes[panel].set_title(f"Satellite {species}")
        axes[panel].set_xlabel("Longitude")
        panel += 1

    # --- Panel 4: Reference or difference ---
    if reference is not None:
        # Interpolate downscaled onto reference grid for comparison
        ds_on_ref = downscaled.interp(
            lat=reference.lat, lon=reference.lon, method="linear",
        )
        diff = ds_on_ref - reference
        valid_diff = diff.values[~np.isnan(diff.values)]
        d_abs = float(np.nanpercentile(np.abs(valid_diff), 95)) if len(valid_diff) > 0 else 1.0
        if d_abs == 0:
            d_abs = 1.0
        axes[panel].pcolormesh(
            diff.lon, diff.lat, diff.values,
            cmap=plt.cm.RdBu_r, shading="auto",
            vmin=-d_abs, vmax=d_abs,
        )
        axes[panel].set_title(f"Downscaled − Reference")
        axes[panel].set_xlabel("Longitude")

    fig.colorbar(im, ax=axes, orientation="horizontal",
                 fraction=0.04, pad=0.12, label=f"{species} emission flux")

    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)

    fig.tight_layout()

    if output_path:
        fig.savefig(output_path)
        logger.info("Spatial comparison saved to %s", output_path)

    return fig


# ---------------------------------------------------------------------------
# Scatter density plot
# ---------------------------------------------------------------------------
def plot_scatter_density(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    species: str = "CO",
    title: str = "",
    output_path: Optional[str | Path] = None,
) -> mpl.figure.Figure:
    """
    Scatter density plot with 1:1 line and statistics overlay.
    """
    valid = ~(np.isnan(y_true) | np.isnan(y_pred))
    yt = y_true[valid]
    yp = y_pred[valid]
    metrics = compute_metrics(yt, yp)

    fig, ax = plt.subplots(figsize=(5.5, 5))

    # 2-D histogram (density)
    vmax_plot = max(np.nanmax(yt), np.nanmax(yp))
    h = ax.hist2d(
        yt, yp,
        bins=100,
        cmap="inferno",
        norm=mcolors.LogNorm(),
        range=[[0, vmax_plot], [0, vmax_plot]],
    )
    fig.colorbar(h[3], ax=ax, label="Count")

    # 1:1 line
    ax.plot([0, vmax_plot], [0, vmax_plot], "w--", lw=1.5, alpha=0.8)

    # Statistics annotation
    stats_text = (
        f"R² = {metrics['r2']:.3f}\n"
        f"RMSE = {metrics['rmse']:.3e}\n"
        f"NMB = {metrics['nmb']:.1f}%\n"
        f"r = {metrics['correlation']:.3f}\n"
        f"N = {metrics['n_valid']:,}"
    )
    ax.text(
        0.05, 0.95, stats_text,
        transform=ax.transAxes,
        verticalalignment="top",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.4", fc="white", alpha=0.85),
    )

    ax.set_xlabel(f"Reference {species} emission")
    ax.set_ylabel(f"Downscaled {species} emission")
    ax.set_title(title or f"{species} Downscaling Validation")
    ax.set_aspect("equal")

    fig.tight_layout()
    if output_path:
        fig.savefig(output_path)
        logger.info("Scatter plot saved to %s", output_path)

    return fig


# ---------------------------------------------------------------------------
# Feature importance
# ---------------------------------------------------------------------------
def plot_feature_importance(
    importance: dict,
    top_n: int = 15,
    species: str = "CO",
    output_path: Optional[str | Path] = None,
) -> mpl.figure.Figure:
    """
    Horizontal bar chart of XGBoost feature importances.
    """
    sorted_items = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    top = sorted_items[:top_n]
    names = [x[0] for x in top][::-1]
    values = [x[1] for x in top][::-1]

    fig, ax = plt.subplots(figsize=(6, 0.35 * len(names) + 1.5))

    bars = ax.barh(names, values, color="#2196F3", edgecolor="white", height=0.7)
    ax.set_xlabel("Feature Importance (gain)")
    ax.set_title(f"Top-{top_n} Features — {species} Downscaling")

    # Value labels
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_width() + max(values) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}",
            va="center", fontsize=9,
        )

    ax.set_xlim(0, max(values) * 1.15)
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path)
        logger.info("Feature importance plot saved to %s", output_path)

    return fig


# ---------------------------------------------------------------------------
# Uncertainty visualisation
# ---------------------------------------------------------------------------
def plot_uncertainty_map(
    downscaled: xr.DataArray,
    uncertainty_low: xr.DataArray,
    uncertainty_high: xr.DataArray,
    species: str = "CO",
    output_path: Optional[str | Path] = None,
) -> mpl.figure.Figure:
    """
    Map of downscaled emissions with uncertainty range overlay.
    """
    unc_range = uncertainty_high - uncertainty_low
    rel_unc = unc_range / (downscaled + 1e-20) * 100

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # Mean estimate
    vmax = float(np.nanpercentile(downscaled.values, 98))
    axes[0].pcolormesh(
        downscaled.lon, downscaled.lat, downscaled.values,
        cmap="YlOrRd", shading="auto", vmin=0, vmax=vmax,
    )
    axes[0].set_title(f"Downscaled {species} (mean)")

    # Absolute uncertainty range
    u_max = float(np.nanpercentile(unc_range.values, 98))
    im1 = axes[1].pcolormesh(
        unc_range.lon, unc_range.lat, unc_range.values,
        cmap="Purples", shading="auto", vmin=0, vmax=u_max,
    )
    axes[1].set_title("Uncertainty Range (P90 − P10)")
    fig.colorbar(im1, ax=axes[1], fraction=0.046)

    # Relative uncertainty (%)
    im2 = axes[2].pcolormesh(
        rel_unc.lon, rel_unc.lat,
        np.clip(rel_unc.values, 0, 200),
        cmap="RdYlGn_r", shading="auto", vmin=0, vmax=200,
    )
    axes[2].set_title("Relative Uncertainty (%)")
    fig.colorbar(im2, ax=axes[2], fraction=0.046, label="%")

    for ax in axes:
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

    fig.suptitle(
        f"{species} Downscaling — Uncertainty Quantification",
        fontsize=14, fontweight="bold", y=1.02,
    )
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path)
        logger.info("Uncertainty map saved to %s", output_path)

    return fig


# ---------------------------------------------------------------------------
# Summary report
# ---------------------------------------------------------------------------
def print_metrics_table(
    metrics: dict,
    species: str = "",
    header: str = "Evaluation Metrics",
) -> str:
    """
    Format metrics as a clean ASCII table for logging.
    """
    lines = [
        f"\n{'=' * 45}",
        f"  {header}" + (f" — {species}" if species else ""),
        f"{'=' * 45}",
    ]
    for key, val in metrics.items():
        if isinstance(val, float):
            if abs(val) > 1000 or (abs(val) < 0.01 and val != 0):
                lines.append(f"  {key:<25s}  {val:>12.4e}")
            else:
                lines.append(f"  {key:<25s}  {val:>12.4f}")
        else:
            lines.append(f"  {key:<25s}  {val!s:>12s}")
    lines.append(f"{'=' * 45}\n")
    table = "\n".join(lines)
    print(table)
    return table
