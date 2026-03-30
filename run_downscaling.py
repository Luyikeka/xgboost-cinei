"""
run_downscaling.py
==================
End-to-end pipeline for XGBoost-based spatial downscaling of
CINEI anthropogenic emission inventories using satellite observations.

This script demonstrates the full workflow:

    1. Load CINEI emission data (0.25°) and satellite retrievals (~0.05°)
    2. Build training features (satellite + spatial + texture)
    3. Train XGBoost mean and quantile regression models
    4. Cross-validate with spatial blocking
    5. Predict high-resolution emissions on the satellite grid
    6. Enforce mass conservation
    7. Evaluate and visualise results

Usage
-----
    python run_downscaling.py \\
        --cinei-dir  /path/to/cinei/data \\
        --sat-dir    /path/to/satellite/data \\
        --output-dir /path/to/output \\
        --species    CO NOx HCHO \\
        --months     1 7

Author : Yijuan Zhang
Project: CINEI – Coupled and Integrated National Emission Inventory
         (https://pypi.org/project/cinei/)
DOI    : 10.5194/gmd-19-217-2026
License: MIT
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import xarray as xr

from data_preprocessing import (
    GridSpec,
    load_cinei_emissions,
    load_satellite_l3,
    prepare_training_data,
)
from xgboost_downscaling import XGBoostDownscaler
from evaluation import (
    compute_metrics,
    plot_spatial_comparison,
    plot_scatter_density,
    plot_feature_importance,
    plot_uncertainty_map,
    print_metrics_table,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-24s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("cinei.downscaling")


# ---------------------------------------------------------------------------
# Synthetic data for demonstration / CI testing
# ---------------------------------------------------------------------------
def create_demo_data(
    domain: str = "south_america",
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    """
    Generate synthetic emission and satellite fields for demonstration.

    The synthetic data mimics realistic spatial patterns:
    - Urban emission hotspots (Gaussian blobs)
    - Background gradient (latitude-dependent)
    - Satellite columns correlated with emissions + noise

    Parameters
    ----------
    domain : str
        ``'south_america'`` or ``'china'``.

    Returns
    -------
    emission_coarse : xr.DataArray  (0.25° grid)
    satellite_fine  : xr.DataArray  (0.05° grid)
    emission_fine   : xr.DataArray  (0.10° reference)
    """
    rng = np.random.RandomState(2025)

    if domain == "south_america":
        bbox = {"lat_min": -30, "lat_max": -10, "lon_min": -55, "lon_max": -35}
        hotspots = [
            (-23.55, -46.63, 5.0),   # São Paulo
            (-22.91, -43.17, 3.5),   # Rio de Janeiro
            (-12.97, -38.51, 2.0),   # Salvador
        ]
    else:  # china
        bbox = {"lat_min": 28, "lat_max": 42, "lon_min": 110, "lon_max": 125}
        hotspots = [
            (39.9, 116.4, 8.0),    # Beijing
            (31.2, 121.5, 7.0),    # Shanghai
            (34.3, 113.6, 4.0),    # Zhengzhou
        ]

    def make_field(lats, lons, scale=1.0):
        lat2d, lon2d = np.meshgrid(lats, lons, indexing="ij")
        field = np.zeros_like(lat2d)

        # Background: latitude-dependent
        field += 0.5 + 0.3 * np.cos(np.deg2rad(lat2d * 2))

        # Hotspots: Gaussian blobs
        for hlat, hlon, intensity in hotspots:
            dist_sq = (lat2d - hlat) ** 2 + (lon2d - hlon) ** 2
            field += intensity * scale * np.exp(-dist_sq / 8.0)

        # Noise
        field += rng.exponential(0.1, field.shape)
        return np.maximum(field, 0)

    # Coarse grid (0.25°)
    coarse_lats = np.arange(bbox["lat_min"] + 0.125, bbox["lat_max"], 0.25)
    coarse_lons = np.arange(bbox["lon_min"] + 0.125, bbox["lon_max"], 0.25)
    emission_coarse = xr.DataArray(
        make_field(coarse_lats, coarse_lons, scale=1.0),
        dims=["lat", "lon"],
        coords={"lat": coarse_lats, "lon": coarse_lons},
        attrs={"units": "kg/m²/s", "long_name": "CO emission flux"},
    )

    # Fine satellite grid (0.05°)
    fine_lats = np.arange(bbox["lat_min"] + 0.025, bbox["lat_max"], 0.05)
    fine_lons = np.arange(bbox["lon_min"] + 0.025, bbox["lon_max"], 0.05)
    sat_base = make_field(fine_lats, fine_lons, scale=0.8)
    # Add satellite-specific noise and bias
    satellite_fine = xr.DataArray(
        sat_base * (1 + 0.2 * rng.randn(*sat_base.shape)),
        dims=["lat", "lon"],
        coords={"lat": fine_lats, "lon": fine_lons},
        attrs={"units": "mol/m²", "long_name": "CO total column"},
    )

    # Reference fine grid (0.10°)
    ref_lats = np.arange(bbox["lat_min"] + 0.05, bbox["lat_max"], 0.10)
    ref_lons = np.arange(bbox["lon_min"] + 0.05, bbox["lon_max"], 0.10)
    emission_fine = xr.DataArray(
        make_field(ref_lats, ref_lons, scale=1.0),
        dims=["lat", "lon"],
        coords={"lat": ref_lats, "lon": ref_lons},
        attrs={"units": "kg/m²/s", "long_name": "CO emission flux (reference)"},
    )

    logger.info(
        "Demo data created — coarse: %s  satellite: %s  reference: %s",
        emission_coarse.shape, satellite_fine.shape, emission_fine.shape,
    )
    return emission_coarse, satellite_fine, emission_fine


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def run_pipeline(
    cinei_dir: Path | None,
    sat_dir: Path | None,
    output_dir: Path,
    species_list: list[str],
    months: list[int],
    demo: bool = False,
    n_cv_folds: int = 5,
) -> None:
    """Execute the full downscaling pipeline."""

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "models").mkdir(exist_ok=True)
    (output_dir / "figures").mkdir(exist_ok=True)
    (output_dir / "data").mkdir(exist_ok=True)

    for species in species_list:
        for month in months:
            logger.info(
                "\n%s\n  Species: %s  |  Month: %02d\n%s",
                "=" * 60, species, month, "=" * 60,
            )

            # ----- Load data -----
            if demo:
                domain = "south_america"
                emis_coarse, sat_fine, emis_fine = create_demo_data(domain)
            else:
                cinei_file = cinei_dir / f"CINEI_v2.1_{species}_2019.nc"
                sat_file = sat_dir / f"TROPOMI_{species}_L3_{month:02d}_2019.nc"

                emis_coarse = load_cinei_emissions(cinei_file, species, month=month)
                sat_fine = load_satellite_l3(sat_file, species)
                emis_fine = None  # No fine reference in production

            # ----- Prepare training data -----
            logger.info("Building training features ...")
            dataset = prepare_training_data(
                emis_coarse, sat_fine,
                emission_fine=emis_fine if demo else None,
                species=species,
                month=month,
            )
            logger.info(
                "Feature matrix: %s  |  Features: %d",
                dataset.features.shape, len(dataset.feature_names),
            )

            # ----- Cross-validation -----
            downscaler = XGBoostDownscaler(
                species=species,
                enforce_mass_conservation=True,
            )

            logger.info("Running %d-fold cross-validation ...", n_cv_folds)
            cv_metrics = downscaler.cross_validate(dataset, n_folds=n_cv_folds)
            print_metrics_table(
                {k: v["mean"] for k, v in cv_metrics.items()},
                species=species,
                header=f"CV Results (month={month:02d})",
            )

            # ----- Train final model -----
            logger.info("Training final model ...")
            train_info = downscaler.fit(dataset)

            # ----- Predict on fine grid -----
            logger.info("Predicting on satellite-resolution grid ...")
            result = downscaler.predict(sat_fine, emis_coarse, month=month)
            result.cv_metrics = cv_metrics

            # ----- Evaluate against reference -----
            if emis_fine is not None:
                from data_preprocessing import conservative_regrid
                # Regrid both to the reference grid for comparison
                ds_regridded = conservative_regrid(
                    result.downscaled,
                    emis_fine.lat.values,
                    emis_fine.lon.values,
                )
                metrics = compute_metrics(
                    emis_fine.values.ravel(),
                    ds_regridded.values.ravel(),
                )
                print_metrics_table(
                    metrics, species=species,
                    header=f"Validation vs Reference (month={month:02d})",
                )

                # Scatter plot
                plot_scatter_density(
                    emis_fine.values.ravel(),
                    ds_regridded.values.ravel(),
                    species=species,
                    title=f"{species} Downscaling — Month {month:02d}",
                    output_path=output_dir / "figures" / f"scatter_{species}_m{month:02d}.png",
                )

            # ----- Visualisation -----
            plot_spatial_comparison(
                emis_coarse, result.downscaled,
                satellite=sat_fine,
                reference=emis_fine,
                species=species,
                title=f"CINEI {species} Downscaling — Month {month:02d}",
                output_path=output_dir / "figures" / f"spatial_{species}_m{month:02d}.png",
            )

            plot_feature_importance(
                result.feature_importance,
                species=species,
                output_path=output_dir / "figures" / f"importance_{species}_m{month:02d}.png",
            )

            plot_uncertainty_map(
                result.downscaled,
                result.uncertainty_low,
                result.uncertainty_high,
                species=species,
                output_path=output_dir / "figures" / f"uncertainty_{species}_m{month:02d}.png",
            )

            # ----- Save outputs -----
            downscaler.save(output_dir / "models" / f"{species}_m{month:02d}")

            ds_out = xr.Dataset({
                f"{species}_downscaled": result.downscaled,
                f"{species}_unc_low": result.uncertainty_low,
                f"{species}_unc_high": result.uncertainty_high,
            })
            ds_out.attrs = {
                "title": f"CINEI v2.1 XGBoost-downscaled {species} emissions",
                "source": "CINEI (https://pypi.org/project/cinei/)",
                "method": "XGBoost spatial downscaling with TROPOMI satellite data",
                "mass_conserved": str(result.mass_conserved),
                "doi": "10.5194/gmd-19-217-2026",
            }
            nc_path = output_dir / "data" / f"CINEI_downscaled_{species}_m{month:02d}.nc"
            ds_out.to_netcdf(nc_path)
            logger.info("Saved downscaled NetCDF → %s", nc_path)

    logger.info("\nPipeline complete.  Outputs in %s", output_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "XGBoost spatial downscaling of CINEI emission inventories "
            "using satellite observations."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with demo (synthetic) data
  python run_downscaling.py --demo --output-dir ./output

  # Run with real data
  python run_downscaling.py \\
      --cinei-dir  /data/cinei/v2.1 \\
      --sat-dir    /data/tropomi/L3 \\
      --output-dir ./output \\
      --species CO NOx HCHO \\
      --months 1 4 7 10
        """,
    )
    parser.add_argument(
        "--cinei-dir", type=Path, default=None,
        help="Directory containing CINEI v2.1 NetCDF files.",
    )
    parser.add_argument(
        "--sat-dir", type=Path, default=None,
        help="Directory containing satellite L3 NetCDF files.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("./output"),
        help="Output directory for models, figures, and data.",
    )
    parser.add_argument(
        "--species", nargs="+", default=["CO", "NOx", "HCHO"],
        help="Species to downscale.",
    )
    parser.add_argument(
        "--months", nargs="+", type=int, default=[1, 7],
        help="Months to process (1–12).",
    )
    parser.add_argument(
        "--demo", action="store_true",
        help="Use synthetic demo data (no real files needed).",
    )
    parser.add_argument(
        "--cv-folds", type=int, default=5,
        help="Number of cross-validation folds.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.demo and (args.cinei_dir is None or args.sat_dir is None):
        logger.error(
            "Must provide --cinei-dir and --sat-dir, or use --demo."
        )
        sys.exit(1)

    run_pipeline(
        cinei_dir=args.cinei_dir,
        sat_dir=args.sat_dir,
        output_dir=args.output_dir,
        species_list=args.species,
        months=args.months,
        demo=args.demo,
        n_cv_folds=args.cv_folds,
    )


if __name__ == "__main__":
    main()
