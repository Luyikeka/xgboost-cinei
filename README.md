# CINEI Spatial Downscaling with XGBoost

**XGBoost-based spatial downscaling of anthropogenic emission inventories using satellite observations**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![CINEI on PyPI](https://img.shields.io/pypi/v/cinei)](https://pypi.org/project/cinei/)
[![DOI](https://img.shields.io/badge/DOI-10.5194%2Fgmd--19--217--2026-blue)](https://doi.org/10.5194/gmd-19-217-2026)

## Overview

This repository provides a machine-learning pipeline for enhancing the spatial resolution of [CINEI](https://pypi.org/project/cinei/) (Coupled and Integrated National Emission Inventory) anthropogenic emission data using high-resolution satellite observations from TROPOMI and OMI.

**Motivation**: Next-generation Earth system models such as [ICON](https://mpimet.mpg.de/en/science/modeling) (MPI-M) and [AIFS](https://www.ecmwf.int/en/about/media-centre/aifs) (ECMWF) require kilometre-resolution emission inputs. CINEI provides 0.25° gridded emissions; satellite instruments observe atmospheric columns at 3.5–7 km. This tool bridges the gap by learning the statistical mapping between satellite-observed column densities and surface emission fluxes, then applying it at satellite resolution.

### Method

```
┌─────────────────────┐      ┌──────────────────────┐
│  CINEI Emissions     │      │  Satellite Columns    │
│  (0.25° / 0.10°)    │      │  (3.5–7 km)           │
│  CO, NOₓ, HCHO      │      │  TROPOMI / OMI        │
└─────────┬───────────┘      └──────────┬────────────┘
          │                              │
          ▼                              ▼
    ┌─────────────────────────────────────────┐
    │  Feature Engineering                     │
    │  • Satellite column densities            │
    │  • Spatial coords (trig-encoded)         │
    │  • Texture features (multi-scale)        │
    │  • Temporal encoding (month)             │
    │  • Satellite-to-emission ratios          │
    └──────────────────┬──────────────────────┘
                       │
                       ▼
    ┌─────────────────────────────────────────┐
    │  XGBoost Regression                      │
    │  • Mean model (squared error)            │
    │  • Quantile models (P10 / P90)           │
    │  • 5-fold cross-validation               │
    └──────────────────┬──────────────────────┘
                       │
                       ▼
    ┌─────────────────────────────────────────┐
    │  Post-processing                         │
    │  • Mass conservation enforcement         │
    │  • Non-negativity constraint             │
    │  • Uncertainty quantification            │
    └──────────────────┬──────────────────────┘
                       │
                       ▼
    ┌─────────────────────────────────────────┐
    │  High-Resolution Emission Field          │
    │  (~0.05° / ~5 km)                        │
    │  + Pixel-level uncertainty bounds        │
    └─────────────────────────────────────────┘
```

## Supported Species

| Species | Satellite Product | Emission Source |
|---------|------------------|-----------------|
| CO      | TROPOMI CO total column | CINEI v2.1, CEDS, EDGAR |
| NO₂/NOₓ | TROPOMI / OMI tropospheric NO₂ | CINEI v2.1, CEDS, EDGAR |
| HCHO    | TROPOMI / OMI tropospheric HCHO | CINEI v2.1, CEDS, EDGAR |

## Installation

```bash
# Clone the repository
git clone https://github.com/Luyikeka/cinei-downscaling.git
cd cinei-downscaling

# Install dependencies
pip install -r requirements.txt

# (Optional) Install CINEI
pip install cinei
```

### Requirements

- Python ≥ 3.9
- xgboost ≥ 2.0
- xarray, numpy, scipy, scikit-learn
- matplotlib (for visualisation)
- netCDF4 or h5netcdf (for I/O)

## Quick Start

### Demo with Synthetic Data

```bash
python run_downscaling.py --demo --output-dir ./demo_output --species CO
```

This generates synthetic emission hotspots and satellite fields over South America, trains XGBoost models, and produces evaluation figures.

### Production Run

```bash
python run_downscaling.py \
    --cinei-dir  /data/cinei/v2.1 \
    --sat-dir    /data/tropomi/L3 \
    --output-dir ./output \
    --species CO NOx HCHO \
    --months 1 4 7 10 \
    --cv-folds 5
```

## Project Structure

```
cinei-downscaling/
├── data_preprocessing.py     # Data loading, regridding, feature engineering
├── xgboost_downscaling.py    # XGBoost model: training, prediction, mass conservation
├── evaluation.py             # Metrics, spatial plots, scatter density, uncertainty maps
├── run_downscaling.py        # CLI entry point and pipeline orchestration
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## Key Features

- **Mass conservation**: post-hoc scaling ensures downscaled totals match original coarse-grid values within each cell
- **Uncertainty quantification**: quantile regression (P10/P90) provides pixel-level confidence intervals
- **Multi-scale texture features**: local mean, std, and gradient magnitude at 3×3, 5×5, and 9×9 windows
- **Trigonometric spatial encoding**: captures periodic geographic patterns without boundary artefacts
- **Log-space training**: handles the right-skewed distribution of emission fluxes
- **Publication-quality figures**: spatial comparison panels, scatter density plots, feature importance charts

## Data Sources

- **CINEI v2.1**: Zhang et al. (2026), *Geosci. Model Dev.*, [doi:10.5194/gmd-19-217-2026](https://doi.org/10.5194/gmd-19-217-2026)
- **CINEI data on PANGAEA**: [doi:10.1594/PANGAEA.974347](https://doi.org/10.1594/PANGAEA.974347)
- **CINEI Python package**: [pypi.org/project/cinei](https://pypi.org/project/cinei/)
- **TROPOMI**: Sentinel-5P satellite, ESA/Copernicus
- **OMI**: Aura satellite, NASA

## Citation

If you use this code, please cite:

```bibtex
@article{zhang2026cinei,
  title   = {The Coupled and Integrated National Emission Inventory (CINEI)},
  author  = {Zhang, Yijuan and others},
  journal = {Geoscientific Model Development},
  volume  = {19},
  pages   = {217--},
  year    = {2026},
  doi     = {10.5194/gmd-19-217-2026}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.

## Author

**Yijuan Zhang**
Institute of Environmental Physics (IUP), University of Bremen /
Max Planck Institute for Meteorology (MPI-M)

- GitHub: [@Luyikeka](https://github.com/Luyikeka)
- CINEI Documentation: [luyikeka.github.io/cinei](https://luyikeka.github.io/cinei/)
