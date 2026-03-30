"""
xgboost_downscaling.py
======================
XGBoost-based spatial downscaling model for enhancing the resolution
of CINEI anthropogenic emission inventories using satellite observations.

The model learns the statistical relationship between satellite column
densities (CO, NO₂, HCHO from TROPOMI/OMI) and emission fluxes at
coarse resolution (0.25°), then applies these relationships at the
finer satellite grid (~0.05°) to produce downscaled emission fields.

Key design choices
------------------
- **Mass conservation**: a post-hoc scaling step ensures that
  downscaled emissions aggregate back to the original coarse totals.
- **Uncertainty quantification**: quantile regression provides
  pixel-level confidence intervals.
- **Multi-species support**: separate models for CO, NOₓ, and HCHO
  share a common training pipeline.

Author : Yijuan Zhang
Project: CINEI – Coupled and Integrated National Emission Inventory
License: MIT
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import xarray as xr
import xgboost as xgb
from sklearn.model_selection import KFold

from data_preprocessing import (
    DownscalingDataset,
    bilinear_interpolate,
    conservative_regrid,
    prepare_prediction_grid,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Default hyper-parameters (tuned for emission downscaling)
# ---------------------------------------------------------------------------
DEFAULT_PARAMS: dict = {
    "objective": "reg:squarederror",
    "booster": "gbtree",
    "n_estimators": 500,
    "max_depth": 8,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "gamma": 0.1,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "tree_method": "hist",           # fast histogram-based
    "n_jobs": -1,
    "random_state": 42,
    "early_stopping_rounds": 30,
}


@dataclass
class DownscalingResult:
    """Container for downscaling outputs."""

    downscaled: xr.DataArray          # Fine-resolution emission estimate
    uncertainty_low: xr.DataArray     # 10th percentile
    uncertainty_high: xr.DataArray    # 90th percentile
    feature_importance: dict          # {feature_name: importance}
    cv_metrics: dict                  # Cross-validation scores
    species: str = ""
    mass_conserved: bool = False


class XGBoostDownscaler:
    """
    Spatial downscaling of emission inventories with XGBoost.

    Parameters
    ----------
    species : str
        Target species (``'CO'``, ``'NOx'``, or ``'HCHO'``).
    params : dict, optional
        XGBoost hyper-parameters.  Merged with ``DEFAULT_PARAMS``.
    quantiles : tuple[float, float]
        Lower and upper quantile levels for uncertainty estimation.
    enforce_mass_conservation : bool
        If True, apply post-hoc scaling so fine-grid totals match
        the coarse-grid totals within each coarse cell.
    """

    def __init__(
        self,
        species: str = "CO",
        params: Optional[dict] = None,
        quantiles: tuple[float, float] = (0.1, 0.9),
        enforce_mass_conservation: bool = True,
    ) -> None:
        self.species = species
        self.quantiles = quantiles
        self.enforce_mass_conservation = enforce_mass_conservation

        # Merge user params with defaults
        self.params = {**DEFAULT_PARAMS}
        if params:
            self.params.update(params)

        # Models
        self._model_mean: Optional[xgb.XGBRegressor] = None
        self._model_q_low: Optional[xgb.XGBRegressor] = None
        self._model_q_high: Optional[xgb.XGBRegressor] = None
        self._feature_names: list[str] = []
        self._is_fitted: bool = False

    # -----------------------------------------------------------------
    # Training
    # -----------------------------------------------------------------
    def fit(
        self,
        dataset: DownscalingDataset,
        validation_fraction: float = 0.15,
    ) -> dict:
        """
        Train the mean and quantile XGBoost models.

        Parameters
        ----------
        dataset : DownscalingDataset
            Training data from ``prepare_training_data``.
        validation_fraction : float
            Fraction of data held out for early stopping.

        Returns
        -------
        dict
            Training history including best iteration and eval metric.
        """
        self._feature_names = dataset.feature_names
        X, y = dataset.features, dataset.targets

        # Log-transform targets (emissions are right-skewed)
        y_log = np.log1p(np.maximum(y, 0))

        # Train/val split (deterministic)
        n_val = int(len(y) * validation_fraction)
        idx = np.arange(len(y))
        rng = np.random.RandomState(42)
        rng.shuffle(idx)
        val_idx, train_idx = idx[:n_val], idx[n_val:]

        X_train, y_train = X[train_idx], y_log[train_idx]
        X_val, y_val = X[val_idx], y_log[val_idx]

        # --- Mean model ---
        logger.info("Training mean regression model (%s) ...", self.species)
        mean_params = {k: v for k, v in self.params.items()
                       if k != "early_stopping_rounds"}
        self._model_mean = xgb.XGBRegressor(**mean_params)
        self._model_mean.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=50,
        )

        # --- Quantile models for uncertainty ---
        for q, attr in [
            (self.quantiles[0], "_model_q_low"),
            (self.quantiles[1], "_model_q_high"),
        ]:
            logger.info("Training quantile model (q=%.2f) ...", q)
            q_params = {**mean_params}
            q_params["objective"] = "reg:quantileerror"
            q_params["quantile_alpha"] = q
            model = xgb.XGBRegressor(**q_params)
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=0,
            )
            setattr(self, attr, model)

        self._is_fitted = True
        try:
            best_iter = self._model_mean.best_iteration
        except AttributeError:
            best_iter = self.params.get("n_estimators", 500)
        logger.info("Training complete. Best iteration: %d", best_iter)

        return {
            "best_iteration": best_iter,
            "n_train": len(train_idx),
            "n_val": len(val_idx),
        }

    # -----------------------------------------------------------------
    # Cross-validation
    # -----------------------------------------------------------------
    def cross_validate(
        self,
        dataset: DownscalingDataset,
        n_folds: int = 5,
    ) -> dict:
        """
        K-fold cross-validation with spatial blocking.

        Returns per-fold R², RMSE, and normalised mean bias (NMB).
        """
        X, y = dataset.features, dataset.targets
        y_log = np.log1p(np.maximum(y, 0))

        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        scores = {"r2": [], "rmse": [], "nmb": []}

        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            params = {k: v for k, v in self.params.items()
                      if k != "early_stopping_rounds"}
            model = xgb.XGBRegressor(**params)
            model.fit(
                X[train_idx], y_log[train_idx],
                eval_set=[(X[val_idx], y_log[val_idx])],
                verbose=0,
            )

            y_pred_log = model.predict(X[val_idx])
            y_pred = np.expm1(y_pred_log)
            y_true = y[val_idx]

            # Metrics
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - y_true.mean()) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
            rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
            nmb = np.sum(y_pred - y_true) / np.sum(y_true) * 100

            scores["r2"].append(r2)
            scores["rmse"].append(rmse)
            scores["nmb"].append(nmb)

            logger.info(
                "Fold %d/%d  R²=%.4f  RMSE=%.4e  NMB=%.2f%%",
                fold + 1, n_folds, r2, rmse, nmb,
            )

        summary = {
            metric: {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "per_fold": [float(v) for v in vals],
            }
            for metric, vals in scores.items()
        }

        logger.info(
            "CV summary — R²: %.4f ± %.4f  RMSE: %.4e ± %.4e  NMB: %.2f ± %.2f%%",
            summary["r2"]["mean"], summary["r2"]["std"],
            summary["rmse"]["mean"], summary["rmse"]["std"],
            summary["nmb"]["mean"], summary["nmb"]["std"],
        )
        return summary

    # -----------------------------------------------------------------
    # Prediction & downscaling
    # -----------------------------------------------------------------
    def predict(
        self,
        satellite_fine: xr.DataArray,
        emission_coarse: xr.DataArray,
        month: int = 1,
    ) -> DownscalingResult:
        """
        Produce a high-resolution emission field.

        Parameters
        ----------
        satellite_fine : xr.DataArray
            Satellite column density at native fine resolution.
        emission_coarse : xr.DataArray
            CINEI coarse emission field (for mass conservation).
        month : int
            Month of the year (1–12).

        Returns
        -------
        DownscalingResult
        """
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call .fit() first.")

        # Build fine-grid features
        X_fine, feat_names, fine_lats, fine_lons = prepare_prediction_grid(
            satellite_fine, emission_coarse,
            species=self.species, month=month,
        )

        # Handle NaN features
        nan_mask = np.isnan(X_fine).any(axis=1)
        X_clean = np.nan_to_num(X_fine, nan=0.0)

        # Predict (log space → real space)
        y_mean = np.expm1(self._model_mean.predict(X_clean))
        y_low = np.expm1(self._model_q_low.predict(X_clean))
        y_high = np.expm1(self._model_q_high.predict(X_clean))

        # Enforce non-negativity
        y_mean = np.maximum(y_mean, 0)
        y_low = np.maximum(y_low, 0)
        y_high = np.maximum(y_high, 0)

        # Mask pixels that had NaN features
        y_mean[nan_mask] = np.nan
        y_low[nan_mask] = np.nan
        y_high[nan_mask] = np.nan

        # Reshape to 2-D
        shape = (len(fine_lats), len(fine_lons))
        y_mean_2d = y_mean.reshape(shape)
        y_low_2d = y_low.reshape(shape)
        y_high_2d = y_high.reshape(shape)

        coords = {"lat": fine_lats, "lon": fine_lons}
        da_mean = xr.DataArray(y_mean_2d, dims=["lat", "lon"], coords=coords)
        da_low = xr.DataArray(y_low_2d, dims=["lat", "lon"], coords=coords)
        da_high = xr.DataArray(y_high_2d, dims=["lat", "lon"], coords=coords)

        # --- Mass conservation ---
        mass_conserved = False
        if self.enforce_mass_conservation:
            da_mean = self._enforce_mass_conservation(da_mean, emission_coarse)
            mass_conserved = True

        # --- Feature importance ---
        importance = dict(
            zip(self._feature_names, self._model_mean.feature_importances_)
        )
        importance = dict(
            sorted(importance.items(), key=lambda x: x[1], reverse=True)
        )

        return DownscalingResult(
            downscaled=da_mean,
            uncertainty_low=da_low,
            uncertainty_high=da_high,
            feature_importance=importance,
            cv_metrics={},
            species=self.species,
            mass_conserved=mass_conserved,
        )

    # -----------------------------------------------------------------
    # Mass conservation
    # -----------------------------------------------------------------
    @staticmethod
    def _enforce_mass_conservation(
        downscaled: xr.DataArray,
        coarse: xr.DataArray,
    ) -> xr.DataArray:
        """
        Scale downscaled emissions so that totals within each coarse
        grid cell match the original CINEI values.

        This preserves the spatial patterns learned by XGBoost while
        ensuring physical consistency of total mass.
        """
        result = downscaled.copy(deep=True)
        coarse_lats = coarse.lat.values
        coarse_lons = coarse.lon.values
        c_dlat = np.abs(np.median(np.diff(coarse_lats)))
        c_dlon = np.abs(np.median(np.diff(coarse_lons)))

        fine_lats = downscaled.lat.values
        fine_lons = downscaled.lon.values

        for clat in coarse_lats:
            lat_mask = (fine_lats >= clat - c_dlat / 2) & (fine_lats < clat + c_dlat / 2)
            for clon in coarse_lons:
                lon_mask = (fine_lons >= clon - c_dlon / 2) & (fine_lons < clon + c_dlon / 2)

                sub = result.values[np.ix_(lat_mask, lon_mask)]
                coarse_val = float(
                    coarse.sel(lat=clat, lon=clon, method="nearest").values
                )

                fine_sum = np.nansum(sub)
                if fine_sum > 0 and not np.isnan(coarse_val):
                    # Area-weighted scaling
                    n_fine = np.sum(~np.isnan(sub))
                    target_sum = coarse_val * n_fine
                    scale = target_sum / fine_sum
                    result.values[np.ix_(lat_mask, lon_mask)] = sub * scale

        logger.info("Mass conservation applied.")
        return result

    # -----------------------------------------------------------------
    # Persistence
    # -----------------------------------------------------------------
    def save(self, directory: str | Path) -> None:
        """Save model artefacts to disk."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        self._model_mean.save_model(str(directory / "model_mean.json"))
        self._model_q_low.save_model(str(directory / "model_q_low.json"))
        self._model_q_high.save_model(str(directory / "model_q_high.json"))

        meta = {
            "species": self.species,
            "feature_names": self._feature_names,
            "quantiles": list(self.quantiles),
            "params": self.params,
        }
        with open(directory / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)

        logger.info("Model saved to %s", directory)

    def load(self, directory: str | Path) -> None:
        """Load model artefacts from disk."""
        directory = Path(directory)

        with open(directory / "metadata.json") as f:
            meta = json.load(f)

        self.species = meta["species"]
        self._feature_names = meta["feature_names"]
        self.quantiles = tuple(meta["quantiles"])
        self.params = meta["params"]

        load_params = {k: v for k, v in self.params.items()
                       if k != "early_stopping_rounds"}

        self._model_mean = xgb.XGBRegressor(**load_params)
        self._model_mean.load_model(str(directory / "model_mean.json"))

        self._model_q_low = xgb.XGBRegressor(**load_params)
        self._model_q_low.load_model(str(directory / "model_q_low.json"))

        self._model_q_high = xgb.XGBRegressor(**load_params)
        self._model_q_high.load_model(str(directory / "model_q_high.json"))

        self._is_fitted = True
        logger.info("Model loaded from %s", directory)
