"""
MLTrainer - Core training orchestration for FPL ML models.

Provides unified training interface for:
- Unified models (all positions)
- Position-specific models
- Hyperparameter optimization
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.model_selection import RandomizedSearchCV

from .config import TrainingConfig, PositionTrainingConfig
from .pipelines import (
    build_pipeline,
    get_param_space,
)

# Import from existing utilities (will be moved eventually)
import sys

_project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(_project_root / "scripts"))

from ml_training_utils import (  # noqa: E402
    TemporalCVSplitter,
    load_training_data,
    engineer_features,
    create_temporal_cv_splits,
    spearman_correlation_scorer,
    fpl_weighted_huber_scorer_sklearn,
    fpl_topk_scorer_sklearn,
    fpl_captain_scorer_sklearn,
    fpl_hauler_capture_scorer_sklearn,
    fpl_hauler_ceiling_scorer_sklearn,
)

# Custom scorer mapping
CUSTOM_SCORERS = {
    "spearman": spearman_correlation_scorer,
    "fpl_weighted_huber": fpl_weighted_huber_scorer_sklearn,
    "fpl_top_k_ranking": fpl_topk_scorer_sklearn,
    "fpl_captain_pick": fpl_captain_scorer_sklearn,
    "fpl_hauler_capture": fpl_hauler_capture_scorer_sklearn,
    "fpl_hauler_ceiling": fpl_hauler_ceiling_scorer_sklearn,  # Hauler-first with variance preservation
}


class MLTrainer:
    """
    Unified ML training orchestrator.

    Handles data loading, feature engineering, hyperparameter optimization,
    and model training for different configurations.
    """

    def _resolve_scorer(self, scorer_name: str):
        """
        Resolve scorer string to sklearn-compatible scorer.

        Custom FPL scorers (fpl_weighted_huber, spearman, etc.) are mapped
        to their actual sklearn make_scorer functions.

        Args:
            scorer_name: Name of scorer (sklearn string or custom name)

        Returns:
            Scorer compatible with sklearn's RandomizedSearchCV
        """
        if scorer_name in CUSTOM_SCORERS:
            return CUSTOM_SCORERS[scorer_name]
        # Return as-is for sklearn built-in scorers (neg_mean_absolute_error, etc.)
        return scorer_name

    def __init__(self, config: TrainingConfig):
        """
        Initialize trainer with configuration.

        Args:
            config: Training configuration
        """
        self.config = config
        self._data_cache: Optional[Tuple] = None
        self._features_cache: Optional[Tuple[pd.DataFrame, np.ndarray, List[str]]] = (
            None
        )

    def _load_data(self) -> Tuple:
        """Load training data (cached)."""
        if self._data_cache is None:
            logger.info(
                f"📦 Loading data GW{self.config.start_gw}-{self.config.end_gw}..."
            )
            self._data_cache = load_training_data(
                self.config.start_gw,
                self.config.end_gw,
                verbose=self.config.verbose > 0,
            )
        return self._data_cache

    def _engineer_features(self) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
        """Engineer features (cached)."""
        if self._features_cache is None:
            data = self._load_data()
            logger.info("🔧 Engineering features...")
            self._features_cache = engineer_features(
                *data, verbose=self.config.verbose > 0
            )
        return self._features_cache

    def select_features(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        feature_names: List[str],
    ) -> List[str]:
        """
        Select features based on configured strategy.

        Args:
            X: Feature matrix
            y: Target variable
            feature_names: All available feature names

        Returns:
            List of selected feature names
        """
        strategy = self.config.feature_selection

        if strategy == "none":
            logger.info(f"   Using all {len(feature_names)} features")
            return feature_names

        penalty_features = [
            "is_primary_penalty_taker",
            "is_penalty_taker",
            "is_corner_taker",
            "is_fk_taker",
        ]

        if strategy == "correlation":
            logger.info("   Removing highly correlated features (|r| > 0.95)...")
            corr_matrix = X[feature_names].corr().abs()
            upper_tri = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            )

            to_drop = set()
            for column in upper_tri.columns:
                if column in to_drop:
                    continue
                high_corr = upper_tri[column][upper_tri[column] > 0.95].index.tolist()
                if self.config.keep_penalty_features and column in penalty_features:
                    to_drop.update([c for c in high_corr if c not in penalty_features])
                else:
                    to_drop.update(high_corr)

            selected = [f for f in feature_names if f not in to_drop]
            logger.info(
                f"   Removed {len(to_drop)} correlated features, kept {len(selected)}"
            )

        elif strategy == "permutation":
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.inspection import permutation_importance

            logger.info("   Computing permutation importance...")
            rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            rf.fit(X[feature_names], y)

            perm_imp = permutation_importance(
                rf, X[feature_names], y, n_repeats=5, random_state=42, n_jobs=-1
            )

            feature_importance = pd.DataFrame(
                {
                    "feature": feature_names,
                    "importance": perm_imp.importances_mean,
                }
            ).sort_values("importance", ascending=False)

            top_features = feature_importance.head(60)["feature"].tolist()

            if self.config.keep_penalty_features:
                for pf in penalty_features:
                    if pf not in top_features and pf in feature_names:
                        top_features.append(pf)

            selected = top_features
            logger.info(
                f"   Selected top {len(selected)} features by permutation importance"
            )

        elif strategy == "rfe-smart":
            from sklearn.ensemble import ExtraTreesRegressor
            from sklearn.feature_selection import RFE

            logger.info("   Running smart RFE...")
            estimator = ExtraTreesRegressor(
                n_estimators=100, random_state=42, n_jobs=-1
            )

            if self.config.keep_penalty_features:
                non_penalty = [f for f in feature_names if f not in penalty_features]
                selector = RFE(estimator, n_features_to_select=56, step=0.1)
                selector.fit(X[non_penalty], y)
                selected_non_penalty = [
                    f for f, keep in zip(non_penalty, selector.support_) if keep
                ]
                selected = selected_non_penalty + penalty_features
            else:
                selector = RFE(estimator, n_features_to_select=60, step=0.1)
                selector.fit(X[feature_names], y)
                selected = [
                    f for f, keep in zip(feature_names, selector.support_) if keep
                ]

            logger.info(f"   RFE selected {len(selected)} features")
        else:
            selected = feature_names

        # Ensure penalty features included if requested
        if self.config.keep_penalty_features:
            for pf in penalty_features:
                if pf not in selected and pf in feature_names:
                    selected.append(pf)

        return selected

    def train_unified(
        self,
        best_params: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Train a unified model on all positions.

        Args:
            best_params: Optional pre-determined hyperparameters (skips search)

        Returns:
            Tuple of (trained_pipeline, metadata_dict)
        """
        features_df, target, all_feature_names = self._engineer_features()

        # Prepare data
        X = features_df[all_feature_names].copy()
        y = target

        # Feature selection
        logger.info("🔧 Feature selection...")
        selected_features = self.select_features(X, y, all_feature_names)

        if best_params is not None:
            # Use provided params, skip search
            logger.info(
                f"🎯 Training {self.config.regressor} with provided hyperparameters..."
            )
            pipeline = build_pipeline(
                self.config.regressor,
                selected_features,
                self.config.preprocessing,
                self.config.random_seed,
                best_params,
            )
            pipeline.fit(X, y)

            metadata = {
                "best_params": best_params,
                "best_score": None,
                "selected_features": selected_features,
                "n_samples": len(y),
            }
        else:
            # Run hyperparameter search
            logger.info(f"🔍 Optimizing {self.config.regressor} hyperparameters...")

            cv_splits, cv_data = create_temporal_cv_splits(
                features_df,
                max_folds=self.config.cv_folds,
                verbose=self.config.verbose > 0,
            )

            X_cv = cv_data[all_feature_names].copy()
            y_cv = target[cv_data["_original_index"].values]

            pipeline = build_pipeline(
                self.config.regressor,
                selected_features,
                self.config.preprocessing,
                self.config.random_seed,
            )

            param_dist = get_param_space(self.config.regressor)
            cv_splitter = TemporalCVSplitter(cv_splits)
            scorer = self._resolve_scorer(self.config.scorer)

            search = RandomizedSearchCV(
                pipeline,
                param_distributions=param_dist,
                n_iter=self.config.n_trials,
                cv=cv_splitter,
                scoring=scorer,
                n_jobs=self.config.n_jobs,
                random_state=self.config.random_seed,
                verbose=self.config.verbose,
                return_train_score=True,
            )

            search.fit(X_cv, y_cv)

            logger.info(f"   Best CV score: {search.best_score_:.4f}")

            # Retrain on all data with best params
            logger.info("🎯 Retraining on all data...")
            pipeline = search.best_estimator_
            pipeline.fit(X, y)

            metadata = {
                "best_params": search.best_params_,
                "best_score": search.best_score_,
                "selected_features": selected_features,
                "n_samples": len(y),
            }

        return pipeline, metadata

    def train_position(
        self,
        position: str,
        regressor: Optional[str] = None,
        best_params: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Train a model for a single position.

        Args:
            position: Position code (GKP, DEF, MID, FWD)
            regressor: Override regressor (default: use config)
            best_params: Optional pre-determined hyperparameters

        Returns:
            Tuple of (trained_pipeline, metadata_dict)
        """
        regressor = regressor or self.config.regressor

        features_df, target, all_feature_names = self._engineer_features()

        # Filter to position
        pos_mask = features_df["position"] == position
        features_pos = features_df[pos_mask].reset_index(drop=True)
        target_pos = target[pos_mask]

        logger.info(f"📍 Training {position} model ({len(features_pos)} samples)...")

        # Add position-specific features
        pos_feature_names = list(all_feature_names)
        pos_additions = self._get_position_feature_additions(position)

        for feat_name, feat_func in pos_additions:
            if feat_name not in features_pos.columns:
                features_pos[feat_name] = feat_func(features_pos)
                pos_feature_names.append(feat_name)

        # Remove duplicates
        pos_feature_names = list(dict.fromkeys(pos_feature_names))

        X = features_pos[pos_feature_names].copy()
        y = target_pos

        # Feature selection
        selected_features = self.select_features(X, y, pos_feature_names)

        if best_params is not None:
            pipeline = build_pipeline(
                regressor,
                selected_features,
                self.config.preprocessing,
                self.config.random_seed,
                best_params,
            )
            pipeline.fit(X, y)

            metadata = {
                "position": position,
                "regressor": regressor,
                "best_params": best_params,
                "best_score": None,
                "selected_features": selected_features,
                "n_samples": len(y),
            }
        else:
            # Need CV splits for this position
            cv_splits, cv_data = create_temporal_cv_splits(
                features_pos, max_folds=self.config.cv_folds, verbose=False
            )

            X_cv = cv_data[pos_feature_names].copy()
            y_cv = (
                target_pos[cv_data["_original_index"].values]
                if "_original_index" in cv_data.columns
                else target_pos
            )

            # Add position features to cv_data too
            for feat_name, feat_func in pos_additions:
                if feat_name not in X_cv.columns:
                    X_cv[feat_name] = feat_func(X_cv)

            pipeline = build_pipeline(
                regressor,
                selected_features,
                self.config.preprocessing,
                self.config.random_seed,
            )

            param_dist = get_param_space(regressor)
            cv_splitter = TemporalCVSplitter(cv_splits)
            scorer = self._resolve_scorer(self.config.scorer)

            search = RandomizedSearchCV(
                pipeline,
                param_distributions=param_dist,
                n_iter=self.config.n_trials,
                cv=cv_splitter,
                scoring=scorer,
                n_jobs=self.config.n_jobs,
                random_state=self.config.random_seed,
                verbose=0,
            )

            search.fit(X_cv, y_cv)

            # Retrain on all position data
            pipeline = search.best_estimator_
            pipeline.fit(X, y)

            metadata = {
                "position": position,
                "regressor": regressor,
                "best_params": search.best_params_,
                "best_score": search.best_score_,
                "selected_features": selected_features,
                "n_samples": len(y),
            }

        return pipeline, metadata

    def train_all_positions(
        self,
        regressors: Optional[List[str]] = None,
    ) -> Dict[str, Tuple[Any, Dict[str, Any]]]:
        """
        Train models for all positions, picking best regressor for each.

        Args:
            regressors: List of regressors to try (default: from config)

        Returns:
            Dict mapping position -> (best_pipeline, metadata)
        """
        if isinstance(self.config, PositionTrainingConfig):
            positions = self.config.positions
            regressors = regressors or self.config.regressors
        else:
            positions = ["GKP", "DEF", "MID", "FWD"]
            regressors = regressors or [self.config.regressor]

        results = {}

        for position in positions:
            logger.info(f"\n{'=' * 50}")
            logger.info(f"🎯 Optimizing {position}")
            logger.info(f"{'=' * 50}")

            best_model = None
            best_metadata = {"best_score": float("-inf")}

            for reg in regressors:
                logger.info(f"\n   Testing {reg}...")
                try:
                    model, metadata = self.train_position(position, regressor=reg)

                    score = metadata.get("best_score", float("-inf"))
                    if score is not None and score > best_metadata.get(
                        "best_score", float("-inf")
                    ):
                        best_model = model
                        best_metadata = metadata
                        logger.info(f"   ✅ {reg}: score = {score:.4f} (new best)")
                    else:
                        logger.info(f"   {reg}: score = {score}")

                except Exception as e:
                    logger.warning(f"   ❌ {reg} failed: {e}")

            if best_model is not None:
                results[position] = (best_model, best_metadata)
                logger.info(
                    f"\n🏆 {position} best: {best_metadata['regressor']} (score: {best_metadata['best_score']:.4f})"
                )
            else:
                logger.error(f"   No model trained for {position}")

        return results

    def save_model(
        self,
        pipeline: Any,
        metadata: Dict[str, Any],
        name_prefix: Optional[str] = None,
    ) -> Path:
        """
        Save trained model and metadata.

        Args:
            pipeline: Trained sklearn pipeline
            metadata: Training metadata
            name_prefix: Optional prefix for filename

        Returns:
            Path to saved model
        """
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if name_prefix:
            base_name = f"{name_prefix}_gw{self.config.start_gw}-{self.config.end_gw}_{timestamp}"
        else:
            base_name = f"{self.config.regressor}_gw{self.config.start_gw}-{self.config.end_gw}_{timestamp}"

        # Save pipeline
        pipeline_path = output_dir / f"{base_name}_pipeline.joblib"
        joblib.dump(pipeline, pipeline_path)

        # Save metadata
        metadata_to_save = {
            **metadata,
            "config": self.config.model_dump()
            if hasattr(self.config, "model_dump")
            else vars(self.config),
            "created_at": datetime.now().isoformat(),
        }

        metadata_path = output_dir / f"{base_name}.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata_to_save, f, indent=2, default=str)

        logger.info(f"💾 Saved: {pipeline_path.name}")

        return pipeline_path

    def _get_position_feature_additions(
        self, position: str
    ) -> List[Tuple[str, callable]]:
        """Get position-specific feature additions."""
        additions = {
            "GKP": [
                (
                    "saves_x_opp_xg",
                    lambda df: df.get("rolling_5gw_saves", 0)
                    * df.get("opponent_rolling_5gw_xg", 0),
                ),
                (
                    "clean_sheet_potential",
                    lambda df: df.get("clean_sheet_probability_enhanced", 0)
                    * df.get("rolling_5gw_minutes", 0)
                    / 90,
                ),
            ],
            "DEF": [
                (
                    "cs_x_minutes",
                    lambda df: df.get("clean_sheet_probability_enhanced", 0)
                    * df.get("rolling_5gw_minutes", 0)
                    / 90,
                ),
                (
                    "goal_threat_def",
                    lambda df: df.get("rolling_5gw_xg", 0)
                    + df.get("rolling_5gw_threat", 0) / 100,
                ),
            ],
            "MID": [
                (
                    "xgi_combined",
                    lambda df: df.get("rolling_5gw_xg", 0)
                    + df.get("rolling_5gw_xa", 0),
                ),
                (
                    "creativity_x_threat",
                    lambda df: df.get("rolling_5gw_creativity", 0)
                    * df.get("rolling_5gw_threat", 0)
                    / 1000,
                ),
            ],
            "FWD": [
                (
                    "xg_x_minutes",
                    lambda df: df.get("rolling_5gw_xg", 0)
                    * df.get("rolling_5gw_minutes", 0)
                    / 90,
                ),
                (
                    "goal_involvement",
                    lambda df: df.get("rolling_5gw_xg", 0)
                    + 0.5 * df.get("rolling_5gw_xa", 0),
                ),
            ],
        }
        return additions.get(position, [])
