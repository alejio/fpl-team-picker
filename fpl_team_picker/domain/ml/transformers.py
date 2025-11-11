"""Custom sklearn transformers for FPL ML pipelines."""

from typing import List
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class FeatureSelector(BaseEstimator, TransformerMixin):
    """
    Select specific features by name from a DataFrame.

    This transformer allows ML pipelines to be self-contained - they know which
    features they need and can extract them from the full feature set.

    This is critical for deployment: MLExpectedPointsService can pass all 118
    features to any model, and the pipeline will internally select the subset
    it was trained on.

    Parameters
    ----------
    feature_names : list of str
        Names of features to select

    Examples
    --------
    >>> from sklearn.pipeline import Pipeline
    >>> from sklearn.preprocessing import StandardScaler
    >>> from sklearn.ensemble import RandomForestRegressor
    >>>
    >>> # Pipeline that accepts 117 features but only uses 60
    >>> pipeline = Pipeline([
    ...     ('feature_selector', FeatureSelector(['feature1', 'feature2', ...])),
    ...     ('scaler', StandardScaler()),
    ...     ('regressor', RandomForestRegressor())
    ... ])
    >>>
    >>> # Can pass all features - selector extracts the ones it needs
    >>> X_all = df[all_117_features]
    >>> pipeline.fit(X_all, y)
    >>> pipeline.predict(X_all)
    """

    def __init__(self, feature_names: List[str]):
        self.feature_names = feature_names

    def fit(self, X, y=None):
        """
        Fit transformer (no-op for feature selection).

        Parameters
        ----------
        X : DataFrame, shape (n_samples, n_features)
            Input features
        y : array-like, optional
            Target variable (ignored)

        Returns
        -------
        self : FeatureSelector
        """
        return self

    def transform(self, X):
        """
        Select only the features the model was trained on.

        Parameters
        ----------
        X : DataFrame, shape (n_samples, n_features_input)
            Input features (can include extra features)

        Returns
        -------
        X_selected : DataFrame, shape (n_samples, n_features_selected)
            Selected features only
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                f"FeatureSelector requires DataFrame input, got {type(X).__name__}"
            )

        # Check if all required features are present
        missing_features = set(self.feature_names) - set(X.columns)
        if missing_features:
            raise ValueError(
                f"Missing required features: {sorted(missing_features)[:10]}..."
                if len(missing_features) > 10
                else f"Missing required features: {sorted(missing_features)}"
            )

        return X[self.feature_names]

    def get_feature_names_out(self, input_features=None):
        """
        Get output feature names for transformation.

        Parameters
        ----------
        input_features : array-like, optional
            Input feature names (ignored)

        Returns
        -------
        feature_names_out : ndarray of str
            Output feature names
        """
        return np.array(self.feature_names)
