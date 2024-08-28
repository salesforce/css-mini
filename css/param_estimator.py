from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor as KNR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler

from css.core import FittableMeta, UserExtendableNamedConfigMixin

if TYPE_CHECKING:
    from sklearn.base import BaseEstimator

    from css._typing import (
        Array1DFloat,
        DistParams2DArray,
        PeerDims2DArray,
    )


def get_param_model(
    param_estimator_config_name: str, *args, **kwargs
) -> ParamBaseEstimator:
    return ParamBaseEstimator.subclass_from_config_name(param_estimator_config_name)(
        *args, **kwargs
    )


def _log_shift(x: Array1DFloat) -> Array1DFloat:
    LOG_SHIFT = 1e-5
    return np.log(x + LOG_SHIFT)


class ParamBaseEstimator(
    ABC,
    UserExtendableNamedConfigMixin,
    metaclass=FittableMeta,
    post_fit_methods=["predict"],
    post_fit_attrs=["r2", "domain", "range"],
):
    _model: BaseEstimator

    def __init__(self, drop_null: bool = False) -> None:
        self._drop_null = drop_null

    def fit(self, X: PeerDims2DArray, Y: DistParams2DArray) -> None:
        """Fit model."""
        if self._drop_null:
            nan_idx = np.any(np.isnan(Y), axis=1)
            X = X[~nan_idx]
            Y = Y[~nan_idx]

        self.domain = {
            "a_min": X.min(axis=0),
            "a_max": X.max(axis=0),
        }
        self.range = {
            "a_min": Y.min(axis=0),
            "a_max": Y.max(axis=0),
        }
        self.model.fit(X, Y)
        self.r2 = self._model.score(X, Y)

    def predict(self, X: PeerDims2DArray) -> DistParams2DArray:
        """Predict parameters given individual's clustering information."""
        X_clipped = np.clip(X, **self.domain)
        Y_hat = self._model.predict(X_clipped)
        np.clip(Y_hat, **self.range)
        return Y_hat

    @property
    def model(self) -> BaseEstimator:
        return self._model


class ParamLinearModel(ParamBaseEstimator):
    CONFIG_NAME: str = "linear"

    def __init__(self, drop_null: bool = False, log: bool = False) -> None:
        super().__init__(drop_null)
        operations = [
            ("scaler", MinMaxScaler()),
            ("glm", LinearRegression()),
        ]
        if log:
            operations.insert(1, ("log", FunctionTransformer(_log_shift)))
        self._model = Pipeline(operations)


class ParamNearestNeighborsModel(ParamBaseEstimator):
    CONFIG_NAME: str = "nearestneighbors"

    def __init__(self, drop_null: bool = False, k: int | None = None, **kwargs) -> None:
        super().__init__(drop_null)
        self.k = k or 5
        self._model = Pipeline(
            [
                ("scaler", MinMaxScaler()),
                ("knr", KNR(weights="distance", n_neighbors=self.k)),
            ]
        )
