from __future__ import annotations

from abc import ABC, abstractmethod
import logging
from typing import (
    TYPE_CHECKING,
    Final,
    Generic,
    Sequence,
    TypeVar,
)

import numpy as np
import pandas as pd
import scipy.stats as ss
from sklearn.neighbors import NearestNeighbors as NN
from sklearn.preprocessing import StandardScaler

from css.core import FittableMeta, UserExtendableNamedConfigMixin
from css.param_estimator import ParamBaseEstimator, get_param_model

if TYPE_CHECKING:
    from css._typing import (
        Array1DFloat,
        Array1DFloatOrInt,
        DistParams2DArray,
        PeerDims2DArray,
        PeerDimsSets3DArray,
        ScipyRVType,
    )


class DifferentScoringScalesException(Exception):
    """Objects in some score combination have different scoring scales."""


class ScoringScale(ABC, UserExtendableNamedConfigMixin):
    """Base class for scoring output representation.

    This injects the mapping of raw cumulative density function (CDF) outputs in [0,1]
    to human-understandable scores.
    """

    @abstractmethod
    def score(self, values: Array1DFloat) -> Array1DFloat:
        """Map values from [0,1] to desired output."""
        ...

    def __eq__(self, other: object) -> bool:
        return type(self) == type(other)

    def __bool__(self) -> bool:
        return True


class ScoringOutOf10(ScoringScale):
    """Scores on a rounded 10 pt scale.

    Args:
        min_incr: Arbitrary decimal based rounding. For example, a value of ``.5``
            would walk up the scale by increments of ``.5``.
    """

    CONFIG_NAME = "outof10"

    def __init__(self, min_incr: float = 0.1) -> None:
        self.min_incr = min_incr

    def round(self, values: Array1DFloat) -> Array1DFloat:
        return (values / self.min_incr).round() * self.min_incr

    def score(self, values: Array1DFloat) -> Array1DFloat:
        values = self.round(values * 10)
        return values

    def __eq__(self, other: object) -> bool:
        return (
            super().__eq__(other)
            and hasattr(other, "min_incr")
            and self.min_incr == other.min_incr
        )


class BaseScoringInterface(
    ABC,
    UserExtendableNamedConfigMixin,
    metaclass=FittableMeta,
    post_fit_methods=["score"],
):
    """Base class for scoring interfaces.

    A scoring interface can be a metric or a combination of metrics that are combined to
    roll up low-level scores to human-understandable scores. This represents the basic
    structure of all CSS scoring mechanisms.

    Attributes:
        name (str): The name of the scoring interface.
    """

    name: str

    @abstractmethod
    def fit(self, dataframe: pd.DataFrame) -> BaseScoringInterface:
        """Fit using provided data.

        By showing a scoring interface the data, it can learn the underlying historical
        patterns and use them to create a statistical model against which we can
        evaluate data relative to itself and new data
        """
        ...

    @abstractmethod
    def score(
        self, dataframe: pd.DataFrame, scoring_scale: ScoringScale | None = None
    ) -> pd.DataFrame:
        """Map data to scores.

        Data may be the training data or new data that we would like to evaluate.
        """
        ...

    @property
    @abstractmethod
    def min_required_columns(self) -> Sequence[str]:
        """Minimum required columns to run training/prediction.

        All scoring interfaces have deep dependence on named dataframes. This can be
        used to label unlabeled dataframes such that they can be used with this package.
        """
        ...

    @property
    def _logger(self) -> logging.Logger:
        return logging.getLogger(f"{__name__}:::{self.name}")


class BaseMetric(BaseScoringInterface, post_fit_attrs=["nonzero_count"]):
    """Base class for metrics used in scoring.

    A Metric is a tool that takes in raw data and maps it to a score via a ScoringScale.

    Attributes:
        name: The name of the metric.
        nonzero_count: number of non-valuable entries in the metric.
    """

    nonzero_count: int
    _extended_logger_name: str

    def __init__(
        self,
        name: str,
    ) -> None:
        self.name = name
        self._extended_logger_name = name

    @property
    def _logger(self) -> logging.Logger:
        return logging.getLogger(f"{__name__}:::{self._extended_logger_name}")

    def add_parent_to_logger(self, parent_logger_name: str) -> None:
        self._extended_logger_name = f"{parent_logger_name}::{self.name}"

    @property
    def min_required_columns(self) -> Sequence[str]:
        return [self.name]


def _sample_linspace(
    array: Array1DFloatOrInt,
    trim: bool,
    sample_size: int,
) -> Array1DFloatOrInt:
    """Get equally spaced samples from array."""
    if np.issubdtype(array.dtype, np.integer):
        unique_values = np.unique(array)
        if unique_values.size <= sample_size:
            return unique_values
        else:
            min_, max_ = array.min(), array.max()
            return np.linspace(min_, max_, sample_size).astype(int)

    elif np.issubdtype(array.dtype, np.floating):
        if trim:
            min_, max_ = np.quantile(array, 0.01), np.quantile(array, 0.99)
        else:
            min_, max_ = array.min(), array.max()
        return np.linspace(min_, max_, sample_size)

    else:
        raise TypeError(
            "Could not interpret clustering data type. "
            "Try coercing to int or float first."
        )


def _neighbors(
    peer_dims: PeerDims2DArray, samples: PeerDims2DArray, k: int
) -> PeerDimsSets3DArray:
    """Get nearest neighbors of representatives from observed data."""
    scaler: StandardScaler = StandardScaler().fit(peer_dims)
    knn: NN = NN(n_neighbors=k).fit(scaler.transform(peer_dims))
    return knn.kneighbors(scaler.transform(samples), return_distance=False)


class InterpolatedPeerMetric(
    BaseMetric,
    post_fit_attrs=["reps", "nonzero_count"],
):
    """
    A class representing an interpolated peer metric.

    Args:
        name: The name of the metric. This should match the column name in the
            dataframe.
        peer_dims: The dimensions to compare peers against. These should match
            column
            names in dataframes passed during fit and predict.
        n_neighbors: The number of real neighbors of a representative to consider when
            fitting a distribution for scoring against peers.
        param_estimator: The parameter estimator that will interpolate parameter values.
            Defaults to "linear".
        param_estimator_kwargs: Keyword arguments for the parameter estimator.
        trim_samples: Whether to trim the samples with float peer comparison dims. This
            will trim off top and bottom 1% to protect from anomalies.
        min: Hard-codes the minimum value of the metric.
        max: Hard-codes the maximum value of the metric.
        trim_min: Remove values less than `min` from the distribution. This is useful
            for removing bottom-coding.
        trim_max: Remove values less than `max` from the distribution. This is useful
            for removing top-coding.
        n_reps_per_dim: The number of representative samples per dimension.
        semantic_flip: As a default, higher metric values are considered better. This
            allows us to flip that so lower values are better. The logic for this
            should be written in the subclass.
        **kwargs: Additional keyword arguments for ``BaseMetric``.

    Attributes:
        R2_GREEN_GE (Final): The R^2 threshold for logging a warning.
        R2_YELLOW_GE (Final): The yellow R^2 threshold for logging a critical warning.
            It is not recommended to proceed with values below this threshold.
        reps: The representatives that are used as a mesh to interpolate.
        param_model: The model that predicts parameters based on ``peer_dims``.
        nonzero_count: The number of non-zero values in the metric.
        CONFIG_NAME: The string name of the object that can be referenced via config.
    """

    _dist_cls: ScipyRVType = ss.norm

    R2_GREEN_GE: Final = 0.75
    R2_YELLOW_GE: Final = 0.50
    CONFIG_NAME: str

    def __init__(
        self,
        name: str,
        *,
        peer_dims: list[str],
        n_neighbors: int,
        param_estimator: str = "linear",
        param_estimator_kwargs: dict | None = None,
        trim_samples: bool = False,
        min: float | None = None,
        max: float | None = None,
        trim_min: bool = False,
        trim_max: bool = False,
        floc: float | None = None,
        fscale: float | None = None,
        n_reps_per_dim: int = 15,
        semantic_flip: bool = False,
        **kwargs,
    ) -> None:
        self._peer_dims = peer_dims
        self._n_neighbors = n_neighbors
        self.param_model: ParamBaseEstimator = get_param_model(
            param_estimator, **(param_estimator_kwargs or {})
        )
        self._trim_samples = trim_samples
        self._n_reps_per_dim = n_reps_per_dim
        self._min = min
        self._max = max
        self._trim_min = trim_min
        self._trim_max = trim_max
        self._fscale = fscale
        self._floc = floc
        self._semantic_flip = semantic_flip
        super().__init__(name, **kwargs)
        self._validate_metric_params()

    def _validate_metric_params(self) -> None:
        if self._trim_max is True and self._max is None:
            raise ValueError("Cannot use `trim_max` without setting `max`")
        if self._trim_min is True and self._min is None:
            raise ValueError("Cannot use `trim_min` without setting `min`")

    def _create_reps(
        self,
        array: PeerDims2DArray,
    ) -> PeerDims2DArray:
        rep_dims = []
        for dim_col in array.T:
            dim_sample = _sample_linspace(
                dim_col,
                trim=self._trim_samples,
                sample_size=self._n_reps_per_dim,
            )
            rep_dims.append(dim_sample)

        # Cartesian product of representative arrays
        reps = np.array(np.meshgrid(*rep_dims))
        reps_explode = reps.T.reshape(-1, len(rep_dims))
        return reps_explode

    def _prepare_data(self, data: Array1DFloat) -> Array1DFloat:
        if self._trim_min and self._min is not None:
            data = data[data > self._min]
        elif self._min is not None:
            data = data[data >= self._min]

        if self._trim_max and self._max is not None:
            data = data[data < self._max]
        elif self._max is not None:
            data = data[data <= self._max]

        return data

    def _fit_dist(self, data: Array1DFloat, rep: Array1DFloatOrInt):
        data_prepared = self._prepare_data(data)
        try:
            params = self._dist_cls.fit(data_prepared)
            return params
        except ss.FitError:
            location = ",".join(
                f"{c}={s}" for c, s in zip(self._peer_dims, rep, strict=True)
            )
            self._logger.warning(
                f"Unable to fit distribution at location ({location}). Skipping...",
            )

    def _build_param_dataset(
        self,
        dataframe: pd.DataFrame,
    ) -> tuple[PeerDims2DArray, DistParams2DArray]:
        metric_series = dataframe[self.name]
        peer_dims = dataframe[self._peer_dims].values

        reps = self._create_reps(peer_dims)
        neighbor_sets = _neighbors(peer_dims, reps, self._n_neighbors)

        params_all = []
        valid_reps = []
        for rep, neighbor_set in zip(reps, neighbor_sets, strict=True):
            metric_neighbors = metric_series.iloc[neighbor_set]
            params = self._fit_dist(metric_neighbors, rep)
            valid_reps.append(rep)
            try:
                params_all.append(list(params))
            except TypeError as exc:
                raise ss.FitError(
                    "No local distributions were fit. Perhaps this isn't the "
                    "right distribution to choose."
                ) from exc

        return np.array(valid_reps), np.array(params_all)

    def fit(self, dataframe: pd.DataFrame) -> InterpolatedPeerMetric:
        """Fits mesh of statistical models to interpolate metric values.

        We take a mesh of representative points in the space of the peer dimensions then
        gather a set of neighbors for each representative. We fit a distribution to the
        neighbors of each representative. The parameters of these distributions are then
        interpolated to predict the parameters of the distribution for a given metric
        value. From the cumulative density function (CDF) of this distribution, we can
        then score the metric value.

        Args:
            dataframe: DataFrame holding both metric values and clusting information.
        """
        self._logger.info(f"Fitting on {dataframe.shape[0]} values...")
        reps, fit_params = self._build_param_dataset(dataframe)
        self.reps = reps
        self.param_model.fit(reps, fit_params)

        r2 = self.param_model.r2
        if r2 > self.R2_GREEN_GE:
            self._logger.info(f"Model fit: R^2={r2:.2f}")
        elif r2 > self.R2_YELLOW_GE:
            self._logger.warning(f"Model fit: R^2={r2:.2f}")
        else:
            self._logger.critical(f"High risk model fit! R^2={r2:.2f}")

        self.nonzero_count = (dataframe[self.name] > 0).sum()
        return self

    def _compute_dist(
        self,
        metric_values: Array1DFloat,
        peer_dims: PeerDims2DArray,
        clip_min_max: bool = True,
    ) -> Array1DFloat:
        def axis_func(row):
            x = row[0]
            params = row[1:]
            return self._dist_cls.cdf(x, *params)

        combined = np.c_[metric_values, peer_dims]
        metric_p = np.apply_along_axis(axis_func, 1, combined)

        if clip_min_max:
            if self._min is not None:
                metric_p = np.where(metric_values <= self._min, 0, metric_p)
            if self._max is not None:
                metric_p = np.where(metric_values >= self._max, 1, metric_p)
        return metric_p

    def score(
        self, dataframe: pd.DataFrame, scoring_scale: ScoringScale | None = None
    ) -> pd.DataFrame:
        """Map metric values to scores.

        Use fit parameter model to predict distribution parameters based on peer
        dimensions then use the interpolated parameters to score metric values via CDF
        output and scoring scale.

        Args:
            dataframe: DataFrame holding both metric values and clusting information.
        """
        self._logger.info(f"Scoring {dataframe.shape[0]} metric values.")
        metric_series = dataframe[self.name]
        metric_series.name = "metric"
        metric_array = metric_series.to_numpy(np.float64)

        params_hat = self.param_model.predict(dataframe[self._peer_dims].values)
        metric_p = self._compute_dist(metric_array, params_hat)
        if self._semantic_flip:
            metric_p = 1 - metric_p
        if scoring_scale:
            metric_p = scoring_scale.score(metric_p)

        scores_series = pd.Series(
            metric_p, index=metric_series.index, name="metric_score"
        )
        scores_io = pd.concat([metric_series, scores_series], axis=1)
        return scores_io

    @property
    def min_required_columns(self) -> Sequence[str]:
        return [self.name, *self._peer_dims]


class FixedOrientationPeerMetric(InterpolatedPeerMetric):
    """Adds functionality for distributions with fixed orientation.

    These metrics have a fixed orientation. They must be flipped to account for the
    shape of the underlying histogram. For example, exponential distributions expect
    higher density closer to zero.

    Note: ``max`` and ``min`` are required when flipping.

    Args:
        flip : Flip x-axis.
    """

    def __init__(self, *args, flip: bool = False, **kwargs):
        self._flip = flip
        super().__init__(*args, **kwargs)

    def _validate_metric_params(self):
        if self._flip and (self._max is None or self._min is None):
            raise ValueError(
                f"{self._dist_name}: {self.name}: "
                f"Please set `max` and `min` if flipping distribution."
            )
        super()._validate_metric_params()

    def _prepare_data(self, data: Array1DFloat) -> Array1DFloat:
        """
        Trim off top-/bottom-coding and outliers. Flip then clip.
        """
        if self._flip:
            data = np.abs(data - self._max)
        return super()._prepare_data(data)

    def _compute_dist(
        self,
        metric_values: Array1DFloat,
        peer_dims: PeerDims2DArray,
        clip_min_max: bool = True,
    ) -> Array1DFloat:
        if self._flip:
            self._max: float
            self._min: float
            metric_values_pre = metric_values.copy()
            metric_values = np.abs(metric_values - self._max)

        # Will clip_min_max below if flipped.
        metric_p = super()._compute_dist(
            metric_values,
            peer_dims,
            clip_min_max=(not self._flip),
        )
        if self._flip:
            metric_p = 1 - metric_p
            if clip_min_max:
                metric_p = np.where(metric_values_pre <= self._min, 0, metric_p)
                metric_p = np.where(metric_values_pre >= self._max, 1, metric_p)

        return metric_p


class NormalMetric(InterpolatedPeerMetric):
    """
    Scoring for Metric whose percentile scores are best interpolated with a normal
    distribution.
    """

    CONFIG_NAME = "ipmnormal"
    _dist_cls: ScipyRVType = ss.norm

    def _validate_metric_params(self):
        if self._floc is not None or self._fscale is not None:
            conjunction = "or" if bool(self._floc) ^ bool(self._fscale) else "and"
            self._logger.warning(
                f"`floc` (mean) {conjunction} `fscale` (variance) are set in config. "
                f"Mean {conjunction} Variance not fitting."
            )
        super()._validate_metric_params()


class GammaMetric(FixedOrientationPeerMetric):
    """
    Scoring for Metric whose percentile scores are best interpolated with a gamma
    distribution.
    """

    CONFIG_NAME = "ipmgamma"
    _dist_cls: ScipyRVType = ss.gamma

    def __init__(self, *args, **kwargs) -> None:
        # Setting floc to min
        if "floc" not in kwargs and "min" in kwargs:
            kwargs["floc"] = kwargs["min"]
        super().__init__(*args, **kwargs)


class ExponentialMetric(GammaMetric):
    """
    Scoring for Metric whose percentile scores are best interpolated with a
    exponential distribution.
    """

    CONFIG_NAME = "ipmexponential"
    _dist_cls: ScipyRVType = ss.expon

    def _validate_metric_params(self):
        if self._floc is not None or self._fscale is not None:
            conjunction = "or" if bool(self._floc) ^ bool(self._fscale) else "and"
            self._logger.warning(
                f"`floc` {conjunction} `fscale` are set in config. "
                f"floc {conjunction} fscale not fitting."
            )
        super()._validate_metric_params()


class BetaMetric(InterpolatedPeerMetric):
    """
    Scoring for Metric whose percentile scores are best interpolated with a beta
    distribution.
    """

    CONFIG_NAME = "ipmbeta"
    _dist_cls: ScipyRVType = ss.beta

    def __init__(self, *args, **kwargs) -> None:
        # Setting floc and fscale are basically required to
        # fit beta. We set the former to be the min and the
        # latter to be the difference between max and min
        # if they exist.
        if "fscale" not in kwargs and "max" in kwargs:
            kwargs["fscale"] = kwargs["max"]

            if "min" in kwargs:
                kwargs["fscale"] -= kwargs["min"]

        if "floc" not in kwargs and "min" in kwargs:
            kwargs["floc"] = kwargs["min"]

        super().__init__(*args, **kwargs)


_ChildrenT = TypeVar("_ChildrenT", bound=BaseScoringInterface)


class BaseWeightedCombinationScorer(
    Generic[_ChildrenT],
    BaseScoringInterface,
    post_fit_methods=["weights"],
):
    """Base class for weighted combination scorers.

    Used to create tree structures of scorers that can be combined via dot product with
    weights.

    Args:
        name: The name of the score group. This can be anything and doesn't need  to
            exist in fit/score dataframe.
        children: A sequence of child objects that will be combined by the scorer.
        weights: The weights assigned to each child. If None, weights may be inferred.

    Attributes:
        scoring_scale: The scoring scale used to transform scores. If None, the raw CDF
            scores are returned.
        level_name: The name of the current level in the hierarchy.
        child_level_name: The name of the child level in the hierarchy.


    """

    scoring_scale: ScoringScale | None
    level_name: str
    child_level_name: str
    _children: dict[str, _ChildrenT]
    _weights: pd.Series | None

    def __init__(
        self,
        name: str,
        children: Sequence[_ChildrenT],
        weights: Sequence[float] | None = None,
        scoring_scale: ScoringScale | None = None,
    ) -> None:
        self.name = name
        self.scoring_scale = scoring_scale
        self._combined_column_name = f"{self.level_name}_score"
        self._child_combined_column_name = f"{self.child_level_name}_score"

        self.children = {child.name: child for child in children}
        self._set_weights(weights)

    @property
    def children(self) -> dict[str, _ChildrenT]:
        return self._children

    @children.setter
    def children(self, children: dict[str, _ChildrenT]) -> None:
        self._children = children

    @abstractmethod
    def _set_weights(self, weights: Sequence[float] | None = None) -> None: ...

    def weights(self):
        return self._weights

    def _combine_children(self, children_scores: dict[str, pd.DataFrame]):
        """Combines the summaries of multiple children into a single DataFrame.

        Args:
            children_scores: A dictionary containing the output scoring dataframe of
                each child.

        Returns:
            pd.DataFrame: A DataFrame with an additional column level and column for the
                weighted combination.

        """
        children_dfs = []
        for child_name, child_data in children_scores.items():
            midx_child = pd.concat(
                [child_data], keys=[child_name], names=[self.child_level_name], axis=1
            )
            children_dfs.append(midx_child)

        return pd.concat(children_dfs, axis=1)

    def _add_dot_product(self, children_data: pd.DataFrame) -> pd.DataFrame:
        """Adds the dot product of children scores and weights to the DataFrame."""
        children_scores = (
            children_data.xs(
                self._child_combined_column_name, level=1, axis=1, drop_level=True
            )
            .fillna(0)
            .to_numpy(np.float64)
        )
        children_data[self._combined_column_name] = children_scores.dot(self.weights())
        return children_data

    def fit(self, dataframe: pd.DataFrame) -> BaseWeightedCombinationScorer:
        """Fit for all descendants.

        Args:
            dataframe: DataFrame holding both metric values and peer information. Must
            include column names that match the names of the ``BaseMetric`` descendants.
        """
        for obj in self.children.values():
            obj.fit(dataframe)
        return self

    def score(
        self,
        dataframe: pd.DataFrame,
        scoring_scale: ScoringScale | None = None,
    ) -> pd.DataFrame:
        """Score over provided dataframe.

        Args:
            dataframe (pd.DataFrame): DataFrame holding both metric values for all
                children and peer information.

        Returns:
            pd.DataFrame: DataFrame holding both inputs and scores. Columns are
                multi-index representing hierarchy.
        """
        scoring_scale = scoring_scale or self.scoring_scale
        summaries = {}
        for name, obj in self.children.items():
            summary = obj.score(dataframe, scoring_scale)
            summaries[name] = summary
        summary = self._combine_children(summaries)
        summary = self._add_dot_product(summary)
        return summary

    @property
    def min_required_columns(self) -> Sequence[str]:
        merged_columns = {
            c for child in self.children.values() for c in child.min_required_columns
        }
        return sorted(merged_columns)

    def __getitem__(self, child_name: str) -> _ChildrenT:
        return self.children[child_name]


class Component(BaseWeightedCombinationScorer[BaseMetric]):
    """Represents a component comprised of multiple metrics.

    A component is a weighted combination of multiple ``BaseMetrics``. It is a logical
    grouping of metrics that are combined to form a single score via a weighted average.
    """

    CONFIG_NAME: str = "weightedcomponent"
    level_name: str = "component"
    child_level_name: str = "metric"
    _children: dict[str, BaseMetric]

    @property
    def children(self) -> dict[str, BaseMetric]:
        try:
            return self._children
        except AttributeError:
            self._children = {}
            return self._children

    @children.setter
    def children(self, children: dict[str, BaseMetric]) -> None:
        """Sets the children of the component.

        Adds the parent name to each child's logger.
        """
        if len(children) == 0:
            raise ValueError("No children passed.")
        old_children = self.children
        try:
            self._children = {}
            for name, child in children.items():
                child.add_parent_to_logger(self.name)
                self.children[name] = child
        except Exception as exc:
            self._children = old_children
            raise exc

    def _set_weights(self, weights: Sequence[float] | None = None) -> None:
        if weights is not None:
            weights_series = pd.Series(weights, index=list(self.children.keys()))
            self._weights = weights_series / weights_series.sum()
        else:
            self._logger.warning("No weights given. Dynamically setting weights.")
            self._weights = None

    def weights(self) -> pd.Series:
        """Generate weights if not set.

        Weights are generated by taking the proportion of non-zero values in each
        metric.
        """
        if self._weights is None:
            # Has to be set lazily because the BaseMetric has
            # to be fit before .nonzero_count() is accessible.
            weights = pd.Series(
                {
                    metric_name: metric.nonzero_count
                    for metric_name, metric in self.children.items()
                }
            )
            weights = pd.Series(weights, index=list(self.children.keys()))
            self._weights = weights / weights.sum()
            return self._weights
        else:
            return self._weights


class GlobalScore(BaseWeightedCombinationScorer[Component]):
    """Holds the outermost level of the scoring hierarchy.

    Top-level object comprised of multiple components. Components are combined via
    weighted average.
    """

    CONFIG_NAME: str = "weightedglobal"
    NAME: Final = "global"
    level_name: str = "global"
    child_level_name: str = "component"
    _children: dict[str, Component]

    def __init__(
        self,
        children: Sequence[Component],
        weights: Sequence[float] | None = None,
        scoring_scale: ScoringScale | None = None,
    ) -> None:
        self.name = "global"
        super().__init__(
            GlobalScore.NAME, children, weights, scoring_scale=scoring_scale
        )

    def _set_weights(self, weights: Sequence[float] | None = None) -> None:
        if weights is None:
            self._logger.warning("No weights given. Using equal weighting.")
            weights = [1] * len(self.children)
        weights_series = pd.Series(weights, index=list(self.children.keys()))
        self._weights = weights_series / weights_series.sum()
