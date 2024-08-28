from __future__ import annotations

import inspect
from operator import itemgetter
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Mapping,
    Sequence,
)

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)
import yaml

from css.score import (
    BaseMetric,
    BaseScoringInterface,
    Component,
    GlobalScore,
    ScoringOutOf10,
    ScoringScale,
)

if TYPE_CHECKING:
    from pathlib import Path

    from css.core import UserExtendableNamedConfigMixin

_ANY_KWARGS = "*"
_ANY_KWARGS_SET = set(_ANY_KWARGS)


def _get_all_kwargs(cls: type) -> set[str]:
    """Get all keyword arguments for a class including parents.

    Returns a set of kwarg names for checking compatibility with underlying classes. If
    a set with ``*`` is returned, it means that the class accepts any keyword arguments
    because the parent dead-ends with **kwargs"""
    kwargs = set()
    signature = inspect.signature(cls.__init__)  # type: ignore
    for name, param in signature.parameters.items():
        # TODO: This should check whether the parent kwargs are
        # actually passed to parent __init__ method
        if param.kind == param.VAR_KEYWORD:
            if cls.__bases__ == (object,):
                return _ANY_KWARGS_SET
            for cls_super in cls.__bases__:
                parent_args = _get_all_kwargs(cls_super)
                kwargs.update(parent_args)
        elif name not in ("self", "args"):
            kwargs.add(name)

    if _ANY_KWARGS in kwargs:
        return _ANY_KWARGS_SET
    return kwargs


class ScoringInterfaceObjectConfigBaseModel(BaseModel):
    """Base class for Pydantic models that correspond directly with an scoring object.

    Attributes:
        type_config_name: The name of the class in the config file.
        kwargs: Keyword arguments passed to the scoring object
    """

    model_config = ConfigDict(validate_default=True, extra="forbid")
    type_base: ClassVar[type[UserExtendableNamedConfigMixin]]
    type_config_name: str = Field(
        description="The config name of the object to create. "
        "For metrics, this would might be 'ipmnormal'. For custom classes, you can "
        "assign a name to a class variable `CONFIG_NAME` and reference it here.",
    )
    kwargs: Mapping[str, Any] = Field(
        default_factory=dict,
        description="Keyword arguments to pass to the scoring object.",
    )

    @model_validator(mode="after")
    def _check_kwargs_match_type(
        self: ScoringInterfaceObjectConfigBaseModel,
    ) -> ScoringInterfaceObjectConfigBaseModel:
        """Verify that the keyword arguments match the type of the scoring object."""
        metric_type = self.type_base.subclass_from_config_name(self.type_config_name)
        available_kwargs = _get_all_kwargs(metric_type)
        if available_kwargs == _ANY_KWARGS_SET:
            return self
        passed_kwargs = set(self.kwargs.keys())
        if diff := passed_kwargs - available_kwargs:
            raise TypeError(f"Unexpected keyword arg(s) passed: {diff}.")
        return self


class MetricConfigModel(ScoringInterfaceObjectConfigBaseModel):
    """Configuration model for Metrics.

    Unlike others, ``type_config_name`` must be passed; there is no default value.
    """

    type_base = BaseMetric


class ComponentConfigModel(ScoringInterfaceObjectConfigBaseModel):
    """Configuration model for Components.

    Defaults to ``Component`` defined in ``css.score``.
    """

    type_base = Component
    type_config_name: str = Field(
        default=Component.CONFIG_NAME,
        description="The config name of the component to create. "
        "This is not required and is used only for custom components.",
    )
    metrics: Sequence[str] = Field(
        description="The metrics to include in this component. "
        "Metrics can only be in one component."
    )
    weights: Sequence[float] | None = Field(
        default=None,
        description="Weights for metrics. Must match `metrics` in length. "
        "Defaults to dynamic weight creation.",
    )

    @model_validator(mode="after")
    def _check_same_length(cls, model):
        """Check that metrics and weights have the same length."""
        if model.weights is not None and len(model.metrics) != len(model.weights):
            raise ValueError("Metrics and weights should have same number of elements.")
        return model


class GlobalScoreConfigModel(ScoringInterfaceObjectConfigBaseModel):
    """Configuration model for GlobalScore.

    Defaults to ``GlobalScore`` defined in ``css.score``.
    """

    type_base = GlobalScore
    type_config_name: str = Field(
        default=GlobalScore.CONFIG_NAME,
        description="The config name of the global score to create. "
        "This is not required and is used only for custom scoring groupings.",
    )
    weights: Sequence[float] | None = Field(
        default=None,
        description="Weights for components. Must match number of components "
        "in length. Defaults to equal weighting.",
    )


class ScoringScaleConfigModel(ScoringInterfaceObjectConfigBaseModel):
    """Configuration model for ScoringScale."""

    type_base = ScoringScale
    type_config_name: str = Field(
        default=ScoringOutOf10.CONFIG_NAME,
        description="The config name of the scoring scale to use. "
        "This is not required and is used only for custom scoring scales.",
    )


class ConfigModel(BaseModel, validate_default=True, extra="forbid"):
    """Configuration model for the scoring system."""

    metrics: Mapping[str, MetricConfigModel] = Field(
        description="Mapping of metric names to metric configurations."
    )
    components: Mapping[str, ComponentConfigModel] = Field(
        default_factory=dict,
        description="Mapping of component names to component configurations.",
    )
    global_score: GlobalScoreConfigModel = Field(
        default_factory=GlobalScoreConfigModel,
        description="Configuration for the global score. Not required, but "
        "this is useful for custom weighting of components.",
    )
    scoring_scale: ScoringScaleConfigModel = Field(
        default_factory=ScoringScaleConfigModel,
        description="Configuration for the scoring scale. Not required.",
    )

    @model_validator(mode="after")
    def _check_metrics_to_components_mapping(self: ConfigModel) -> ConfigModel:
        """Metrics must be in exactly one component unless only one metric exists."""
        for metric in self.metrics:
            metric_count = 0
            for component in self.components.values():
                if metric in component.metrics:
                    metric_count += 1
                if metric_count > 1:
                    raise ValueError(f"Metric {metric} in more than one component.")
            if metric_count == 0 and len(self.components) > 0 and len(self.metrics) > 1:
                # We still let a user pass a single metric
                # and create scoring structure.
                raise ValueError(f"Metric {metric} was not included in a component.")
        return self

    @model_validator(mode="after")
    def _check_global_weights_match_components(cls, model):
        """Global weights must align with components."""
        if (weights := model.global_score.weights) is not None:
            if len(weights) != len(model.components):
                raise ValueError(
                    "Components and weights should have same number of elements."
                )
        return model

    @classmethod
    def from_yaml(cls: type[ConfigModel], path: str | Path) -> ConfigModel:
        """Create config directly from a YAML file."""
        with open(path) as f:
            return ConfigModel.model_validate(yaml.safe_load(f))

    def to_yaml(self, path: str | Path) -> None:
        """Write the configuration to a YAML file."""
        with open(path, "w") as f:
            f.write(yaml.dump(self.model_dump()))

    def to_obj(self) -> BaseScoringInterface:
        """Convert the configuration to a scoring object.

        Returns:
            ScoreInterface: The scoring object which may be a ``Metric`` or
                ``GlobalScore``.
        """
        metrics = {}
        for name, mmodel in self.metrics.items():
            metric_type = BaseMetric.subclass_from_config_name(mmodel.type_config_name)
            metrics[name] = metric_type(name=name, **mmodel.kwargs)
            if not self.components:
                # There must be only one in this case
                return metrics[name]

        if self.scoring_scale is not None:
            scoring_scale_type = ScoringScale.subclass_from_config_name(
                self.scoring_scale.type_config_name
            )
            scoring_scale = scoring_scale_type(**self.scoring_scale.kwargs)
        else:
            scoring_scale = None

        components = {}
        for name, cmodel in self.components.items():
            component_type = Component.subclass_from_config_name(
                cmodel.type_config_name
            )
            if len(cmodel.metrics) == 1:
                children = [metrics[cmodel.metrics[0]]]
            else:
                children = itemgetter(*cmodel.metrics)(metrics)
            components[name] = component_type(
                name=name, children=children, weights=cmodel.weights, **cmodel.kwargs
            )

        global_score_type = GlobalScore.subclass_from_config_name(
            self.global_score.type_config_name
        )
        return global_score_type(
            children=list(components.values()),
            weights=self.global_score.weights,
            scoring_scale=scoring_scale,
            **self.global_score.kwargs,
        )
