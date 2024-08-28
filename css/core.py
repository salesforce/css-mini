from __future__ import annotations

from abc import ABCMeta
from functools import wraps
from typing import (
    Any,
    Callable,
    Generic,
    Iterable,
    ParamSpec,
    Protocol,
    TypeVar,
)


class FitStateError(Exception):
    """Protects fittable object state.

    Represents the attempt to perform post-fit actions in pre-fit state, or vice-versa.
    """


class IncorrectlyDefinedFittableClass(Exception):
    """Malformed user object defined as fittable.

    If the user is attempting to extend this package and create a new family of
    fittable objects, they must have a method named ``fit`` that managed state.
    """


class TracksFit(Protocol):
    """Represents fittable object interfaces.

    Note: The user does not need to worry about adding this in their class definition.
    It is managed in the metaclass.
    """

    __is_fit: bool

    def is_fit(self) -> bool:
        """Accessor for name-mangled fit boolean."""
        ...


_T = TypeVar("_T", bound="TracksFit")
_ValueT = TypeVar("_ValueT")


class _PostFitAttr(Generic[_ValueT]):
    """Descriptor for attributes that will be assigned post-fit.

    They may or may not exist prior to the fit state change, but either way they
    are inaccessible. This also creates a nicer interface because instance of a
    confusing ``AttributeError``, the user is clearly pointed to the fact that the
    attribute hasn't been added yet. It is not recommended to use this directly, and
    instead only use the ``FittableMeta`` metaclass as the sole entry point.

    Attributes:
        name: The name of the attribute.

    """

    name: str

    def __init__(self, name: str) -> None:
        self.name = name

    def __get__(self, instance: TracksFit, owner: FittableMeta) -> _ValueT:
        if instance.is_fit():
            try:
                return instance.__dict__[self.name]
            except KeyError as exc:
                raise AttributeError(
                    "Something went wrong during fitting! "
                    f"Attribute `{self.name}` not assigned during fit."
                ) from exc
        else:
            raise FitStateError(
                "Fit method not yet called. "
                f"Call `fit` before accessing `{self.name}`!"
            )

    def __set__(self, instance: TracksFit, value: _ValueT):
        instance.__dict__[self.name] = value


_P = ParamSpec("_P")
_RetT = TypeVar("_RetT")
_AnyFunc = Callable[_P, _RetT]


class FittableMeta(ABCMeta):
    """Metaclass to manage object with pre- and post-fit states.

    Tracks the structure of an object as the state changes from pre-fit to post-fit.
    Each class using this as its metaclass should have a single fit function that
    manages this state.

    Args:
        post_fit_methods: String names of functions that cannot be run until the fit
            method is run.
        post_fit_attrs: String names of attributes that cannot be run until the fit
            method is run.
    """

    def __new__(
        mcs: type[_T],
        name: str,
        bases: tuple[type, ...],
        namespace: dict[str, Any],
        post_fit_methods: Iterable[str] | None = None,
        post_fit_attrs: Iterable[str] | None = None,
        **kwargs,
    ) -> _T:
        post_fit_methods = [] if post_fit_methods is None else post_fit_methods
        post_fit_attrs = [] if post_fit_attrs is None else post_fit_attrs

        def is_fit(self) -> bool:
            return self.__is_fit

        namespace["is_fit"] = is_fit

        try:
            namespace["fit"] = FittableMeta.wrap_fit(namespace["fit"])
        except KeyError as exc:
            # Check if a parent has already defined a fit method and that it
            # has been properly wrapped.
            if not any(isinstance(b, FittableMeta) for b in bases):
                raise IncorrectlyDefinedFittableClass(
                    "This class must have a `fit` method"
                ) from exc

        for method in post_fit_methods:
            namespace[method] = FittableMeta.wrap_post_fit(namespace[method])
        for attr in post_fit_attrs:
            namespace[attr] = _PostFitAttr(attr)

        inst = type.__new__(mcs, name, bases, namespace, **kwargs)
        inst.__is_fit = False
        return inst

    @staticmethod
    def wrap_fit(func: _AnyFunc) -> _AnyFunc:
        """Decorator function that wraps a fit method to ensure it is only called once.

        Args:
            func: The fit method to be wrapped.

        Raises:
            FitStateError: If the fit method is called more than once.
        """

        @wraps(func)
        def update_fit(self, *args, **kwargs):
            if self.__is_fit:
                raise FitStateError(
                    "Fit method called twice. This object should only be fit once!"
                )
            raw_output = func(self, *args, **kwargs)
            self.__is_fit = True
            return raw_output

        return update_fit

    @staticmethod
    def wrap_post_fit(func: _AnyFunc) -> _AnyFunc:
        """Ensure that class is only called after the fit method has been called.

        Args:
            func: The method to be wrapped.

        Raises:
            FitStateError: If the `fit` method has not been called before calling
                the wrapped method.
        """

        @wraps(func)
        def update_post_fit(self, *args, **kwargs):
            if not self.__is_fit:
                this_func_name = func.__name__
                raise FitStateError(f"Call `fit` before calling `{this_func_name}`!")
            return func(self, *args, **kwargs)

        return update_post_fit


def _get_all_subclass_descendants(cls: type) -> list[type]:
    all_subclasses = [cls]

    for subclass in cls.__subclasses__():
        all_subclasses.append(subclass)
        all_subclasses.extend(_get_all_subclass_descendants(subclass))

    return all_subclasses


_V = TypeVar("_V", bound="UserExtendableNamedConfigMixin")


class UserExtendableNamedConfigMixin:
    """Allows users to access extended classes by name in config.

    Much of this package is text config driven. This means that if a user extends the
    ``css`` package, they won't be able to add their own classes if they want to use a
    config. This allows them to name their subclasses and let them be found at runtime
    by the config driven execution.
    """

    CONFIG_NAME: str

    @classmethod
    def subclass_from_config_name(cls: type[_V], config_name: str) -> type[_V]:
        """Create an instance of subclass by calling its string name (``CONFIG_NAME``).

        This is and should stay dynamic because a user may interactively add subclasses
        through REPL systems like Jupyter.

        Args:
            config_name: should match a subclass's ``CONFIG_NAME``.
        """
        subclass_config_name_map = {}
        for type_ in _get_all_subclass_descendants(cls):
            if name := getattr(type_, "CONFIG_NAME", ""):
                subclass_config_name_map[name] = type_
        try:
            return subclass_config_name_map[config_name]
        except KeyError as exc:
            raise KeyError(
                "Passed config_name does not match any subclass CONFIG_NAME"
            ) from exc

    @classmethod
    def available_config_names(cls: type[_V]) -> list[str]:
        """Get all available config names from the subclasses."""
        config_names = [
            type_.CONFIG_NAME
            for type_ in _get_all_subclass_descendants(cls)
            if hasattr(type_, "CONFIG_NAME")
        ]
        return config_names
