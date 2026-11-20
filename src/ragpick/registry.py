"""Component registry and entry-point plugin loading.

Register a component with the decorator::

    from ragpick import register

    @register("embedder", "my-embedder")
    class MyEmbedder: ...

and build one with ``registry.create("embedder", "my-embedder", **params)``.

Third-party packages expose components through the ``ragpick.components``
entry-point group. The entry-point *name* is ``"<kind>.<name>"`` and its value
points at the factory (usually a class)::

    [project.entry-points."ragpick.components"]
    "embedder.cohere" = "ragpick_cohere:CohereEmbedder"

An entry point whose name has no dot is simply imported, so a module can register
any number of components with the decorator as an import side effect.
"""

from __future__ import annotations

import importlib
import inspect
import logging
from collections.abc import Callable
from importlib import metadata
from typing import Any, TypeVar

from ragpick.errors import ComponentConfigError, ComponentNotFoundError

logger = logging.getLogger(__name__)

ENTRY_POINT_GROUP = "ragpick.components"

#: Component kinds known to ragpick. Registering under a new kind is allowed.
KINDS: tuple[str, ...] = (
    "loader",
    "chunker",
    "embedder",
    "store",
    "keyword_index",
    "retriever",
    "reranker",
    "generator",
)

F = TypeVar("F", bound=Callable[..., Any])


class Registry:
    """Maps ``(kind, name)`` pairs to component factories.

    Only the global :data:`registry` loads ragpick's built-in components
    (``load_builtins=True``); separate instances start empty, which is useful for
    isolated tests. Both load entry-point plugins on first lookup.
    """

    def __init__(self, *, load_builtins: bool = False) -> None:
        self._factories: dict[str, dict[str, Callable[..., Any]]] = {k: {} for k in KINDS}
        self._builtins_loaded = not load_builtins
        self._plugins_loaded = False

    # -- registration -----------------------------------------------------
    def add(
        self, kind: str, name: str, factory: Callable[..., Any], *, override: bool = False
    ) -> None:
        """Register ``factory`` under ``kind``/``name``.

        Raises:
            ValueError: if the name is taken and ``override`` is false.
        """
        bucket = self._factories.setdefault(kind, {})
        existing = bucket.get(name)
        if existing is not None and existing is not factory and not override:
            raise ValueError(f"{kind} '{name}' is already registered (pass override=True)")
        bucket[name] = factory

    def remove(self, kind: str, name: str) -> None:
        """Unregister ``kind``/``name`` (no error if it is not registered)."""
        self._factories.get(kind, {}).pop(name, None)

    def register(self, kind: str, name: str, *, override: bool = False) -> Callable[[F], F]:
        """Class/function decorator form of :meth:`add`."""

        def decorator(factory: F) -> F:
            self.add(kind, name, factory, override=override)
            return factory

        return decorator

    # -- lookup -------------------------------------------------------------
    def _ensure_loaded(self) -> None:
        if not self._builtins_loaded:
            self._builtins_loaded = True
            importlib.import_module("ragpick.components")
            importlib.import_module("ragpick.adapters")
        if not self._plugins_loaded:
            self.load_plugins()

    def load_plugins(self) -> None:
        """Load components exposed through the ``ragpick.components`` entry points.

        Broken plugins are logged and skipped; they never prevent ragpick from working.
        """
        self._plugins_loaded = True
        for ep in metadata.entry_points(group=ENTRY_POINT_GROUP):
            try:
                obj = ep.load()
            except Exception as exc:  # a broken plugin must not break ragpick
                logger.warning("Failed to load ragpick plugin %r: %s", ep.name, exc)
                continue
            if "." in ep.name:
                kind, name = ep.name.split(".", 1)
                if name in self._factories.get(kind, {}):
                    logger.warning("Plugin %r ignored: %s '%s' already exists", ep.name, kind, name)
                    continue
                self.add(kind, name, obj)
                logger.debug("Loaded plugin component %s '%s'", kind, name)

    def kinds(self) -> list[str]:
        """All kinds that have at least one registered component, plus the built-in kinds."""
        self._ensure_loaded()
        return list(self._factories)

    def names(self, kind: str) -> list[str]:
        """Sorted names registered for ``kind``."""
        self._ensure_loaded()
        return sorted(self._factories.get(kind, {}))

    def has(self, kind: str, name: str) -> bool:
        """Whether ``kind``/``name`` is registered."""
        self._ensure_loaded()
        return name in self._factories.get(kind, {})

    def get(self, kind: str, name: str) -> Callable[..., Any]:
        """Return the factory for ``kind``/``name``.

        Raises:
            ComponentNotFoundError: listing the available names.
        """
        self._ensure_loaded()
        if kind not in self._factories:
            raise ComponentNotFoundError(
                f"Unknown component kind '{kind}'. Known kinds: {', '.join(self._factories)}"
            )
        try:
            return self._factories[kind][name]
        except KeyError:
            available = ", ".join(self.names(kind)) or "(none)"
            raise ComponentNotFoundError(
                f"Unknown {kind} '{name}'. Available {kind}s: {available}"
            ) from None

    def create(self, kind: str, name: str, /, **params: Any) -> Any:
        """Instantiate a component, validating ``params`` against its signature.

        Raises:
            ComponentNotFoundError: unknown kind/name.
            ComponentConfigError: parameters do not match the factory's signature.
        """
        factory = self.get(kind, name)
        try:
            signature = inspect.signature(factory)
        except (TypeError, ValueError):  # pragma: no cover - builtins without signatures
            signature = None
        if signature is not None:
            try:
                signature.bind(**params)
            except TypeError as exc:
                accepted = [
                    p.name
                    for p in signature.parameters.values()
                    if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
                ]
                raise ComponentConfigError(
                    f"Invalid params for {kind} '{name}': {exc}. "
                    f"Accepted params: {', '.join(accepted) or '(none)'}"
                ) from None
        return factory(**params)

    def accepted_params(self, kind: str, name: str) -> set[str] | None:
        """Names of keyword parameters the factory accepts (``None`` if it takes ``**kwargs``)."""
        signature = inspect.signature(self.get(kind, name))
        names: set[str] = set()
        for p in signature.parameters.values():
            if p.kind is p.VAR_KEYWORD:
                return None
            if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY):
                names.add(p.name)
        return names


#: The global registry used by ragpick.
registry = Registry(load_builtins=True)


def register(kind: str, name: str, *, override: bool = False) -> Callable[[F], F]:
    """Register a component in the global :data:`registry` (decorator)."""
    return registry.register(kind, name, override=override)
