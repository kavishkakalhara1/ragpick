"""Exception hierarchy for ragpick.

Every error raised deliberately by ragpick derives from :class:`RagpickError`, so
callers (and the CLI) can catch one type and show a friendly message.
"""

from __future__ import annotations


class RagpickError(Exception):
    """Base class for all ragpick errors. Messages are meant to be shown to users."""


class ConfigError(RagpickError):
    """The configuration file is missing, malformed, or fails validation."""


class ComponentNotFoundError(RagpickError):
    """A component kind/name pair is not present in the registry."""


class ComponentConfigError(RagpickError):
    """A component was given parameters it does not accept."""


class MissingDependencyError(RagpickError):
    """An optional dependency (extra) required by a component is not installed."""

    def __init__(self, package: str, extra: str, component: str) -> None:
        self.package = package
        self.extra = extra
        super().__init__(
            f"{component} requires the '{package}' package. "
            f'Install it with: pip install "ragpick[{extra}]"'
        )


class MissingCredentialError(RagpickError):
    """A required credential (usually an API key environment variable) is not set."""


class BlueprintError(RagpickError):
    """A blueprint is unknown or not yet available."""


class IndexNotFoundError(RagpickError):
    """A query was issued before anything was ingested."""


class ProviderError(RagpickError):
    """A remote model provider returned an error or an unexpected response."""
