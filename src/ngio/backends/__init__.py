"""Backend configuration for ngio array I/O."""

from ngio.backends._config import (
    BackendType,
    get_backend,
    set_backend,
    set_tensorstore_concurrency,
)

__all__ = [
    "BackendType",
    "get_backend",
    "set_backend",
    "set_tensorstore_concurrency",
]
