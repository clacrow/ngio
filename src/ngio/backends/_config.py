"""Global backend configuration for ngio."""

import os
from typing import Literal

BackendType = Literal["zarr", "tensorstore"]


class _BackendConfig:
    _instance: "_BackendConfig | None" = None
    _backend: BackendType = "zarr"
    _tensorstore_concurrency: int = 128

    def __new__(cls) -> "_BackendConfig":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            env_val = os.environ.get("NGIO_BACKEND", "zarr").lower()
            if env_val in ("zarr", "tensorstore"):
                cls._instance._backend = env_val  # type: ignore[assignment]
        return cls._instance

    @property
    def backend(self) -> BackendType:
        return self._backend

    @backend.setter
    def backend(self, value: BackendType) -> None:
        if value not in ("zarr", "tensorstore"):
            raise ValueError(f"Unsupported backend: {value!r}. Use 'zarr' or 'tensorstore'.")
        self._backend = value

    @property
    def tensorstore_concurrency(self) -> int:
        return self._tensorstore_concurrency

    @tensorstore_concurrency.setter
    def tensorstore_concurrency(self, value: int) -> None:
        if value < 1:
            raise ValueError("Concurrency must be at least 1.")
        self._tensorstore_concurrency = value


_config = _BackendConfig()


def get_backend() -> BackendType:
    """Get the current array I/O backend."""
    return _config.backend


def set_backend(backend: BackendType) -> None:
    """Set the array I/O backend.

    Args:
        backend: Either "zarr" (default) or "tensorstore".
    """
    _config.backend = backend


def set_tensorstore_concurrency(limit: int) -> None:
    """Set the concurrency limit for TensorStore chunk fetching.

    Args:
        limit: Number of parallel chunk operations. Default is 128.
    """
    _config.tensorstore_concurrency = limit
