"""Where ngio's data I/O picks between zarr and TensorStore.

ngio keeps `zarr.Array` as *the* array object everywhere — metadata, shards,
creation, copies. Only the five places that move chunk data ask this module
for the object to read from or write into (`io_array`, `as_dask`). With
`tensorstore.enabled` off, both return zarr's own objects untouched.
"""

import json
import threading
import warnings
from collections import OrderedDict
from typing import TYPE_CHECKING, Any

import dask.array as da
import numpy as np
import zarr

from ngio.backends._spec_builder import Unsupported, build_spec
from ngio.config import TensorStoreConfig, get_config
from ngio.utils._store import NgioStore
from ngio.utils._warnings import NgioUserWarning

if TYPE_CHECKING:
    from ngio.backends._tensorstore_array import TensorStoreArray

#: Open handles, most recently used last. Keyed on the spec *and* the zarr
#: metadata, so an array rewritten with a different layout gets a fresh handle.
_HANDLES: "OrderedDict[tuple, TensorStoreArray]" = OrderedDict()
_MAX_HANDLES = 1024
_CONTEXT: tuple[tuple, Any] | None = None
_LOCK = threading.Lock()
#: Fallback reasons already reported; one plate opens thousands of arrays.
_REPORTED: set[str] = set()


def io_array(zarr_array: zarr.Array) -> "zarr.Array | TensorStoreArray":
    """The object to read `zarr_array`'s data from, or write it into.

    `zarr_array` itself unless the TensorStore backend is enabled and can open
    it; otherwise a `TensorStoreArray` over the same data. Indexing semantics
    are the same either way.
    """
    config = get_config().tensorstore
    if not config.enabled:
        return zarr_array
    handle = _open(zarr_array, config)
    return zarr_array if handle is None else handle


def as_dask(zarr_array: zarr.Array, chunks: tuple[int, ...] | None = None) -> da.Array:
    """`zarr_array` as a dask array, read through the configured backend.

    Args:
        zarr_array: The array to read.
        chunks: The dask block shape; the array's chunks if `None`.
    """
    source = io_array(zarr_array)
    if source is zarr_array:
        return da.from_zarr(zarr_array, chunks=chunks or zarr_array.chunks)
    return da.from_array(
        source,
        chunks=chunks or zarr_array.chunks,
        meta=np.empty((0,) * zarr_array.ndim, dtype=zarr_array.dtype),
        asarray=False,
        fancy=False,
    )


def _warn_fallback(reason: str) -> None:
    if reason in _REPORTED:
        return
    _REPORTED.add(reason)
    warnings.warn(
        f"tensorstore.enabled is set, but ngio falls back to zarr: {reason}. "
        "Reads and writes are correct, just not served by TensorStore.",
        NgioUserWarning,
        stacklevel=4,
    )


def _context(config: TensorStoreConfig) -> Any:
    """One TensorStore context for every handle, rebuilt when its config changes."""
    import tensorstore as ts

    global _CONTEXT
    key = (
        config.data_copy_concurrency,
        config.file_io_concurrency,
        config.request_concurrency,
    )
    if _CONTEXT is not None and _CONTEXT[0] == key:
        return _CONTEXT[1]
    spec: dict[str, Any] = {}
    if config.data_copy_concurrency is not None:
        spec["data_copy_concurrency"] = {"limit": config.data_copy_concurrency}
    if config.file_io_concurrency is not None:
        spec["file_io_concurrency"] = {"limit": config.file_io_concurrency}
    if config.request_concurrency is not None:
        spec["s3_request_concurrency"] = {"limit": config.request_concurrency}
        spec["gcs_request_concurrency"] = {"limit": config.request_concurrency}
    context = ts.Context(spec)
    _CONTEXT = (key, context)
    return context


def _open(
    zarr_array: zarr.Array, config: TensorStoreConfig
) -> "TensorStoreArray | None":
    """A cached `TensorStoreArray` for `zarr_array`, or `None` to use zarr."""
    spec = build_spec(
        NgioStore.ensure(zarr_array.store),
        zarr_array.path,
        zarr_array.metadata.zarr_format,
    )
    if isinstance(spec, Unsupported):
        _warn_fallback(spec.reason)
        return None

    read_only = zarr_array.read_only
    with _LOCK:
        context = _context(config)
        key = (
            json.dumps(spec, sort_keys=True),
            read_only,
            repr(zarr_array.metadata),
            id(context),
        )
        cached = _HANDLES.get(key)
        if cached is not None:
            _HANDLES.move_to_end(key)
            return cached

    import tensorstore as ts

    from ngio.backends._tensorstore_array import TensorStoreArray

    try:
        handle = ts.open(
            spec, context=context, open=True, read=True, write=not read_only
        ).result()
    except ValueError as e:
        # TensorStore reports every open failure (missing metadata, a codec
        # it does not implement, ...) as ValueError.
        _warn_fallback(f"TensorStore could not open the array ({e})")
        return None

    array = TensorStoreArray(handle, chunks=zarr_array.chunks)
    if array.shape != zarr_array.shape or array.dtype != zarr_array.dtype:
        _warn_fallback(
            f"TensorStore sees shape {array.shape} / dtype {array.dtype}, zarr "
            f"sees {zarr_array.shape} / {zarr_array.dtype}"
        )
        return None

    with _LOCK:
        _HANDLES[key] = array
        while len(_HANDLES) > _MAX_HANDLES:
            _HANDLES.popitem(last=False)
    return array


def _clear_handles() -> None:
    """Drop every cached handle and fallback report (for tests)."""
    global _CONTEXT
    with _LOCK:
        _HANDLES.clear()
        _REPORTED.clear()
        _CONTEXT = None
