"""TensorStoreArray adapter exposing a zarr.Array-compatible interface."""

from __future__ import annotations

from typing import Any

import numpy as np
import tensorstore as ts


class TensorStoreArray:
    """Wrapper around a TensorStore array providing zarr.Array-like access.

    Implements the subset of zarr.Array interface used by ngio:
    shape, chunks, dtype, ndim, __getitem__, __setitem__,
    plus compressors/shards delegated to an optional zarr array reference.
    """

    def __init__(
        self,
        ts_array: ts.TensorStore,
        zarr_array_ref: Any = None,
    ) -> None:
        self._ts_array = ts_array
        self._zarr_array_ref = zarr_array_ref
        self._chunks: tuple[int, ...] | None = None

    @classmethod
    def open(
        cls,
        spec: dict,
        concurrency: int = 128,
        zarr_array_ref: Any = None,
    ) -> TensorStoreArray | None:
        """Open a TensorStore array. Returns None if opening fails."""
        try:
            context = ts.Context({"data_copy_concurrency": {"limit": concurrency}})
            ts_array = ts.open(spec, context=context).result()
        except Exception:
            return None
        return cls(ts_array=ts_array, zarr_array_ref=zarr_array_ref)

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(int(s) for s in self._ts_array.shape)

    @property
    def chunks(self) -> tuple[int, ...]:
        if self._chunks is not None:
            return self._chunks
        if self._zarr_array_ref is not None:
            self._chunks = self._zarr_array_ref.chunks
            return self._chunks
        layout = self._ts_array.chunk_layout
        write_chunk = layout.write_chunk
        if write_chunk is not None and write_chunk.shape is not None:
            self._chunks = tuple(int(s) for s in write_chunk.shape)
        else:
            read_chunk = layout.read_chunk
            if read_chunk is not None and read_chunk.shape is not None:
                self._chunks = tuple(int(s) for s in read_chunk.shape)
            else:
                self._chunks = self.shape
        return self._chunks

    @property
    def dtype(self) -> np.dtype:
        return self._ts_array.dtype.numpy_dtype

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def compressors(self) -> Any:
        if self._zarr_array_ref is not None:
            return self._zarr_array_ref.compressors
        return None

    @property
    def shards(self) -> Any:
        if self._zarr_array_ref is not None:
            return self._zarr_array_ref.shards
        return None

    def __getitem__(self, slicing: Any) -> np.ndarray:
        return self._ts_array[slicing].read().result()

    def __setitem__(self, slicing: Any, value: Any) -> None:
        self._ts_array[slicing].write(value).result()

    def __repr__(self) -> str:
        return f"TensorStoreArray(shape={self.shape}, dtype={self.dtype}, chunks={self.chunks})"
