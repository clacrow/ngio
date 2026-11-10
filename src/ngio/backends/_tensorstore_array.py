"""A TensorStore handle behind the indexing interface ngio's I/O paths use."""

from collections.abc import Sequence
from typing import Any

import numpy as np
import tensorstore as ts

_Index = int | slice | list[int] | np.ndarray


class TensorStoreArray:
    """The data side of a zarr array, served by TensorStore.

    Only what ngio's read/write paths and dask need: `shape`, `dtype`, `ndim`,
    `chunks`, `__getitem__` and `__setitem__`. Metadata stays on the
    `zarr.Array` this was opened for.

    Indexing follows zarr rather than TensorStore, which differs in three
    ways that would otherwise surface as errors: negative indices count from
    the end, slices past the end are clipped, and assigned values are cast to
    the array dtype unconditionally. A list index selects orthogonally, like
    zarr's `oindex`.
    """

    def __init__(self, handle: ts.TensorStore, chunks: tuple[int, ...]) -> None:
        self._handle = handle
        self._shape = tuple(int(s) for s in handle.shape)
        self._dtype = np.dtype(handle.dtype.numpy_dtype)
        self._chunks = chunks

    @property
    def handle(self) -> ts.TensorStore:
        return self._handle

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    @property
    def dtype(self) -> np.dtype:
        return self._dtype

    @property
    def ndim(self) -> int:
        return len(self._shape)

    @property
    def chunks(self) -> tuple[int, ...]:
        return self._chunks

    def _view(self, selection: Any) -> ts.TensorStore:
        index, orthogonal = _normalize_selection(selection, self._shape)
        return self._handle.oindex[index] if orthogonal else self._handle[index]

    def __getitem__(self, selection: Any) -> np.ndarray:
        return self._view(selection).read().result()

    def __setitem__(self, selection: Any, value: Any) -> None:
        # `astype` with numpy's default (unsafe) casting, as zarr does.
        data = np.asarray(value).astype(self._dtype, copy=False)
        self._view(selection).write(data).result()

    def __dask_tokenize__(self) -> tuple:
        return ("ngio-tensorstore", self._handle.spec().to_json(), self._chunks)

    def __repr__(self) -> str:
        return (
            f"TensorStoreArray(shape={self.shape}, dtype={self.dtype}, "
            f"chunks={self.chunks})"
        )


def _normalize_selection(
    selection: Any, shape: Sequence[int]
) -> tuple[tuple[_Index, ...], bool]:
    """`selection` with zarr semantics, as in-bounds TensorStore indices.

    Returns the index tuple and whether it needs orthogonal (`oindex`) indexing.
    """
    if not isinstance(selection, tuple):
        selection = (selection,)
    if any(s is Ellipsis for s in selection):
        at = next(i for i, s in enumerate(selection) if s is Ellipsis)
        fill = (slice(None),) * (len(shape) - len(selection) + 1)
        selection = selection[:at] + fill + selection[at + 1 :]
    if len(selection) > len(shape):
        raise IndexError(
            f"Too many indices: array is {len(shape)}-dimensional, "
            f"but {len(selection)} were indexed."
        )
    selection = selection + (slice(None),) * (len(shape) - len(selection))

    out: list[_Index] = []
    orthogonal = False
    for index, size in zip(selection, shape, strict=True):
        if isinstance(index, slice):
            if index.step is not None and index.step < 0:
                # `slice.indices` yields stop=-1 here, which TensorStore
                # would read as a literal coordinate.
                raise IndexError("Negative slice steps are not supported.")
            out.append(slice(*index.indices(size)))
        elif isinstance(index, int | np.integer):
            out.append(_wrap(int(index), size))
        elif isinstance(index, list | np.ndarray):
            orthogonal = True
            out.append(np.asarray([_wrap(int(i), size) for i in index], dtype=np.int64))
        else:
            raise IndexError(f"Unsupported index {index!r} of type {type(index)}.")
    return tuple(out), orthogonal


def _wrap(index: int, size: int) -> int:
    wrapped = index + size if index < 0 else index
    if not 0 <= wrapped < size:
        raise IndexError(f"Index {index} is out of bounds for axis of size {size}.")
    return wrapped
