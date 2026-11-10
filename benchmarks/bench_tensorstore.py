"""Benchmark comparing zarr-python vs tensorstore for array reads/writes.

Usage:
    pixi run -e dev python benchmarks/bench_tensorstore.py /path/to/image.zarr
    pixi run -e dev python benchmarks/bench_tensorstore.py s3://bucket/image.zarr
    pixi run -e dev python benchmarks/bench_tensorstore.py /path/to/image.zarr --trials 10
    pixi run -e dev python benchmarks/bench_tensorstore.py /path/to/image.zarr --concurrency 256
"""

import argparse
import statistics
import time
from collections.abc import Callable

import numpy as np

import ngio


def _time_it(fn: Callable, n_trials: int) -> list[float]:
    times = []
    for _ in range(n_trials):
        start = time.perf_counter()
        fn()
        elapsed = time.perf_counter() - start
        times.append(elapsed)
    return times


def _report(label: str, times: list[float], nbytes: int) -> None:
    med = statistics.median(times)
    p95 = sorted(times)[int(len(times) * 0.95)]
    throughput = (nbytes / 1e6) / med if med > 0 else float("inf")
    print(f"  {label:30s}  median={med:.4f}s  p95={p95:.4f}s  {throughput:.1f} MB/s")


def bench_numpy_read(
    store: ngio.StoreOrGroup, n_trials: int, concurrency: int
) -> None:
    print("\n--- NumPy read (full highest-res image) ---")
    for backend in ("zarr", "tensorstore"):
        ngio.set_backend(backend)  # type: ignore[arg-type]
        if backend == "tensorstore":
            ngio.set_tensorstore_concurrency(concurrency)
        container = ngio.open_ome_zarr_container(store, mode="r")
        image = container.get_image()
        nbytes = int(np.prod(image.shape) * np.dtype(image.dtype).itemsize)

        times = _time_it(lambda: image.get_as_numpy(), n_trials)
        _report(f"[{backend}]", times, nbytes)


def bench_numpy_slice_read(
    store: ngio.StoreOrGroup, n_trials: int, concurrency: int
) -> None:
    print("\n--- NumPy read (single z-slice / 2D plane) ---")
    for backend in ("zarr", "tensorstore"):
        ngio.set_backend(backend)  # type: ignore[arg-type]
        if backend == "tensorstore":
            ngio.set_tensorstore_concurrency(concurrency)
        container = ngio.open_ome_zarr_container(store, mode="r")
        image = container.get_image()

        slice_kwargs: dict = {}
        if image.dimensions.has_axis("z"):
            slice_kwargs["z"] = 0
        if image.dimensions.has_axis("t"):
            slice_kwargs["t"] = 0
        if image.dimensions.has_axis("c"):
            slice_kwargs["c"] = 0

        result = image.get_as_numpy(**slice_kwargs)
        nbytes = result.nbytes

        times = _time_it(lambda: image.get_as_numpy(**slice_kwargs), n_trials)
        _report(f"[{backend}]", times, nbytes)


def bench_dask_read(
    store: ngio.StoreOrGroup, n_trials: int, concurrency: int
) -> None:
    print("\n--- Dask read + compute (full highest-res image) ---")
    for backend in ("zarr", "tensorstore"):
        ngio.set_backend(backend)  # type: ignore[arg-type]
        if backend == "tensorstore":
            ngio.set_tensorstore_concurrency(concurrency)
        container = ngio.open_ome_zarr_container(store, mode="r")
        image = container.get_image()
        nbytes = int(np.prod(image.shape) * np.dtype(image.dtype).itemsize)

        times = _time_it(lambda: image.get_as_dask().compute(), n_trials)
        _report(f"[{backend}]", times, nbytes)


def bench_numpy_write(
    store: ngio.StoreOrGroup, n_trials: int, concurrency: int
) -> None:
    print("\n--- NumPy write (single z-slice / 2D plane) ---")
    for backend in ("zarr", "tensorstore"):
        ngio.set_backend(backend)  # type: ignore[arg-type]
        if backend == "tensorstore":
            ngio.set_tensorstore_concurrency(concurrency)
        container = ngio.open_ome_zarr_container(store, mode="r+")
        image = container.get_image()

        slice_kwargs: dict = {}
        if image.dimensions.has_axis("z"):
            slice_kwargs["z"] = 0
        if image.dimensions.has_axis("t"):
            slice_kwargs["t"] = 0
        if image.dimensions.has_axis("c"):
            slice_kwargs["c"] = 0

        data = image.get_as_numpy(**slice_kwargs)
        nbytes = data.nbytes

        times = _time_it(
            lambda: image.set_array(data, **slice_kwargs), n_trials
        )
        _report(f"[{backend}]", times, nbytes)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark zarr vs tensorstore backends"
    )
    parser.add_argument("store", help="Path or URL to an OME-Zarr container")
    parser.add_argument(
        "--trials", type=int, default=5, help="Number of trials per benchmark"
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=128,
        help="TensorStore concurrency limit",
    )
    parser.add_argument(
        "--skip-write",
        action="store_true",
        help="Skip write benchmarks (useful for read-only stores)",
    )
    args = parser.parse_args()

    print(f"Store: {args.store}")
    print(f"Trials: {args.trials}")
    print(f"TensorStore concurrency: {args.concurrency}")

    ngio.set_backend("zarr")
    container = ngio.open_ome_zarr_container(args.store, mode="r")
    image = container.get_image()
    print(f"Image shape: {image.shape}, dtype: {image.dtype}, chunks: {image.chunks}")

    bench_numpy_read(args.store, args.trials, args.concurrency)
    bench_numpy_slice_read(args.store, args.trials, args.concurrency)
    bench_dask_read(args.store, args.trials, args.concurrency)

    if not args.skip_write:
        bench_numpy_write(args.store, args.trials, args.concurrency)

    ngio.set_backend("zarr")
    print("\nDone. Backend reset to 'zarr'.")


if __name__ == "__main__":
    main()
