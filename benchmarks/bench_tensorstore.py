"""Benchmark ngio's array I/O: zarr-python, zarr + zarrs, and TensorStore.

Usage:
    pixi run -e dev python benchmarks/bench_tensorstore.py /path/to/image.zarr
    pixi run -e dev python benchmarks/bench_tensorstore.py s3://bucket/image.zarr
    pixi run -e dev python benchmarks/bench_tensorstore.py IMG --trials 10 \
        --request-concurrency 256

Each configuration does one untimed warm-up run per benchmark, so the first
configuration does not pay for warming the OS page cache for the others.
`zarrs` only engages on plain local stores (see `ngio.utils._codec_pipeline`).
"""

import argparse
import importlib.util
import statistics
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager

import numpy as np
import zarr

import ngio
from ngio.backends._io import io_array
from ngio.config import TensorStoreConfig, get_config, set_config


def _configurations() -> list[str]:
    configs = ["zarr"]
    if importlib.util.find_spec("zarrs") is not None:
        configs.append("zarrs")
    configs.append("tensorstore")
    return configs


@contextmanager
def _configured(name: str, args: argparse.Namespace) -> Iterator[None]:
    """Run the body under one I/O configuration, restoring the previous one."""
    previous = get_config()
    config = previous.model_copy(deep=True)
    config.tensorstore = TensorStoreConfig(
        enabled=name == "tensorstore",
        data_copy_concurrency=args.data_copy_concurrency,
        request_concurrency=args.request_concurrency,
    )
    pipeline = (
        "zarrs.ZarrsCodecPipeline"
        if name == "zarrs"
        else "zarr.core.codec_pipeline.BatchedCodecPipeline"
    )
    set_config(config)
    try:
        with zarr.config.set({"codec_pipeline.path": pipeline}):
            yield
    finally:
        set_config(previous)


def _time_it(fn: Callable[[], object], n_trials: int) -> list[float]:
    fn()  # warm-up, untimed
    times = []
    for _ in range(n_trials):
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
    return times


def _report(label: str, times: list[float], nbytes: int) -> None:
    med = statistics.median(times)
    p95 = sorted(times)[min(len(times) - 1, int(len(times) * 0.95))]
    throughput = (nbytes / 1e6) / med if med > 0 else float("inf")
    print(f"  {label:14s} median={med:.4f}s  p95={p95:.4f}s  {throughput:8.1f} MB/s")


def _plane_kwargs(image: ngio.Image) -> dict[str, int]:
    """Index 0 on every non-spatial-plane axis present: one 2D plane."""
    return {ax: 0 for ax in ("t", "c", "z") if ax in image.dimensions.axes}


def _run(args: argparse.Namespace, title: str, mode: str, op: str) -> None:
    print(f"\n--- {title} ---")
    for name in _configurations():
        with _configured(name, args):
            container = ngio.open_ome_zarr_container(args.store, mode=mode)  # ty: ignore[invalid-argument-type]
            image = container.get_image()
            served_by = type(io_array(image.zarr_array)).__name__
            if name == "tensorstore" and served_by != "TensorStoreArray":
                print("  [tensorstore] FELL BACK TO ZARR - see the warning above")
            kwargs = _plane_kwargs(image) if "plane" in op else {}
            data = image.get_as_numpy(**kwargs)

            def fn(image=image, kwargs=kwargs, data=data) -> object:
                if op == "write_plane":
                    return image.set_array(data, **kwargs)
                if op == "dask":
                    return image.get_as_dask().compute()
                return image.get_as_numpy(**kwargs)

            _report(f"[{name}]", _time_it(fn, args.trials), data.nbytes)


def main() -> None:
    """Parse arguments and run every benchmark."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("store", help="Path or URL to an OME-Zarr container")
    parser.add_argument("--trials", type=int, default=5, help="Timed runs each")
    parser.add_argument("--data-copy-concurrency", type=int, default=None)
    parser.add_argument("--request-concurrency", type=int, default=None)
    parser.add_argument(
        "--skip-write", action="store_true", help="For read-only stores"
    )
    args = parser.parse_args()

    image = ngio.open_ome_zarr_container(args.store, mode="r").get_image()
    print(f"Store: {args.store}  trials: {args.trials}")
    print(
        f"Image shape: {image.shape}, dtype: {image.dtype}, chunks: {image.chunks}, "
        f"shards: {image.zarr_array.shards}"
    )
    print(f"Configurations: {', '.join(_configurations())}")

    _run(args, "NumPy read (full highest-res image)", "r", "read")
    _run(args, "NumPy read (single 2D plane)", "r", "read_plane")
    _run(args, "Dask read + compute (full highest-res image)", "r", "dask")
    if not args.skip_write:
        _run(args, "NumPy write (single 2D plane)", "r+", "write_plane")

    nbytes = int(np.prod(image.shape) * np.dtype(image.dtype).itemsize)
    print(f"\nDone. Full image: {nbytes / 1e6:.1f} MB.")


if __name__ == "__main__":
    main()
