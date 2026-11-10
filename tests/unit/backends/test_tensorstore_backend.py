import importlib.util
from pathlib import Path

import dask.array as da
import numpy as np
import pytest
import zarr

import ngio
from ngio.backends import _io
from ngio.backends._spec_builder import Unsupported, build_kvstore
from ngio.backends._tensorstore_array import TensorStoreArray
from ngio.config import TensorStoreConfig, get_config
from ngio.utils import NgioUserWarning
from ngio.utils._store import NgioStore


@pytest.fixture
def tensorstore_on(monkeypatch):
    monkeypatch.setattr(get_config().tensorstore, "enabled", True)
    _io._clear_handles()
    yield
    _io._clear_handles()


def _make_image(store, data: np.ndarray, version: str, shards=None):
    return ngio.create_ome_zarr_from_array(
        store,
        data,
        pixelsize=0.5,
        axes_names="zyx",
        ngff_version=version,  # ty: ignore[invalid-argument-type]
        chunks=(1, 16, 16),
        shards=shards,
        levels=3,
        consolidation_mode="dask",
    )


@pytest.mark.parametrize(
    "version, shards", [("0.4", None), ("0.5", None), ("0.5", (1, 32, 32))]
)
@pytest.mark.usefixtures("tensorstore_on")
def test_tensorstore_serves_local_io_and_matches_zarr(tmp_path: Path, version, shards):
    rng = np.random.default_rng(0)
    data = rng.integers(0, 1000, (2, 64, 64), dtype=np.uint16)
    container = _make_image(tmp_path / "img.zarr", data, version, shards)
    image = container.get_image()

    # The point of the backend: it must actually be the one serving the data,
    # not a silent fallback that leaves every test green.
    assert isinstance(_io.io_array(image.zarr_array), TensorStoreArray)
    assert isinstance(image.zarr_array, zarr.Array)

    np.testing.assert_array_equal(image.get_as_numpy(), data)
    np.testing.assert_array_equal(image.get_as_dask().compute(), data)

    patch = np.full((1, 10, 10), 7, dtype=np.uint16)
    image.set_array(patch, z=slice(0, 1), y=slice(5, 15), x=slice(5, 15))
    data[0, 5:15, 5:15] = 7
    image.set_array(
        da.from_array(np.full((2, 32, 64), 3, dtype=np.uint16)), y=slice(0, 32)
    )
    data[:, :32] = 3
    image.consolidate(mode="dask")

    np.testing.assert_array_equal(image.get_as_numpy(), data)
    # Every pyramid level written through TensorStore reads back identically
    # through plain zarr.
    reopened = ngio.open_ome_zarr_container(tmp_path / "img.zarr")
    for path in container.level_paths:
        ts_data = reopened.get_image(path=path).get_as_numpy()
        zarr_data = zarr.open_array(tmp_path / "img.zarr" / path, mode="r")[...]
        np.testing.assert_array_equal(ts_data, zarr_data)


@pytest.mark.usefixtures("tensorstore_on")
def test_tensorstore_array_follows_zarr_indexing(tmp_path: Path):
    array = zarr.create_array(tmp_path / "a.zarr", shape=(4, 5), dtype="uint16")
    array[...] = np.arange(20, dtype=np.uint16).reshape(4, 5)
    ts_array = _io.io_array(array)
    assert isinstance(ts_array, TensorStoreArray)

    expected = array[...]
    np.testing.assert_array_equal(ts_array[-1], expected[-1])
    np.testing.assert_array_equal(ts_array[0:99], expected[0:99])
    np.testing.assert_array_equal(ts_array[..., 1], expected[..., 1])
    np.testing.assert_array_equal(ts_array[[0, 2], 1:3], expected[[0, 2], 1:3])

    # Unsafe casts and scalars are assigned the way zarr assigns them.
    ts_array[0, 0:2] = np.array([1.7, 2.2])
    ts_array[1] = 9
    assert array[0, :2].tolist() == [1, 2]
    assert array[1].tolist() == [9] * 5

    with pytest.raises(IndexError):
        ts_array[4]
    with pytest.raises(IndexError):
        ts_array[::-1]


@pytest.mark.usefixtures("tensorstore_on")
def test_rewritten_array_gets_a_fresh_handle(tmp_path: Path):
    path = tmp_path / "a.zarr"
    first = zarr.create_array(path, shape=(4, 4), dtype="uint8")
    assert _io.io_array(first).shape == (4, 4)
    second = zarr.create_array(path, shape=(8, 8), dtype="uint8", overwrite=True)
    second[...] = 1
    assert _io.io_array(second).shape == (8, 8)
    assert _io.io_array(second)[...].sum() == 64


def test_disabled_backend_returns_zarr(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(get_config().tensorstore, "enabled", False)
    array = zarr.create_array(tmp_path / "a.zarr", shape=(4,), dtype="uint8")
    assert _io.io_array(array) is array


@pytest.mark.usefixtures("tensorstore_on")
def test_memory_store_falls_back_with_a_warning():
    data = np.ones((2, 32, 32), dtype=np.uint16)
    with pytest.warns(NgioUserWarning, match="falls back to zarr: memory stores"):
        container = _make_image({}, data, "0.5")
        image = container.get_image()
        np.testing.assert_array_equal(image.get_as_numpy(), data)
    assert _io.io_array(image.zarr_array) is image.zarr_array


def test_s3fs_explicit_keys_are_not_mirrored():
    s3fs = pytest.importorskip("s3fs")
    fs = s3fs.S3FileSystem(key="k", secret="s", asynchronous=True)
    store = NgioStore(zarr.storage.FsspecStore(fs, path="bucket/prefix"))
    result = build_kvstore(store, "img.zarr/0")
    assert isinstance(result, Unsupported)
    assert "key/secret" in result.reason


def test_s3fs_credential_sources_are_mirrored(monkeypatch):
    s3fs = pytest.importorskip("s3fs")
    monkeypatch.delenv("AWS_ENDPOINT_URL", raising=False)
    monkeypatch.delenv("AWS_ENDPOINT_URL_S3", raising=False)
    for kwargs, creds in [
        ({"anon": True}, {"type": "anonymous"}),
        ({"profile": "lab"}, {"type": "profile", "profile": "lab"}),
        ({}, {"type": "default"}),
    ]:
        fs = s3fs.S3FileSystem(
            asynchronous=True, endpoint_url="http://minio:9000", **kwargs
        )
        store = NgioStore(zarr.storage.FsspecStore(fs, path="bucket/prefix"))
        assert build_kvstore(store, "img.zarr/0") == {
            "driver": "s3",
            "bucket": "bucket",
            "path": "prefix/img.zarr/0/",
            "endpoint": "http://minio:9000",
            "aws_credentials": creds,
        } | ({"aws_region": r} if (r := _env_region()) else {})


def _env_region() -> str | None:
    import os

    return os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")


def test_enabling_without_tensorstore_installed_is_rejected(monkeypatch):
    real_find_spec = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name, *a: None if name == "tensorstore" else real_find_spec(name, *a),
    )
    with pytest.raises(ValueError, match=r"ngio\[tensorstore\]"):
        TensorStoreConfig(enabled=True)


def test_concurrency_settings_reach_the_context(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(get_config(), "tensorstore", TensorStoreConfig(enabled=True))
    _io._clear_handles()
    array = zarr.create_array(tmp_path / "a.zarr", shape=(4,), dtype="uint8")
    first = _io.io_array(array)
    get_config().tensorstore.data_copy_concurrency = 2
    second = _io.io_array(array)
    assert first is not second  # new context, new handle
    assert _io._CONTEXT is not None
    assert _io._CONTEXT[1]["data_copy_concurrency"].to_json() == {"limit": 2}
    _io._clear_handles()
