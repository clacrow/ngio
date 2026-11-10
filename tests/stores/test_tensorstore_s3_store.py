import numpy as np
import pytest
import s3fs
from tests.stores.utils import random_zarr_path

import ngio
from ngio.backends import _io
from ngio.backends._tensorstore_array import TensorStoreArray
from ngio.config import get_config


@pytest.fixture
def tensorstore_on(monkeypatch):
    monkeypatch.setattr(get_config().tensorstore, "enabled", True)
    _io._clear_handles()
    yield
    _io._clear_handles()


@pytest.mark.usefixtures("tensorstore_on")
def test_tensorstore_on_s3(moto_s3_server: dict) -> None:
    # No explicit key/secret: credentials come from the environment moto sets
    # up, which is the source TensorStore can share with s3fs.
    fs = s3fs.S3FileSystem(
        client_kwargs={
            "endpoint_url": moto_s3_server["endpoint_url"],
            "region_name": "us-east-1",
        }
    )
    store = fs.get_mapper(f"{moto_s3_server['bucket_name']}/{random_zarr_path()}")
    data = np.random.default_rng(0).integers(0, 1000, (2, 64, 64), dtype=np.uint16)

    container = ngio.create_ome_zarr_from_array(
        store,
        data,
        pixelsize=0.5,
        axes_names="zyx",
        chunks=(1, 32, 32),
        levels=2,
        consolidation_mode="dask",
    )
    image = container.get_image()
    assert isinstance(_io.io_array(image.zarr_array), TensorStoreArray)

    image.set_array(
        np.full((1, 8, 8), 5, dtype=np.uint16),
        z=slice(0, 1),
        y=slice(0, 8),
        x=slice(0, 8),
    )
    data[0, :8, :8] = 5
    image.consolidate(mode="dask")
    np.testing.assert_array_equal(image.get_as_numpy(), data)
    np.testing.assert_array_equal(image.get_as_dask().compute(), data)

    # Read back through plain zarr: TensorStore's writes are visible to it.
    get_config().tensorstore.enabled = False
    reopened = ngio.open_ome_zarr_container(store)
    np.testing.assert_array_equal(reopened.get_image().get_as_numpy(), data)
