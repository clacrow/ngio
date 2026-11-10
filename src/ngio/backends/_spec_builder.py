"""Convert zarr Store + paths into TensorStore JSON specs."""

from __future__ import annotations

from typing import Any, Literal

from zarr.storage import FsspecStore, LocalStore, MemoryStore, ZipStore


def build_tensorstore_spec(
    store: Any,
    group_path: str,
    array_path: str,
    zarr_format: Literal[2, 3],
) -> dict | None:
    """Build a TensorStore spec from a zarr store and array path.

    Returns None if the store type is not supported by TensorStore,
    indicating the caller should fall back to zarr.
    """
    kvstore = _build_kvstore(store, group_path, array_path)
    if kvstore is None:
        return None

    driver = "zarr3" if zarr_format == 3 else "zarr"
    return {
        "driver": driver,
        "kvstore": kvstore,
        "open": True,
    }


def _build_kvstore(
    store: Any,
    group_path: str,
    array_path: str,
) -> dict | None:
    """Build the kvstore portion of a TensorStore spec."""
    if isinstance(store, LocalStore):
        full_path = store.root / group_path / array_path
        return {"driver": "file", "path": str(full_path)}

    if isinstance(store, FsspecStore):
        return _build_kvstore_from_fsspec(store, group_path, array_path)

    if isinstance(store, MemoryStore | ZipStore):
        return None

    return None


def _build_kvstore_from_fsspec(
    store: FsspecStore,
    group_path: str,
    array_path: str,
) -> dict | None:
    """Build kvstore spec from an FsspecStore by inspecting its filesystem protocol."""
    fs = store.fs
    protocol = fs.protocol if isinstance(fs.protocol, str) else fs.protocol[0]

    base_path = store.path.rstrip("/")
    if group_path:
        base_path = f"{base_path}/{group_path}"
    full_path = f"{base_path}/{array_path}"

    if protocol in ("s3", "s3a"):
        return _build_s3_kvstore(fs, full_path)
    elif protocol in ("gcs", "gs"):
        return _build_gcs_kvstore(full_path)
    elif protocol == "file":
        return {"driver": "file", "path": full_path}

    # HTTP and other protocols: fall back to zarr
    return None


def _build_s3_kvstore(fs: Any, full_path: str) -> dict:
    """Build S3 kvstore spec."""
    parts = full_path.split("/", 1)
    bucket = parts[0]
    path = parts[1] if len(parts) > 1 else ""

    spec: dict[str, Any] = {
        "driver": "s3",
        "bucket": bucket,
        "path": path,
    }

    client_kwargs = getattr(fs, "client_kwargs", None) or {}
    endpoint_url = client_kwargs.get("endpoint_url")
    if endpoint_url:
        spec["endpoint"] = endpoint_url

    if getattr(fs, "anon", False):
        spec["aws_credentials"] = {"anonymous": True}
    else:
        key = getattr(fs, "key", None)
        secret = getattr(fs, "secret", None)
        if key and secret:
            spec["aws_credentials"] = {
                "access_key": key,
                "secret_key": secret,
            }

    return spec


def _build_gcs_kvstore(full_path: str) -> dict:
    """Build GCS kvstore spec."""
    parts = full_path.split("/", 1)
    bucket = parts[0]
    path = parts[1] if len(parts) > 1 else ""

    return {
        "driver": "gcs",
        "bucket": bucket,
        "path": path,
    }


