"""Translate an ngio store + array path into a TensorStore spec.

Store-type dispatch goes through `NgioStore` services (`local_root`,
`fsspec_fs`), never through `isinstance` checks on zarr stores: a handler's
store is an `NgioStore` facade, so checking for `LocalStore` here would never
match.
"""

import os
from dataclasses import dataclass
from typing import Any, Literal

from ngio.utils._store import NgioStore


@dataclass(frozen=True)
class Unsupported:
    """Why an array cannot be opened with TensorStore; ngio falls back to zarr."""

    reason: str


def build_spec(
    store: NgioStore, array_path: str, zarr_format: Literal[2, 3]
) -> dict[str, Any] | Unsupported:
    """The TensorStore spec opening the array at `array_path` in `store`."""
    kvstore = build_kvstore(store, array_path)
    if isinstance(kvstore, Unsupported):
        return kvstore
    return {"driver": "zarr3" if zarr_format == 3 else "zarr", "kvstore": kvstore}


def build_kvstore(store: NgioStore, path: str) -> dict[str, Any] | Unsupported:
    """The TensorStore kvstore spec for the directory `path` inside `store`."""
    path = path.strip("/")
    root = store.local_root
    if root is not None:
        return {"driver": "file", "path": _as_dir(f"{root.as_posix()}/{path}")}

    fs = store.fsspec_fs
    if fs is None:
        return Unsupported(f"{store.store_type} stores are not supported")

    protocols = (fs.protocol,) if isinstance(fs.protocol, str) else fs.protocol
    full_path = f"{store.full_url(path)}"
    if "file" in protocols or "local" in protocols:
        return {"driver": "file", "path": _as_dir(full_path)}
    if "s3" in protocols:
        return _s3_kvstore(fs, full_path)
    if "gs" in protocols or "gcs" in protocols:
        return _gcs_kvstore(fs, full_path)
    return Unsupported(f"fsspec protocol {protocols[0]!r} is not supported")


def _as_dir(path: str) -> str:
    # TensorStore treats a kvstore path ending in "/" as a directory prefix.
    return path.rstrip("/") + "/"


def _split_bucket(full_path: str) -> tuple[str, str]:
    bucket, _, key = full_path.strip("/").partition("/")
    return bucket, _as_dir(key) if key else ""


def _s3_kvstore(fs: Any, full_path: str) -> dict[str, Any] | Unsupported:
    """Mirror an s3fs filesystem's endpoint, region and credential source.

    TensorStore takes credentials only from a provider (environment, profile,
    instance metadata, ...), never as literal keys. A filesystem built with an
    explicit `key`/`secret` therefore cannot be mirrored faithfully, and is
    left to zarr rather than silently opened with other credentials.
    """
    if getattr(fs, "key", None) or getattr(fs, "secret", None):
        return Unsupported(
            "s3fs was given explicit key/secret, which TensorStore cannot take; "
            "provide credentials through the environment or a profile instead"
        )

    bucket, key = _split_bucket(full_path)
    spec: dict[str, Any] = {"driver": "s3", "bucket": bucket, "path": key}

    client_kwargs = getattr(fs, "client_kwargs", None) or {}
    # s3fs leaves both unset when botocore resolves them from the environment.
    endpoint = (
        getattr(fs, "endpoint_url", None)
        or client_kwargs.get("endpoint_url")
        or os.environ.get("AWS_ENDPOINT_URL_S3")
        or os.environ.get("AWS_ENDPOINT_URL")
    )
    if endpoint:
        spec["endpoint"] = endpoint
    region = (
        client_kwargs.get("region_name")
        or os.environ.get("AWS_REGION")
        or os.environ.get("AWS_DEFAULT_REGION")
    )
    if region:
        spec["aws_region"] = region

    profile = (getattr(fs, "storage_options", None) or {}).get("profile")
    if getattr(fs, "anon", False):
        spec["aws_credentials"] = {"type": "anonymous"}
    elif profile:
        spec["aws_credentials"] = {"type": "profile", "profile": profile}
    else:
        spec["aws_credentials"] = {"type": "default"}
    return spec


def _gcs_kvstore(fs: Any, full_path: str) -> dict[str, Any] | Unsupported:
    """GCS with application-default credentials, the only source mirrored."""
    token = (getattr(fs, "storage_options", None) or {}).get("token")
    if token not in (None, "google_default"):
        return Unsupported(
            "gcsfs was given an explicit token, which is not mirrored to "
            "TensorStore; use application-default credentials instead"
        )
    bucket, key = _split_bucket(full_path)
    return {"driver": "gcs", "bucket": bucket, "path": key}
