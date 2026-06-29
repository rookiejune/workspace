"""Resolve workspace paths from machine-level dynamic and static homes.

This module owns the `DYNAMIC_HOME` and `STATIC_HOME` contract for the workspace
package. It exposes stable path helpers for datasets, train outputs, model
roots, debug outputs and the cache roots delegated to external ecosystems.
Missing homes fall back to the Fudan shared defaults with a warning; explicit
dataset paths only override the dataset root.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path


DYNAMIC_HOME_ENV = "DYNAMIC_HOME"
STATIC_HOME_ENV = "STATIC_HOME"
ANYDATASET_HOME_ENV = "ANYDATASET_HOME"
HF_HOME_ENV = "HF_HOME"

FUDAN_HOME = Path("/mnt/pami202/zhuyin")
DEFAULT_STATIC_HOME = FUDAN_HOME
DEFAULT_DYNAMIC_HOME = FUDAN_HOME / "dynamic"
_DEFAULT_HOMES = {
    STATIC_HOME_ENV: DEFAULT_STATIC_HOME,
    DYNAMIC_HOME_ENV: DEFAULT_DYNAMIC_HOME,
}


def dynamic_home() -> Path:
    """Return the configured dynamic workspace home."""

    return _required_home(DYNAMIC_HOME_ENV)


def static_home() -> Path:
    """Return the configured static workspace home."""

    return _required_home(STATIC_HOME_ENV)


def dataset_dir(name: str) -> Path:
    """Return the stable root for one prepared dataset."""

    return static_home() / "datasets" / name


def train_dir(project: str) -> Path:
    """Return the training output root for one project."""

    return dynamic_home() / "train" / project


def models_dir() -> Path:
    """Return the root for manually managed model weights."""

    return static_home() / "models"


def debug_dir(project: str | None = None) -> Path:
    """Return the debug output root, optionally scoped to one project."""

    root = dynamic_home() / "debug"
    if project is None:
        return root
    return root / project


def anydataset_home() -> Path:
    """Return the anydataset internal storage root."""

    return _derived_env_path(ANYDATASET_HOME_ENV, "anydataset")


def hf_home() -> Path:
    """Return the HuggingFace cache root."""

    return _derived_env_path(HF_HOME_ENV, "huggingface")


def configure_environment() -> None:
    """Populate derived ecosystem environment variables from `STATIC_HOME`.

    Explicit `ANYDATASET_HOME` or `HF_HOME` values are respected. Missing values
    are derived from `STATIC_HOME` at call time so static assets and ecosystem
    caches stay under the same machine-level root by default.
    """

    root = static_home()
    _set_default_path(ANYDATASET_HOME_ENV, root / "anydataset")
    _set_default_path(HF_HOME_ENV, root / "huggingface")


def _derived_env_path(name: str, child: str) -> Path:
    value = os.environ.get(name)
    if value is not None:
        if not value:
            raise ValueError(f"{name} must not be empty.")
        return Path(value).expanduser()
    return static_home() / child


def _required_home(name: str) -> Path:
    value = os.environ.get(name)
    if not value:
        if value is None:
            return _default_home(name)
        raise ValueError(f"{name} must not be empty.")
    return Path(value).expanduser()


def _default_home(name: str) -> Path:
    value = _DEFAULT_HOMES[name]
    warnings.warn(
        f"{name} is not set; using Fudan default {value}.",
        RuntimeWarning,
        stacklevel=3,
    )
    os.environ[name] = str(value)
    return value


def _set_default_path(name: str, path: Path) -> None:
    value = os.environ.get(name)
    if value is not None:
        if not value:
            raise ValueError(f"{name} must not be empty.")
        return
    os.environ[name] = str(path)
