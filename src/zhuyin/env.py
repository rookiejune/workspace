"""Resolve workspace paths from machine-level dynamic and static homes.

This module owns the `LOCATION`, `DYNAMIC_HOME` and `STATIC_HOME` contract for
the workspace package. It exposes stable path helpers for datasets, train
outputs, model roots, debug outputs and static cache roots delegated to external
ecosystems. Missing homes fall back to the defaults selected by `LOCATION` with
a warning; explicit dataset paths only override the dataset root.
"""

from __future__ import annotations

import os
import warnings
from enum import StrEnum, auto
from pathlib import Path

LOCATION_ENV = "LOCATION"
DYNAMIC_HOME_ENV = "DYNAMIC_HOME"
STATIC_HOME_ENV = "STATIC_HOME"
ANYDATASET_HOME_ENV = "ANYDATASET_HOME"
ANYTRAIN_HOME_ENV = "ANYTRAIN_HOME"
BPE_CACHE_DIR_ENV = "BPE_CACHE_DIR"
HF_HOME_ENV = "HF_HOME"
HF_HUB_CACHE_ENV = "HF_HUB_CACHE"
HF_DATASETS_CACHE_ENV = "HF_DATASETS_CACHE"
TORCH_HOME_ENV = "TORCH_HOME"
WHISPER_ROOT_ENV = "ANYTRAIN_WHISPER_ROOT"

FUDAN_HOME = Path("/mnt/pami202/zhuyin")
HZ_HOME = Path("/nfs/yin.zhu")


class Location(StrEnum):
    """Known machine locations and their default workspace roots."""

    FUDAN = auto()
    HZ = auto()

    @property
    def default_static_home(self) -> Path:
        if self is Location.FUDAN:
            return FUDAN_HOME
        if self is Location.HZ:
            return HZ_HOME
        raise ValueError(f"unsupported {LOCATION_ENV}: {self.value}")

    @property
    def default_dynamic_home(self) -> Path:
        if self is Location.FUDAN:
            return self.default_static_home / "dynamic"
        if self is Location.HZ:
            return Path("/yin.zhu")
        raise ValueError(f"unsupported {LOCATION_ENV}: {self.value}")

    def default_home(self, name: str) -> Path:
        if name == STATIC_HOME_ENV:
            return self.default_static_home
        if name == DYNAMIC_HOME_ENV:
            return self.default_dynamic_home
        raise ValueError(f"unsupported home variable: {name}")


DEFAULT_LOCATION = Location.FUDAN
DEFAULT_STATIC_HOME = DEFAULT_LOCATION.default_static_home
DEFAULT_DYNAMIC_HOME = DEFAULT_LOCATION.default_dynamic_home


def location() -> Location:
    """Return the configured workspace location."""

    value = os.environ.get(LOCATION_ENV)
    if value is None:
        os.environ[LOCATION_ENV] = DEFAULT_LOCATION.value
        return DEFAULT_LOCATION
    if not value:
        raise ValueError(f"{LOCATION_ENV} must not be empty.")
    try:
        return Location(value)
    except ValueError as e:
        choices = ", ".join(item.value for item in Location)
        raise ValueError(f"{LOCATION_ENV} must be one of: {choices}.") from e


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


def hf_hub_cache() -> Path:
    """Return the HuggingFace Hub cache root."""

    return _env_path_or(HF_HUB_CACHE_ENV, hf_home() / "hub")


def hf_datasets_cache() -> Path:
    """Return the HuggingFace datasets cache root."""

    return _env_path_or(HF_DATASETS_CACHE_ENV, hf_home() / "datasets")


def bpe_cache_dir() -> Path:
    """Return the reusable BPE artifact root."""

    return _derived_env_path(BPE_CACHE_DIR_ENV, "bpe")


def anytrain_home() -> Path:
    """Return the anytrain integration cache root."""

    return _env_path_or(ANYTRAIN_HOME_ENV, static_home())


def torch_home() -> Path:
    """Return the Torch cache root used by optional backends."""

    return _env_path_or(TORCH_HOME_ENV, static_home() / "torch")


def whisper_root() -> Path:
    """Return the Whisper checkpoint root used by speech evaluators."""

    return _env_path_or(WHISPER_ROOT_ENV, static_home() / "whisper")


def configure_environment() -> None:
    """Populate derived ecosystem environment variables from `STATIC_HOME`.

    Explicit cache environment variables are respected. Missing values are
    derived from `STATIC_HOME` at call time so static assets and ecosystem
    caches stay under the same machine-level root by default.
    """

    root = static_home()
    _set_default_path(ANYDATASET_HOME_ENV, root / "anydataset")
    _set_default_path(ANYTRAIN_HOME_ENV, root)
    _set_default_path(BPE_CACHE_DIR_ENV, root / "bpe")
    _set_default_path(HF_HOME_ENV, root / "huggingface")
    _set_default_path(HF_HUB_CACHE_ENV, hf_home() / "hub")
    _set_default_path(HF_DATASETS_CACHE_ENV, hf_home() / "datasets")
    _set_default_path(TORCH_HOME_ENV, root / "torch")
    _set_default_path(WHISPER_ROOT_ENV, root / "whisper")


def _derived_env_path(name: str, child: str) -> Path:
    return _env_path_or(name, static_home() / child)


def _env_path_or(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    if value is not None:
        if not value:
            raise ValueError(f"{name} must not be empty.")
        return Path(value).expanduser()
    return default


def _required_home(name: str) -> Path:
    value = os.environ.get(name)
    if not value:
        if value is None:
            return _default_home(name)
        raise ValueError(f"{name} must not be empty.")
    return Path(value).expanduser()


def _default_home(name: str) -> Path:
    default_location = location()
    value = default_location.default_home(name)
    warnings.warn(
        f"{name} is not set; using {default_location.value} default {value}.",
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
