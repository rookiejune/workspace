"""Materialize WMT19 TTS codec views from the prepared waveform store.

This service owns codec construction, the text fields retained in codec stores,
and the stable output layout. CLI scripts only resolve paths and forward runtime
settings.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path

from anydataset.provider.codec import CodecProvider
from anydataset.store import ViewMaterializer
from anydataset.types import (
    AudioView,
    Modality,
    Role,
    TextMeta,
    TextReq,
    TextView,
)

from zhuyin.datasets._wmt19_tts_io import (
    Stage,
    is_ready_store,
    ready_stage,
    stage,
    store_sample_count,
)
from zhuyin.datasets._wmt19_tts_store import StoreFactory

STABLE_STORE_DIR = "stable"
UNICODEC_STORE_DIR = "unicodec"


class StableCodecFactory:
    def __init__(
        self,
        *,
        version: str,
        pretrained_model: str | None,
        posthoc_bottleneck: str | None,
        normalize: bool,
    ) -> None:
        self.version = version
        self.pretrained_model = pretrained_model
        self.posthoc_bottleneck = posthoc_bottleneck
        self.normalize = normalize

    def __call__(self, device: str) -> CodecProvider:
        from anytrain.codec.stable_codec import StableCodec

        codec = StableCodec.from_pretrained(
            version=self.version,
            pretrained_model=self.pretrained_model,
            device=device,
            posthoc_bottleneck=self.posthoc_bottleneck,
            normalize=self.normalize,
        )
        return CodecProvider(codec, AudioView.STABLE)


class UniCodecFactory:
    def __init__(
        self,
        *,
        cache_dir: Path | None,
        domain: str,
        bandwidth_id: int,
        local_files_only: bool,
    ) -> None:
        self.cache_dir = cache_dir
        self.domain = domain
        self.bandwidth_id = bandwidth_id
        self.local_files_only = local_files_only

    def __call__(self, device: str) -> CodecProvider:
        from anytrain.codec.unicodec import UniCodec

        codec = UniCodec.from_pretrained(
            cache_dir=self.cache_dir,
            device=device,
            domain=self.domain,
            bandwidth_id=self.bandwidth_id,
            local_files_only=self.local_files_only,
        )
        return CodecProvider(codec, AudioView.UNICODEC)


def prepare_stable_codec(
    *,
    root: Path,
    split: str,
    devices: str,
    max_shard_samples: int,
    batch_size: int,
    num_workers: int,
    read_prefetch: int | None,
    write_workers: int,
    write_prefetch: int | None,
    version: str,
    pretrained_model: str | None,
    posthoc_bottleneck: str | None,
    normalize: bool,
) -> Stage:
    return _prepare_codec(
        root=root,
        split=split,
        store_dir=STABLE_STORE_DIR,
        provider_factory=StableCodecFactory(
            version=version,
            pretrained_model=pretrained_model,
            posthoc_bottleneck=posthoc_bottleneck,
            normalize=normalize,
        ),
        devices=devices,
        max_shard_samples=max_shard_samples,
        batch_size=batch_size,
        num_workers=num_workers,
        read_prefetch=read_prefetch,
        write_workers=write_workers,
        write_prefetch=write_prefetch,
    )


def prepare_unicodec(
    *,
    root: Path,
    split: str,
    devices: str,
    max_shard_samples: int,
    batch_size: int,
    num_workers: int,
    read_prefetch: int | None,
    write_workers: int,
    write_prefetch: int | None,
    cache_dir: Path | None,
    domain: str,
    bandwidth_id: int,
    local_files_only: bool,
) -> Stage:
    return _prepare_codec(
        root=root,
        split=split,
        store_dir=UNICODEC_STORE_DIR,
        provider_factory=UniCodecFactory(
            cache_dir=cache_dir,
            domain=domain,
            bandwidth_id=bandwidth_id,
            local_files_only=local_files_only,
        ),
        devices=devices,
        max_shard_samples=max_shard_samples,
        batch_size=batch_size,
        num_workers=num_workers,
        read_prefetch=read_prefetch,
        write_workers=write_workers,
        write_prefetch=write_prefetch,
    )


def _prepare_codec(
    *,
    root: Path,
    split: str,
    store_dir: str,
    provider_factory: Callable[[str], CodecProvider],
    devices: str,
    max_shard_samples: int,
    batch_size: int,
    num_workers: int,
    read_prefetch: int | None,
    write_workers: int,
    write_prefetch: int | None,
) -> Stage:
    output = root / store_dir
    if is_ready_store(output):
        return ready_stage(store_dir, output)

    started_at = time.perf_counter()
    ViewMaterializer(
        output,
        split=split,
        max_shard_samples=max_shard_samples,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=read_prefetch,
        write_workers=write_workers,
        write_prefetch=write_prefetch,
        keep_schema=_text_schema(),
    ).write(
        dataset_factory=StoreFactory(root, split),
        provider_factory=provider_factory,
        devices=devices,
    )
    return stage(
        store_dir,
        output,
        seconds=time.perf_counter() - started_at,
        sample_count=store_sample_count(output),
    )


def _text_schema():
    requirement = TextReq(
        views=frozenset({TextView.TEXT}),
        meta=frozenset({TextMeta.LANG}),
    )
    return {
        (Role.SOURCE, Modality.TEXT): requirement,
        (Role.TARGET, Modality.TEXT): requirement,
    }


__all__ = [
    "STABLE_STORE_DIR",
    "UNICODEC_STORE_DIR",
    "StableCodecFactory",
    "UniCodecFactory",
    "prepare_stable_codec",
    "prepare_unicodec",
]
