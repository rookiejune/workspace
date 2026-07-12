from __future__ import annotations

from pathlib import Path
from typing import Any

from anydataset.types import AudioView, Modality, Role, TextMeta, TextView

from zhuyin.datasets import _wmt19_tts_codec as service
from zhuyin.datasets import _wmt19_tts_store as store


def test_prepare_stable_codec_keeps_source_and_target_text(
    tmp_path: Path,
    monkeypatch,
) -> None:
    calls: dict[str, Any] = {}

    class FakeMaterializer:
        def __init__(self, output: Path, **kwargs: Any) -> None:
            calls["output"] = output
            calls["init"] = kwargs

        def write(self, **kwargs: Any) -> None:
            calls["write"] = kwargs

    monkeypatch.setattr(service, "ViewMaterializer", FakeMaterializer)
    monkeypatch.setattr(service, "is_ready_store", lambda _path: False)
    monkeypatch.setattr(service, "store_sample_count", lambda _path: 7)
    monkeypatch.setattr(store, "wmt19_tts", lambda **kwargs: kwargs)

    stage = service.prepare_stable_codec(
        root=tmp_path,
        split="dev",
        devices="cuda:0",
        max_shard_samples=100,
        batch_size=3,
        num_workers=2,
        read_prefetch=4,
        write_workers=1,
        write_prefetch=2,
        version="speech-16k",
        pretrained_model=None,
        posthoc_bottleneck=None,
        normalize=True,
    )

    assert calls["output"] == tmp_path / "stable"
    schema = calls["init"]["keep_schema"]
    assert set(schema) == {
        (Role.SOURCE, Modality.TEXT),
        (Role.TARGET, Modality.TEXT),
    }
    for requirement in schema.values():
        assert requirement.views == frozenset({TextView.TEXT})
        assert requirement.meta == frozenset({TextMeta.LANG})
    assert calls["write"]["dataset_factory"]() == {
        "root": tmp_path,
        "split": "dev",
    }
    assert isinstance(calls["write"]["provider_factory"], service.StableCodecFactory)
    assert calls["write"]["devices"] == "cuda:0"
    assert stage["path"] == str(tmp_path / "stable")
    assert stage["sample_count"] == 7


def test_prepare_unicodec_keeps_source_and_target_text(
    tmp_path: Path,
    monkeypatch,
) -> None:
    calls: dict[str, Any] = {}

    class FakeMaterializer:
        def __init__(self, output: Path, **kwargs: Any) -> None:
            calls["output"] = output
            calls["init"] = kwargs

        def write(self, **kwargs: Any) -> None:
            calls["write"] = kwargs

    monkeypatch.setattr(service, "ViewMaterializer", FakeMaterializer)
    monkeypatch.setattr(service, "is_ready_store", lambda _path: False)
    monkeypatch.setattr(service, "store_sample_count", lambda _path: 9)
    monkeypatch.setattr(store, "wmt19_tts", lambda **kwargs: kwargs)

    stage = service.prepare_unicodec(
        root=tmp_path,
        split="train",
        devices="cuda:1",
        max_shard_samples=100,
        batch_size=4,
        num_workers=0,
        read_prefetch=None,
        write_workers=1,
        write_prefetch=None,
        cache_dir=tmp_path / "cache",
        domain="0",
        bandwidth_id=0,
        local_files_only=True,
    )

    assert calls["output"] == tmp_path / "unicodec"
    assert set(calls["init"]["keep_schema"]) == {
        (Role.SOURCE, Modality.TEXT),
        (Role.TARGET, Modality.TEXT),
    }
    factory = calls["write"]["provider_factory"]
    assert isinstance(factory, service.UniCodecFactory)
    assert factory.cache_dir == tmp_path / "cache"
    assert calls["write"]["devices"] == "cuda:1"
    assert AudioView.UNICODEC.value == "unicodec"
    assert stage["path"] == str(tmp_path / "unicodec")
    assert stage["sample_count"] == 9
