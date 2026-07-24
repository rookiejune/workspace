from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from anydataset.types import AudioView, Modality, Role, TextMeta, TextView

from zhuyin.datasets import _wmt19_tts_stable as stable

SCRIPTS_DIR = Path(__file__).parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import _wmt19_tts_codec as service  # noqa: E402
import _wmt19_tts_store as store  # noqa: E402


def test_prepare_longcat_keeps_source_and_target_text(
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
    monkeypatch.setattr(service, "store_sample_count", lambda _path: 5)
    monkeypatch.setattr(store, "wmt19_tts", lambda **kwargs: kwargs)

    stage = service.prepare_longcat(
        root=tmp_path,
        split="dev",
        devices="auto",
        max_shard_samples=100,
        batch_size=3,
        num_workers=2,
        read_prefetch=4,
        write_workers=1,
        write_prefetch=2,
    )

    assert calls["output"] == tmp_path / "longcat"
    assert set(calls["init"]["keep_schema"]) == {
        (Role.SOURCE, Modality.TEXT),
        (Role.TARGET, Modality.TEXT),
    }
    for requirement in calls["init"]["keep_schema"].values():
        assert requirement.views == frozenset({TextView.TEXT})
        assert requirement.meta == frozenset({TextMeta.LANG})
    assert calls["write"]["dataset_factory"]() == {
        "root": tmp_path,
        "split": "dev",
    }
    assert isinstance(calls["write"]["provider_factory"], service.LongCatFactory)
    assert calls["write"]["devices"] == "auto"
    assert stage["path"] == str(tmp_path / "longcat")
    assert stage["sample_count"] == 5


def test_longcat_factory_preserves_auto_device_selection(monkeypatch) -> None:
    from anytrain.codec import longcat

    calls: list[dict[str, Any]] = []

    class FakeLongCat:
        @classmethod
        def from_pretrained(cls, **kwargs: Any) -> object:
            calls.append(kwargs)
            return object()

    class FakeProvider:
        def __init__(self, codec: object, output: AudioView) -> None:
            self.codec = codec
            self.output = output

    monkeypatch.setattr(longcat, "LongCat", FakeLongCat)
    monkeypatch.setattr(service, "CodecProvider", FakeProvider)

    provider = service.LongCatFactory()("cpu")

    assert isinstance(provider, FakeProvider)
    assert provider.output is AudioView.LONGCAT
    assert calls == [{"device": None}]


def test_prepare_dac_keeps_source_and_target_text(
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
    monkeypatch.setattr(service, "store_sample_count", lambda _path: 6)
    monkeypatch.setattr(store, "wmt19_tts", lambda **kwargs: kwargs)

    stage = service.prepare_dac(
        root=tmp_path,
        split="train",
        devices="cuda:0",
        max_shard_samples=100,
        batch_size=4,
        num_workers=0,
        read_prefetch=None,
        write_workers=1,
        write_prefetch=None,
        cache_dir=tmp_path / "cache",
        model_type="44khz",
        model_bitrate="8kbps",
        tag="latest",
        n_quantizers=4,
        local_files_only=True,
    )

    assert calls["output"] == tmp_path / "dac"
    assert set(calls["init"]["keep_schema"]) == {
        (Role.SOURCE, Modality.TEXT),
        (Role.TARGET, Modality.TEXT),
    }
    factory = calls["write"]["provider_factory"]
    assert isinstance(factory, service.DACFactory)
    assert factory.cache_dir == tmp_path / "cache"
    assert factory.model_type == "44khz"
    assert factory.model_bitrate == "8kbps"
    assert factory.n_quantizers == 4
    assert calls["write"]["devices"] == "cuda:0"
    assert stage["path"] == str(tmp_path / "dac")
    assert stage["sample_count"] == 6


def test_dac_factory_forwards_pretrained_configuration(monkeypatch, tmp_path: Path) -> None:
    from anytrain.codec import dac

    calls: list[dict[str, Any]] = []

    class FakeDAC:
        @classmethod
        def from_pretrained(cls, **kwargs: Any) -> object:
            calls.append(kwargs)
            return object()

    class FakeProvider:
        def __init__(self, codec: object, output: AudioView) -> None:
            self.codec = codec
            self.output = output

    monkeypatch.setattr(dac, "DAC", FakeDAC)
    monkeypatch.setattr(service, "CodecProvider", FakeProvider)
    factory = service.DACFactory(
        cache_dir=tmp_path,
        model_type="24khz",
        model_bitrate="8kbps",
        tag="latest",
        n_quantizers=4,
        local_files_only=True,
    )

    provider = factory("cuda:1")

    assert isinstance(provider, FakeProvider)
    assert provider.output is AudioView.DAC
    assert calls == [
        {
            "cache_dir": tmp_path,
            "model_type": "24khz",
            "model_bitrate": "8kbps",
            "tag": "latest",
            "device": "cuda:1",
            "n_quantizers": 4,
            "local_files_only": True,
        }
    ]


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
    monkeypatch.setattr(
        service,
        "is_ready_store",
        lambda path: path == tmp_path / "stable",
    )
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
        posthoc_bottleneck=stable.DEFAULT_STABLE_QUANTIZER,
        normalize=True,
    )

    assert calls["output"] == tmp_path / "stable-1x46656_400bps"
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
    assert stage["path"] == str(tmp_path / "stable-1x46656_400bps")
    assert stage["sample_count"] == 7


def test_stable_factory_identity_and_pretrained_configuration(monkeypatch) -> None:
    from anytrain.codec import stable_codec

    calls: list[dict[str, Any]] = []

    class FakeStableCodec:
        @classmethod
        def from_pretrained(cls, **kwargs: Any) -> object:
            calls.append(kwargs)
            return object()

    class FakeProvider:
        def __init__(self, codec: object, output: AudioView) -> None:
            self.codec = codec
            self.output = output

    monkeypatch.setattr(stable_codec, "StableCodec", FakeStableCodec)
    monkeypatch.setattr(service, "CodecProvider", FakeProvider)
    factory = service.StableCodecFactory(
        version="speech-16k-base",
        pretrained_model="local/stable",
        posthoc_bottleneck=stable.StableQuantizer.FSQ_2X15625_700BPS,
        normalize=False,
    )

    provider = factory("cuda:1")

    assert isinstance(provider, FakeProvider)
    assert provider.output is AudioView.STABLE
    assert calls == [
        {
            "version": "speech-16k-base",
            "pretrained_model": "local/stable",
            "device": "cuda:1",
            "posthoc_bottleneck": "2x15625_700bps",
            "normalize": False,
        }
    ]
    assert repr(factory) == (
        "StableCodecFactory(version='speech-16k-base', "
        "pretrained_model='local/stable', "
        "posthoc_bottleneck='2x15625_700bps', normalize=False)"
    )


def test_stable_store_identity_tracks_quantizer() -> None:
    assert stable.store_dir() == "stable-1x46656_400bps"
    assert (
        stable.store_dir(stable.StableQuantizer.FSQ_4X729_1000BPS)
        == "stable-4x729_1000bps"
    )


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
