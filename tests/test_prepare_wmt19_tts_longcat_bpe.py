from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import pytest
from anydataset import AudioItem, AudioView, Modality, Role, Sample, Source, Spec

from scripts import prepare_wmt19_tts_longcat_bpe as script


class Codes:
    def __init__(self, values: Sequence[int]) -> None:
        self.values = list(values)

    def reshape(self, *shape: int) -> Codes:
        assert shape == (-1,)
        return self

    def detach(self) -> Codes:
        return self

    def cpu(self) -> Codes:
        return self

    def tolist(self) -> list[int]:
        return list(self.values)


class FakeDataset:
    def __init__(self, samples: Sequence[Sample]) -> None:
        self.samples = list(samples)
        self.spec = Spec(source=Source.STORE, path="/data/wmt19/longcat", split="train")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Sample:
        return self.samples[index]


def test_longcat_corpus_yields_source_and_target_sequences() -> None:
    dataset = FakeDataset([sample(source=[1, 2, 3], target=[4, 5])])

    corpus = list(script.corpus(dataset))

    assert corpus == [
        [[1], [2], [3]],
        [[4], [5]],
    ]


def test_run_writes_bpe_artifact_and_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = FakeDataset([sample(source=[1, 2, 1, 2], target=[1, 2])])
    calls: list[dict[str, object]] = []

    def fake_wmt19_tts_longcat(**kwargs: object) -> FakeDataset:
        calls.append(kwargs)
        return dataset

    monkeypatch.setattr(script, "wmt19_tts_longcat", fake_wmt19_tts_longcat)

    summary = script.run(
        argparse.Namespace(
            dataset_dir=None,
            split="train",
            cache_dir=tmp_path / "bpe",
            codec_name="longcat",
            vocab_size=100_000,
            min_frequency=0,
            max_token_length=None,
            codebook_sizes=[8192],
            sample_limit=None,
            overwrite=False,
            show_progress=False,
        )
    )

    artifact_dir = tmp_path / "bpe" / "longcat" / "vocab_100k_minfreq_0_maxlen_none_codes_8192"
    meta = json.loads((artifact_dir / script.META_FILE).read_text(encoding="utf-8"))

    assert calls == [{"split": "train"}]
    assert (artifact_dir / script.STATE_FILE).exists()
    assert (artifact_dir / script.TOKENIZER_FILE).exists()
    assert (artifact_dir / script.EVAL_FILE).exists()
    assert meta["datasets"] == [dataset.spec.to_dict()]
    assert meta["min_frequency"] == 0
    assert meta["max_token_length"] is None
    assert summary["actual_vocab_size"] >= 2
    assert summary["artifact_dir"] == str(artifact_dir)
    assert "codec_bpe" not in summary
    assert "tokenizer" not in summary
    assert "meta" not in summary
    assert "reused" not in summary
    assert summary["eval"]["encoded_tokens"] == 2
    assert summary["eval"]["num_sequences"] == 2
    assert summary["eval"]["original_tokens"] == 6
    assert summary["eval"]["compression_factor"] == pytest.approx(3.0)
    assert summary["eval"]["compression_gain"] == pytest.approx(2 / 3)
    assert summary["eval"]["compression_ratio"] == pytest.approx(1 / 3)
    assert summary["eval"]["mean_encoded_length"] == pytest.approx(1.0)
    assert summary["eval"]["mean_original_length"] == pytest.approx(3.0)
    assert json.loads((artifact_dir / script.EVAL_FILE).read_text(encoding="utf-8")) == summary


def test_run_evaluates_existing_bpe_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = FakeDataset([sample(source=[1, 2, 1, 2], target=[1, 2])])
    monkeypatch.setattr(script, "wmt19_tts_longcat", lambda **_: dataset)
    args = argparse.Namespace(
        dataset_dir=None,
        split="train",
        cache_dir=tmp_path / "bpe",
        codec_name="longcat",
        vocab_size=100_000,
        min_frequency=0,
        max_token_length=None,
        codebook_sizes=[8192],
        sample_limit=None,
        overwrite=False,
        show_progress=False,
    )

    first = script.run(args)
    second = script.run(args)

    assert second == first
    assert "reused" not in second


def sample(*, source: Sequence[int], target: Sequence[int]) -> Sample:
    return {
        (Role.SOURCE, Modality.AUDIO): AudioItem(
            views={AudioView.LONGCAT: {"semantic_codes": Codes(source)}}
        ),
        (Role.TARGET, Modality.AUDIO): AudioItem(
            views={AudioView.LONGCAT: {"semantic_codes": Codes(target)}}
        ),
    }
