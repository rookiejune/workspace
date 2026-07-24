"""Train a LongCat semantic BPE tokenizer from the prepared WMT19 TTS store.

The script reads the LongCat `wmt19_tts_codec()` view, extracts codebook 0 from
source and target, trains `anytrain.tokenizer.CodecBPE`, and saves the artifact
under the static workspace BPE cache.
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections.abc import Callable, Iterable, Sequence
from functools import partial
from itertools import islice
from pathlib import Path
from typing import TYPE_CHECKING, cast

from anydataset.types import AudioItem, AudioView, Modality, Role, Sample

from zhuyin.datasets.wmt19_tts import Codec, wmt19_tts_codec
from zhuyin.env import context
from zhuyin.tokenizers._codec_bpe_artifact import (
    DEFAULT_CODEBOOK_SIZES,
    DEFAULT_CODEC_NAME,
    DEFAULT_MAX_TOKEN_LENGTH,
    DEFAULT_MIN_FREQUENCY,
    DEFAULT_VOCAB_SIZE,
    artifact_path,
)
from zhuyin.tokenizers.codec_bpe import codec_bpe_path

if TYPE_CHECKING:
    from anytrain.tokenizer import CodecBPE

STATE_FILE = "codec_bpe.json"
META_FILE = "meta.json"
EVAL_FILE = "eval.json"


def main(argv: Sequence[str] | None = None) -> None:
    """Run the command line tokenizer preparation entry."""

    args = parse_args(argv)
    with context():
        summary = run(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def run(args: argparse.Namespace) -> dict[str, object]:
    """Train or reuse the configured WMT19 TTS LongCat BPE artifact."""

    codebook_sizes = tuple(args.codebook_sizes)
    artifact = artifact_path(
        codec_name=args.codec_name,
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        max_token_length=args.max_token_length,
        codebook_sizes=codebook_sizes,
        sample_limit=args.sample_limit,
    )
    artifact_dir = codec_bpe_path(root=args.bpe_root, artifact=artifact)
    if (
        artifact_dir.exists()
        and not args.overwrite
        and not (artifact_dir / STATE_FILE).exists()
    ):
        raise FileExistsError(
            f"BPE artifact directory exists without {STATE_FILE}: {artifact_dir}"
        )
    if artifact_dir.exists() and args.overwrite:
        shutil.rmtree(artifact_dir)

    raw_dataset = wmt19_tts_codec(
        codec=Codec.LONGCAT,
        root=args.root,
        split=args.split,
    )
    dataset = cast(Sequence[Sample], raw_dataset)
    from anytrain.tokenizer import CodecBPE

    if (artifact_dir / STATE_FILE).exists():
        bpe = CodecBPE.from_pretrained(artifact_dir)
    else:
        bpe = CodecBPE.train(
            corpus_factory(dataset, sample_limit=args.sample_limit),
            codebook_sizes=codebook_sizes,
            vocab_size=args.vocab_size,
            min_frequency=args.min_frequency,
            max_token_length=args.max_token_length,
            show_progress=args.show_progress,
        )
        artifact_dir.mkdir(parents=True, exist_ok=False)
        bpe.save_pretrained(artifact_dir)
        write_meta(artifact_dir, args, raw_dataset.spec.to_dict(), codebook_sizes)
    return summarize(artifact_dir, bpe, dataset, sample_limit=args.sample_limit)


def corpus_factory(
    dataset: Sequence[Sample],
    *,
    sample_limit: int | None = None,
) -> Callable[[], Iterable[list[list[int]]]]:
    """Return a replayable corpus factory for multi-pass BPE training."""

    return partial(corpus, dataset, sample_limit=sample_limit)


def corpus(
    dataset: Sequence[Sample],
    *,
    sample_limit: int | None = None,
) -> Iterable[list[list[int]]]:
    """Yield source and target semantic ids as single-codebook BPE frames."""

    for sample in islice((dataset[index] for index in range(len(dataset))), sample_limit):
        for role in (Role.SOURCE, Role.TARGET):
            frames = frames_for(sample, role)
            if frames:
                yield frames


def frames_for(sample: Sample, role: Role) -> list[list[int]]:
    """Return LongCat semantic ids for one role as BPE frames."""

    audio = cast(AudioItem, sample[role, Modality.AUDIO])
    view = audio.views[AudioView.LONGCAT]
    values = view[:, 0].reshape(-1).detach().cpu().tolist()
    return [[int(value)] for value in values]


def write_meta(
    artifact_dir: Path,
    args: argparse.Namespace,
    dataset_meta: dict[str, object],
    codebook_sizes: tuple[int, ...],
) -> None:
    """Write the tokenizer artifact metadata."""

    meta: dict[str, object] = {
        "codec_name": args.codec_name,
        "vocab_size": args.vocab_size,
        "min_frequency": args.min_frequency,
        "max_token_length": args.max_token_length,
        "codebook_sizes": list(codebook_sizes),
        "datasets": [dataset_meta],
    }
    if args.sample_limit is not None:
        meta["sample_limit"] = args.sample_limit
    (artifact_dir / META_FILE).write_text(
        json.dumps(meta, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def summarize(
    artifact_dir: Path,
    bpe: CodecBPE,
    dataset: Sequence[Sample],
    *,
    sample_limit: int | None,
) -> dict[str, object]:
    """Evaluate a BPE artifact and return the command output payload."""

    payload = _json_payload(
        {
            "actual_vocab_size": bpe.vocab_size,
            "artifact_dir": str(artifact_dir),
            "eval": bpe.evaluate(corpus(dataset, sample_limit=sample_limit)),
        }
    )
    (artifact_dir / EVAL_FILE).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def _json_payload(payload: dict[str, object]) -> dict[str, object]:
    return cast(
        dict[str, object],
        json.loads(json.dumps(payload, ensure_ascii=False, sort_keys=True)),
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments."""

    parser = argparse.ArgumentParser(description="Train LongCat semantic BPE for WMT19 TTS.")
    parser.add_argument("--root", type=Path)
    parser.add_argument("--split", default="train")
    parser.add_argument("--bpe-root", type=Path)
    parser.add_argument("--codec-name", default=DEFAULT_CODEC_NAME)
    parser.add_argument("--vocab-size", type=int, default=DEFAULT_VOCAB_SIZE)
    parser.add_argument("--min-frequency", type=int, default=DEFAULT_MIN_FREQUENCY)
    parser.add_argument("--max-token-length", type=int, default=DEFAULT_MAX_TOKEN_LENGTH)
    parser.add_argument(
        "--codebook-sizes",
        type=int,
        nargs="+",
        default=list(DEFAULT_CODEBOOK_SIZES),
    )
    parser.add_argument("--sample-limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--show-progress", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    main()
