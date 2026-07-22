"""Run WMT19 TTS filtering tasks through one public script entry."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from typing import Any

Entry = Callable[[Sequence[str]], None]


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    args.entry(args.args)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter the prepared WMT19 TTS store.")
    subparsers = parser.add_subparsers(dest="task", required=True)
    task_parser(
        subparsers,
        "speech",
        "apply speech-quality filtering",
        run_speech,
    )
    task_parser(
        subparsers,
        "translation",
        "apply translation-quality filtering",
        run_translation,
    )
    task_parser(
        subparsers,
        "speech-translation",
        "chain translation and speech filtering",
        run_speech_translation,
    )
    args, rest = parser.parse_known_args(argv)
    args.args = rest
    return args


def run_speech(args: Sequence[str]) -> None:
    if __package__:
        from . import _filter_wmt19_tts_speech as speech_filter
    else:
        import _filter_wmt19_tts_speech as speech_filter

    speech_filter.main(args)


def run_translation(args: Sequence[str]) -> None:
    if __package__:
        from . import _filter_wmt19_tts_translation as translation_filter
    else:
        import _filter_wmt19_tts_translation as translation_filter

    translation_filter.main(args)


def run_speech_translation(args: Sequence[str]) -> None:
    if __package__:
        from . import _filter_wmt19_tts_speech_translation as speech_translation_filter
    else:
        import _filter_wmt19_tts_speech_translation as speech_translation_filter

    speech_translation_filter.main(args)


def task_parser(
    subparsers: Any,
    name: str,
    help_text: str,
    entry: Entry,
) -> None:
    parser = subparsers.add_parser(name, help=help_text, add_help=False)
    parser.set_defaults(entry=entry)


if __name__ == "__main__":
    main()
