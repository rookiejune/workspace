"""Run WMT19 TTS codec-view preparation tasks through one public script entry."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from typing import Any

Entry = Callable[[Sequence[str]], None]


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    args.entry(args.args)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare WMT19 TTS codec views."
    )
    subparsers = parser.add_subparsers(dest="codec", required=True)
    codec_parser(
        subparsers,
        "longcat",
        "materialize the LongCat codec view",
        run_longcat,
    )
    codec_parser(
        subparsers,
        "dac",
        "materialize the DAC codec view",
        run_dac,
    )
    codec_parser(
        subparsers,
        "stable",
        "materialize the Stable Codec view",
        run_stable,
    )
    codec_parser(
        subparsers,
        "unicodec",
        "materialize the UniCodec view",
        run_unicodec,
    )
    args, rest = parser.parse_known_args(argv)
    args.args = rest
    return args


def run_longcat(args: Sequence[str]) -> None:
    if __package__:
        from . import _prepare_wmt19_tts_longcat as longcat_prepare
    else:
        import _prepare_wmt19_tts_longcat as longcat_prepare

    longcat_prepare.main(args)


def run_dac(args: Sequence[str]) -> None:
    run_codec("dac", args)


def run_stable(args: Sequence[str]) -> None:
    run_codec("stable", args)


def run_unicodec(args: Sequence[str]) -> None:
    run_codec("unicodec", args)


def run_codec(codec: str, args: Sequence[str]) -> None:
    if __package__:
        from . import _prepare_wmt19_tts_codec as codec_prepare
    else:
        import _prepare_wmt19_tts_codec as codec_prepare

    codec_prepare.main([codec, *args])


def codec_parser(
    subparsers: Any,
    name: str,
    help_text: str,
    entry: Entry,
) -> None:
    parser = subparsers.add_parser(name, help=help_text, add_help=False)
    parser.set_defaults(entry=entry)


if __name__ == "__main__":
    main()
