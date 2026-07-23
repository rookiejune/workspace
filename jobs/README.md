# Workspace Jobs

`scripts/` contains the location-neutral Python entry points. WMT19 TTS prepare
work is split into three public entries: TTS waveform materialization
(`scripts/prepare_wmt19_tts.py`), codec-view materialization
(`scripts/prepare_wmt19_tts_codec_view.py`), and BPE training
(`scripts/prepare_wmt19_tts_bpe.py`). Jobs select subcommands where needed and
supply concrete launch defaults. `jobs/` contains
shell launchers grouped by location; wrappers may set machine paths, Python
interpreters, devices, and submission defaults before calling the shared
script.

Common environment:

| Variable | Meaning |
| --- | --- |
| `WORKSPACE_ROOT` | The `workspace/` project root. |
| `REPOS_ROOT` | Parent directory containing `workspace/`, `third_party`, and experiment repos. |
| `LOCATION` | Resolved machine profile; currently only `fudan` has workspace defaults. |
| `STATIC_HOME` | Resolved stable asset root. |
| `DYNAMIC_HOME` | Resolved training/debug output root. |
| `PYTHONPATH` | Prepends `workspace/src` and required `third_party` packages. |

Root `jobs/env.sh` only resolves the shared workspace environment. Location
wrappers source their own `jobs/<location>/env.sh`, which pins the
location and then sources the root env. Do not put location-specific launch
defaults in `scripts/`.

Implemented Fudan wrappers:

```bash
jobs/fudan/prepare_wmt19_tts.sh
jobs/fudan/prepare_wmt19_tts_chunks_500k.sh
jobs/fudan/prepare_wmt19_tts_longcat.sh
jobs/fudan/prepare_wmt19_tts_dac.sh
jobs/fudan/prepare_wmt19_tts_stable.sh
jobs/fudan/prepare_wmt19_tts_unicodec.sh
jobs/fudan/filter_wmt19_tts_speech.sh
jobs/fudan/filter_wmt19_tts_translation.sh
jobs/fudan/filter_wmt19_tts_speech_translation.sh
jobs/fudan/prepare_wmt19_tts_longcat_bpe.sh
jobs/fudan/speech_to_speech_env.sh
```

`jobs/fudan/speech_to_speech_env.sh` contains the Fudan machine defaults shared
by `speech-to-speech/jobs/011/*.sh` and `speech-to-speech/jobs/013/*.sh`: Python selection, Hugging Face cache roots,
Qwen checkpoint discovery, and optional dataset-root override helpers. Keep
these physical launch defaults in workspace instead of the experiment repo.

`jobs/hz/` and `jobs/us/` are placeholders. Add wrappers
there only when the matching `src/zhuyin/_locations/<location>.py` profile and
any required loader source/tests are implemented in the same change.

Target CLI path contract:

| Option | Meaning |
| --- | --- |
| `--root` | WMT19 TTS dataset root containing `base/`, `longcat`, and other views. |
| `--bpe-root` | BPE artifact root; defaults to `$BPE_CACHE_DIR`, then `$STATIC_HOME/bpe`. |
| `--split` | Logical dataset split, defaulting to `train`. |

All WMT19 prepare and filter scripts use `--root` with the same meaning. The
BPE script also uses `--root` for its input dataset and keeps output on the
separate `--bpe-root` axis.

The Stable Codec Fudan wrapper selects the default posthoc preset
`1x46656_400bps`; its output is `stable-1x46656_400bps/`, never the legacy
native-FSQ `stable/` store. A later `--posthoc-bottleneck` argument in `"$@"`
selects another supported preset and therefore another preset-named sibling
store.

Debug BPE runs can keep their own artifact names by limiting samples:

```bash
jobs/fudan/prepare_wmt19_tts_longcat_bpe.sh --sample-limit 1000
```
