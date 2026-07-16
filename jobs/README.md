# Workspace Jobs

`env.sh` locates this repository and prepends the local `PYTHONPATH`.
Python path helpers resolve `LOCATION`, `STATIC_HOME` and `DYNAMIC_HOME` without
mutating the process environment. Scripts may enter `zhuyin.env.context()` when
third-party code requires those values as environment variables. Job submission
should first enter the intended Python environment and then run the wrapper directly.

Common variables:

| Variable | Meaning |
| --- | --- |
| `WORKSPACE_ROOT` | The `workspace/` project root. |
| `REPOS_ROOT` | Parent directory containing `workspace/`, `third_party/`, and experiment repos. |
| `PYTHONPATH` | Prepends `workspace/src` and required `third_party` packages. |

Machine presets such as `LOCATION=us`, `LOCATION=hz` and `LOCATION=fudan` are
resolved by `zhuyin.env` inside Python. `with zhuyin.env.context():` temporarily
sets those workspace variables and unsets values that were absent before entering
the block. If `LOCATION` is unset, Python detects the first available marker in
this order: `/share5_video`, `/nfs/yin.zhu`, `/mnt`. Explicit `STATIC_HOME` or
`DYNAMIC_HOME` still override those defaults.

Common wrappers:

```bash
jobs/prepare_wmt19_tts.sh
jobs/prepare_wmt19_tts_longcat.sh
jobs/prepare_wmt19_tts_dac.sh
jobs/prepare_wmt19_tts_stable.sh
jobs/prepare_wmt19_tts_unicodec.sh
jobs/filter_wmt19_tts_speech.sh
jobs/filter_wmt19_tts_translation.sh
jobs/filter_wmt19_tts_speech_translation.sh
jobs/prepare_wmt19_tts_longcat_bpe.sh
```

Target CLI path contract:

| Option | Meaning |
| --- | --- |
| `--root` | WMT19 TTS dataset root containing `base/`, `longcat/`, and other views. |
| `--bpe-root` | BPE artifact root; defaults to `$BPE_CACHE_DIR`, then `$STATIC_HOME/bpe`. |
| `--split` | Logical dataset split, defaulting to `train`. |

All WMT19 prepare and filter scripts use `--root` with the same meaning. The BPE
script also uses `--root` for its input dataset and keeps its output location on
the separate `--bpe-root` axis. Wrappers do not translate these options or add
compatibility aliases.

The Stable Codec wrapper explicitly selects the default posthoc preset
`1x46656_400bps`; its output is `stable-1x46656_400bps/`, never the legacy native-FSQ
`stable/` store. A later `--posthoc-bottleneck` argument in `"$@"` selects another
supported preset and therefore another preset-named sibling store.

The LongCat BPE wrapper writes to `--bpe-root` when it is explicit, otherwise to
`$BPE_CACHE_DIR`, which defaults to `$STATIC_HOME/bpe` inside Python. The default
BPE vocab size is 100k; `codes_8192` in the artifact name records the LongCat
semantic codebook size, not the target BPE vocab size. Debug runs can keep their
own artifact names by limiting the number of samples:

```bash
jobs/prepare_wmt19_tts_longcat_bpe.sh --sample-limit 1000
```
