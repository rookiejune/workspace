# Workspace Jobs

`env.sh` only locates this repository and fills `PYTHONPATH` when needed. Python
code owns `LOCATION`, `STATIC_HOME`, `DYNAMIC_HOME` and derived cache variables
through `zhuyin.env`, so job submission should first enter the intended Python
environment and then run the wrapper directly.

Common variables:

| Variable | Meaning |
| --- | --- |
| `WORKSPACE_ROOT` | The `workspace/` project root. |
| `REPOS_ROOT` | Parent directory containing `workspace/`, `third_party/`, and experiment repos. |
| `PYTHONPATH` | Adds `workspace/src` and required `third_party` packages when unset. |

Machine presets such as `LOCATION=us`, `LOCATION=hz` and `LOCATION=fudan` are
resolved by `zhuyin.env` inside Python. If `LOCATION` is unset, Python detects
the first available marker in this order: `/share5_video`, `/nfs/yin.zhu`,
`/mnt`. Explicit `STATIC_HOME` or `DYNAMIC_HOME` still override those defaults.

Common wrappers:

```bash
jobs/prepare_wmt19_tts.sh
jobs/prepare_wmt19_tts_longcat.sh
jobs/filter_wmt19_tts_speech.sh
jobs/filter_wmt19_tts_translation.sh
jobs/filter_wmt19_tts_speech_translation.sh
jobs/prepare_wmt19_tts_longcat_bpe.sh
```

The LongCat BPE wrapper writes to `$BPE_CACHE_DIR`, which defaults to
`$STATIC_HOME/bpe` inside Python. The default BPE vocab size is 100k; `codes_8192` in the
artifact name records the LongCat semantic codebook size, not the target BPE
vocab size. Debug runs can keep their own artifact names by limiting the number
of samples:

```bash
jobs/prepare_wmt19_tts_longcat_bpe.sh --sample-limit 1000
```
