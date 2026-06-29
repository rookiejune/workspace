# Workspace Jobs

`env.sh` owns shared machine and cache variables for experiments in this
workspace. Project-level job wrappers should source it, then add only their own
`src` path, output root, experiment name, and Hydra overrides.

Common variables:

| Variable | Meaning |
| --- | --- |
| `WORKSPACE_ROOT` | The `workspace/` project root. |
| `REPOS_ROOT` | Parent directory containing `workspace/`, `third_party/`, and experiment repos. |
| `LOCATION` | Machine preset used for default homes. Supports `fudan` and `hz`; defaults to `fudan`. |
| `STATIC_HOME` | Shared stable assets root. Defaults to the selected `LOCATION` preset. |
| `DYNAMIC_HOME` | Shared dynamic output root. Defaults to the selected `LOCATION` preset. |
| `ANYDATASET_HOME` | Anydataset cache and logs. Defaults to `$STATIC_HOME/anydataset`. |
| `ANYTRAIN_HOME` | Anytrain root. Defaults to `$STATIC_HOME`. |
| `BPE_CACHE_DIR` | BPE/tokenizer artifacts. Defaults to `$STATIC_HOME/bpe`. |
| `HF_HOME` | Hugging Face cache root. Defaults to `$STATIC_HOME/huggingface`. |
| `HF_HUB_CACHE` | Hugging Face Hub cache. Defaults to `$HF_HOME/hub`. |
| `HF_DATASETS_CACHE` | Hugging Face datasets cache. Defaults to `$HF_HOME/datasets`. |
| `TORCH_HOME` | Torch cache. Defaults to `$STATIC_HOME/torch`. |
| `ANYTRAIN_WHISPER_ROOT` | Whisper cache/root used by evaluators. Defaults to `$STATIC_HOME/whisper`. |
| `HF_ENDPOINT` | Hugging Face endpoint. Defaults to `https://hf-mirror.com`. |
| `WORKSPACE_PYTHON` | Shared Python executable. Remote `py312` is preferred; local `torch2.12` is the fallback. |
| `PYTHONPATH` | Adds `workspace/src` and required `third_party` packages. |

Location presets:

| Location | `STATIC_HOME` | `DYNAMIC_HOME` |
| --- | --- | --- |
| `fudan` | `/mnt/pami202/zhuyin` | `/mnt/pami202/zhuyin/dynamic` |
| `hz` | `/nfs/yin.zhu` | `/yin.zhu` |

Common wrappers:

```bash
jobs/prepare_wmt19_tts.sh
jobs/prepare_wmt19_tts_longcat.sh
jobs/filter_wmt19_tts.sh
jobs/prepare_wmt19_tts_longcat_bpe.sh
```

The LongCat BPE wrapper writes to `$BPE_CACHE_DIR`, which defaults to
`$STATIC_HOME/bpe`. The default BPE vocab size is 100k; `codes_8192` in the
artifact name records the LongCat semantic codebook size, not the target BPE
vocab size. Debug runs can keep their own artifact names by limiting the number
of samples:

```bash
jobs/prepare_wmt19_tts_longcat_bpe.sh --sample-limit 1000
```
