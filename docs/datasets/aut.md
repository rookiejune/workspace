# Prepared AuT

`zhuyin.datasets.aut` 读取离线生成的 WMT19 TTS waveform 与
Qwen2.5-Omni audio-tower（AuT）target。它只提供 map-style 读取和严格验证，不在线加载
teacher，不重采样 waveform，也不补回未存储的文本、speaker 或原始文件路径。

## Public API

```python
from zhuyin.datasets.aut import (
    CHECKPOINT,
    DEFAULT_REVISION,
    FEATURE_DIM,
    SAMPLE_RATE,
    TRANSFORMERS_VERSION,
    Sample,
    prepared_aut,
)

dataset = prepared_aut(split="train")
sample = dataset[0]
```

固定契约为：

- dataset：WMT19 TTS；
- checkpoint：`Qwen/Qwen2.5-Omni-7B`；
- revision：`ae9e1690543ffd5c0221dc27f79834d0294cba00`；
- Transformers：`5.14.1`；
- target：audio tower 最终 `float32 [feature_frame, 3584]` features；
- audio：`float32 [1, time]`、16 kHz mono；
- timing：100 Hz Mel，长度向上取整，一层 stride-2 convolution 后再做 stride-2 pooling，
  frame time 取覆盖区间末端。

默认 store 路径为：

```text
$STATIC_HOME/datasets/prepared_aut/wmt19_tts/
  qwen2_5-omni-7b/<revision>/<split>/
```

显式 `root` 始终表示 `prepared_aut/wmt19_tts` 数据集根，而不是 revision 或 split 目录：

```python
dataset = prepared_aut(
    root="/data/prepared_aut/wmt19_tts",
    revision=DEFAULT_REVISION,
    split="dev",
)
```

## Sample Contract

`Sample` 只包含：

```text
sample_id             str
audio_sha256          str     64-char lowercase hex
waveform              float32 [1, time]
waveform_length       int64   scalar
sample_rate           int     16000
aut_features          float32 [feature_frame, 3584]
aut_feature_mask      bool    [feature_frame]
audio_placeholders    int64   scalar
```

单样本 artifact 不做 padding，因此 `aut_feature_mask` 必须全部为 true。有效 mask 数、
`audio_placeholders`、feature frame 数和 waveform 按 Qwen2.5 timing 算出的整数长度必须完全
相等。loader 不新增 `aut_feature_length`，避免与 mask 重复表达同一个事实。

## Store Contract

每个 split 是一次完成后不再原地修改的 store：

```text
<split>/
  .ready
  manifest.json
  samples.jsonl
  samples/
    000000.pt
    ...
```

`manifest.json` 的顶层键严格固定为 `schema_version`、`sample_count`、`dataset`、
`teacher`、`timing`、`audio`、`storage` 和 `fields`。revision 必须是 40 位小写十六进制
commit hash。各嵌套对象也拒绝缺失或额外键；完整
schema 以 `tests/test_aut.py::_manifest()` 的可执行 fixture 为准。关键固定值包括：

```json
{
  "schema_version": 1,
  "teacher": {
    "family": "qwen2_5_omni",
    "feature_name": "audio_tower_final_projected",
    "feature_dtype": "float32"
  },
  "timing": {
    "name": "qwen2_5_conv2_pool2_v1",
    "mel_rate_hz": 100,
    "frame_reference": "end",
    "mel_length_rounding": "ceil",
    "conv_stride": 2,
    "pool_stride": 2
  },
  "storage": {
    "format": "torch_weights_only_v1",
    "index": "samples.jsonl"
  }
}
```

`samples.jsonl` 每行只有 `sample_id`、`audio_sha256`、`path`。`path` 必须是直接位于
`samples/*.pt` 的 POSIX 相对路径；重复 ID、目录逃逸、缺失 payload 和 manifest/index
count 不一致都会在 dataset 构造时失败。

`.pt` payload 只有 `waveform`、`waveform_length`、`aut_features`、
`aut_feature_mask`、`audio_placeholders`，并通过 `torch.load(..., weights_only=True)` 读取。
`audio_sha256` 是 payload 中 contiguous CPU `float32 [1, time]` waveform 按 C 顺序排列的
原始 bytes 的 SHA-256，不是 `.pt` 文件本身的 hash。payload 的键、dtype、shape、有限值、
整数 timing 和 SHA 在取样时逐项校验。
