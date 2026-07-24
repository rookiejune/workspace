# Qwen TTS Speech

Qwen TTS speech 由三层独立能力组合：

1. `anydataset.dataset` 给任意 map-style canonical dataset 的一个或多个 text item 增加
   `TextView.SPEAKERS`。
2. `scripts._qwen_tts_speech` 选择参与 TTS 的 text references，并使用
   `QwenTTSProvider` 和 `ModalityMaterializer` 生成同 role audio。
3. `zhuyin.datasets.qwen_tts_speech` 只读取 prepared speaker-grid store。

## Speaker View

`TextView.SPEAKERS` 是独立的生成条件 view。单条 sample 中它是一个 speaker id 字符串，
不放进 text meta，也不依赖 Qwen。生成后的 audio item 使用
`AudioMeta.SPEAKER_ID` 记录实际 speaker。

普通 assignment 不改变数据集长度，并支持多个 text references。例如同时为 WMT19
source/target text 指定 speaker：

```python
from anydataset.dataset import SpeakerAssignment
from anydataset.types import Modality, Role
from scripts._qwen_tts_speech import materialize_qwen_tts_speech

materialize_qwen_tts_speech(
    text_dataset_factory=text_dataset_factory,
    assignments={
        (Role.SOURCE, Modality.TEXT): SpeakerAssignment(
            ("vivian", "ryan"),
            mode="cycle",
        ),
        (Role.TARGET, Modality.TEXT): SpeakerAssignment(
            ("ryan",),
            mode="cycle",
        ),
    },
    output_dir="/data/qwen_tts",
)
```

workflow 会先把输入 sample 投影到 assignments 中显式列出的 text references，因此原数据集
中其他 audio/image/text item 不会被 `ModalityMaterializer` 误判为本次 TTS 输入。未列出的
text reference 不生成 audio。

## Speaker Grid

需要为每条 text 生成完整 speaker 集合时，使用单 reference Cartesian workflow：

```python
from scripts._qwen_tts_speech import materialize_qwen_tts_speaker_grid

materialize_qwen_tts_speaker_grid(
    text_dataset_factory=text_dataset_factory,
    speaker_ids=("vivian", "ryan"),
    output_dir="/data/qwen_tts_speaker_grid",
    split="train",
)
```

物理 store 顺序为：

```text
text_index=0, speaker=vivian
text_index=0, speaker=ryan
text_index=1, speaker=vivian
text_index=1, speaker=ryan
```

每条 flat sample 的 text meta 保存 `TextMeta.SOURCE_INDEX`。这样 materializer 的 resume、
sharding、失败重跑和质量过滤仍按单条 waveform 处理。

## Grouped Loader

prepared grid 通过 workspace loader 按 text 聚合：

```python
from zhuyin.datasets.qwen_tts_speech import qwen_tts_speaker_grid

dataset = qwen_tts_speaker_grid(
    root="/data/qwen_tts_speaker_grid",
    speaker_ids=("vivian", "ryan"),
)
sample = dataset[0]
```

聚合后的 sample：

- text item 保留原文和 `TextMeta.SOURCE_INDEX`，移除单条 flat sample 的 speaker view。
- `AudioView.WAVEFORM` 仍是 `(waveform_tensor, sample_rate)`，speaker 是 tensor 第 0 维。
- `AudioView.SPEAKERS` 定义 speaker 轴顺序。
- `AudioView.SPEAKER_LENGTHS` 保存每个 waveform 在 padding 前的真实长度。
- grouped audio 不再使用单值语义的 `AudioMeta.SPEAKER_ID`；speaker 轴只由
  `AudioView.SPEAKERS` 表达。

speaker id 顺序是物理 store 契约的一部分。读取时必须传入生成时的相同顺序；source index、
text speaker view、已存在的 audio speaker meta、采样率或 waveform shape 不匹配时 loader
明确报错，不静默重排或重标。

Fudan smoke 入口会生成 2 条文本 × 2 个 speaker，并导出 grouped wav：

```text
jobs/fudan/prepare_qwen_tts_speaker_grid_smoke.sh
```
