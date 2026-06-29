# Common Voice

`zhuyin.datasets.common_voice` 暴露 Common Voice 这个物理数据集入口。它不管理
run 目录、speaker vocabulary 或具体实验的逻辑数据集命名。

## Default

- 默认语种：`en`
- 默认 split：`train`
- 默认根目录：`$STATIC_HOME/datasets/common_voice`

入口按这个顺序解析根目录：

1. `Args(root=...)`
2. `$STATIC_HOME/datasets/common_voice`

调用入口时会根据 `STATIC_HOME` 补齐缺失的 `ANYDATASET_HOME=$STATIC_HOME/anydataset`
和 `HF_HOME=$STATIC_HOME/huggingface`。如果显式设置过 `ANYDATASET_HOME` 或 `HF_HOME`，
则保留显式值。

## Workspace API

```python
from anydataset import AudioMeta, Modality, Role
from zhuyin.datasets.common_voice import Args, common_voice

dataset = common_voice(Args(split="train"))
sample = next(iter(dataset))
audio = sample[(Role.DEFAULT, Modality.AUDIO)]
speaker = audio.meta[AudioMeta.SPEAKER_ID]
```

Common Voice 的 `client_id` 已由 anydataset preset 收进 default audio 的
`AudioMeta.SPEAKER_ID`。如果后续实验需要 speaker vocabulary，应由具体实验从数据集样本
统计生成，而不是作为 Common Voice loader 的独立入口。

静态资产路径统一由 `STATIC_HOME` 表达。旧入口如果仍依赖专用路径，手动建立符号链接。
