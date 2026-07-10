# Common Voice

`zhuyin.datasets.common_voice` 暴露 Common Voice 这个物理数据集入口。它不管理
run 目录、speaker vocabulary 或具体实验的逻辑数据集命名。

## Default

- 默认 split：`train`
- 默认根目录：`$STATIC_HOME/datasets/common_voice`

入口按这个顺序解析根目录：

1. `common_voice(root=...)`
2. `$STATIC_HOME/datasets/common_voice`
3. `LOCATION` 自动探测出的默认 `$STATIC_HOME/datasets/common_voice`，同时发 warning

如果 `root` 指向 `cv-corpus-*` 下面的具体语种目录，语种由目录名推断；否则交给
anydataset 的 Common Voice preset 从根目录结构推断最新语料版本和默认语种。

加载入口本身不写入第三方缓存变量。`with zhuyin.env.context():` 只临时注入
`LOCATION`、`STATIC_HOME` 和 `DYNAMIC_HOME`，具体第三方变量由对应脚本或调用环境显式设置。

## Workspace API

```python
from anydataset import AudioMeta, Modality, Role
from zhuyin.datasets.common_voice import common_voice

dataset = common_voice(split="train")
sample = next(iter(dataset))
audio = sample[(Role.DEFAULT, Modality.AUDIO)]
speaker = audio.meta[AudioMeta.SPEAKER_ID]
```

Common Voice 的 `client_id` 已由 anydataset preset 收进 default audio 的
`AudioMeta.SPEAKER_ID`。如果后续实验需要 speaker vocabulary，应由具体实验从数据集样本
统计生成，而不是作为 Common Voice loader 的独立入口。

静态资产路径统一由 `STATIC_HOME` 表达。旧入口如果仍依赖专用路径，手动建立符号链接。
