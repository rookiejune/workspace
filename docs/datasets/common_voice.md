# Common Voice

`zhuyin.datasets.common_voice` 暴露 Common Voice 这个物理数据集入口。它不管理
run 目录、speaker vocabulary 或具体实验的逻辑数据集命名。

## Default

- 默认 split：`train`
- 默认根目录：`$STATIC_HOME/datasets/common_voice`

入口按这个顺序解析根目录：

1. `common_voice(root=...)`
2. `zhuyin.env.static_home() / "datasets/common_voice"`

`static_home()` 优先读取 `$STATIC_HOME`；未设置时使用 `LOCATION` 对应默认值并发
`RuntimeWarning`。路径解析是纯函数调用，不修改进程环境。显式传 `root` 时不解析
workspace 默认路径，也不产生默认路径 warning。

如果 `root` 指向 `cv-corpus-*` 下面的具体语种目录，语种由目录名推断；否则交给
anydataset 的 Common Voice preset 从根目录结构推断最新语料版本和默认语种。

加载入口不写入 workspace 或第三方环境变量。第三方变量由对应脚本或调用环境显式设置。

## Workspace API

```python
from anydataset import AudioMeta, Modality, Role
from zhuyin.datasets.common_voice import common_voice

dataset = common_voice(split="train")
sample = next(iter(dataset))
audio = sample[(Role.DEFAULT, Modality.AUDIO)]
speaker = audio.meta[AudioMeta.SPEAKER_ID]
```

临时读取其他物理目录时直接传 `root`：

```python
dataset = common_voice(root="/data/common_voice", split="dev")
```

Common Voice 的 `client_id` 已由 anydataset preset 收进 default audio 的
`AudioMeta.SPEAKER_ID`。如果后续实验需要 speaker vocabulary，应由具体实验从数据集样本
统计生成，而不是作为 Common Voice loader 的独立入口。

静态资产路径统一由 `STATIC_HOME` 表达。旧入口如果仍依赖专用路径，手动建立符号链接。
