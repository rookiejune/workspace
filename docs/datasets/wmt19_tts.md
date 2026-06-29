# WMT19 TTS

当前数据集根目录是 `$STATIC_HOME/datasets/wmt19_tts`。`wmt19_tts()` 返回真实 TTS
store，source/target 两侧包含文本和 waveform。
`wmt19_tts_longcat()` 返回 LongCat-only delta store，供 speech-to-speech 训练读取
原始 LongCat codes。

默认 TTS 入口：

```text
store://$STATIC_HOME/datasets/wmt19_tts/base:train
```

对应 workspace 入口：

```python
from zhuyin.datasets.wmt19_tts import wmt19_tts

dataset = wmt19_tts()
```

默认 LongCat 训练入口：

```text
store://$STATIC_HOME/datasets/wmt19_tts/longcat-delta:train
```

对应 workspace 入口：

```python
from zhuyin.datasets.wmt19_tts import wmt19_tts_longcat

dataset = wmt19_tts_longcat()
```

交互式检查入口：

```text
workspace/notebooks/datasets/wmt19_tts.ipynb
```

这个 notebook 会加载默认数据集、打印长度，并按 index 取具体 sample，检查 source/target
两侧 audio 的 `AudioView.LONGCAT` keys、`semantic_codes` / `acoustic_codes` shape，并用
LongCat decoder 还原 source/target 波形供试听。

如果 `STATIC_HOME` 未设置，入口会使用复旦共享默认 `/mnt/pami202/zhuyin` 并发
warning。调用入口时会根据 `STATIC_HOME` 或该默认值补齐缺失的 `ANYDATASET_HOME` 和
`HF_HOME`。
临时使用其他数据集根目录时，直接传 `dataset_dir=...`。

121 上当前物理 GPU 3 能被 `nvidia-smi` 枚举，但 PyTorch CUDA 初始化会报
`DeferredCudaCallError: device=3, num_gpus=3`。notebook 的首个导入单元在未手动设置
`CUDA_VISIBLE_DEVICES` 时会默认使用 `0,1,2`；如果 kernel 已经 import 过 `torch`，需要
重启 kernel 后重新运行首格才会生效。

LongCat 训练契约和 speech-to-speech 保持一致：source 和 target 两侧 audio 都包含
`AudioView.LONGCAT`，其中至少有 `semantic_codes` 和 `acoustic_codes`。需要文本或
waveform 时使用 `wmt19_tts()`。
