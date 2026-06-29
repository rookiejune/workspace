# WMT19 TTS LongCat

当前 speech-to-speech 长跑使用的是 WMT19 zh-en 文本，经 MOSS-TTS 生成 source/target
语音，再用 LongCat 生成 `AudioView.LONGCAT` 后的 store。

默认训练入口：

```text
store://$STATIC_HOME/datasets/wmt19_tts/longcat-delta:train
```

对应 workspace 入口：

```python
from zhuyin.datasets.wmt19_tts import wmt19_tts

dataset = wmt19_tts()
```

交互式检查入口：

```text
workspace/notebooks/datasets/wmt19_tts.ipynb
```

这个 notebook 会加载默认数据集、打印长度，并按 index 取具体 sample，检查 source/target
两侧 audio 的 `AudioView.LONGCAT` keys、`semantic_codes` / `acoustic_codes` shape，并用
LongCat decoder 还原 source/target 波形供试听。

默认数据集根目录是 `$STATIC_HOME/datasets/wmt19_tts`，默认 store 子目录是
`longcat-delta`。如果 `STATIC_HOME` 未设置，入口会使用复旦共享默认
`/mnt/pami202/zhuyin` 并发 warning。调用入口时会根据 `STATIC_HOME` 或该默认值补齐
缺失的 `ANYDATASET_HOME` 和 `HF_HOME`。
临时使用其他数据集根目录时，直接传 `dataset_dir=...`。

数据契约和 speech-to-speech 保持一致：source 和 target 两侧 audio 都包含
`AudioView.LONGCAT`，其中至少有 `semantic_codes` 和 `acoustic_codes`。
