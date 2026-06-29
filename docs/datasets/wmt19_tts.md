# WMT19 TTS

`wmt19_tts()` 返回逻辑 WMT19 zh-en TTS 对象，source/target 两侧包含文本和
waveform。`wmt19_tts_longcat()` 返回同一逻辑数据集的 LongCat 视图，供
speech-to-speech 读取原始 LongCat codes。

入口会根据 `LOCATION` 选择默认物理 profile，但返回的逻辑 sample 契约保持一致。
显式传 `dataset_dir=...` 时只覆盖当前 profile 的物理 root；需要强制读取标准 store
时，同时传 `profile=...STORE`。

默认 TTS 入口：

```text
LOCATION=fudan  store://$STATIC_HOME/datasets/wmt19_tts/base:train
LOCATION=hz     sharded_csv:///nfs/yin.zhu/train/text_to_speech/moss_tts_hz_export:train
```

对应 workspace 入口：

```python
from zhuyin.datasets.wmt19_tts import wmt19_tts

dataset = wmt19_tts()
```

默认 LongCat 训练入口：

```text
LOCATION=fudan  store://$STATIC_HOME/datasets/wmt19_tts/longcat:train
LOCATION=hz     hf-disk:///nfs/yin.zhu/datasets/wmt19_tts_longcat_codes_text_cleaned:train
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

如果 `STATIC_HOME` 未设置，入口会使用 `LOCATION` 对应默认值并发 warning；
`LOCATION` 缺失时会按 `/share5_video`、`/nfs/yin.zhu`、`/mnt` 的顺序探测默认位置。
调用入口时会根据 `STATIC_HOME` 或该默认值补齐缺失的 `ANYDATASET_HOME`、
`ANYTRAIN_HOME`、`BPE_CACHE_DIR`、`HF_HOME`、`HF_HUB_CACHE`、`HF_DATASETS_CACHE`、
`TORCH_HOME` 和 `ANYTRAIN_WHISPER_ROOT`。
临时使用其他物理根目录时，传 `dataset_dir=...`；需要强制选择物理加载方案时，
传 `profile=...`。

## 合成 TTS

WMT19 TTS 生成脚本位于 workspace：

```bash
PYTHONPATH=src:../third_party/anydataset/src:../third_party/anytrain/src \
python scripts/prepare_wmt19_tts.py
```

默认输出到 `$STATIC_HOME/datasets/wmt19_tts`。其中 `base` 是 source/target 文本和
waveform 的 TTS store；报告放在 `reports/` 下。生成 target audio
时，脚本会先生成 source audio，再用 source audio 作为 target 的 MOSS reference。
target 阶段默认只取 source audio 前 8 秒作为 reference，避免 MOSS tokenizer 对长
reference 做二次编码时 OOM；中间 store 名会带 `target-audio-ref8s`。
可提交任务入口是：

```bash
jobs/prepare_wmt19_tts.sh
```

## 准备 LongCat

LongCat 编码脚本消费 `base`，写出供 `wmt19_tts_longcat()` 读取的 `longcat` store：

```bash
PYTHONPATH=src:../third_party/anydataset/src:../third_party/anytrain/src \
python scripts/prepare_wmt19_tts_longcat.py
```

可提交任务入口是：

```bash
jobs/prepare_wmt19_tts_longcat.sh
```

## 质量过滤

语音质量过滤消费合成脚本生成的 `base` store，不重新合成音频：

```bash
PYTHONPATH=src:../third_party/anydataset/src:../third_party/anytrain/src \
python scripts/filter_wmt19_tts.py
```

输出写到 `$STATIC_HOME/datasets/wmt19_tts/reports/filter_summary.json` 和
`speech_quality_metrics.jsonl`。需要调阈值时只重跑过滤脚本。可提交任务入口是：

```bash
jobs/filter_wmt19_tts.sh
```

121 上当前物理 GPU 3 能被 `nvidia-smi` 枚举，但 PyTorch CUDA 初始化会报
`DeferredCudaCallError: device=3, num_gpus=3`。notebook 的首个导入单元在未手动设置
`CUDA_VISIBLE_DEVICES` 时会默认使用 `0,1,2`；如果 kernel 已经 import 过 `torch`，需要
重启 kernel 后重新运行首格才会生效。

LongCat 训练契约和 speech-to-speech 保持一致：source 和 target 两侧 audio 都包含
`AudioView.LONGCAT`，其中至少有 `semantic_codes` 和 `acoustic_codes`。需要文本或
waveform 时使用 `wmt19_tts()`。

## LongCat BPE

可以直接用 workspace 脚本从默认 LongCat store 训练 speech-to-speech 使用的 BPE：

```bash
PYTHONPATH=src:../third_party/anytrain/src:../third_party/anydataset/src \
python scripts/prepare_wmt19_tts_longcat_bpe.py
```

可提交任务入口是：

```bash
jobs/prepare_wmt19_tts_longcat_bpe.sh
```

脚本默认从 `wmt19_tts_longcat(split="train")` 读取 source 和 target 两侧完整
`semantic_codes`，训练 `anytrain.tokenizer.CodecBPE` 的 100k vocab。训练参数中
`vocab_size`、`min_frequency`、`show_progress` 和
`max_token_length` 对齐 `tokenizers.trainers.BpeTrainer`。输出根目录优先使用
`$BPE_CACHE_DIR`；未设置时由 workspace 配到
`$STATIC_HOME/bpe`，并写出
`longcat/vocab_100k_minfreq_0_maxlen_none_codes_8192/{codec_bpe.json,tokenizer.json,meta.json,eval.json}`。
其中 `vocab_100k` 是目标 BPE vocab size，`codes_8192` 是 LongCat semantic codebook
大小。命令输出只保留 `artifact_dir`、`actual_vocab_size` 和 `eval` 压缩统计。

调试小样本可以显式限制 sample 数，避免污染正式 artifact 名：

```bash
PYTHONPATH=src:../third_party/anytrain/src:../third_party/anydataset/src \
python scripts/prepare_wmt19_tts_longcat_bpe.py --sample-limit 1000
```

代码里通过 workspace 入口拿路径或 tokenizer：

```python
from zhuyin.tokenizers.longcat import longcat_bpe, longcat_bpe_path

path = longcat_bpe_path()
bpe = longcat_bpe()
```
