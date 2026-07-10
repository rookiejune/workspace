# WMT19 TTS

`wmt19_tts()` 返回逻辑 WMT19 zh-en TTS 对象，source/target 两侧包含文本和
waveform。`wmt19_tts_codec()` 返回同一数据集的 codec 视图。

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

默认 LongCat codec 视图：

```text
LOCATION=fudan  store://$STATIC_HOME/datasets/wmt19_tts/longcat:train
LOCATION=hz     hf-disk:///nfs/yin.zhu/datasets/wmt19_tts_longcat_codes_text_cleaned:train
```

对应 workspace 入口：

```python
from zhuyin.datasets.wmt19_tts import Codec, wmt19_tts_codec

longcat = wmt19_tts_codec()
stable = wmt19_tts_codec(codec=Codec.STABLE)
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
加载入口本身不写入第三方缓存变量。`with zhuyin.env.context():` 只临时注入
`LOCATION`、`STATIC_HOME` 和 `DYNAMIC_HOME`；`HF_HOME`、`BPE_CACHE_DIR` 或
`ANYTRAIN_WHISPER_ROOT` 这类变量由具体脚本按需读取或由调用环境显式设置。
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

大规模合成不要一次性写到单个 `base` store；最终目录和中间
`work/base/{source-audio,target-audio-*}` 会同时占空间。长跑时按 chunk 写到独立目录，
并在成功后清理中间 waveform：

```bash
jobs/prepare_wmt19_tts.sh \
  --root /mnt/pami201/zhuyin/datasets/wmt19_tts_chunks/chunk-000000 \
  --offset 0 \
  --limit 10000 \
  --cleanup-work
```

50w 样本建议拆成 50 个 1w chunk。每个 chunk 可独立 resume；如果某个 chunk 失败，
重跑同一个 `--root` 即可继续未完成的 fragment。

## 准备 LongCat

LongCat 编码脚本消费 `base`，写出供 `wmt19_tts_codec(codec=Codec.LONGCAT)` 读取的
`longcat` store：

```bash
PYTHONPATH=src:../third_party/anydataset/src:../third_party/anytrain/src \
python scripts/prepare_wmt19_tts_longcat.py
```

可提交任务入口是：

```bash
jobs/prepare_wmt19_tts_longcat.sh
```

## 质量过滤

质量过滤有语音和翻译两个原子入口，以及一个串联二者的组合入口。它们都消费合成脚本
生成的 `base` store，不重新合成音频。

语音质量过滤检查合成音频的 UTMOS、ASR chrF、长度和峰值：

```bash
PYTHONPATH=src:../third_party/anydataset/src:../third_party/anytrain/src \
python scripts/filter_wmt19_tts_speech.py
```

输出写到 `$STATIC_HOME/datasets/wmt19_tts/reports/speech_filter_summary.json` 和
`speech_quality_metrics.jsonl`。可提交任务入口是：

```bash
jobs/filter_wmt19_tts_speech.sh
```

翻译质量过滤检查 source/target 文本句对，输出 `clean`、`usable`、`review` 和
`reject` 四档；默认训练选择是 `clean` + `usable`：

```bash
PYTHONPATH=src:../third_party/anydataset/src:../third_party/anytrain/src \
python scripts/filter_wmt19_tts_translation.py
```

输出写到 `$STATIC_HOME/datasets/wmt19_tts/reports/translation_filter_summary.json` 和
`translation_quality_metrics.jsonl`。需要调阈值时只重跑对应过滤脚本。可提交任务入口是：

```bash
jobs/filter_wmt19_tts_translation.sh
```

组合过滤入口按同一批规则串联两个缓存阶段，默认先跑更快的文本翻译过滤，再对保留
样本跑语音过滤；如需复用已有完整语音缓存，可传 `--order speech,translation`：

```bash
PYTHONPATH=src:../third_party/anydataset/src:../third_party/anytrain/src \
python scripts/filter_wmt19_tts_speech_translation.py
```

组合报告写到 `$STATIC_HOME/datasets/wmt19_tts/reports/speech_translation/`，其中
`summary.json` 记录顺序、两个 rule name、各阶段 counts 和最终选择数量；两阶段
metrics 分别写成 `translation_quality_metrics.jsonl` 和
`speech_quality_metrics.jsonl`。可提交任务入口是：

```bash
jobs/filter_wmt19_tts_speech_translation.sh
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

脚本默认从 `wmt19_tts_codec(codec=Codec.LONGCAT, split="train")` 读取 source 和 target 两侧完整
`semantic_codes`，训练 `anytrain.tokenizer.CodecBPE` 的 100k vocab。训练参数中
`vocab_size`、`min_frequency`、`show_progress` 和
`max_token_length` 对齐 `tokenizers.trainers.BpeTrainer`。输出根目录优先使用
`$BPE_CACHE_DIR`；未设置时写到 `$STATIC_HOME/bpe`，并写出
`longcat/vocab_100k_minfreq_0_maxlen_none_codes_8192/{codec_bpe.json,meta.json,eval.json}`。
其中 `vocab_100k` 是目标 BPE vocab size，`codes_8192` 是 LongCat semantic codebook
大小。命令输出只保留 `artifact_dir`、`actual_vocab_size` 和 `eval` 压缩统计。

调试小样本可以显式限制 sample 数，避免污染正式 artifact 名：

```bash
PYTHONPATH=src:../third_party/anytrain/src:../third_party/anydataset/src \
python scripts/prepare_wmt19_tts_longcat_bpe.py --sample-limit 1000
```

代码里通过 workspace 入口拿路径或 tokenizer：

```python
from zhuyin.tokenizers.codec_bpe import codec_bpe, codec_bpe_path

path = codec_bpe_path()
bpe = codec_bpe()
```
