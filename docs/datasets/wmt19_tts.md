# WMT19 TTS

`wmt19_tts()` 返回逻辑 WMT19 zh-en TTS 对象，source/target 两侧包含文本和 waveform。
`wmt19_tts_codec()` 返回同一数据集的 codec 视图；DAC、Stable Codec 和 UniCodec 也提供
不要求调用方传枚举的具名入口。物理来源由 loader 私有选择，不改变逻辑 sample 契约。

目标公开接口：

```python
from zhuyin.datasets.wmt19_tts import (
    dataset_root,
    wmt19_tts,
    wmt19_tts_codec,
    wmt19_tts_dac,
    wmt19_tts_stable,
    wmt19_tts_unicodec,
)

tts = wmt19_tts()
longcat = wmt19_tts_codec()
dac = wmt19_tts_dac()
stable = wmt19_tts_stable()
unicodec = wmt19_tts_unicodec()
store = wmt19_tts(root=dataset_root())
```

参数规则：

- 显式传 `root`：读取该根目录下的标准 store。
- `root=None`：根据当前 location 和逻辑视图选择默认物理来源。
- HZ 默认从固定 export 读取 TTS 和 LongCat；这些路径由 WMT19 模块私有管理。
- DAC、Stable Codec 和 UniCodec 没有 HZ export，所有 location 默认读取标准 store。
- 在 HZ 强制读取标准 store 时传 `root=dataset_root()`。

`root` 始终指 WMT19 TTS 数据集根目录，其下包含 `base/`、`longcat/`、`dac/`、`stable/` 等视图，
不指向具体 view 目录。

默认 TTS 入口：

```text
LOCATION=fudan  store://$STATIC_HOME/datasets/wmt19_tts/base:train
LOCATION=hz     sharded_csv:///nfs/yin.zhu/train/text_to_speech/moss_tts_hz_export:train
```

对应 workspace 入口：

```python
from zhuyin.datasets.wmt19_tts import dataset_root, wmt19_tts

dataset = wmt19_tts()
store = wmt19_tts(root=dataset_root())
temporary = wmt19_tts(root="/data/wmt19_tts", split="dev")
```

默认 LongCat codec 视图：

```text
LOCATION=fudan  store://$STATIC_HOME/datasets/wmt19_tts/longcat:train
LOCATION=hz     hf-disk:///nfs/yin.zhu/datasets/wmt19_tts_longcat_codes_text_cleaned:train
```

对应 workspace 入口：

```python
from zhuyin.datasets.wmt19_tts import (
    Codec,
    dataset_root,
    wmt19_tts_codec,
    wmt19_tts_dac,
    wmt19_tts_stable,
    wmt19_tts_unicodec,
)

longcat = wmt19_tts_codec()
dac = wmt19_tts_dac()
stable = wmt19_tts_stable()
unicodec = wmt19_tts_unicodec()
store_longcat = wmt19_tts_codec(codec=Codec.LONGCAT, root=dataset_root())
```

交互式检查入口：

```text
workspace/notebooks/datasets/wmt19_tts.ipynb
```

这个 notebook 会加载默认数据集、打印长度，并按 index 取具体 sample，检查 source/target
两侧 audio 的 `AudioView.LONGCAT` `[frame, codebook]` shape，并用 LongCat decoder
还原 source/target 波形供试听。

标准 store 需要默认路径且 `STATIC_HOME` 未设置时，`static_home()` 使用 `LOCATION`
对应默认值并发 warning；`LOCATION` 缺失时会按 `/share5_video`、`/nfs/yin.zhu`、`/mnt`
的顺序探测默认位置。路径解析不修改进程环境。

加载入口本身不写入第三方缓存变量。`HF_HOME`、`BPE_CACHE_DIR` 或
`ANYTRAIN_WHISPER_ROOT` 由具体脚本按需读取或由调用环境显式设置。临时使用其他标准 store
根目录时传 `root=...`。

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
`longcat` store。标准 store 的 codes 直接使用 anytrain codec 契约定义的
`[frame, codebook]` 整数 Tensor；HZ 旧 export 的 semantic/acoustic 字段只在 HZ loader
边界转换。旧的 dict 格式标准 store 不再兼容，需要重新物化：

```bash
PYTHONPATH=src:../third_party/anydataset/src:../third_party/anytrain/src \
python scripts/prepare_wmt19_tts_longcat.py
```

可提交任务入口是：

```bash
jobs/prepare_wmt19_tts_longcat.sh
```

## 准备 DAC

DAC 脚本消费 `base`，写出供 `wmt19_tts_dac()` 读取的 `dac` store。该 store 保留
source/target 文本及语言信息，codes 使用统一的 `[frame, codebook]` 布局。默认使用官方
`44khz` / `8kbps` checkpoint；`n_quantizers` 在物化前固定，store 内始终保存完整、有序的
已配置码本。

```bash
PYTHONPATH=src:../third_party/anydataset/src:../third_party/anytrain/src \
python scripts/prepare_wmt19_tts_dac.py
```

离线使用已有 checkpoint 时传 `--local-files-only`，需要其它官方配置时传
`--model-type`、`--model-bitrate` 或 `--n-quantizers`。可提交任务入口是：

```bash
jobs/prepare_wmt19_tts_dac.sh
```

## 准备 Stable Codec

Stable Codec 脚本消费 `base`，写出供 `wmt19_tts_stable()` 读取的 `stable` store。
该 store 保留 source/target 文本及语言信息，并增加
`AudioView.STABLE`；它不依赖 HZ 专用 export。

```bash
PYTHONPATH=src:../third_party/anydataset/src:../third_party/anytrain/src \
python scripts/prepare_wmt19_tts_stable.py
```

可提交任务入口是：

```bash
jobs/prepare_wmt19_tts_stable.sh
```

## 准备 UniCodec

UniCodec 脚本消费 `base`，写出供 `wmt19_tts_unicodec()` 读取的 `unicodec` store。
该 store 保留 source/target 文本及语言信息，codes 使用统一的
`[frame, codebook]` 布局；speech 默认使用 domain `0` 和 bandwidth id `0`。

```bash
PYTHONPATH=src:../third_party/anydataset/src:../third_party/anytrain/src \
python scripts/prepare_wmt19_tts_unicodec.py
```

可提交任务入口是：

```bash
jobs/prepare_wmt19_tts_unicodec.sh
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

LongCat 训练契约和 speech-to-speech 保持一致：source 和 target 两侧 audio 都包含
`AudioView.LONGCAT`，逻辑值统一为 `[frame, codebook]` Tensor，第 0 个 codebook 是
semantic code，其余 codebook 是 acoustic code。标准 store 的旧物理 dict 在 loader
边界不再转换，需要重新物化为 anytrain 统一 Tensor；需要文本或 waveform 时使用
`wmt19_tts()`。

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

脚本默认从 `wmt19_tts_codec(codec=Codec.LONGCAT, split="train")` 读取 source 和 target 两侧
完整 LongCat codes，并使用第 0 个 semantic codebook 训练 `anytrain.tokenizer.CodecBPE`
的 100k vocab。训练参数中
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

代码里先通过 workspace 入口定位准确 artifact，再显式加载 tokenizer：

```python
from zhuyin.tokenizers.codec_bpe import codec_bpe, codec_bpe_path

path = codec_bpe_path()
bpe = codec_bpe(path)
```

`codec_bpe_path()` 的根目录按显式 `root`、`BPE_CACHE_DIR`、`$STATIC_HOME/bpe` 的顺序
解析，`artifact` 参数表示其中的稳定相对路径。vocab size、codebook size 和训练参数到
artifact 名的转换留在 BPE 训练服务。训练脚本使用 `--root` 指定输入 WMT19 TTS 数据集根，
使用 `--bpe-root` 指定输出 artifact 根，避免同一个参数同时表达输入和输出路径。
