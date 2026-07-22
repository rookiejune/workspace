# WMT19 TTS

`wmt19_tts()` 返回逻辑 WMT19 zh-en TTS 对象，source/target 两侧包含文本和 waveform。
`wmt19_tts_codec()` 返回同一数据集的 codec 视图；DAC、Stable Codec 和 UniCodec 也提供
不要求调用方传枚举的具名入口。物理来源由 loader 私有选择，不改变逻辑 sample 契约。

目标公开接口：

```python
from zhuyin.datasets.wmt19_tts import (
    DEFAULT_STABLE_QUANTIZER,
    StableQuantizer,
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
- `root=None`：目前只实现 Fudan 默认来源，读取标准 store。
- 其他 location 不保留 dataset source 目录；启用时单独补齐 source 实现和测试。
- 非 Fudan location 临时读取已有标准 store 时传 `root=...`。

`root` 始终指 WMT19 TTS 数据集根目录，其下包含 `base/`、`longcat/`、`dac/`、
`stable-1x46656_400bps/` 等视图，不指向具体 view 目录。Stable Codec 视图的
目录名始终包含 quantizer preset。

默认 TTS 入口：

```text
LOCATION=fudan  store://$STATIC_HOME/datasets/wmt19_tts/base:train
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

标准 store 需要默认路径且 `STATIC_HOME` 未设置时，`static_home()` 使用 Fudan
默认值并发 warning；`LOCATION` 缺失时会按 Fudan marker 探测，探测不到时仍回退
到 Fudan。路径解析不修改进程环境。

加载入口本身不写入第三方缓存变量。`HF_HOME`、`BPE_CACHE_DIR` 或
`ANYTRAIN_WHISPER_ROOT` 由具体脚本按需读取或由调用环境显式设置。临时使用其他标准 store
根目录时传 `root=...`。

## Store schema

标准 `base/` 和 codec view 只接受 anydataset store schema v2。当前 prepare 流程通过
`DatasetWriter` 直接生成 v2；loader 遇到 v1 会明确报错，不在读取或训练过程中自动迁移，
也不回退到旧 schema。这样不会把长时间复制或半成品发布隐藏在数据读取副作用里。

存量 v1 store 使用 anydataset 的离线迁移入口生成独立副本：

```bash
ROOT=/mnt/pami202/zhuyin/datasets/wmt19_tts
STAGING=/mnt/pami202/zhuyin/datasets/wmt19_tts-schema-v2-staging

PYTHONPATH=../third_party/anydataset/src \
python -m anydataset.store.maintenance migrate \
  "$ROOT/base" "$STAGING/base"
```

迁移期间冻结源 store 的 writer，并预留至少一份源 store 的额外空间。迁移完成后，在 staging
root 下只读引用对应 codec view，用 `wmt19_tts(root=...)` 和 `wmt19_tts_codec(root=...)`
验收 manifest、sample count 和真实 payload。验收通过后在同一文件系统内将原 `base/` 改名为
回滚副本，再把 v2 目录改名为稳定的 `base/`；默认入口复验通过前不要删除回滚副本。

复旦 canonical `base/` 已于 2026-07-22 从 v1 迁移到 v2。1000 条 base/LongCat 样本在切换
前后均通过真实 loader 全量反序列化，默认 speech DataModule collate 和两步 GPU smoke 通过；
原 v1 store 暂时保留在 `base-schema-v1-backup-20260722/`。

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
jobs/fudan/prepare_wmt19_tts.sh
```

大规模合成不要一次性写到单个 `base` store；最终目录和中间
`work/base/{source-audio,target-audio-*}` 会同时占空间。长跑时按 chunk 写到独立目录，
并在成功后清理中间 waveform：

```bash
jobs/fudan/prepare_wmt19_tts.sh \
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
`[frame, codebook]` 整数 Tensor。旧的 dict 格式标准 store 不再兼容，需要重新物化：

```bash
PYTHONPATH=src:../third_party/anydataset/src:../third_party/anytrain/src \
python scripts/prepare_wmt19_tts_codec_view.py longcat
```

可提交任务入口是：

```bash
jobs/fudan/prepare_wmt19_tts_longcat.sh
```

## 准备 DAC

DAC 脚本消费 `base`，写出供 `wmt19_tts_dac()` 读取的 `dac` store。该 store 保留
source/target 文本及语言信息，codes 使用统一的 `[frame, codebook]` 布局。默认使用官方
`44khz` / `8kbps` checkpoint；`n_quantizers` 在物化前固定，store 内始终保存完整、有序的
已配置码本。

```bash
PYTHONPATH=src:../third_party/anydataset/src:../third_party/anytrain/src \
python scripts/prepare_wmt19_tts_codec_view.py dac
```

离线使用已有 checkpoint 时传 `--local-files-only`，需要其它官方配置时传
`--model-type`、`--model-bitrate` 或 `--n-quantizers`。可提交任务入口是：

```bash
jobs/fudan/prepare_wmt19_tts_dac.sh
```

## 准备 Stable Codec

Stable Codec 脚本消费 `base`，默认使用 posthoc FSQ preset
`1x46656_400bps`，写出供 `wmt19_tts_stable()` 读取的
`stable-1x46656_400bps` store。该 store 保留 source/target 文本及语言信息，
并增加 `AudioView.STABLE`；它不依赖 location 专用 export。

```bash
PYTHONPATH=src:../third_party/anydataset/src:../third_party/anytrain/src \
python scripts/prepare_wmt19_tts_codec_view.py stable
```

支持的其他 posthoc preset 可通过 `--posthoc-bottleneck` 显式选择，每个 preset
使用独立 sibling store：

```bash
python scripts/prepare_wmt19_tts_codec_view.py stable \
  --posthoc-bottleneck 2x15625_700bps
```

对应的非默认视图通过具名 loader 选择：

```python
stable_700bps = wmt19_tts_stable(
    quantizer=StableQuantizer.FSQ_2X15625_700BPS,
)
```

旧 `stable/` 目录使用训练期 native FSQ `17^6 = 24,137,569`，它有量化约束，但没有
应用目标 posthoc preset。新 prepare 的 ready 检查和 loader 不会复用该产物；需要
用新 preset 重新物化。summary 文件同样包含 store identity，默认为
`reports/prepare_stable-1x46656_400bps_summary.json`。

可提交任务入口是：

```bash
jobs/fudan/prepare_wmt19_tts_stable.sh
```

## 准备 UniCodec

UniCodec 脚本消费 `base`，写出供 `wmt19_tts_unicodec()` 读取的 `unicodec` store。
该 store 保留 source/target 文本及语言信息，codes 使用统一的
`[frame, codebook]` 布局；speech 默认使用 domain `0` 和 bandwidth id `0`。

```bash
PYTHONPATH=src:../third_party/anydataset/src:../third_party/anytrain/src \
python scripts/prepare_wmt19_tts_codec_view.py unicodec
```

可提交任务入口是：

```bash
jobs/fudan/prepare_wmt19_tts_unicodec.sh
```

## 质量过滤

质量过滤在同一个公开 Python 入口下有语音和翻译两个原子子命令，以及一个串联二者的组合子命令。它们都消费合成脚本
生成的 `base` store，不重新合成音频。

语音质量过滤检查合成音频的 UTMOS、ASR chrF、长度和峰值：

```bash
PYTHONPATH=src:../third_party/anydataset/src:../third_party/anytrain/src \
python scripts/filter_wmt19_tts.py speech
```

输出写到 `$STATIC_HOME/datasets/wmt19_tts/reports/speech_filter_summary.json` 和
`speech_quality_metrics.jsonl`。可提交任务入口是：

```bash
jobs/fudan/filter_wmt19_tts_speech.sh
```

翻译质量过滤检查 source/target 文本句对，输出 `accept` 和 `reject` 两档；默认训练选择是
`accept`：

```bash
PYTHONPATH=src:../third_party/anydataset/src:../third_party/anytrain/src \
python scripts/filter_wmt19_tts.py translation
```

输出写到 `$STATIC_HOME/datasets/wmt19_tts/reports/translation_filter_summary.json` 和
`translation_quality_metrics.jsonl`。需要调阈值时只重跑对应过滤脚本。可提交任务入口是：

```bash
jobs/fudan/filter_wmt19_tts_translation.sh
```

组合过滤入口按同一批规则串联两个缓存阶段，默认先跑更快的文本翻译过滤，再对保留
样本跑语音过滤；如需复用已有完整语音缓存，可传 `--order speech,translation`：

```bash
PYTHONPATH=src:../third_party/anydataset/src:../third_party/anytrain/src \
python scripts/filter_wmt19_tts.py speech-translation
```

组合报告写到 `$STATIC_HOME/datasets/wmt19_tts/reports/speech_translation/`，其中
`summary.json` 记录顺序、两个 rule name、各阶段 counts 和最终选择数量；两阶段
metrics 分别写成 `translation_quality_metrics.jsonl` 和
`speech_quality_metrics.jsonl`。可提交任务入口是：

```bash
jobs/fudan/filter_wmt19_tts_speech_translation.sh
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
python scripts/prepare_wmt19_tts_bpe.py
```

可提交任务入口是：

```bash
jobs/fudan/prepare_wmt19_tts_longcat_bpe.sh
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
python scripts/prepare_wmt19_tts_bpe.py --sample-limit 1000
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
