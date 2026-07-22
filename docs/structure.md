# Workspace Structure

`workspace` 把机器 profile、物理路径和已处理资产包装成稳定的逻辑对象。调用方只依赖
公开入口的参数和返回对象，不依赖机器目录、第三方缓存布局或脚本中间产物。

## Target Layout

```text
src/zhuyin/
  env.py                         # 机器 profile、workspace root 和临时环境上下文
  _locations/
    fudan.py                     # 复旦机器默认 homes 和 marker
    hz.py                        # 杭州 no-implement 占位；启用时单独补 profile 和测试
    us.py                        # US no-implement 占位；启用时单独补 profile 和测试
  datasets/
    aut.py                       # WMT19 TTS prepared Qwen2.5 AuT target 入口
    _aut_store.py                # prepared AuT manifest/index/payload 严格读取
    common_voice.py              # Common Voice 逻辑数据集入口
    wmt19_tts.py                 # WMT19 TTS 与 codec 视图入口
    _wmt19_tts_prepare.py        # prepare 服务
    _wmt19_tts_codec.py          # LongCat / DAC / Stable Codec / UniCodec view 生成服务
    _wmt19_tts_stable.py         # Stable Codec quantizer preset 与 store identity
    _wmt19_tts_filter.py         # filter 服务
    _wmt19_tts_bpe.py            # BPE 语料与训练服务
    _wmt19_tts_io.py             # store 状态和报告 IO
    _wmt19_tts_store.py          # 标准 store root 和 dataset factory
  tokenizers/
    _codec_bpe_artifact.py       # BPE 训练默认值和 artifact 命名
    codec_bpe.py                 # tokenizer artifact 定位与加载
scripts/
  prepare_wmt19_tts.py           # WMT19 TTS waveform store 入口
  prepare_wmt19_tts_codec_view.py # WMT19 codec view 统一入口：longcat/dac/stable/unicodec
  prepare_wmt19_tts_bpe.py       # WMT19 LongCat semantic BPE 入口
  filter_wmt19_tts.py            # WMT19 过滤任务统一入口：speech/translation/speech-translation
  _*.py                          # 入口复用的私有 argparse、服务调用和 summary 打印 helper
jobs/
  env.sh                         # 共享 workspace 环境解析，不含任务逻辑
  locations/
    fudan/*.sh                   # Fudan 提交入口，调用统一 scripts
    hz/README.md                 # HZ no-implement 占位
    us/README.md                 # US no-implement 占位
docs/
  datasets/                      # 公开数据集契约
  structure.md                   # workspace 总体边界和迁移顺序
```

同一个公开导入路径只能有一种物理实现。`zhuyin.env` 使用单文件 `env.py`，不同时保留
`env.py` 和 `env/` 包。公开模块最多到第二级，不在 `zhuyin` 或 `zhuyin.datasets`
聚合导出具体入口。

## Public Contracts

### Environment

`zhuyin.env` 是机器相关公开信息的唯一所有者，并从私有
`src/zhuyin/_locations/fudan.py` 读取当前已实现 location 的本地配置，包括：

- `Location.FUDAN` 的 `STATIC_HOME`、`DYNAMIC_HOME` 默认值。
- Fudan 自动探测所需的文件系统 marker。
- `context()`、`location()`、`static_home()`、`dynamic_home()` 和 `datasets_home()`。

其他 location 只在 `_locations/` 保留 no-implement 占位文件，不注册到运行时 profile，
不预填真实路径或数据源；需要启用时在同一改动里补齐 profile、loader source 和测试。

数据集 export、模型 checkpoint 等具体资产路径不属于 `zhuyin.env`；它们由对应 loader 或
资源模块根据 workspace root 解析。

环境函数按以下顺序纯解析路径：

1. 当前进程环境变量。
2. `LOCATION=fudan` 对应的默认值；使用默认 home 时发 `RuntimeWarning`。

`location()` 读取 `$LOCATION`，缺失时根据 marker 探测。`static_home()`、
`dynamic_home()` 和 `datasets_home()` 返回解析结果但不修改 `os.environ`。公开 loader
直接调用这些纯函数获得默认路径，不通过临时环境注入完成普通路径计算。

`context()` 只临时修改 `LOCATION`、`STATIC_HOME`、`DYNAMIC_HOME` 和调用方显式传入的
额外变量，退出时恢复现场。显式 override、当前环境和 location 默认值使用同一套纯解析
逻辑。`context()` 只用于确实要求环境变量的第三方代码或脚本，不是 loader 解析路径的
必要前置。它不自动派生 `HF_HOME`、`BPE_CACHE_DIR`、`TORCH_HOME` 或模型专用路径。

### Dataset Loaders

公开 loader 返回稳定逻辑对象，并隐藏机器专用协议和 export 路径。默认参数必须能够构造
目标对象；缺少文件、请求的视图没有可用物理来源或参数组合无效时直接抛出清晰异常。

Common Voice 只有一个标准 store root，不公开机器 profile：

```python
common_voice(*, root=None, split="train")
```

Prepared AuT 是独立的 map-style tensor store，不走 `anydataset`：

```python
prepared_aut(*, root=None, split="train", revision=DEFAULT_REVISION)
```

默认读取
`datasets_home() / "prepared_aut/wmt19_tts/qwen2_5-omni-7b/<revision>/<split>"`；显式
`root` 表示 `prepared_aut/wmt19_tts` 数据集根。teacher checkpoint/revision、Transformers
版本、feature schema、Qwen2.5 timing 和 16 kHz mono waveform 都由 manifest 固定，loader
不提供兼容或在线转换分支。完整 artifact 和 sample 契约见
[`docs/datasets/aut.md`](datasets/aut.md)。

WMT19 TTS 的公开入口不暴露物理 profile：

```python
dataset_root(root=None) -> Path
wmt19_tts(*, root=None, split="train")
wmt19_tts_codec(*, codec=Codec.LONGCAT, root=None, split="train")
wmt19_tts_dac(*, root=None, split="train")
wmt19_tts_stable(*, root=None, split="train", quantizer=DEFAULT_STABLE_QUANTIZER)
wmt19_tts_unicodec(*, root=None, split="train")
```

参数规则：

- 显式 `root` 始终读取标准 store。
- `root=None` 时，目前只实现 Fudan 默认来源，读取标准 store。
- 其他 location 不保留 dataset source 目录；需要启用时单独补齐 source 实现和测试。
- 非 Fudan location 临时读取已有标准 store 时，显式传 `root=...`。

`root` 始终表示标准 WMT19 TTS 数据集根目录，其下包含 `base/`、`longcat/`、`dac/`、
`stable-1x46656_400bps/` 等逻辑视图。Stable Codec 的目录名包含 posthoc
quantizer preset；旧 `stable/` native-FSQ store 不在公开 loader 的兼容范围内。`root` 不表示
具体 view 目录，也不覆盖 location 默认来源。

### Tokenizer Artifacts

artifact 定位和对象加载分成两个明确入口：

```python
codec_bpe_path(*, root=None, artifact=DEFAULT_ARTIFACT) -> Path
codec_bpe(path) -> CodecBPE
```

`codec_bpe_path()` 的 root 按以下顺序解析：

1. 显式 `root`。
2. `$BPE_CACHE_DIR`。
3. `static_home() / "bpe"`。

`codec_bpe()` 只加载调用方给出的准确 artifact 路径，不再次解释训练参数或环境变量。
`artifact` 是 artifact root 下的稳定相对路径。vocab size、codebook size 和 trainer 参数到
artifact 名的转换属于 BPE 训练服务，不暴露给普通加载方。这样训练配置和对象加载不会形成
两套重叠参数入口。

## Private Services

WMT19 TTS 的 prepare、codec、filter 和 BPE 规则放在
`src/zhuyin/datasets/_wmt19_tts_*.py`。
私有服务层可以依赖 `anydataset` 和 `anytrain` 的公开接口，但不依赖 `argparse.Namespace`，
不解析 CLI 参数，也不打印命令行 summary。

公开 prepare 脚本按 TTS、codec-view 和 BPE 三件事分开；codec-view 内部再用子命令选择
LongCat、DAC、Stable Codec 或 UniCodec。同一件事不再保留多个 location-neutral Python
入口。私有 helper 使用 `_` 前缀，只供公开入口和测试复用。

脚本只负责：

1. 定义该入口独有的 argparse 参数。
2. 把参数转换成服务层的严格类型参数。
3. 调用服务并打印或写出顶层 summary。

重复的 store ready 检查、manifest 读取和 JSON/JSONL IO 放在 `_wmt19_tts_io.py`。业务规则
不放进 IO helper。新增结构优先使用 `TypedDict`、普通不可变值或简单 callable class；需要
新增 dataclass 时按仓库约定先确认。

## Jobs

job wrapper 按 location 放在 `jobs/locations/<location>/`。wrapper 先 source 对应
location 的 `env.sh`，进入项目根目录，并调用真实 Python 脚本。所有 wrapper 的
Python 命令末尾保留 `"$@"`。机器路径、Python 解释器和设备默认值只放在 location
job 层，不写进统一 `scripts/`。

WMT19 数据处理脚本统一使用 `--root` 表示数据集根目录。BPE 输出根使用 `--bpe-root`，与
输入数据集根分开；未传时由 `BPE_CACHE_DIR` 或 workspace 默认路径解析。

## Migration Order

迁移按公开契约向内推进，每一步完成后都应恢复完整测试收集：

1. 合并 `zhuyin.env` 实现，确保公开导入只有一个物理来源。
2. 移除公开 dataset profile，一次性迁移 WMT19 loader、所有脚本调用和测试。
3. 一次性迁移 `codec_bpe_path(root=..., artifact=...)`、BPE 训练脚本和测试。
4. 把 prepare/codec/filter/BPE 纯逻辑移入私有服务模块，删除脚本间重复 helper。
5. 最后整理 job 参数和运行文档，不在接口迁移期间保留静默兼容分支。

每一步的完成标准是：公开入口可以导入，默认参数可以构造对象，相关脚本可以执行
`--help`，测试可以完整收集，Ruff 和目标 Python 版本检查通过。
