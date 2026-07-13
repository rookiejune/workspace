# Workspace Structure

`workspace` 把机器 profile、物理路径和已处理资产包装成稳定的逻辑对象。调用方只依赖
公开入口的参数和返回对象，不依赖机器目录、第三方缓存布局或脚本中间产物。

## Target Layout

```text
src/zhuyin/
  env.py                         # 机器 profile、workspace root 和临时环境上下文
  datasets/
    common_voice.py              # Common Voice 逻辑数据集入口
    wmt19_tts.py                 # WMT19 TTS 与 codec 视图入口
    _wmt19_tts_prepare.py        # prepare 服务
    _wmt19_tts_codec.py          # LongCat / DAC / Stable Codec / UniCodec view 生成服务
    _wmt19_tts_filter.py         # filter 服务
    _wmt19_tts_bpe.py            # BPE 语料与训练服务
    _wmt19_tts_io.py             # store 状态和报告 IO
    _wmt19_tts_store.py          # 标准 store root 和 dataset factory
  tokenizers/
    _codec_bpe_artifact.py       # BPE 训练默认值和 artifact 命名
    codec_bpe.py                 # tokenizer artifact 定位与加载
scripts/
  *.py                           # argparse、服务调用和 summary 打印
jobs/
  *.sh                           # 环境激活、cd 和唯一 Python 调用
docs/
  datasets/                      # 公开数据集契约
  structure.md                   # workspace 总体边界和迁移顺序
```

同一个公开导入路径只能有一种物理实现。`zhuyin.env` 使用单文件 `env.py`，不同时保留
`env.py` 和 `env/` 包。公开模块最多到第二级，不在 `zhuyin` 或 `zhuyin.datasets`
聚合导出具体入口。

## Public Contracts

### Environment

`zhuyin.env` 是机器相关信息的唯一所有者，包括：

- `Location` 和各 location 的 `STATIC_HOME`、`DYNAMIC_HOME` 默认值。
- location 自动探测所需的文件系统 marker。
- `context()`、`location()`、`static_home()`、`dynamic_home()` 和 `datasets_home()`。

数据集 export、模型 checkpoint 等具体资产路径不属于 `zhuyin.env`；它们由对应 loader 或
资源模块根据 workspace root 解析。

环境函数按以下顺序纯解析路径：

1. 当前进程环境变量。
2. `LOCATION` 对应的默认值；使用默认 home 时发 `RuntimeWarning`。

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

WMT19 TTS 的公开入口不暴露物理 profile：

```python
dataset_root(root=None) -> Path
wmt19_tts(*, root=None, split="train")
wmt19_tts_codec(*, codec=Codec.LONGCAT, root=None, split="train")
wmt19_tts_dac(*, root=None, split="train")
wmt19_tts_stable(*, root=None, split="train")
wmt19_tts_unicodec(*, root=None, split="train")
```

参数规则：

- 显式 `root` 始终读取标准 store。
- `root=None` 时，loader 根据当前 location 和请求的逻辑视图选择默认物理来源。
- HZ 的 TTS 和 LongCat export 路径是 WMT19 模块的私有实现，不进入 `zhuyin.env`。
- DAC、Stable Codec 和 UniCodec 没有 HZ export，默认读取标准 store。
- 需要在 HZ 强制读取标准 store 时，显式传 `root=dataset_root()`。

`root` 始终表示标准 WMT19 TTS 数据集根目录，其下包含 `base/`、`longcat/`、`dac/`、`stable/`
等逻辑视图。它不表示具体 view 目录，也不覆盖 HZ export 路径。

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

脚本只负责：

1. 定义该入口独有的 argparse 参数。
2. 把参数转换成服务层的严格类型参数。
3. 调用服务并打印或写出顶层 summary。

重复的 store ready 检查、manifest 读取和 JSON/JSONL IO 放在 `_wmt19_tts_io.py`。业务规则
不放进 IO helper。新增结构优先使用 `TypedDict`、普通不可变值或简单 callable class；需要
新增 dataclass 时按仓库约定先确认。

## Jobs

job wrapper 只 source `workspace/jobs/env.sh`，进入项目根目录，并调用真实 Python 脚本。
所有 wrapper 的 Python 命令末尾保留 `"$@"`。机器路径优先通过 workspace profile 或环境
变量表达，不在通用 wrapper 中硬编码。

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
