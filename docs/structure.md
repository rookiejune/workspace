# Workspace Structure

`workspace` 的首要职责是把机器 profile、物理路径和已处理资产包装成稳定逻辑对象。
目录结构按调用层级组织，避免把业务规则、CLI 参数和 job 环境混在同一层。

## Current Target

```text
src/zhuyin/
  env/                    # workspace base env、profile 值和临时注入
  datasets/
    common_voice.py       # 数据集加载入口
    wmt19_tts.py          # WMT19 TTS 逻辑对象和 codec 视图加载入口
    _profiles.py          # 物理 profile 和默认机器映射
  tokenizers/
    codec_bpe.py          # 已准备 tokenizer artifact 的路径和加载入口
scripts/
  *.py                    # 只做 argparse、调用 src 服务、打印 summary
jobs/
  *.sh                    # 只做环境激活、cd、最终 Python 调用
docs/
  datasets/               # 公开数据集契约
  structure.md            # 目录和迁移约定
```

## Next Refactor

WMT19 TTS 的 prepare/filter/BPE 逻辑现在主要在 `scripts/` 里。后续迁移时按下面的
私有服务层拆开，脚本保持薄入口：

```text
src/zhuyin/datasets/
  _wmt19_tts_prepare.py   # base store 和 LongCat store 生成
  _wmt19_tts_filter.py    # translation/speech filter 规则和报告 payload
  _wmt19_tts_bpe.py       # semantic code corpus、BPE 训练和 artifact metadata
```

迁移顺序：

1. 先移动无 argparse 依赖的纯 helper，并保持脚本行为不变。
2. 再把 dataset factory 和 provider factory 移出脚本。
3. 最后让脚本只保留 `parse_args()`、`main()` 和 summary 打印。

加载入口仍保留在 `zhuyin.datasets.wmt19_tts`，不把 prepare/filter/BPE 作为公开 dataset
loader 导出。需要复用私有处理逻辑时从 `_wmt19_tts_*` 模块调用。

## Public Contracts

公开入口按对象职责分层：

- `zhuyin.env` 只负责解析当前机器 profile、`STATIC_HOME`、`DYNAMIC_HOME` 和临时环境变量注入。
- `zhuyin.datasets.*` 负责把物理数据源包装成稳定逻辑数据集。
- `zhuyin.tokenizers.*` 负责定位和加载已经准备好的 tokenizer artifact。

调用方不应依赖 `workspace` 内部目录名、临时文件名、脚本中间产物或第三方库的隐式副作用。
新增公开入口时，应同步补充对应模块文档，说明输入、返回对象、默认路径和失败边界。

### Environment Contract

`zhuyin.env.context()` 是进入 workspace 路径环境的标准边界。它会根据当前环境和显式
override 解析：

```text
LOCATION
STATIC_HOME
DYNAMIC_HOME
```

解析顺序：

1. 显式传入 `context(LOCATION=..., STATIC_HOME=..., DYNAMIC_HOME=...)`。
2. 当前进程已有环境变量。
3. `LOCATION` 对应的默认 profile 路径；使用默认路径时发 `RuntimeWarning`。

`static_home()`、`dynamic_home()` 和 `datasets_home()` 只读取当前进程环境。调用这些函数前，
调用方必须已经设置对应环境变量，或处在 `with zhuyin.env.context():` 中。这样可以避免在
普通函数调用里悄悄修改进程环境。

常规数据集入口应在内部使用 `zhuyin.env.context()` 解析默认路径；脚本、job 和 notebook
可以显式包一层 context 来固定本次运行的机器 profile。第三方缓存变量，例如 `HF_HOME`、
`BPE_CACHE_DIR` 和模型专用 checkpoint root，不属于 `zhuyin.env` 的自动注入范围，由具体
脚本或 job 按需设置。
