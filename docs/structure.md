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
