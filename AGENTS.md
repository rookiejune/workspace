# Workspace

## 定位

- `workspace` 提供可复用的加载入口，用来直接拿到已经处理好的逻辑对象，包括但不限于数据集、模型和实验中反复使用的资源。
- 这里隐藏跨项目调用、路径查找、缓存位置和第三方库适配细节；调用方应只关心入口函数的参数和返回对象。
- `workspace` 依赖 `third_party/` 中的通用能力，不复制或内嵌 `third_party/` 的内部实现。

## 对外接口

- 所有入口都应当有可用默认值，主路径和缓存位置优先用环境变量覆盖。
- 外部可见模块最多到第二级，不做顶层聚合导出。

```python
# Do
from zhuyin.datasets.wmt19_tts import wmt19_tts

# Don't
from zhuyin.datasets import wmt19_tts
from zhuyin import wmt19_tts
```

- 新增入口时，优先打磨函数签名和返回类型；不要让调用方依赖内部目录结构、临时文件名或第三方对象的隐式副作用。
- 对外入口要明确失败原因。缺少数据、环境变量错误、远程资源不可用时直接抛出清晰异常，不用兼容逻辑静默降级。
- 返回对象应尽量是稳定的逻辑对象；如果必须暴露第三方类型，应在模块文档里说明边界和版本假设。
- 不在公开数据集入口暴露机器名或物理协议 profile；默认物理来源由 loader 根据 location 和逻辑视图私有选择，显式 `root` 表示标准 store。

## 目录约定

- `src/` 存加载入口和轻量适配逻辑，只负责提供已经处理好的具体逻辑对象、路径解析和稳定返回契约。
- `src/zhuyin/datasets/` 存数据集相关入口；新增数据集按对象或任务建子模块。
- `scripts/` 提供可复用的构建、过滤、格式转换和模型推理 workflow。WMT19 prepare 入口按 TTS、codec-view、BPE 三件事分开；同一件事只保留一个公开脚本入口，具体动作通过子命令区分；私有 helper 使用 `_` 前缀。脚本应可传参、模块化并复用 `src` 入口拿到具体对象，不写 location 专属路径、镜像或设备默认值。
- `jobs/` 按 location 存与 `scripts/` 对应的可提交任务。shell wrapper 只负责环境激活、机器相关变量和最终 Python 调用，Python 命令末尾保留 `"$@"`。
- `notebooks/` 存交互式调试，目录和命名尽量与 `src/` 对称。
    1. 不要通过sys.path.insert加载包，而是应该在环境里安装。
- 大文件、中间输出和临时检查结果不要放进该目录；调试输出统一放到顶层 `debug/`。

## 实现约定

- 每个对外模块都要有模块文档，说明它提供的能力、输入输出和边界，不只记录内部实现步骤。
- 加载入口只在本层处理加载、路径、缓存和对象组装；构建、过滤、物化和报告规则放在 `scripts/` 的 workflow/helper 中。
- 通用数据结构、读写协议和可复用算法优先放在 `third_party/`；仍然依赖本工作区对象命名、路径契约或资产布局的逻辑留在 `scripts/`。
- `location()`、`static_home()`、`dynamic_home()` 和 `datasets_home()` 纯解析并返回值，不修改 `os.environ`；`context()` 只用于确实依赖环境变量的第三方代码或脚本。
- `zhuyin.env` 负责公开解析机器 home 和探测 marker；当前只实现 Fudan，具体 location 常量放在 `src/zhuyin/_locations/` 对应文件中，数据集 export、模型 checkpoint 等具体资产路径归对应资源模块。
- 不在 `src` 中写大量训练、评测、构建或实验分支逻辑；这些应进入 `scripts/`、具体工程或实验目录。
- 类型提示要覆盖入口参数和返回值。需要避免重 import 时，用 `TYPE_CHECKING` 隔离。
- 发现不属于当前任务所有权的 `third_party/` 或其他工程 `src/` 问题时，先按顶层 `AGENTS.md` 给出推荐方案并确认，不直接越界修改。

## 调试与验证

- 先做能跑通的最小闭环，再补齐生产路径、远程路径和错误信息。
- 新增或修改加载入口时，至少用一个轻量脚本、测试或 notebook 验证默认参数能构造目标对象。
- 涉及本地 Python、虚拟环境、代理、缓存或临时文件约定时，先阅读顶层 `docs/dev-env.md`。
- 涉及复旦远程服务器、GPU、SSH、远程数据集或远程训练流程时，先阅读顶层 `docs/fdu-remote.md`。
- 远程 job 失败时，wrapper 应尽量保留现场并进入便于调试的 loop，而不是直接清理上下文。
