# CEN-TS

CEN-TS v6 研究将文本转换为可验证事件叙述后输入 TaTS 的预测流程。
当前活跃实现为 P2 文本变体与预测适配、P3A 事件抽取 pilot，以及 v6 P0/P1 检查与运行模块。
本次目录重构以 `feature/cen-event-extractor-pilot-v6` 的 `1a7d341` 为基线，不改变预测计算、训练协议、提示词、数据划分或既有实验产物。

## 目录与唯一入口

统一从仓库根目录运行 **`python scripts/run.py <stage> [参数]`**。
入口使用当前 `sys.executable` 启动对应的 `scripts/v6/` 阶段脚本，并将工作目录固定到仓库根目录。
可用阶段见 `python scripts/run.py --help`，阶段参数见例如 `python scripts/run.py p2-run --help`。

| 路径 | 职责 |
| --- | --- |
| `src/cen_ts/` | 唯一业务包：事件 schema、抽取、缓存、校验、叙述、文本变体和 v6 runtime |
| `vendor/tats/` | 固定版本 TaTS 预测源码及已有最小适配；内部保留上游 `models/layers/exp/utils` import 约定 |
| `scripts/` | 统一调度入口和阶段运行脚本 |
| `configs/v6/`、`tests/` | 当前配置、离线回归和本地模型集成检查 |
| `archive/legacy/` | 不再被 v6 引用的旧包、脚本、配置、测试和旧 README；不安装、不默认收集测试 |
| `third_party/TaTS/` | 原版只读对照，用于源码审计及 P1b/P2 等价性比较 |
| `prompts/`、`data/`、`artifacts/` | 提示词、原有数据及本地缓存，路径保留 |
| `results/`、`reports/` | 原有实验输出，路径与历史内容保留 |

从旧 `src/cen_tats/` 迁入的模块仅有 `runtime/` 和 `evaluation/forecast_metrics.py`，它们仍被 v6 入口或测试引用；
`src/cents/` 无 v6 引用，已完整归档。历史包不再放入 Python 搜索路径。

## 环境准备

使用 Python 3.10+。已有 TaTS/CUDA 环境可直接激活复用；新环境可运行：

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# PowerShell: .venv/Scripts/Activate.ps1
python -m pip install -e .
python -m pip install -e ".[tats,dev]"
```

基础安装仅安装 `cen_ts` 业务包及其基础依赖；`tats` extra 包含 torch、transformers、sktime 等预测依赖，
`dev` extra 提供 pytest。也可 `python -m pip install -r requirements.txt`。
GPU 验证应先安装适配本机驱动的 PyTorch CUDA 版本；当前已验证的 TaTS 环境使用 PyTorch 2.7.0+cu128。
本次不升级现有实验环境。

GPT-2 严格从本地加载。默认目录为仓库的 `models/gpt2/`；也可设置环境变量 `CEN_TS_GPT2_PATH`，
或向 P2 命令传 `--llm_path <本地目录>`。P0/P1 的 `tats.model_path` 支持配置覆盖，
P2 子进程默认使用当前解释器，也支持 `--python <解释器>`。
相对模型路径按仓库根目录解析，不依赖运行时工作目录。示例：

```powershell
$env:CEN_TS_GPT2_PATH = "models/gpt2"
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
```

`configs/v6/*.local.yaml` 是被忽略的本机覆盖配置，不纳入版本管理。
预检优先读取 `preflight.local.yaml`（如存在），否则读取示例配置。
缺失本地权重时，模型集成测试会明确 skip；已有但损坏或不完整的模型仍会报错。
P2 首批等价性检查需要 CUDA、本地 GPT-2 以及固定的原版 TaTS checkout。

## 固定上游与模型源码

TaTS 上游为 `https://github.com/iDEA-iSAIL-Lab-UIUC/TaTS`，
固定 commit 为 `a053503674c61c54d101d01d47c9d680288a7c9a`。
来源及全部模型 SHA-256 见 [vendor/tats/UPSTREAM.json](vendor/tats/UPSTREAM.json)，许可证保留在 vendor 目录。
`third_party/TaTS/` 不参与业务包安装，不修改其源码；缺失对照时自行建立固定 checkout：

```bash
git clone https://github.com/iDEA-iSAIL-Lab-UIUC/TaTS.git third_party/TaTS
git -C third_party/TaTS checkout --detach a053503674c61c54d101d01d47c9d680288a7c9a
python scripts/run.py p2-prepare
```

正常 clone 本仓库已包含 vendor 下全部模型源码。`p2-prepare` 仅从固定的本地上游 commit 恢复缺失原版模型，
保留所有已存在的文件，重复执行不会清空目录或覆盖业务代码、预测适配、实验清单。
其他 vendor 文件若缺失，应从本仓库版本管理恢复。
根目录 `/models/` 忽略本地权重；`vendor/tats/models/*.py` 则纳入版本管理。
模型文件保留迁移前字节；清单分别记录上游 Git blob 和本机原始换行格式的哈希，源码比较只归一化 CRLF。

## P2

生成文本变体（不会训练或调用 API）：

```bash
python scripts/run.py p2-variant --source_csv vendor/tats/data/Environment.csv --output_csv data/v6/p2/Environment_raw.csv --mode raw --text_column fact --manifest_path results/v6/p2/environment_raw_manifest.json
```

同一入口支持 `constant`、`shuffled` 模式；各自使用独立输出和 manifest 路径。
协议说明保留在 `configs/v6/p2/cen_tats_p2.yaml`，实际运行参数由阶段 CLI 指定。
raw 预测使用原始 CSV，保留官方 split 与数值窗口。

仅执行首批输入、forward、loss、gradient 等价性比较（不调用优化器步骤、不正式训练）：

```bash
python scripts/run.py p2-parity --output results/refactor/project-layout/p2_parity.json
```

比较使用真实上游和 vendor 实现，保留相同 seed、GPT-2 层数、tokenization、pooling、projection 和 prior mix。
缺少依赖会明确失败。可加 `--compare-saved-training` 只读比较既有训练指标和预测文件；默认不依赖这些大文件。
历史 `results/v6/p2/p1b_parity.json` 不会被此命令覆盖。

正式预测运行命令如下，**会启动训练**，本次结构重构不执行：

```bash
python scripts/run.py p2-run --mode raw --prior_weight 0.5 --train_epochs 5 --run_id p2_raw_new_run
```

## P3A

审计和采样是离线阶段：

```bash
python scripts/run.py p3a-audit --config configs/v6/p3a/event_extraction_pilot.example.yaml
python scripts/run.py p3a-sample --config configs/v6/p3a/event_extraction_pilot.example.yaml
```

需真实抽取时，通过本机环境设置 `OPENAI_API_KEY`、`OPENAI_BASE_URL`、
`CEN_EXTRACTOR_MODEL` 和预算变量（详见 `src/cen_ts/api_config.py`），再运行以下阶段。
**probe 和 cache miss 的 pilot 会调用 API**，本次重构不执行：

```bash
python scripts/run.py p3a-probe --config configs/v6/p3a/event_extraction_pilot.example.yaml
python scripts/run.py p3a-run --config configs/v6/p3a/event_extraction_pilot.example.yaml
python scripts/run.py p3a-cost --config configs/v6/p3a/event_extraction_pilot.example.yaml
python scripts/run.py p3a-report
```

提示词继续使用 `prompts/v6/event_extraction/`；训练内采样、缓存键、预算检查和验证/测试隔离保持原实现。
各阶段默认结果路径保留，重跑产物生成命令前应确认其输出范围。

## 验证

```bash
python -m pytest tests -q
python scripts/run.py p2-parity --output results/refactor/project-layout/p2_parity.json
python scripts/run.py layout-check
```

pytest 包括当前业务单元测试、布局/初始化测试、本地模型检查及历史产物回归。
历史产物断言不代表本次重新训练；当前计算等价性由单独的 P2 parity 命令验证。
本轮迁移清单和验证记录见 [reports/refactor/project-layout.md](reports/refactor/project-layout.md)。
