# CEN-TS project layout refactor

已在 `refactor/project-layout` 完成代码结构和工程配置重构。基线为
`feature/cen-event-extractor-pilot-v6` 的 `1a7d341384d09dfec2de126acb116360e4a92a3b`；
只读查询远端确认该提交仍为 v6 分支最新 HEAD。开始时工作区干净，无待保护的未提交变更。
本轮改动暂存供审阅，未创建提交、未推送远端。

## 迁移与引用审计

| 原路径 | 当前路径 | 依据 |
| --- | --- | --- |
| `tats_cen/cen_ts/` | `src/cen_ts/` | P2/P3A 当前业务实现，原有业务文件内容保持不变 |
| `src/cen_tats/runtime/` | `src/cen_ts/runtime/` | v6 P0/P1 入口、预检及模型测试仍直接依赖 |
| `src/cen_tats/evaluation/forecast_metrics.py` | `src/cen_ts/evaluation/forecast_metrics.py` | P1 runtime 的直接依赖，无旧包的其他传递依赖 |
| `src/cen_tats/` 剩余模块、`src/cents/` | `archive/legacy/src/` | 无 v6 活跃引用；与历史入口、配置、测试一起归档 |
| `tats_cen/` 剩余预测代码及数据快照 | `vendor/tats/` | 保留原版预测实现和已有最小适配 |
| `scripts/v5/`、旧根级阶段脚本、v5/旧测试、非 v6 配置 | `archive/legacy/` 下对应目录 | 从活跃入口、安装发现和默认测试收集中移除 |
| 原根目录 `legacy/` | `archive/legacy/pre_v5/` | 保留历史源码 |
| P2 `20_create_tats_cen_fork.py` / `22_run_tats_cen.py` | `20_prepare_vendor.py` / `22_run_tats.py` | 初始化安全化、预测入口路径更新 |

完整的 241 个文件迁移映射及内容检查见
[`migration_audit.json`](../../results/refactor/project-layout/migration_audit.json)。
`results/`、`reports/`、`prompts/`、原有划分和实验数据路径保留；旧 README 归档保存。

## 工程修改

- 唯一推荐入口为 `python scripts/run.py <stage> [参数]`。调度器使用 `sys.executable`，
  工作目录固定到仓库根；P2 子进程可通过 `--python` 覆盖解释器。
- `cen_ts.paths` 统一项目根、vendor、固定上游及本地 GPT-2 路径。默认读取
  `CEN_TS_GPT2_PATH`，否则使用仓库 `models/gpt2/`；CLI 或配置可显式覆盖。
- `.gitignore` 将笼统的 `models/` 改为根级 `/models/`，纳入 10 个 TaTS 模型文件。
  本机这些文件原本齐全，只是未被跟踪；全部与固定上游源码一致。缺文件恢复能力已在临时目录测试。
- 上游固定为 `a053503674c61c54d101d01d47c9d680288a7c9a`。
  `vendor/tats/UPSTREAM.json` 记录每个模型的上游 Git blob SHA-256 和原始 checkout SHA-256。
  `.gitattributes` 保留现有模型字节、CRLF 和上游原有空白，源码审计仅归一化换行。
- 初始化只从固定的本地上游 commit 补缺失原版模型，使用独占创建；现有文件均保留。
  不删除目录，不覆盖业务代码、已有最小适配或原有实验清单。
- `pyproject.toml` 仅发现 `cen_ts` 及其子包；TaTS 依赖放入 `tats` extra，测试依赖放入 `dev` extra。
  去掉已归档流程的 OpenAI SDK/tabulate 依赖，将 torch 下界调整为已验证的 2.7，补全预测依赖。
- README 给出目录职责、环境准备、统一入口以及 P2/P3A 命令，区分离线检查和会训练/调用 API 的阶段。
- P2 parity 的默认输出改为独立重构验证目录，不再覆盖历史 `results/v6/p2/p1b_parity.json`；
  可选 `--compare-saved-training` 仅读取已有训练结果，默认不需要旧预测大文件。

## 验证结果

| 检查 | 结果 |
| --- | --- |
| 当前业务与工程测试 | **62 passed，0 failed，0 skipped**；见 [`pytest.xml`](../../results/refactor/project-layout/pytest.xml) |
| Editable install | 在仓库内新建继承现有 TaTS 依赖的 `.venv`，`python -m pip install -e . --no-build-isolation --no-index` 成功 |
| 安装范围 | distribution 顶层包仅 `cen_ts`；`cen_tats`、`cents` 均不可导入 |
| 依赖一致性 | `python -m pip check`：No broken requirements found |
| 初始化幂等性 | 实际连续执行两次，均保留全部已有模型；临时目录恢复及已有业务/适配文件保护测试通过 |
| P2 实时首批等价性 | 真实上游与 vendor 的数值输入、token、pooling、projection、组合输入、初始模型状态全部一致 |
| P2 forward / loss / gradient | 最大绝对差均为 **0.0**，`p2_parity=PASS`；见 [`p2_parity.json`](../../results/refactor/project-layout/p2_parity.json) |
| 迁移完整性 | 241 个映射目标均存在，无非预期源码变化 |
| 实验产物、提示词及划分保护 | 基线记录的 **594 个文件 SHA-256 全部未变**；见 [`baseline.json`](../../results/refactor/project-layout/baseline.json) |
| 模型与原版对照 | 10 个模型均被 Git 跟踪，固定上游源码哈希一致；`third_party/TaTS` HEAD 正确且 clean |
| 残留与语法 | 活跃旧包目录、旧包 import、硬编码机器路径和 Python 语法错误均为 0 |
| Git whitespace | `git diff --cached --check` 通过；固定上游模型保留原始空白 |

测试复用本机已有 Python 3.11 / PyTorch 2.7.0+cu128 / Transformers 5.13.1 / CUDA / 本地 GPT-2；
只创建项目内验证用虚拟环境，没有升级原有实验环境。P2 使用官方的六层 GPT-2 配置；
加载时未使用的第 7–12 层权重提示在两侧一致，与原有协议相同。

## 范围与剩余项

没有阻塞验收的未验证项。按要求未运行正式训练、付费 API、真实 P3A 抽取或多种子完整实验；
历史训练指标断言仅用于读取回归，不作为本次重新训练的证据。
本轮验证覆盖本机 Windows/CUDA 环境，未另行部署 Linux 环境。

归档代码、历史报告/manifest 中的旧包名及旧机器路径作为溯源保留；本机被忽略的
`*.local.yaml` 配置也保留，属于显式本机配置。它们不构成新的活跃包或源码中的硬编码依赖。
