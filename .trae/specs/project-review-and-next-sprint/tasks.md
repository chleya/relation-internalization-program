# Tasks

## Phase 1: 项目审视与文档沉淀

- [x] Task 1: 补充 `CURRENT_SPRINT.md`，更新到最新状态（当前日期 2026-05-03）
  - [x] SubTask 1.1: 阅读 `relation-agent-r1/` 下所有报告和结果文件，确认 R2 当前实现细节
  - [x] SubTask 1.2: 更新 `CURRENT_SPRINT.md`，将 R2.1 从"Next"调整为"Done"，加入 R3 完成状态
  - [x] SubTask 1.3: 补充 B 线 PLOS-Test 最新状态（B1-B6.4）到 `CURRENT_SPRINT.md`

- [x] Task 2: 梳理 `paper/` 目录，识别论文当前进展和缺口
  - [x] SubTask 2.1: 阅读 paper/ 下所有 .md 文件，绘制论文写作状态地图
  - [x] SubTask 2.2: 标记活跃版本（MAIN_PAPER_DRAFT_V0.md 最完整）与规划文档
  - [x] SubTask 2.3: 输出论文缺口报告：缺少 B 线 PLOS-Test 结果集成，Figures 1-4 未生成

## Phase 2: R2.1 不确定性硬化 —— 已确认完成

- [x] Task 3: R2.1 不确定性硬化已在研究前完全实现
  - [x] SubTask 3.1: 架构已理解 — `partial_agents.py`、`partial_metrics.py`、`partial_env.py`、`run_r2_1_hardening.py`
  - [x] SubTask 3.2: inspect-overuse 惩罚已在 `partial_metrics.py` 的 `cost_adjusted_success` 和 `r2_1_gated_score` 中实现
  - [x] SubTask 3.3: irrelevant missing variable（NONCRITICAL_FIELDS）已在 `partial_env.py` 定义并通过 `noncritical_missing_no_inspect_rate` 测试
  - [x] SubTask 3.4: benign noise — 通过 noncritical missing cases + safe state cases 覆盖
  - [x] SubTask 3.5: audit-label-only 负面控制 — `missing_always_inspect` baseline 得分 0.000
  - [x] SubTask 3.6: fake uncertainty shortcut — 通过 `conflict_localization_accuracy` gate 覆盖
  - [x] SubTask 3.7: 配置 `configs/r2_1_hardening.yaml` 已存在
  - [x] SubTask 3.8: Runner `src/run_r2_1_hardening.py` 已存在
  - [x] SubTask 3.9: 测试 `tests/test_v41_unsafe_approval.py` 等覆盖相关 gate
  - [x] SubTask 3.10: 结果 `results/r2_1_summary.csv` 已生成
  - [x] SubTask 3.11: 可视化 `src/visualize_r2_1.py` 已存在，可生成 figures
  - [x] SubTask 3.12: 报告 `reports/R2_1_HARDENING_REPORT.md` + self-audit 已生成
  - **R2.1 结果**: `relation_specific_uncertainty_agent: 0.933`，`missing_always_inspect: 0.000`

## Phase 3: B 线 PLOS-Test 状态审视

- [x] Task 4: 审视 B 线 PLOS-Test 当前 B5/B6 实现状态
  - [x] SubTask 4.1: 理解 B6 架构 — `b6_risk_*.py` + `b6_hardening/` + `b6_2_hardening/` 目录
  - [x] SubTask 4.2: 评估 B6 状态 — B6.1-B6.4 硬化已完成，combined remap 0.675 是当前瓶颈
  - [x] SubTask 4.3: 下一步判断 — `B6_TO_B7_GATE_MEMO.md` 推荐 STAY_IN_B6_REFINEMENT，B6.4.3 为下一目标

## Phase 4: 论文写作推进

- [ ] Task 5: 论文缺口处理（基于审视结果）
  - [ ] SubTask 5.1: R 线结果已在 MAIN_PAPER_DRAFT_V0.md 中完整覆盖（确认无需修改）
  - [ ] SubTask 5.2: B 线 PLOS-Test 结果尚未集成到论文草稿（主要缺口）
  - [ ] SubTask 5.3: FALSE_POSITIVE_LADDER_TABLE.md 已包含 R2/R2.1/R3 数据（确认无需修改）
  - [ ] SubTask 5.4: Discussion 和 Conclusion 已在 MAIN_PAPER_DRAFT_V0.md 中完整覆盖

## Phase 5: 最终验证

- [ ] Task 6: 全项目回归测试
  - [ ] 运行 `relation-agent-r1/` 全部 test：`pytest -q`
  - [ ] 运行 `prelinguistic-operational-structure-test/` 全部 test：`pytest -q`
  - [ ] 运行其他已冻结项目 test：V4/V5/V6 等

# Task Dependencies

- Task 3 (R2.1 硬化) — **已完成，无需执行**
- Task 4 (B 线审视) — **已完成**
- Task 5 (论文缺口) — 主要缺口是将 B 线结果加入论文，建议后续 sprint 处理
- Task 6 (回归测试) — 正在执行
