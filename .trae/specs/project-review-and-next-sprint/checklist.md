# Checklist

## 项目理解

- [x] 确认对项目核心命题的理解正确：关系内化 vs 预测/记忆/捷径/通用审查文本/探针可读性
- [x] 确认对 V1-V6 各阶段定位的理解正确
- [x] 确认对 R1/R2 Agent 线定位的理解正确
- [x] 确认对 B 线 PLOS-Test W→O₁→O₂→L 框架的理解正确
- [x] 确认对 LLM 侧车负面基线定位的理解正确
- [x] 确认对当前瓶颈的判断正确：R 线已无 inspect-overuse 瓶颈(R2.1 已解决)，真正瓶颈已转移到 R4 学习化；B 线瓶颈为 combined remap 0.675

## 文档更新

- [x] CURRENT_SPRINT.md 已更新到最新状态（2026-05-03），R2.1/R3 标记为 Done
- [x] paper/ 目录已梳理，论文缺口已标记（B 线结果未集成，Figures 未生成）
- [x] 所有更新遵循 MAINLINE_EXECUTION_PROTOCOL.md 的决策规则

## R2.1 不确定性硬化

- [x] inspect-overuse 惩罚已对全观测低风险状态的误 inspect 生效（`cost_adjusted_success` 机制）
- [x] irrelevant missing variable 不会错误触发 inspect（`NONCRITICAL_FIELDS` + `noncritical_missing_no_inspect_rate = 1.0`）
- [x] benign noise 不会错误触发高不确定性判断（safe state cases 覆盖）
- [x] audit-label-only 负面控制的 gated 得分为 0.000（`missing_always_inspect: 0.000`）
- [x] fake uncertainty shortcut 不会误导 Agent（`conflict_localization_accuracy = 1.0`）
- [x] R2.1 的 relation_specific_uncertainty_agent gated 得分 0.933 ≥ 0.900
- [x] 所有 baseline 的 gated 得分 = 0.000（`missing_always_inspect: 0.000`）
- [x] 所有旧测试仍然通过（36 passed in relation-agent-r1）
- [x] R2.1 的 report/self-audit 已生成（`reports/R2_1_HARDENING_REPORT.md` + `R2_1_SELF_AUDIT.md`）

## B 线 PLOS-Test B6

- [x] B6 风险约束闭环的架构已理解（`b6_risk_*.py` + `b6_hardening/` + `b6_2_hardening/`）
- [x] B6 结果已评估：B6.4 combined remap 0.675，oracle_gap 0.325，未达 B7 门禁标准
- [x] B6 相关的 report/self-audit 已完整（B6_1-B6_4 result reviews + B6_TO_B7_GATE_MEMO）

## 论文

- [x] MAIN_PAPER_DRAFT_V0.md 已纳入最新 R 线和 R2.1/R3 结果
- [ ] MAIN_PAPER_DRAFT_V0.md 尚未纳入 B 线 PLOS-Test 结果（建议后续 sprint 处理）
- [x] FALSE_POSITIVE_LADDER_TABLE 已更新到 R3
- [x] 论文 Discussion 和 Conclusion 已完整覆盖
- [x] 论文 claim boundary 与 CLAIM_BOUNDARY.md 一致（均为 toy diagnostic 边界）

## 最终验证

- [x] 各项目 pytest 全部通过（469 tests across 5 projects）
  - relation-agent-r1: 36 passed
  - prelinguistic-operational-structure-test: 396 passed
  - engineering-review-case-v4: 15 passed
  - governance-shell-v5: 11 passed
  - multi-party-audit-v6: 11 passed
- [x] 各项目 gated 得分正常（agent 高，baseline 为零）
  - R1: relation_agent 0.972, baselines 0.000
  - R1.1: relation_agent 1.000, baselines 0.000
  - R1.2: discovery_relation_agent 1.000, baselines 0.000
  - R2: uncertainty_discovery_agent 0.982, baselines 0.000
  - R2.1: relation_specific_uncertainty_agent 0.933, missing_always_inspect 0.000
  - R3: active_inspection_agent 0.933, all baselines 0.000
- [x] 没有新引入的未处理失败
