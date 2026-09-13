# 项目审视与下一阶段计划 Spec

## Why

项目经历了从 V1 到 V6 的诊断/治理壳体阶段，以及 R1/R2 核心 Agent 线和 B 线 PLOS-Test 的并行推进。当前需要统一理解项目现状，并基于已有的长期路线图和冻结理论，主动制定接下去的执行计划。

## What Changes

- 汇总项目核心命题：关系内化（Relation Internalization），即系统是否暴露可用的内部关系（可转移、可反事实、可编辑、可审计、可指导行动）
- 梳理已完成的阶段：V1-V6（已冻结的诊断/治理壳体） + R1/R2（当前核心 Agent 线） + B 线 PLOS-Test（B5 闭环已完成，B6 风险约束正在进行中） + LLM 侧车（已冻结为负面基线） + 论文草稿线
- 明确当前瓶颈：两条主线各有不同瓶颈
  - **R 线瓶颈**：R2 已有 partial-observability baseline，但 `uncertainty_discovery_agent` 得分 0.982 可能来自 inspect-overuse。R2.1 需要对其进行不确定性硬化。
  - **B 线瓶颈**：B5 闭环操作已实现，但需要进一步区分 epistemic（信息）价值和 pragmatic（干预）价值；当前的 trace 更新和反馈修正是否真正工作。
- 制定下一阶段任务：R2.1 不确定性硬化 + B6 风险约束闭环 + 论文主线整理

## Impact

- Affected specs: 无（此为首次整体审视与规划）
- Affected code: 
  - `relation-agent-r1/` — R 线核心，R2.1 不确定性硬化将在此展开
  - `prelinguistic-operational-structure-test/` — B 线 B6 风险约束闭环
  - `paper/` — 论文主线整理

## ADDED Requirements（新增规划）

### Requirement: R2.1 不确定性硬化

R2 的 `uncertainty_discovery_agent` 得分为 0.982，但可能存在 inspect 过度使用的问题。R2.1 需要对其硬化。

#### Scenario: inspect-overuse 检测
- **WHEN** 系统面对低风险的全观测状态
- **THEN** 不应触发不必要的 inspect，否则应受惩罚

#### Scenario: 无关缺失变量
- **WHEN** 缺失的变量与风险关系链无关
- **THEN** 不应触发 inspect 或 takeover

#### Scenario: benign noise
- **WHEN** 噪声不影响风险关系链
- **THEN** 不应被误判为需要 inspect 的高不确定性状态

#### Scenario: audit-label-only 负面控制
- **WHEN** Agent 仅输出审计标签而无具体关系链证据
- **THEN** 其不确定性审计得分应为零

#### Scenario: fake uncertainty shortcut
- **WHEN** 存在与关系链不确定性无关的虚假不确定性信号
- **THEN** Agent 不应依赖此信号触发 inspect

### Requirement: B6 风险约束闭环

B5 已实现了 epistemic-pragmatic 闭环（观察→检查/跳过→更新 trace→干预/跳过→观察结果→修订 trace），B6 需要在此基础上加入风险约束。

#### Scenario: 风险知晓的行动选择
- **WHEN** 干预行动有不同风险等级
- **THEN** 系统应能区分高风险和低风险干预，并在不确定性高时避开高风险行动

#### Scenario: 行动力掩码（actionability mask）
- **WHEN** 某些行动不可逆或成本极高
- **THEN** 系统应在决定前识别这些约束并纳入决策

#### Scenario: 规划预算约束
- **WHEN** inspect 和 intervene 共享有限预算
- **THEN** 系统应合理分配 epistemic 和 pragmatic 行动之间的预算

### Requirement: 论文主线整理

论文主线文件在 `paper/` 目录下已有多份草稿和规划文档，需要甄别最新状态并补充缺失内容。

#### Scenario: 论文状态梳理
- **WHEN** 查看 paper/ 目录
- **THEN** 应能识别哪些文档是当前活跃的，哪些是历史版本，并确定还需要补充的内容
