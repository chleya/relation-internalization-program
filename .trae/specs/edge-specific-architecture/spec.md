# Edge-Specific Architecture for Content-Level Emergence

## Why

28 轮实验证明：当前的 GNN 架构（共享 `w_msg` 变换 + 边权重标量）只能达到**架构级涌现**（message passing 有用，benefit=0.041，OOD 泛化 116%），但永远无法达到**内容级涌现**（学习哪些边更重要）。

根因分析：
1. **w_msg 共享**：所有源对象的变换相同 → 温度差信息在任意边上等价
2. **O(1/N) 梯度盲**：单个边参数贡献 ~1/N，有限差分梯度 ≈10⁻¹¹，float32 无法解析
3. **边权重标量太弱**：sigmoid(w_edge) 是一个标量门控，只决定"是否传递"，不区分"传递什么"
4. **四路尝试均失败**：L1 正则、Top-K 竞争、非对称世界、纯边嵌入（移除 w_msg）→ 所有权重坍缩到 0.5

## What Changes

- **新增** `EdgeSpecificGNN` 架构类：边特定消息变换 + 可学习边嵌入
- **新增** `HyperEdgeGNN` 架构类：超网络从边嵌入生成边特定变换权重
- **新增** `AttnRouterGNN` 架构类：多头注意力路由 + 边级 key/query
- **新增** 三个独立实验脚本：每种架构 vs 非边特定 baseline → 测边权重-真实系数相关性
- **新增** 统一评估框架：`edge_recovery_metrics()` 计算 AUC、Spearman r、avg_precision@K
- 所有现有脚本 **不改动**，新架构脚本独立运行

## Impact

- Affected specs: 无（全新能力线）
- Affected code: 
  - 新文件：`run_edge_specific_gnn.py`、`run_hypernet_gnn.py`、`run_attn_router_gnn.py`
  - 新结果目录：`results/edge_specific/`、`results/hypernet/`、`results/attn_router/`
  - 不改动 `gn_model.py`，新架构在各自的实验脚本中内联定义

---

## ADDED Requirements

### Requirement 1: EdgeSpecificGNN — Per-Edge Message Transform

系统 SHALL 提供一种 GNN 架构，其中每个有向边 (src→tgt) 拥有独立的可学习消息变换矩阵 `w_msg[s,t]`（形状 N_OBJ × N_OBJ × N_FEAT × HID），取代共享的 `w_msg`。

前向传播：
```
for each target i, source j (i≠j):
    msgs[i] += feats[j] @ w_msg[j,i]
```

#### Scenario: Edge-specific transform learns content-level structure
- **WHEN** 在有向因果图世界（pairwise random coefficients）上训练 EdgeSpecificGNN 80 epochs
- **AND** 固定 `w_self`, `w_out` 为共享参数，仅 `w_msg[s,t]` 为边特定
- **THEN** Spearman r(边权重范数, 真实系数) ≥ 0.4
- **AND** avg_precision@K 显著高于随机水平 (0.25)

#### Scenario: 消融对照 — 移除 w_msg 边特定性
- **WHEN** 同样世界 + 训练 80 epochs，但将所有 `w_msg[j,i]` 绑定为同一矩阵
- **THEN** Spearman r 回到 ≈0（基线水平）

---

### Requirement 2: HyperEdgeGNN — Hypernetwork-Generated Edge Weights

系统 SHALL 提供一种架构，用一个小型超网络从边嵌入生成边特定的消息变换权重，减少参数量爆炸：

```
edge_emb[j,i] ∈ R^D           # 每个边对的可学习嵌入
w_msg[j,i] = hypernet(edge_emb[j,i])  # 超网络输出 (N_FEAT, HID)
```

超网络结构：Linear(D→H) + ReLU + Linear(H→N_FEAT*HID) + reshape

#### Scenario: Hypernetwork generalizes to unseen edges
- **WHEN** 在 5-object 世界上训练 HyperEdgeGNN
- **AND** 在 6-object 世界（含未见边）上评估
- **THEN** OOD 边加权与真实系数的相关性 > 0

---

### Requirement 3: AttnRouterGNN — Multi-Head Attention with Per-Edge Routing

系统 SHALL 提供一种基于多头注意力的边特定路由架构：

```
对于每个注意头 h:
    Q_i^h = feats[i] @ W_q^h
    K_j^h = feats[j] @ W_k^h
    α_ij^h = softmax_j(Q_i^h · K_j^h / √d)

对于每个边 j→i:
    edge_route[j,i] = softmax(MLP(edge_emb[j,i]))  # K 维门控
    msgs[i] += Σ_h α_ij^h · edge_route[j,i]_h · (feats[j] @ W_v^h)
```

头的注意力权重 α 和边级路由 edge_route 共同决定每条边传递多少信息。

#### Scenario: Attention + routing disambiguates edges
- **WHEN** 在有向因果图世界上训练 AttnRouterGNN 80 epochs
- **THEN** 高系数边的平均 attention α 加权显著高于低系数边
- **AND** edge_route 在不同边间产生可区分的路由模式

---

### Requirement 4: 统一评估 — edge_recovery_metrics()

系统 SHALL 提供一个统一的边结构恢复评估函数，接受「边权重/重要性矩阵」和「真实系数矩阵」，返回：

- **AUC**：将非对角边按预测权重排序，计算 top-K 中高系数边命中率曲线下面积
- **Spearman r**：秩相关系数
- **avg_precision@K**：前 K=高系数边数 的位置上 precision 均值
- **全权重矩阵**：供可视化

---

### Requirement 5: 三路对比实验

系统 SHALL 在相同的 pairwise 随机系数世界上运行三路实验：

1. `run_edge_specific_gnn.py`：EdgeSpecificGNN vs 共享 `w_msg` baseline
2. `run_hypernet_gnn.py`：HyperEdgeGNN 不同嵌入维度 + OOD 泛化
3. `run_attn_router_gnn.py`：AttnRouterGNN 不同头数 + 边路由分析

每个实验输出到独立的结果子目录，包含 `metrics.json` 和完整终端日志。
