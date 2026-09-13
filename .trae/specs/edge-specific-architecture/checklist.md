# Checklist: Edge-Specific Architecture

## 统一评估框架
- [x] `edge_recovery_metrics()` 返回 AUC、Spearman r、avg_precision@K
- [x] AUC 计算对边数 ≤2 的情况有合理的边界处理
- [ ] Spearman r 对完全均匀向量返回 0（而非 NaN） — **BUG: 基线uniform weights给出r=0.49而非0，std检测未触发**

## EdgeSpecificGNN
- [x] `EdgeSpecificGNN` 类中每个有向边有独立的 `w_msg[j,i]` 矩阵
- [x] `EdgeGNNBaseline` 类中只用一个共享 `w_msg` 矩阵
- [x] 两个模型在相同数据、相同超参数下训练
- [x] 实验结果写入 `results/edge_specific/metrics.json`
- [x] 终端输出包含训练 loss 曲线和边权重分析

## HyperEdgeGNN
- [x] `HyperEdgeGNN` 用超网络从边嵌入生成 w_msg
- [x] 超网络结构：Linear+ReLU+Linear+reshape
- [x] 至少测试 2 种嵌入维度（D=4, D=8）
- [x] OOD 测试：5-obj 训练 → 6-obj 评估
- [x] 实验结果写入 `results/hypernet/metrics.json`

## AttnRouterGNN
- [x] `AttnRouterGNN` 用多头注意力计算 α_ij
- [x] 边路由 MLP 输出 K 维 softmax 门控（K=头数）
- [ ] edge_route 在不同边间产生可观测的差异 — **FAILED: 所有route坍缩为均匀 [0.25,...]**

## 汇总
- [x] 三个实验均成功运行（无崩溃、无 NaN）
- [x] 汇总结果如下表
- [x] 根据汇总结果给出明确判断：**内容级涌现不可达**
