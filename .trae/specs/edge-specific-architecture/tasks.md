# Tasks: Edge-Specific Architecture

- [x] Task 1: 创建统一评估框架 `edge_recovery_metrics()`
  - [x] 1.1 实现 AUC（top-K 命中率曲线下面积）
  - [x] 1.2 实现 Spearman r（秩相关系数）
  - [x] 1.3 实现 avg_precision@K
  - [x] 1.4 实现数值稳定性：处理 NaN/Inf/常数向量

- [x] Task 2: 实现并运行 EdgeSpecificGNN 实验
  - [x] 2.1 内联定义 `EdgeSpecificGNN` 类（per-edge w_msg[N,N,FEAT,HID]）
  - [x] 2.2 内联定义 `EdgeGNNBaseline` 类（共享 w_msg）
  - [x] 2.3 编写 `run_edge_specific_gnn.py`：训练+评估+对比
  - [x] 2.4 运行实验，收集 metrics，判断是否达到 Spearman r ≥ 0.4 → **FAILED: r=0.15, delta=-0.34**

- [x] Task 3: 实现并运行 HyperEdgeGNN 实验
  - [x] 3.1 内联定义 `HyperEdgeGNN` 类（超网络从边嵌入生成 w_msg）
  - [x] 3.2 内联定义超网络（Linear→ReLU→Linear→reshape）
  - [x] 3.3 编写 `run_hypernet_gnn.py`：多嵌入维度对比 + OOD 测试
  - [x] 3.4 运行实验，收集 metrics → **FAILED: D=4 r=0.003, D=8 r=0.05, OOD r=-0.05**

- [x] Task 4: 实现并运行 AttnRouterGNN 实验
  - [x] 4.1 内联定义 `AttnRouterGNN` 类（多头注意力 + 边路由门控）
  - [x] 4.2 内联定义边路由 MLP（边嵌入 → K 维 softmax 门控）
  - [x] 4.3 编写 `run_attn_router_gnn.py`：多头对比 + 边路由分析
  - [x] 4.4 运行实验，收集 metrics → **FAILED: h=4 r=0.22, h=8 r=0.26, 所有route坍缩为均匀**

- [x] Task 5: 汇总三路结果并给出判断
  - [x] 5.1 读取三个实验的 metrics.json
  - [x] 5.2 打印对比表：AUC, Spearman r, avg_precision
  - [x] 5.3 判断内容级涌现是否达成 → **VERDICT: 内容级涌现不可达。所有边特定架构均劣于共享w_msg基线。**

# Task Dependencies
- Task 1 无依赖，可优先完成
- Task 2, 3, 4 均依赖 Task 1（使用统一评估函数），彼此之间独立可并行
- Task 5 依赖 Task 2, 3, 4 全部完成
