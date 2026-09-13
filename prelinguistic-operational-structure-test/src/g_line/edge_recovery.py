from __future__ import annotations

import numpy as np


def edge_recovery_metrics(edge_weights, true_coef, n_obj, is_strong=None):
    if is_strong is None:
        is_strong = (true_coef >= 0.35).astype(np.int32)

    non_diag = ~np.eye(n_obj, dtype=bool)

    ew_flat = edge_weights[non_diag]
    tc_flat = true_coef[non_diag]
    strong_flat = is_strong[non_diag]

    n_edges = len(ew_flat)
    n_strong = int(np.sum(strong_flat))

    sort_idx = np.argsort(-ew_flat)
    sorted_strong = strong_flat[sort_idx]

    if n_strong > 0 and n_edges > 0:
        max_k = min(n_strong * 2, n_edges)
        ks = np.arange(1, max_k + 1, dtype=np.float64)
        tp_cumsum = np.cumsum(sorted_strong[:max_k].astype(np.float64))
        auc = float(np.mean(tp_cumsum / ks))
    else:
        auc = 0.0

    if n_edges < 2 or np.std(ew_flat) < 1e-12 or np.std(tc_flat) < 1e-12:
        spearman_r = 0.0
    else:
        order_ew = np.argsort(ew_flat)
        rank_ew = np.empty(n_edges, dtype=np.float64)
        rank_ew[order_ew] = np.arange(n_edges, dtype=np.float64)

        order_tc = np.argsort(tc_flat)
        rank_tc = np.empty(n_edges, dtype=np.float64)
        rank_tc[order_tc] = np.arange(n_edges, dtype=np.float64)

        rank_ew_centered = rank_ew - np.mean(rank_ew)
        rank_tc_centered = rank_tc - np.mean(rank_tc)

        numerator = np.sum(rank_ew_centered * rank_tc_centered)
        denominator = np.sqrt(np.sum(rank_ew_centered ** 2) * np.sum(rank_tc_centered ** 2))

        if denominator < 1e-12:
            spearman_r = 0.0
        else:
            spearman_r = float(numerator / denominator)

    if n_strong > 0:
        ks = np.arange(1, n_strong + 1, dtype=np.float64)
        tp_cumsum = np.cumsum(sorted_strong[:n_strong].astype(np.float64))
        avg_precision = float(np.mean(tp_cumsum / ks))
    else:
        avg_precision = 0.0

    return auc, spearman_r, avg_precision
