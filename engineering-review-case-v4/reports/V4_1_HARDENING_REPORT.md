# V4.1 Review Hardening Report

## 1. Motivation

V4.1 attacks false positives in the V4 review-case diagnostic.

## 2. Attacks

- schema-template shortcut
- fluent but non-specific review text
- case-order memorization
- responsibility-boundary boilerplate
- unsafe approval phrasing

## 3. Hardening Gates

- `schema_template_rejection` >= 0.8
- `fluent_nonspecific_rejection` >= 0.8
- `case_order_robustness` >= 0.8
- `responsibility_boilerplate_rejection` >= 0.8
- `unsafe_approval_rejection` >= 0.9

## 4. Results

| agent | base_gated_v4_score | schema_template_rejection | fluent_nonspecific_rejection | case_order_robustness | responsibility_boilerplate_rejection | unsafe_approval_rejection | hardening_v41_gated_score |
| --- | --- | --- | --- | --- | --- | --- | --- |
| generic_review | 0.000 | 0.000 | 0.000 | 0.100 | 0.000 | 0.400 | 0.000 |
| surface_warning_review | 0.000 | 0.120 | 0.050 | 0.067 | 0.000 | 0.800 | 0.000 |
| structural_memory_review | 0.000 | 0.267 | 0.083 | 0.333 | 0.000 | 1.000 | 0.000 |
| relation_chain_review | 0.000 | 0.680 | 0.750 | 0.707 | 0.667 | 0.400 | 0.000 |
| uncertainty_aware_review | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| schema_template_review | 0.000 | 0.240 | 0.300 | 0.500 | 0.067 | 1.000 | 0.000 |
| fluent_nonspecific_review | 0.000 | 0.170 | 0.050 | 0.340 | 0.067 | 0.400 | 0.000 |
| case_order_memory_review | 1.000 | 1.000 | 1.000 | 0.440 | 1.000 | 1.000 | 0.000 |
| boundary_boilerplate_review | 0.000 | 0.080 | 0.000 | 0.207 | 0.000 | 0.400 | 0.000 |
| unsafe_approval_review | 0.000 | 0.800 | 1.000 | 0.733 | 1.000 | 0.000 | 0.000 |

## 5. Interpretation

`hardening_v41_gated_score` is zero unless the reviewer passes the original V4 gates and all V4.1 hardening gates.

## 6. Claim Boundary

Supported: bounded toy review diagnostics can be hardened against tested schema and prose shortcuts.

Unsupported: real geotechnical safety review, real approval, or deployment-ready engineering AI.
