# Manifest

Date: 2026-04-29

## Included

| Folder | Role | Status |
| --- | --- | --- |
| `relation-agent-r1` | R1/R2 core non-LLM relation-internalization agent | current core agent line, partial-observability stage implemented |
| `relation-internalization-test` | V1 static food-world relation table | frozen stage |
| `neural-relation-probe` | neural probe and extraction | frozen stage |
| `slope-relation-toy` | static slope relation-chain review | frozen stage |
| `temporal-slope-relation-toy` | V2/V2.1 delayed temporal relation diagnostic plus V3 uncertainty/takeover | frozen through V3.2 |
| `engineering-review-case-v4` | V4 bounded engineering-review-case diagnostic | frozen through V4.2 |
| `governance-shell-v5` | V5 toy governance shell for logs/gates/replay/responsibility | frozen through V5.1 |
| `multi-party-audit-v6` | V6 toy multi-party audit resolution | frozen through V6.1 |
| `RELATION_INTERNALIZATION_PROJECT_SUMMARY.md` | consolidated report | current |
| `RELATION_INTERNALIZATION_V1_V5_FINAL_REPORT.md` | V1-V5 final staged report | current |
| `RELATION_INTERNALIZATION_V1_V6_FINAL_REPORT.md` | V1-V6 final staged report | current |
| `LONG_TERM_ROADMAP.md` | long-term stage ladder | current |
| `MAINLINE_EXECUTION_PROTOCOL.md` | execution and adaptation rules | current |
| `CURRENT_SPRINT.md` | active sprint state | current |
| `PROJECT_EVOLUTION.md` | discussion-to-mainline evolution | current |
| `CLAIM_BOUNDARY.md` | supported and unsupported claims | current |
| `NEXT_MAINLINE_TASKS.md` | next implementation route, R2.1 | current |

## Excluded

Runtime caches were not copied:

```text
.pytest_cache
__pycache__
*.pyc
```

## Current Verification Snapshot

```text
relation-agent-r1: 17 passed
relation-internalization-test: 17 passed
neural-relation-probe: 7 passed
slope-relation-toy: 12 passed
temporal-slope-relation-toy: 25 passed
engineering-review-case-v4: 15 passed
governance-shell-v5: 11 passed
multi-party-audit-v6: 11 passed
```

## Current Research Boundary

Supported:

```text
The diagnostics distinguish prediction, memory, surface shortcuts, generic review text,
static relation chains, delayed temporal relation chains, uncertainty-aware takeover,
bounded review-case relation audit, governance-shell replay, and multi-party
audit resolution. R1 adds the current core agent claim: in a toy process world,
a non-LLM agent can actively explore, learn relation links, use them for action,
answer counterfactuals, and change behavior after internal relation edits. R1.1
hardens this against nuisance candidates, process-rule reversal, intervention
cost tradeoffs, candidate expansion, and no-exploration ablation. R1.2 replaces
the hand-written TRUE_LINK candidate table with transition-enumerated relation
candidates, adds unmarked nuisance tests, synthetic new-link discovery, and
adaptive low-coverage exploration. R2 adds missing/noisy observations,
inspection-before-action, unsafe automation checks, and relation-specific
uncertainty audit.
```

Not supported:

```text
real geotechnical modeling
unrestricted relation discovery
safety-calibrated deployment
```
