# V2.1 Reviewer Hardening Report

## 1. Motivation

V2 needed hardening because fixed delay templates and limited predefined candidates could create false positives.

## 2. New Attacks

- variable delay
- false temporal shortcut
- multi-link delay edit
- temporal audit consistency
- anti-template generalization

## 3. Agents

surface_temporal, structural_memory_temporal, instant_relation_chain, delayed_relation_chain, learned_delayed_links

## 4. Metrics and Gates

```text
{'variable_delay_success': 0.8, 'false_delay_shortcut_rejection': 0.8, 'multi_link_delay_edit_success': 0.8, 'temporal_audit_consistency': 0.9, 'anti_template_generalization': 0.8}
```

## 5. Results

```text
                            seed  variable_delay_success  false_delay_shortcut_rejection  multi_link_delay_edit_success  temporal_audit_consistency  anti_template_generalization  hardening_gated_score
agent                                                                                                                                                                                                   
delayed_relation_chain       2.0                   1.000                           1.000                            1.0                         1.0                         1.000                    1.0
instant_relation_chain       2.0                   0.426                           0.380                            0.0                         0.0                         0.383                    0.0
learned_delayed_links        2.0                   1.000                           1.000                            1.0                         1.0                         1.000                    1.0
structural_memory_temporal   2.0                   0.985                           0.980                            0.0                         0.0                         0.975                    0.0
surface_temporal             2.0                   0.296                           0.291                            0.0                         0.0                         0.291                    0.0
```

## 6. Interpretation

Agents with non-zero hardening scores pass all V2.1 hardening gates.

## 7. Failure Cases

Failures should be read directly from hardening_summary.csv. A zero hardening score means at least one hardening gate failed.

## 8. Claim Boundary

Supported:
- delayed relation-chain diagnostic hardened against fixed-template shortcuts.

Unsupported:
- real geotechnical time-series modeling.
- unrestricted temporal relation discovery.
- deployment-ready engineering safety AI.
