# B6.3 Trace Repair Report

## Trace Repair Result
| condition | score | conflict detection | downgrade | repair attempt | inspect recovery | gain over state_only | gain over mask_only |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| wrong_trace | 0.753 | 1.000 | 1.000 | 1.000 | 1.000 | -0.247 | -0.206 |
| wrong_trace_state_ambiguous | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.247 | 0.310 |
| hide_public_state_cue | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.247 | 0.247 |
| missing_trace | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 | 0.247 | 0.247 |
| ambiguous_trace | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.247 | 0.247 |
| low_confidence_trace | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.247 | 0.247 |

## Interpretation
Trace repair is only structural evidence when it beats state_only/mask_only and degrades under trace/history/public-state ablations. If repair mainly follows public state cues, the claim remains narrow.
