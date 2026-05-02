# B6.4 Transfer and Anti-Overfit Generalization Plan

## Purpose

B6.4 should test whether the operational structures diagnosed through B6.3.1 survive toy-to-toy transfer.

The purpose is not to add a new capability or move toward real-world control. The purpose is to test anti-overfit generalization: whether trace repair, fallback risk inference, delayed credit assignment, and hidden indirect discovery remain partially functional when surface cues are remapped.

## Why B6.4

B6.3.1 sharpened targeted diagnostics:

- wrong_trace_no_public_state reduces the state_only shortcut
- feedback/history necessity is clearer in synthetic required splits
- delayed credit buffer necessity is clearer in delay5-required splits
- hidden indirect discovery works in synthetic exploration/outcome-history diagnostics

But B6.3.1 remains vulnerable to split-specific fitting. B6.4 should ask whether the same operational structure transfers when the toy environment changes while preserving the latent operational relation.

## Proposed Transfer Axes

1. Visual remapping:
   - change visual appearance of regions
   - remap saliency patterns
   - preserve latent trace/action structure

2. Dynamics remapping:
   - alter transition style
   - preserve causal relation class
   - test whether delayed credit still works

3. Risk cue remapping:
   - change risk marker encoding
   - hide or invert previous public cues
   - test fallback risk inference

4. Delay profile remapping:
   - change delay length distribution
   - introduce mixed delay profiles
   - test credit buffer transfer

5. Hidden indirect path remapping:
   - change indirect path appearance
   - remove direct candidate shortcuts
   - preserve outcome-history discoverability

## Candidate Evaluation

Compare:

- b63_1_policy
- state_only
- mask_only
- trace_only
- random
- always_abstain
- oracle

Core metrics:

- transfer_score
- transfer_drop_from_original
- wrong_trace_transfer_repair_score
- feedback_history_transfer_score
- delayed_credit_transfer_score
- hidden_indirect_transfer_score
- gain_over_state_only
- gain_over_mask_only
- gap_to_oracle
- leakage_count

## Interpretation Rules

If performance transfers while state_only/mask_only degrade, this supports stronger toy evidence for operational structure.

If performance collapses with cue remapping, B6.3.1 should be interpreted as split-specific diagnostic success rather than transferable structure.

If oracle remains high but b63_1_policy collapses, the transfer split is valid and the mechanism did not transfer.

## Claim Boundary

B6.4 would still be toy-to-toy transfer. It must not claim real-world risk intelligence, robotics capability, safety certification, construction-site autonomy, or engineering deployment.

The strongest allowed claim after a successful B6.4 would be:

```text
Some toy diagnostic operational structures transfer across controlled PLOS remappings.
```

It would still not prove language-free cognition solved.

