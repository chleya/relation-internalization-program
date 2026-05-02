# B6.4 Codex Start Prompt

```text
Current task: start B6.4 Transfer and Anti-Overfit Generalization.

Do not implement B7.
Do not claim real-world risk intelligence.
Do not claim robotics capability.
Do not claim safety certification.
Do not claim construction-site autonomy or engineering deployment.
Do not merge main.
Do not modify PR #2.
Do not rewrite B6/B6.1/B6.2/B6.3/B6.3.1 historical results.

Project:
relation-internalization-program / prelinguistic-operational-structure-test

Stage:
B6.4 Transfer and Anti-Overfit Generalization

Core question:
Do the operational structures discovered in B6.3.1 transfer under toy-to-toy remapping, or are they overfit to the current PLOS splits?

Context:
B6.3.1 supports only toy PLOS diagnostic evidence for stronger wrong-trace pressure, targeted feedback/history necessity, targeted delayed credit buffer necessity, and synthetic hidden-indirect exploration/outcome-history diagnostics.

B6.3.1 does not prove:
- solved robust trace repair
- general feedback/history necessity
- general delayed credit solved
- real-world causal discovery
- real-world risk intelligence
- robotics capability
- safety certification
- deployable engineering control

B6.4 must test transfer, not add a new real-world setting.

Do not add:
- real robots
- real construction-site deployment
- safety certification
- 3D environments
- large language models
- language tasks
- B7

B6.4 should add toy-to-toy remapping diagnostics:
- dynamics remapping
- visual remapping
- risk cue remapping
- delay profile remapping
- indirect path remapping
- mask visibility remapping

Evaluation setup:
- train/config on original PLOS
- evaluate on remapped PLOS
- keep the environment toy-scale
- preserve clean model_input / evaluator_ground_truth / oracle_baseline_view separation
- preserve leakage guards and poisoned-evaluator invariance checks
- report split-level and baseline-level results, not just aggregate means

Compare:
- b63_1_policy
- state_only
- mask_only
- trace_only
- random
- always_abstain
- conservative_uncertainty
- oracle

Required outputs:
- transfer_score
- transfer_drop
- anti_overfit_score
- remap_generalization_score
- cue_remap_sensitivity
- dynamics_remap_sensitivity
- delay_remap_sensitivity
- indirect_path_remap_sensitivity
- baseline_transfer_gap
- oracle_gap

Suggested new files:
- src/b6_4_transfer/
- src/run_b6_4_transfer.py
- src/run_b6_4_result_review.py
- src/visualize_b6_4.py
- configs/b6_4_transfer.yaml
- tests/test_b6_4_*.py
- results/b6_4_transfer_summary.csv
- results/b6_4_transfer_records.csv
- results/b6_4_transfer_metrics.json
- results/b6_4_result_review.json
- figures/b6_4_transfer_plot.png
- reports/B6_4_TRANSFER_GENERALIZATION_REPORT.md
- reports/B6_4_RESULT_REVIEW.md

Minimum diagnostics:
1. Dynamics remap:
   Change toy latent transition dynamics while preserving the high-level operational relation.

2. Visual remap:
   Change visual appearance, cue position, color, saliency, or region encoding without changing latent operational structure.

3. Risk cue remap:
   Change how risk/irreversibility/cost cues appear in public state and mask-visible fields.

4. Delay profile remap:
   Change delayed outcome timing and delay signatures.

5. Indirect path remap:
   Change which indirect paths are valid and how they are revealed through exploration/history.

6. Mask visibility remap:
   Vary public mask visibility so explicit operational cues do not trivially transfer.

Metrics must distinguish:
- true transfer from oracle/evaluator leakage
- transfer from state-only shortcut
- transfer from mask-only shortcut
- transfer from conservative abstention
- transfer from candidate-search fallback
- transfer from score caps

Claim boundary:
Even if B6.4 passes, only claim toy-to-toy operational structure transfer evidence.

Do not claim:
- general language-free cognition
- real-world risk intelligence
- robotics capability
- safety certification
- construction-site autonomy
- deployable engineering control

Validation to run:
cd prelinguistic-operational-structure-test
python -m src.run_b6_4_transfer --config configs/b6_4_transfer.yaml --seed 0
python -m src.run_b6_4_result_review
pytest -q
python -m src.visualize_b6_4 --summary results/b6_4_transfer_summary.csv
git diff --check

Final response must include:
1. Files changed.
2. Tests run.
3. Transfer score and transfer drop.
4. Which remaps transferred.
5. Which remaps failed.
6. Baseline transfer gaps.
7. Leakage/metric integrity status.
8. Whether B6.4 is submit-ready as diagnostic branch.
9. Remaining blockers.
10. Conservative next step.
```
