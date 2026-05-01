# B5: Epistemic-Pragmatic Closed-Loop Operation  
## Codex 可执行任务书

```text
Project:
relation-internalization-program / prelinguistic-operational-structure-test

Stage:
B5 Epistemic-Pragmatic Closed-Loop Operation

Purpose:
B4.2 resolved the fixed-action-type shortcut and showed that private delayed trace can guide both intervention-region selection and differentiated action-type selection in the toy PLOS environment.

B5 must test the next bottleneck:
Can the system use private delayed trace in a two-step closed loop where it must decide when to inspect, update trace after inspection, intervene based on the updated trace, observe consequences, and revise trace?

Core question:
Can private delayed operational trace support a minimal epistemic-pragmatic closed loop?

Do not change:
- PLOS v1 claim
- B1.1 claim
- B2 claim
- B2.1 claim
- B2.1a claim
- B2.2 claim
- B2.3 claim
- B3 claim
- B3.1 claim
- B3.2 claim
- B4 claim
- B4.1 claim
- B4.2 claim
- historical scoring
- existing reports
- existing commands

Do not add:
- LLM
- language labels
- complex video world
- 3D
- real video
- robot control
- continuous control
- large action space
- real engineering deployment claims

Keep the 64x64 toy world.
Add B5 as a minimal two-step closed-loop diagnostic.
```

---

## 1. B5 position in the evidence ladder

B3 tested:

```text
where should I inspect?
```

B4 tested:

```text
where should I intervene?
```

B4.2 tested:

```text
which action type should I use?
```

B5 must test:

```text
should I inspect first, then intervene after updating trace?
or should I intervene immediately?
or should I do nothing?
```

B5 is not simply another action test.  
It is the first closed-loop test.

---

## 2. Core loop

B5 loop:

```text
observe
→ choose inspect / skip
→ update trace
→ choose intervention / skip
→ observe consequence
→ revise trace
```

Minimal two-step protocol:

```text
t0: observe partial state
t1: choose epistemic action: inspect or skip
t2: update trace from inspection result
t3: choose pragmatic action: intervene or skip
t4: observe consequence
t5: revise trace / record feedback
```

---

## 3. Epistemic vs pragmatic value

B5 must explicitly separate:

```text
epistemic value:
  value of information gained by inspection

pragmatic value:
  value of outcome improved by intervention
```

Use a simple value decomposition:

```text
closed_loop_value =
  epistemic_gain
  + pragmatic_gain
  - inspection_cost
  - intervention_cost
  - wrong_inspect_penalty
  - wrong_intervention_penalty
```

Do not collapse inspect and intervene into one scalar without recording both components.

---

## 4. New B5 tasks

B5 must include:

```text
1. inspect-vs-intervene timing
2. trace update after inspection
3. intervention after update
4. feedback correction after consequence
5. wrong-inspect penalty
6. wrong-intervene penalty
7. planning budget
8. epistemic/pragmatic value decomposition
9. random / saliency / short-horizon / inspect-always / intervene-immediately / oracle baselines
```

---

## 5. File modifications

Add:

```text
prelinguistic-operational-structure-test/
│
├── src/
│   ├── b5_closed_loop_env.py
│   ├── b5_epistemic_pragmatic_values.py
│   ├── b5_closed_loop_policy.py
│   ├── b5_trace_update.py
│   ├── b5_feedback_revision.py
│   ├── b5_closed_loop_metrics.py
│   ├── b5_closed_loop_baselines.py
│   ├── b5_planning_budget.py
│   ├── b5_outputs.py
│   ├── run_b5_closed_loop.py
│   └── visualize_b5.py
│
├── configs/
│   └── b5_closed_loop.yaml
│
├── tests/
│   ├── test_b5_closed_loop_env.py
│   ├── test_b5_epistemic_pragmatic_values.py
│   ├── test_b5_closed_loop_policy.py
│   ├── test_b5_trace_update.py
│   ├── test_b5_feedback_revision.py
│   ├── test_b5_baselines.py
│   ├── test_b5_planning_budget.py
│   └── test_b5_scoring.py
│
├── results/
│   ├── b5_closed_loop_summary.csv
│   ├── b5_closed_loop_records.csv
│   ├── b5_epistemic_pragmatic_values.csv
│   ├── b5_trace_update_results.csv
│   ├── b5_feedback_revision_results.csv
│   ├── b5_baseline_comparison.csv
│   └── b5_planning_budget_results.csv
│
├── figures/
│   ├── b5_closed_loop_scores.png
│   ├── b5_epistemic_vs_pragmatic.png
│   ├── b5_trace_update.png
│   ├── b5_intervention_after_update.png
│   ├── b5_feedback_revision.png
│   ├── b5_baseline_comparison.png
│   └── b5_planning_budget.png
│
└── reports/
    ├── B5_EPISTEMIC_PRAGMATIC_CLOSED_LOOP.md
    └── B5_CLOSED_LOOP_SELF_AUDIT.md
```

Allowed small modifications:

```text
src/models/recurrent_flow_checkpoint_model.py
src/models/field_memory_model.py
src/models/schema_memory_model.py
src/b3_inspection_policy.py
src/b4_intervention_policy.py
src/b42_action_type_policy.py
```

Only add:

```text
closed-loop trace state export
trace update hook
feedback revision hook
epistemic/pragmatic value provenance
planning budget accounting
```

Do not rewrite historical stages.

---

## 6. Config file

Add:

```text
configs/b5_closed_loop.yaml
```

Suggested content:

```yaml
base_config: configs/b42_action_type_disambiguation.yaml

target_models:
  - recurrent_flow_checkpoint_model
  - field_memory_model
  - schema_memory_model

baselines:
  - random_closed_loop_baseline
  - saliency_closed_loop_baseline
  - short_horizon_closed_loop_baseline
  - inspect_always_baseline
  - intervene_immediately_baseline
  - oracle_closed_loop_baseline

seeds: [0, 1, 2]

b5:
  n_train: 2000
  n_test: 500
  n_ood: 500

  frame_size: 64
  grid_size: 8
  past_frames: 8
  future_frames: 12

  inspect_budget: 1
  intervention_budget: 1
  planning_budget:
    max_candidate_inspections: 8
    max_candidate_interventions: 8
    max_rollout_evaluations: 16

  costs:
    inspection_cost: 0.10
    intervention_cost: 0.20
    wrong_inspect_penalty: 0.30
    wrong_intervention_penalty: 0.50

  tasks:
    inspect_vs_intervene_timing: true
    trace_update_after_inspection: true
    intervention_after_update: true
    feedback_correction: true
    wrong_inspect_penalty: true
    wrong_intervention_penalty: true
    planning_budget: true
    epistemic_pragmatic_split: true

  gates:
    inspect_timing_accuracy: 0.70
    epistemic_value_alignment: 0.70
    trace_update_accuracy: 0.70
    post_inspection_intervention_accuracy: 0.70
    pragmatic_value_alignment: 0.70
    feedback_revision_accuracy: 0.65

    closed_loop_gain_over_inspect_always: 0.15
    closed_loop_gain_over_intervene_immediately: 0.15
    closed_loop_gain_over_random: 0.20
    closed_loop_gain_over_saliency: 0.15
    closed_loop_gain_over_short_horizon: 0.15

    wrong_inspect_penalty_sensitivity: 0.20
    wrong_intervention_penalty_sensitivity: 0.20

    planning_budget_compliance: 1.00
    oracle_closed_loop_score: 0.95
    random_closed_loop_score_max: 0.25

  scoring:
    zero_if_trace_update_fails: true
    zero_if_intervention_after_update_fails: true
    zero_if_epistemic_pragmatic_not_separated: true
    zero_if_planning_budget_exceeded: true
```

---

## 7. B5 closed-loop environment

Add:

```text
src/b5_closed_loop_env.py
```

### 7.1 Episode types

B5 must create at least four episode types:

```text
1. inspect-needed:
   initial trace is ambiguous;
   inspection reveals key delayed trace information;
   intervention after inspection improves outcome.

2. intervene-now:
   trace is already sufficiently certain;
   inspection cost is unnecessary;
   immediate intervention is better.

3. do-not-act:
   intervention has low or negative pragmatic value;
   best decision is no intervention.

4. misleading-saliency:
   visually salient region is not epistemically or pragmatically valuable.
```

### 7.2 Core functions

```python
def make_b5_closed_loop_episode(
    config: dict,
    seed: int,
    episode_type: str,
) -> dict:
    """
    Generate one closed-loop episode.
    Must include partial observation, hidden trace uncertainty,
    possible inspection targets, possible interventions, and final outcome.
    """
```

```python
def apply_b5_inspection(
    episode: dict,
    inspect_region: int | None,
    config: dict,
) -> dict:
    """
    Apply inspection and return inspection observation.
    If inspect_region is None, return no new information.
    """
```

```python
def apply_b5_intervention(
    episode: dict,
    action: dict | None,
    updated_trace_state: dict,
    config: dict,
) -> dict:
    """
    Apply intervention after trace update and return consequence.
    """
```

```python
def observe_b5_consequence(
    episode: dict,
    intervened_state: dict,
    config: dict,
) -> dict:
    """
    Return post-intervention consequence used for feedback revision.
    """
```

### 7.3 Episode ground truth

```python
ground_truth = {
    "episode_type": str,
    "needs_inspection": bool,
    "best_inspect_region": int | None,
    "best_intervention_after_inspection": dict | None,
    "best_intervention_without_inspection": dict | None,
    "should_intervene_immediately": bool,
    "should_do_nothing": bool,
    "epistemic_values": dict[int, float],
    "pragmatic_values_before_inspection": dict,
    "pragmatic_values_after_inspection": dict,
    "oracle_closed_loop_plan": {
        "inspect_region": int | None,
        "intervention_action": dict | None,
    },
}
```

---

## 8. Epistemic-pragmatic values

Add:

```text
src/b5_epistemic_pragmatic_values.py
```

### 8.1 Core functions

```python
def compute_epistemic_value(
    episode: dict,
    inspect_region: int,
    config: dict,
) -> float:
    """
    Value of inspecting region as uncertainty reduction.
    """
```

```python
def compute_pragmatic_value(
    episode: dict,
    intervention_action: dict,
    trace_state: dict,
    config: dict,
) -> float:
    """
    Value of intervention action as outcome improvement.
    """
```

```python
def compute_closed_loop_value(
    epistemic_gain: float,
    pragmatic_gain: float,
    inspection_cost: float,
    intervention_cost: float,
    penalties: dict,
) -> float:
    """
    Closed-loop value = epistemic + pragmatic - costs - penalties.
    """
```

```python
def decompose_policy_value(
    episode: dict,
    plan: dict,
    config: dict,
) -> dict:
    """
    Return epistemic component, pragmatic component, costs, penalties, total value.
    """
```

### 8.2 Required outputs

```text
epistemic_gain
pragmatic_gain
inspection_cost
intervention_cost
wrong_inspect_penalty
wrong_intervention_penalty
closed_loop_value
```

---

## 9. Trace update after inspection

Add:

```text
src/b5_trace_update.py
```

### 9.1 Purpose

B5 must verify that inspection changes trace state, and that updated trace improves later intervention.

### 9.2 Core functions

```python
def initial_trace_state(
    model,
    episode: dict,
    config: dict,
) -> dict:
    """
    Extract initial private delayed trace state before inspection.
    """
```

```python
def update_trace_after_inspection(
    model,
    trace_state: dict,
    inspection_observation: dict,
    config: dict,
) -> dict:
    """
    Update private trace after inspection.
    """
```

```python
def evaluate_trace_update_accuracy(
    initial_trace: dict,
    updated_trace: dict,
    ground_truth: dict,
    config: dict,
) -> dict:
    """
    Measure whether updated trace moves closer to true delayed trace.
    """
```

### 9.3 Metrics

```text
trace_update_accuracy
trace_uncertainty_reduction
trace_target_shift_correctness
post_inspection_trace_alignment
```

Gates:

```text
trace_update_accuracy >= 0.70
trace_uncertainty_reduction >= 0.20
```

---

## 10. Closed-loop policy

Add:

```text
src/b5_closed_loop_policy.py
```

### 10.1 Core interface

```python
def closed_loop_policy(
    model,
    episode: dict,
    config: dict,
) -> dict:
    """
    Two-step policy:
      1. decide inspect_region or skip
      2. update trace after inspection
      3. decide intervention action or skip
      4. record feedback revision after consequence

    Must not use oracle closed-loop plan.
    """
```

Output:

```python
{
    "inspect_decision": {
        "inspect_region": int | None,
        "inspect_score": float,
        "epistemic_value_estimate": float,
        "policy_source": str,
    },
    "trace_before_inspection": dict,
    "inspection_observation": dict | None,
    "trace_after_inspection": dict,
    "intervention_decision": {
        "action": dict | None,
        "pragmatic_value_estimate": float,
        "policy_source": str,
    },
    "consequence": dict,
    "trace_after_feedback": dict,
    "value_decomposition": dict,
    "planning_budget_used": dict,
    "provenance": dict,
}
```

### 10.2 Required distinctions

Policy must explicitly distinguish:

```text
inspect because uncertain
intervene because trace is actionable
skip inspect because epistemic value < cost
skip intervention because pragmatic value < cost/risk
```

---

## 11. Feedback revision

Add:

```text
src/b5_feedback_revision.py
```

### 11.1 Purpose

B5 must check that consequence feedback revises trace, rather than the system only making a one-shot action.

### 11.2 Core functions

```python
def revise_trace_after_consequence(
    model,
    trace_state: dict,
    consequence: dict,
    config: dict,
) -> dict:
    """
    Revise trace after observing intervention consequence.
    """
```

```python
def evaluate_feedback_revision(
    trace_before_feedback: dict,
    trace_after_feedback: dict,
    consequence: dict,
    ground_truth: dict,
    config: dict,
) -> dict:
    """
    Measure whether feedback improves or corrects trace for future decisions.
    """
```

### 11.3 Metrics

```text
feedback_revision_accuracy
feedback_uncertainty_reduction
future_decision_improvement_after_feedback
```

Gate:

```text
feedback_revision_accuracy >= 0.65
```

---

## 12. Planning budget

Add:

```text
src/b5_planning_budget.py
```

### 12.1 Purpose

Prevent brute-force search.

### 12.2 Core functions

```python
def track_planning_budget(
    candidate_inspections_evaluated: int,
    candidate_interventions_evaluated: int,
    rollout_evaluations: int,
    config: dict,
) -> dict:
    """
    Track planning/search budget.
    """
```

```python
def planning_budget_compliant(
    budget_record: dict,
    config: dict,
) -> bool:
    """
    Return whether the policy stays within budget.
    """
```

### 12.3 Metrics

```text
candidate_inspections_evaluated
candidate_interventions_evaluated
rollout_evaluations
planning_budget_compliance
```

Gate:

```text
planning_budget_compliance == 1.00
```

---

## 13. Baselines

Add:

```text
src/b5_closed_loop_baselines.py
```

Baselines:

```python
def random_closed_loop_baseline(episode: dict, config: dict, seed: int) -> dict:
    ...

def saliency_closed_loop_baseline(episode: dict, config: dict) -> dict:
    ...

def short_horizon_closed_loop_baseline(episode: dict, config: dict) -> dict:
    ...

def inspect_always_baseline(episode: dict, config: dict) -> dict:
    ...

def intervene_immediately_baseline(episode: dict, config: dict) -> dict:
    ...

def oracle_closed_loop_baseline(episode: dict, config: dict) -> dict:
    ...
```

Baseline meanings:

```text
random:
  random inspect/intervene choices.

saliency:
  inspect/intervene at visually salient region.

short_horizon:
  choose based on immediate next-frame cues.

inspect_always:
  always inspect even when not worth cost.

intervene_immediately:
  skip inspection and intervene immediately.

oracle:
  upper bound using oracle closed-loop plan.
```

---

## 14. Metrics

Add:

```text
src/b5_closed_loop_metrics.py
```

Core metrics:

```python
def inspect_timing_accuracy(predicted: dict, ground_truth: dict) -> float:
    ...

def epistemic_value_alignment(policy_output: dict, ground_truth: dict) -> float:
    ...

def trace_update_accuracy(updated_trace: dict, ground_truth: dict) -> float:
    ...

def post_inspection_intervention_accuracy(policy_output: dict, ground_truth: dict) -> float:
    ...

def pragmatic_value_alignment(policy_output: dict, ground_truth: dict) -> float:
    ...

def feedback_revision_accuracy(policy_output: dict, ground_truth: dict) -> float:
    ...

def closed_loop_gain_over_baseline(model_score: float, baseline_score: float) -> float:
    ...

def wrong_inspect_penalty_sensitivity(correct_value: float, wrong_inspect_value: float) -> float:
    ...

def wrong_intervention_penalty_sensitivity(correct_value: float, wrong_intervention_value: float) -> float:
    ...

def b5_closed_loop_score(metrics: dict, gates: dict) -> float:
    """
    Return 0 if any core B5 gate fails.
    """
```

Core gates:

```text
inspect_timing_accuracy >= 0.70
epistemic_value_alignment >= 0.70
trace_update_accuracy >= 0.70
post_inspection_intervention_accuracy >= 0.70
pragmatic_value_alignment >= 0.70
feedback_revision_accuracy >= 0.65

closed_loop_gain_over_inspect_always >= 0.15
closed_loop_gain_over_intervene_immediately >= 0.15
closed_loop_gain_over_random >= 0.20
closed_loop_gain_over_saliency >= 0.15
closed_loop_gain_over_short_horizon >= 0.15

wrong_inspect_penalty_sensitivity >= 0.20
wrong_intervention_penalty_sensitivity >= 0.20

planning_budget_compliance == 1.00
oracle_closed_loop_score >= 0.95
random_closed_loop_score <= 0.25
```

If any core gate fails:

```text
b5_closed_loop_score = 0.0
```

---

## 15. Runner

Add:

```text
src/run_b5_closed_loop.py
```

Pseudo-code:

```python
from __future__ import annotations

import argparse
from pathlib import Path
import yaml

from .b5_closed_loop_env import make_b5_datasets
from .b5_closed_loop_policy import evaluate_closed_loop_policy
from .b5_closed_loop_baselines import evaluate_b5_baselines
from .b5_closed_loop_metrics import b5_closed_loop_score
from .b5_outputs import write_b5_outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/b5_closed_loop.yaml")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))

    summary, records = run_b5_closed_loop(config, seed=args.seed)
    write_b5_outputs(summary, records)

    best = max(row.get("b5_closed_loop_score", 0.0) for row in summary)
    print(f"best_b5_closed_loop_score={best:.3f}")


if __name__ == "__main__":
    main()
```

Core function:

```python
def run_b5_closed_loop(
    config: dict,
    seed: int = 0,
) -> tuple[list[dict], list[dict]]:
    """
    For each target model:
      1. generate closed-loop episodes
      2. compute epistemic/pragmatic oracle values
      3. run closed-loop policy
      4. inspect or skip
      5. update trace after inspection
      6. intervene or skip
      7. observe consequence
      8. revise trace after feedback
      9. compare baselines
      10. enforce planning budget
      11. compute B5 score
    """
```

---

## 16. Outputs

Generate:

```text
results/b5_closed_loop_summary.csv
results/b5_closed_loop_records.csv
results/b5_epistemic_pragmatic_values.csv
results/b5_trace_update_results.csv
results/b5_feedback_revision_results.csv
results/b5_baseline_comparison.csv
results/b5_planning_budget_results.csv
```

Summary fields:

```text
model
seed
inspect_timing_accuracy
epistemic_value_alignment
trace_update_accuracy
trace_uncertainty_reduction
post_inspection_intervention_accuracy
pragmatic_value_alignment
feedback_revision_accuracy
closed_loop_gain_over_inspect_always
closed_loop_gain_over_intervene_immediately
closed_loop_gain_over_random
closed_loop_gain_over_saliency
closed_loop_gain_over_short_horizon
wrong_inspect_penalty_sensitivity
wrong_intervention_penalty_sensitivity
planning_budget_compliance
oracle_closed_loop_score
random_closed_loop_score
b5_closed_loop_score
```

Records fields:

```text
model
seed
episode_id
episode_type
needs_inspection
predicted_inspect_region
oracle_inspect_region
inspect_skipped
epistemic_gain
inspection_cost
trace_before_region
trace_after_inspection_region
trace_update_correct
predicted_intervention_action_type
predicted_intervention_region
oracle_intervention_action_type
oracle_intervention_region
pragmatic_gain
intervention_cost
consequence_value
trace_after_feedback_region
feedback_revision_correct
closed_loop_value
baseline_name
baseline_value
planning_budget_used
gate_pass
note
```

---

## 17. Visualization

Add:

```text
src/visualize_b5.py
```

Run:

```bash
python -m src.visualize_b5 \
  --summary results/b5_closed_loop_summary.csv
```

Output:

```text
figures/b5_closed_loop_scores.png
figures/b5_epistemic_vs_pragmatic.png
figures/b5_trace_update.png
figures/b5_intervention_after_update.png
figures/b5_feedback_revision.png
figures/b5_baseline_comparison.png
figures/b5_planning_budget.png
```

Requirements:

```text
matplotlib only
no seaborn
one chart per figure
simple readable charts
```

---

## 18. Report template

Add:

```text
reports/B5_EPISTEMIC_PRAGMATIC_CLOSED_LOOP.md
```

Must contain:

```text
# B5 Epistemic-Pragmatic Closed-Loop Operation

## 1. Purpose
B5 tests whether B4.2 private delayed traces can support a two-step closed loop:
observe → inspect → update trace → intervene → observe consequence → revise trace.

## 2. Background
PLOS v1: short-horizon checkpoint candidate.
B1.1: delayed checkpoint failure.
B2: trace-bearing delayed checkpoint.
B2.1: trace hardening.
B2.1a: score degeneracy.
B2.2: shared selector problem.
B2.3: private selector reconstruction.
B3: trace-guided active inspection.
B3.1: inspection degeneracy audit.
B3.2: mechanism-disambiguating active inspection.
B4: trace-guided intervention.
B4.1: fixed action-type shortcut discovered.
B4.2: action-type disambiguation.
B5: epistemic-pragmatic closed-loop operation.

## 3. Closed-loop task
- observe
- inspect or skip
- update trace
- intervene or skip
- observe consequence
- revise trace

## 4. Epistemic vs pragmatic value
- epistemic value = information gain from inspection
- pragmatic value = outcome improvement from intervention

## 5. Baselines
- random
- saliency
- short-horizon
- inspect-always
- intervene-immediately
- oracle

## 6. Results
Tables:
- model summary
- value decomposition
- trace update
- intervention after update
- feedback revision
- baseline comparison
- planning budget

## 7. Interpretation
If B5 passes:
  private delayed trace supports minimal closed-loop operation in the toy PLOS world.
If B5 fails:
  current system supports one-shot inspect/intervention, but not closed-loop update and reuse.

## 8. Claim boundary
Do not claim real control.
Do not claim robotics capability.
Do not claim engineering deployment.
Do not claim human-like active inference.
Do not claim language-free cognition solved.
```

---

## 19. Self-audit template

Add:

```text
reports/B5_CLOSED_LOOP_SELF_AUDIT.md
```

Content:

```text
# B5 Self-Audit

## What This Improves
- Moves from one-shot action to closed-loop operation.
- Separates epistemic and pragmatic value.
- Tests trace update after inspection.
- Tests intervention after updated trace.
- Tests feedback revision after consequence.
- Adds inspect-vs-intervene timing.
- Adds planning budget.
- Adds no-inspect / intervene-immediately baselines.

## Remaining Weaknesses
- Still a 64x64 toy world.
- Only two-step closed loop.
- Inspection and intervention values are simulator-defined.
- Trace update may still be hand-designed.
- No real robot control.
- No real engineering environment.
- Passing does not prove general active intelligence.

## False Positive Risks
- Model may always inspect first.
- Model may always intervene immediately.
- Trace update may be a direct oracle-like patch.
- Feedback revision may be scripted.
- Epistemic/pragmatic values may leak evaluator assumptions.
- Planning budget may be too loose.
- Baselines may be too weak.

## Required Failure Checks
1. inspect-always baseline matches model
2. intervene-immediately baseline matches model
3. trace update does not improve intervention
4. feedback does not revise trace
5. wrong inspect is not penalized
6. wrong intervention is not penalized
7. planning budget is exceeded
8. oracle closed-loop score is low
9. random baseline passes
```

---

## 20. Tests

Add:

```text
tests/test_b5_closed_loop_env.py
tests/test_b5_epistemic_pragmatic_values.py
tests/test_b5_closed_loop_policy.py
tests/test_b5_trace_update.py
tests/test_b5_feedback_revision.py
tests/test_b5_baselines.py
tests/test_b5_planning_budget.py
tests/test_b5_scoring.py
```

### test_b5_closed_loop_env.py

Check:

```text
episode has needs_inspection
episode has oracle_closed_loop_plan
episode has epistemic_values
episode has pragmatic values before/after inspection
inspection can be applied
intervention can be applied after trace update
```

### test_b5_epistemic_pragmatic_values.py

Check:

```text
epistemic value finite
pragmatic value finite
closed_loop_value decomposes into epistemic + pragmatic - costs - penalties
```

### test_b5_closed_loop_policy.py

Check:

```text
policy returns inspect decision
policy returns intervention decision
policy records trace before/after inspection
policy records value decomposition
policy does not use oracle plan
```

### test_b5_trace_update.py

Check:

```text
trace update changes trace when inspection reveals new information
trace uncertainty reduction is computed
trace update accuracy in valid range
```

### test_b5_feedback_revision.py

Check:

```text
feedback revision returns valid trace
feedback revision accuracy computed
future decision improvement computed
```

### test_b5_baselines.py

Check:

```text
random baseline runs
saliency baseline runs
short-horizon baseline runs
inspect-always baseline runs
intervene-immediately baseline runs
oracle baseline runs
oracle >= random in synthetic sanity case
```

### test_b5_planning_budget.py

Check:

```text
budget record is produced
budget compliance detects exceeded budget
compliant policy passes
```

### test_b5_scoring.py

Check:

```text
trace update failure -> score 0
post-inspection intervention failure -> score 0
planning budget exceeded -> score 0
oracle low -> score 0
random high -> score 0
all gates pass -> score > 0
```

---

## 21. Run commands

```bash
cd prelinguistic-operational-structure-test

pytest -q

python -m src.run_b5_closed_loop \
  --config configs/b5_closed_loop.yaml \
  --seed 0

python -m src.visualize_b5 \
  --summary results/b5_closed_loop_summary.csv
```

Historical commands must still run:

```bash
python -m src.run_sweep --config configs/sweep.yaml
python -m src.run_hardening --config configs/sweep.yaml --seed 0
python -m src.run_b11_flow_hardening --config configs/b11_flow_hardening.yaml --seed 0
python -m src.run_b2_delayed_checkpoint --config configs/b2_delayed_checkpoint.yaml --seed 0
python -m src.run_b21_trace_hardening --config configs/b21_trace_hardening.yaml --seed 0
python -m src.run_b21a_degeneracy_audit --config configs/b21a_degeneracy_audit.yaml --seed 0
python -m src.run_b22_selector_disentanglement --config configs/b22_selector_disentanglement.yaml --seed 0
python -m src.run_b23_private_selector --config configs/b23_private_selector.yaml --seed 0
python -m src.run_b3_active_inspection --config configs/b3_active_inspection.yaml --seed 0
python -m src.run_b31_inspection_audit --config configs/b31_inspection_audit.yaml --seed 0
python -m src.run_b32_mechanism_inspection --config configs/b32_mechanism_inspection.yaml --seed 0
python -m src.run_b4_intervention --config configs/b4_intervention.yaml --seed 0
python -m src.run_b41_intervention_audit --config configs/b41_intervention_audit.yaml --seed 0
python -m src.run_b42_action_type_disambiguation --config configs/b42_action_type_disambiguation.yaml --seed 0
```

---

## 22. Failure criteria

### Failure A: inspect timing fails

```text
inspect_timing_accuracy < 0.70
```

Meaning:

```text
The system does not know when information is worth acquiring.
```

### Failure B: epistemic value alignment fails

```text
epistemic_value_alignment < 0.70
```

Meaning:

```text
The system does not inspect based on information value.
```

### Failure C: trace update fails

```text
trace_update_accuracy < 0.70
```

Meaning:

```text
Inspection does not update delayed trace correctly.
```

### Failure D: intervention after update fails

```text
post_inspection_intervention_accuracy < 0.70
```

Meaning:

```text
Updated trace does not improve intervention.
```

### Failure E: pragmatic value alignment fails

```text
pragmatic_value_alignment < 0.70
```

Meaning:

```text
Intervention is not aligned with outcome improvement.
```

### Failure F: feedback revision fails

```text
feedback_revision_accuracy < 0.65
```

Meaning:

```text
The system does not use consequence feedback to revise trace.
```

### Failure G: closed-loop not better than simpler policies

```text
closed_loop_gain_over_inspect_always < 0.15
or closed_loop_gain_over_intervene_immediately < 0.15
or closed_loop_gain_over_random < 0.20
```

Meaning:

```text
Closed-loop behavior is not meaningfully better than simpler strategies.
```

### Failure H: wrong inspect/intervention not penalized

```text
wrong_inspect_penalty_sensitivity < 0.20
or wrong_intervention_penalty_sensitivity < 0.20
```

Meaning:

```text
Task does not strongly require correct epistemic or pragmatic choices.
```

### Failure I: planning budget exceeded

```text
planning_budget_compliance < 1.00
```

Meaning:

```text
The system relies on excessive search instead of compact operational structure.
```

### Failure J: oracle or random sanity fails

```text
oracle_closed_loop_score < 0.95
or random_closed_loop_score > 0.25
```

Meaning:

```text
B5 task or scoring is degenerate.
```

---

## 23. Passing interpretation

If B5 passes, write only:

```text
B5 supports that, in the toy PLOS environment, private delayed operational trace can support a minimal epistemic-pragmatic closed loop: the system can decide when to inspect, update trace from inspection, intervene based on updated trace, observe consequence, and revise trace under budget.
```

Chinese:

```text
B5 表明，在 toy PLOS 环境中，private delayed operational trace 可以支持一个最小 epistemic-pragmatic 闭环：系统能决定何时检查，根据检查更新 trace，基于更新后的 trace 干预，观察后果，并在预算约束下修正 trace。
```

Do not claim:

```text
real-world control
robotic intelligence
engineering deployment readiness
human-like active inference
language-free cognition solved
general AGI
```

---

## 24. Failing interpretation

If B5 fails, write:

```text
B5 shows that current private delayed traces support one-shot inspection and intervention, but not yet closed-loop operational update. The next bottleneck is converting one-shot trace use into feedback-corrected trace reuse.
```

Chinese:

```text
B5 表明，当前 private delayed trace 可以支持一次性检查和干预，但尚不能支持闭环操作更新。下一瓶颈是把一次性 trace 使用推进为反馈校正后的 trace 复用。
```

---

## 25. Short Codex prompt

```text
Implement B5 Epistemic-Pragmatic Closed-Loop Operation.

Context:
B4.2 passed and resolved the fixed-action-type shortcut.
The next bottleneck is closed-loop trace update and epistemic-pragmatic action selection.

Goal:
Implement a minimal two-step closed-loop protocol:
observe → inspect/skip → update trace → intervene/skip → observe consequence → revise trace.

Do not change historical claims.
Do not expand the 64x64 toy world.
Do not add LLM or language labels.

Add:
1. inspect-vs-intervene timing
2. epistemic/pragmatic value decomposition
3. trace update after inspection
4. intervention after update
5. feedback correction after consequence
6. wrong-inspect/wrong-intervene penalties
7. planning budget
8. random/saliency/short-horizon/inspect-always/intervene-immediately/oracle baselines

Add files:
- src/b5_closed_loop_env.py
- src/b5_epistemic_pragmatic_values.py
- src/b5_closed_loop_policy.py
- src/b5_trace_update.py
- src/b5_feedback_revision.py
- src/b5_closed_loop_metrics.py
- src/b5_closed_loop_baselines.py
- src/b5_planning_budget.py
- src/b5_outputs.py
- src/run_b5_closed_loop.py
- src/visualize_b5.py
- configs/b5_closed_loop.yaml
- tests/test_b5_*.py
- reports/B5_EPISTEMIC_PRAGMATIC_CLOSED_LOOP.md
- reports/B5_CLOSED_LOOP_SELF_AUDIT.md

Outputs:
- results/b5_closed_loop_summary.csv
- results/b5_closed_loop_records.csv
- results/b5_epistemic_pragmatic_values.csv
- results/b5_trace_update_results.csv
- results/b5_feedback_revision_results.csv
- results/b5_baseline_comparison.csv
- results/b5_planning_budget_results.csv
- figures/b5_*.png

Core gates:
- inspect_timing_accuracy >= 0.70
- epistemic_value_alignment >= 0.70
- trace_update_accuracy >= 0.70
- post_inspection_intervention_accuracy >= 0.70
- pragmatic_value_alignment >= 0.70
- feedback_revision_accuracy >= 0.65
- closed_loop_gain_over_inspect_always >= 0.15
- closed_loop_gain_over_intervene_immediately >= 0.15
- closed_loop_gain_over_random >= 0.20
- closed_loop_gain_over_saliency >= 0.15
- closed_loop_gain_over_short_horizon >= 0.15
- wrong_inspect_penalty_sensitivity >= 0.20
- wrong_intervention_penalty_sensitivity >= 0.20
- planning_budget_compliance == 1.00
- oracle_closed_loop_score >= 0.95
- random_closed_loop_score <= 0.25

If any core gate fails:
b5_closed_loop_score = 0.0

Passing B5 means:
private delayed operational trace supports a minimal epistemic-pragmatic closed loop in the toy PLOS environment.

Failing B5 means:
current traces support one-shot inspection/intervention but not closed-loop update and reuse.
```
