# B5 Self-Audit

## What This Improves

- Moves from one-shot action to closed-loop operation.
- Separates epistemic and pragmatic value.
- Tests trace update after inspection.
- Tests intervention after updated trace.
- Tests feedback revision after consequence.
- Adds inspect-vs-intervene timing.
- Adds planning budget.
- Adds inspect-always / intervene-immediately baselines.

## Remaining Weaknesses

- Still a 64x64 toy world.
- Only two-step closed loop.
- Inspection and intervention values are simulator-defined.
- Trace update remains evaluator-designed.
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
4. inspection does not reduce uncertainty
5. feedback revision does not change trace
6. planning budget is exceeded
7. oracle closed-loop score is low
8. random baseline passes
