# B-Line Research Program

## 1. Research Question

Can a model form operational structure from continuous 2D dynamics without
language labels, and can that structure be verified by behavior, structural
intervention, and OOD generalization?

## 2. W/O1/O2/L Definitions

```text
W  = world process / fact-occurrence layer
O1 = trajectory schema layer
O2 = operational structure layer
L  = language overlay
```

Facts happen before language. Motion, occlusion, collision, force-field effects,
and trajectory deviations occur at W before they are named.

O1 is the pre-linguistic trajectory schema extracted from trajectory fragments:
continuity, occlusion persistence, motion tendency, contact changes motion,
anomalous velocity change, and critical transition points.

O2 is operational structure derived from O1: object identity, event boundaries,
relation locality, force-field regions, inspection-value regions, and critical
checkpoints.

L is not the target in v1. Language can later cover, express, transmit, audit,
and reshape operational structure, but language fluency is not evidence of
operational structure.

## 3. Why Language Is Not The Target

This B-line tests `W -> O1 -> O2`, not `L -> L`. A text system can describe
events without carrying the operational structure that makes those events usable
for intervention, inspection, or OOD action.

## 4. Why Accuracy Is Insufficient

Prediction accuracy can come from smoothness, memory, or local visual continuity.
Tracking accuracy can come from color cues. Inspection accuracy can come from
saliency. None of those alone proves operational structure.

## 5. Why Structural Intervention Is Required

A candidate structure should be causally involved in behavior. If masking,
swapping, or perturbing a claimed structure has no local and predictable effect,
the structure is not evidence. If perturbation causes global chaos, the structure
is not locally operational.

## 6. Difference From A-Line Symbolic Relation Internalization

The A-line studies explicit relation agents, symbolic gates, audits, edits, and
inspection policies. This B-line removes language and symbolic relation tables
from v1 and asks whether pre-linguistic operational structure can emerge from
continuous dynamics.

## 7. Why Field-First Hybrid Substrate

A pure slot model assumes objects too early. A pure sequence model is difficult
to intervene on. A pure pixel predictor can pass short-term prediction without
structure. A field-first substrate stays closer to continuous W while exposing
velocity, force response, uncertainty, inspection value, event maps, and
relation locality for intervention.

## 8. Branch Map

```text
pixel_predictor       frame prediction baseline
trajectory_memory     nearest-neighbor trajectory memory baseline
world_model           latent sequence baseline
slot_model            object-biased second-order candidate
field_model           field-form operational structure candidate
schema_model          field-first hybrid schema main candidate
```

## 9. First-Version Scope Control

v1 uses a 64x64 2D toy world with two balls, occlusion/crossing,
collision/bounce, one hidden local force field, and one budgeted inspect action.
It has no 3D, no deformable bodies, no language labels, no explicit symbolic
rules, and no LLM.
