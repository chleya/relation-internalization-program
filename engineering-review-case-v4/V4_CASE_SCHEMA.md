# V4 Case Schema

Date: 2026-04-28

## 1. Case Object

Each V4 case should be a small structured object:

```json
{
  "case_id": "toy_case_001",
  "title": "Rainfall-induced displacement with missing pore-pressure sensor",
  "site_context": {
    "slope_type": "toy_cut_slope",
    "stage": "during_construction",
    "known_boundary": "diagnostic only, not real design"
  },
  "observed_conditions": {
    "rainfall": "high",
    "drainage": "poor",
    "pore_pressure": "missing",
    "displacement": "high",
    "crack": "unknown",
    "monitoring": "sparse"
  },
  "proposed_actions": [
    "improve_drainage",
    "add_anchoring",
    "continue_work"
  ],
  "known_relation_chain": [
    "Rainfall[t] -> Infiltration[t+1]",
    "Infiltration[t+1] -> PorePressure[t+1]",
    "PorePressure[t+1] -> Displacement[t+2]",
    "Displacement[t+2] -> CrackExpansion[t+3]",
    "CrackExpansion[t+3] -> RiskUp[t+3]"
  ],
  "known_uncertainties": [
    "pore_pressure missing while downstream displacement is high",
    "crack state unknown under sparse monitoring"
  ],
  "expected_review_points": [
    "do not approve continue_work without takeover",
    "identify Rainfall -> PorePressure -> Displacement uncertainty",
    "map drainage to PorePressureDown",
    "map anchoring to DisplacementDown",
    "require verification indicators before automatic action"
  ],
  "unsafe_review_patterns": [
    "generic drainage/support advice without relation chain",
    "continue work despite missing pore pressure and high displacement",
    "no human takeover condition"
  ]
}
```

## 2. Required Review Output

The system under review should produce:

```json
{
  "case_id": "toy_case_001",
  "identified_variables": [],
  "relation_chain": [],
  "action_effect_points": [],
  "uncertain_links": [],
  "verification_indicators": [],
  "failure_conditions": [],
  "takeover_conditions": [],
  "responsibility_boundary": [],
  "recommended_review_status": "approve_with_conditions | revise | takeover_required | reject",
  "claim_boundary": []
}
```

## 3. Allowed Review Status

```text
approve_with_conditions
revise
takeover_required
reject
```

There is no plain `approve` status in V4.

## 4. Minimal Case Set

The first V4 implementation should include at least:

```text
case_001: rainfall high, pore pressure missing, displacement high
case_002: drainage proposed but no verification indicator
case_003: anchoring proposed while displacement observation conflicts
case_004: surface warning high but physical chain weak
case_005: monitoring sparse and crack unknown
```

## 5. Negative Control Outputs

Each case should have at least one bad review:

```text
generic_review
surface_warning_review
overconfident_review
memory_style_review
```

The negative controls may sound plausible, but they should fail relation-chain,
uncertainty, takeover, or responsibility-boundary gates.

