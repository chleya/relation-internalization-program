from __future__ import annotations

CONDITIONS = [
    "clean_reference",
    "visual_remap",
    "risk_cue_remap",
    "dynamics_remap",
    "delay_profile_remap",
    "indirect_path_remap",
    "mask_visibility_remap",
    "combined_remap",
]

POLICY_NAMES = [
    "b63_1_policy_reference",
    "b64_transfer_policy",
    "state_only",
    "mask_only",
    "trace_only",
    "random",
    "always_abstain",
    "conservative_uncertainty",
    "oracle",
]

REMAP_TYPES = {
    "clean_reference": "none",
    "visual_remap": "visual",
    "risk_cue_remap": "risk_cue",
    "dynamics_remap": "dynamics",
    "delay_profile_remap": "delay_profile",
    "indirect_path_remap": "indirect_path",
    "mask_visibility_remap": "mask_visibility",
    "combined_remap": "combined",
}
