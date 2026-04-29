# V3 Self-Audit

## What V3 improves

- Adds missing sensor observations.
- Adds delayed noisy observations.
- Adds takeover as an explicit action.
- Measures unsafe automation.
- Requires uncertain relation audit.

## Remaining weaknesses

- Still a toy diagnostic.
- Candidate physical links are predefined.
- Noise model is simple and synthetic.
- Takeover ground truth is rule-defined.
- No real geotechnical monitoring data.
- No deployment claim.

## False positive risks

- An agent may learn the uncertainty generator rather than robust engineering uncertainty.
- Audit text may still be formatted correctly without real-world domain understanding.
- Oracle delayed chain has hand-coded relation structure.

## Next after V3

Only after V3 is stable should the project consider a small V4 engineering review case, with no safety deployment claim.
