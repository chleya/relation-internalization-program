# V2.1 Limitations

Date: 2026-04-28

## Main Limitations

- Candidate links are predefined.
- Delay values are limited to a small finite set.
- The generator is deterministic or near-deterministic.
- There is no real monitoring uncertainty model.
- There is no numerical slope mechanics.
- There is no real geotechnical data.
- There is no deployment claim.

## Specific Risk

`learned_delayed_links` passes V2.1, but it still learns within this fixed candidate set:

```text
Rainfall -> PorePressure
PorePressure -> Displacement
Displacement -> Crack
Drainage -> PorePressureDown
Anchoring -> DisplacementDown
```

Therefore, V2.1 does not demonstrate unrestricted temporal relation discovery.

## Reviewer Risk

The strongest remaining reviewer objection is:

```text
The learned agent may be learning the toy generator's candidate-link scoring pattern rather than a general temporal relation concept.
```

That objection is valid. V2.1 should be presented as a hardened toy diagnostic, not a general engineering model.
