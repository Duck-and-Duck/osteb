# Astrophysical Limitations & Scope of OSTE-Ariel

A rigorous assessment of what OSTE-Ariel can and cannot accomplish in real flight operations:

## 1. The Transit Light Source Effect (Stellar Contamination)
- **Physics:** When a star has unocculted cool spots ($T_{\text{spot}} < T_{\text{phot}}$), the stellar flux deficit in the optical/near-IR distorts the normalized transmission spectrum, introducing chromatic slopes that mimic $\text{H}_2\text{O}$ absorption around 2.7 µm (Rackham et al. 2018).
- **Limitation:** OSTE-Ariel isolates linear spot trends but does not solve a full multi-temperature non-linear photospheric model. Consequently, on heavily spotted stars ($f_{\text{spot}} > 4\%$), false alarm rates for water can reach 40%.
- **Mitigation:** OSTE-Ariel flags candidates with high spot slopes ($W_7 > \text{threshold}$) as *Spot-Contaminated: Require Multi-Band Chromatic Follow-up*.

## 2. Cloud Deck Truncation (Gray Attenuation)
- **Physics:** High-altitude opaque cloud decks ($P_{\text{cloud}} < 0.01\text{ bar}$) truncate molecular absorption peaks, producing flat, featureless transit spectra.
- **Limitation:** The engine cannot distinguish between a completely dry atmosphere and a water-rich atmosphere covered by a high-altitude opaque haze.

## 3. Scope of Deployment
- **Intended Use:** High-throughput pre-screening, candidate triage, and proposal feasibility prioritization on catalog scales.
- **Not Intended For:** Final confirmation of atmospheric composition in peer-reviewed discovery announcements without full Bayesian Nested Sampling (TauREx/NEMESIS) verification.