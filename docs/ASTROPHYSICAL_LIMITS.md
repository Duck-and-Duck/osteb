# Astrophysical Scope & Methodological Disclaimers

## 1. The Transit Light Source Effect
Unocculted cool stellar spots ($T_{\text{spot}} < T_{\text{phot}}$) introduce chromatic slopes mimicking water absorption at 2.7 µm. 
OSTE-Ariel employs narrow-band local differential indexing ($\Delta\lambda \approx 0.4\text{ }\mu\text{m}$ flanking wings) to eliminate linear spot curvature. However, on stars with extreme spot filling fractions ($f_{\text{spot}} > 8\%$), non-linear photospheric contamination requires multi-epoch chromatic follow-up.

## 2. Cloud Deck Attenuation
High-altitude opaque photochemical hazes (e.g. GJ 1214b) truncate molecular absorption peaks. While OSTE-Ariel correctly flags flat atmospheres without false alarms, it cannot determine whether a featureless spectrum is caused by high-altitude clouds or a dry atmosphere without auxiliary Rayleigh scattering diagnostics.

## 3. Scope of Software
OSTE-Ariel is a **survey triage co-pilot**. It is designed to rapidly screen large catalogs (TESS, Kepler, Ariel, PLATO) on desktop hardware to prioritize targets for expensive space observatory scheduling.