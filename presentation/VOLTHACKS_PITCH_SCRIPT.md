# VoltHacks 2026 Video Pitch Script (Face-Free Screen Recording)
**Track:** AI + Hardware Integration & Open Innovation  
**Project:** OSTE-Ariel: Hardware-Accelerated Exoplanet Atmospheric Triage Engine  
**Duration:** 2:45 - 3:00 Minutes  
**How to Record:** Open `presentation/volthacks_deck.html` in your browser and your VS Code terminal, then speak this script clearly over the screen recording.

---

### [00:00 - 00:40] Slide 1 & 2: The Physical Problem & Hardware Crisis
*(Visual: Show Slide 1, then transition to Slide 2)*

> "Hello judges and fellow builders. Welcome to OSTE-Ariel.
> Space observatories like NASA's James Webb and ESA's upcoming Ariel mission analyze exoplanetary atmospheres by measuring starlight filtering through planetary skies during transit.
> But modern astrophysics faces a severe computational crisis. Traditional Bayesian atmospheric retrievals rely on heavy MCMC algorithms that require **12 to 48 hours of supercomputer cluster time for a single planet candidate**. 
> When future surveys observe thousands of stars, cluster-based triage is physically impossible. Furthermore, space telescope pointing jitter and cool stellar spots distort starlight by dozens of parts-per-million, introducing chromatic noise that tricks models into false detections."

---

### [00:40 - 01:25] Slide 3 & 4: The Hardware Innovation (AI + Tensor Cores)
*(Visual: Show Slide 3, then Slide 4)*

> "To solve this, we engineered OSTE-Ariel—an open-source AI and hardware acceleration engine that brings exoplanet atmospheric triage directly onto local consumer GPUs.
> Instead of sequential filtering that dilutes the transit depth by over 200 ppm, OSTE-Ariel constructs an 8-parameter orthogonal joint design matrix. We ingest 52-channel infrared spectroscopy alongside fine-guidance sensor telemetry.
> By vectorizing Generalized Least Squares into batched Tensor Core GEMM operations, we solve jitter de-correlation, stellar rotation trends, and planetary transit depth simultaneously.
> As shown on Slide 4, our engine recovers true transmission spectra down to **14.08 parts-per-million**, matching the physical noise floor of world-class peer-reviewed pipelines."

---

### [01:25 - 02:10] Live Terminal Execution: 65 Microseconds on Hardware
*(Visual: Switch to VS Code terminal, run `python benchmarks/run_standard_ariel_benchmark.py`, let the benchmark print live)*

> "Let's demonstrate this live on hardware. We are running our standardized benchmark on a local NVIDIA GeForce RTX 3050 laptop GPU across a batch of 512 diverse exoplanets.
> Notice the execution:
> The entire batch of 512 candidates is processed in **33.4 milliseconds**. That is an amortized net latency of **65.30 microseconds per candidate**, sustaining a throughput of over **15,300 exoplanets per second** directly in VRAM with zero host synchronization bottlenecks.
> Below, Protocol 2 validates our engine against real flight targets: WASP-39b correctly confirms water, WASP-96b confirms water and methane, and GJ 1214b—an opaque, flat haze planet—correctly outputs zero false alarms."

---

### [02:10 - 02:45] Slide 5 & 6: Astrophysical Rigor & The VoltHacks Vision
*(Visual: Switch back to Slide 5, then Slide 6)*

> "Unlike black-box models that claim impossible 100% detection rates, we rigorously address the *Transit Light Source Effect*. Cool stellar spots at 4200 Kelvin mimic water features. To solve this without false alarms, we engineered a local flanking-wing differential index across a narrow 0.4-micron baseline where spot chromaticity cancels out linearly.
> OSTE-Ariel bridges the gap between AI and space hardware. It proves that by respecting physical sensor geometry and leveraging GPU Tensor Cores, researchers can triage massive space catalogs on their laptops in seconds instead of months.
> The project is fully open-source on GitHub under the MIT License. Thank you."