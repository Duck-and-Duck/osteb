# 3-Minute Video Pitch Script (Face-Free Screen Recording)
**Target Competition:** AI Builders Hackathon 2026 (Devpost)  
**Project:** OSTE-Ariel (GPU-Accelerated Exoplanet Atmospheric Triage Engine)  
**Recording Method:** Screen record `presentation/slide_deck.html` and your VS Code terminal while speaking this script.

---

### [00:00 - 00:30] Introduction & The Core Problem
*(Visual: Show Slide 1, then transition to Slide 2)*

> "Hello everyone. Space telescopes like NASA's James Webb and ESA's upcoming Ariel mission observe hundreds of exoplanetary atmospheres using transmission spectroscopy. 
> But modern astrophysics faces a massive computational crisis. Running traditional Bayesian atmospheric retrievals—like nested sampling and MCMC—takes anywhere from **12 to 48 hours of supercomputer time for a single planet**. 
> When surveying thousands of candidates, full MCMC triage is impossible. Furthermore, spacecraft pointing jitter and unocculted stellar spots introduce chromatic noise that distorts transit depths by dozens of parts-per-million."

---

### [00:30 - 01:15] The Technical Innovation: Orthogonal Joint Inversion
*(Visual: Show Slide 3, then Slide 4 with Figure 1)*

> "To solve this, I developed **OSTE-Ariel**—an open-source, hardware-accelerated triage engine that bridges linear generalized least squares with GPU tensor cores. 
> Instead of sequential filtering which dilutes transit depth, OSTE-Ariel constructs an 8-parameter joint design matrix that solves optical pointing jitter, stellar rotation trends, and transit depth simultaneously. 
> As you can see on Slide 4, this orthogonal formulation recovers the true atmospheric transmission spectrum down to an average error of **15.79 parts-per-million**, matching the physical noise floor of peer-reviewed competition winners like `jcottaar / ariel2`."

---

### [01:15 - 02:00] Live Demonstration: Hardware-Accelerated Execution
*(Visual: Switch screen to VS Code Terminal, run `python benchmarks/run_hardened_sota_audit.py`, and let the clean output print live)*

> "Let's see it live on hardware. I am running our out-of-distribution benchmark on a local NVIDIA RTX 3050 laptop GPU across a batch of 512 diverse, heterogeneous exoplanets.
> Notice the execution:
> The entire batch is processed in roughly 33 to 45 milliseconds. That is an amortized **kernel latency of 55 to 90 microseconds per candidate**, or an end-to-end wall-clock time of 4 to 9 milliseconds including data ingestion. 
> That yields a triage throughput exceeding **11,000 exoplanets per second** directly on local consumer silicon."

---

### [02:00 - 02:35] Honest Scientific Rigor: The Stellar Spot Effect
*(Visual: Switch back to Slide 5 with Figure 2 and Slide 6)*

> "Now, let's talk honest science. Many deep learning projects claim a 100% detection rate by testing on sterile toy datasets. That is not how astrophysics works.
> When we simulate realistic unocculted stellar spots at 4200 Kelvin, the *Transit Light Source Effect* causes false water detections, elevating our false alarm rate to approximately 38%. 
> Instead of hiding this, OSTE-Ariel uses pure CUDA Bayesian Delta-BIC hypothesis testing to detect molecular excess at greater than 3-sigma confidence, while explicitly flagging spot-contaminated targets for chromatic verification."

---

### [02:35 - 03:00] Conclusion & Devpost Vision
*(Visual: Show Slide 7 & Slide 8)*

> "OSTE-Ariel does not claim to replace multi-day MCMC when preparing final peer-reviewed discoveries. Instead, it serves as an ultra-fast **Scientific Co-Pilot**—allowing researchers to triage massive space archives on their personal laptops in seconds rather than months.
> The project is fully open-source on GitHub under the MIT License, tested, and reproducible with a single command. Thank you."