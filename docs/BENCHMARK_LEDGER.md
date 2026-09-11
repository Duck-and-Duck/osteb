# Benchmark Ledger & Hardware Profiling

All evaluations performed on an **NVIDIA GeForce RTX 3050 6GB Laptop GPU** running PyTorch 2.7.1+cu118 on Windows 11.

## Summary Performance Profile
- **Batch Processing Volume:** 512 Candidates in parallel.
- **VRAM Tensor Core Batch Time:** ~33.4 to 39.0 ms (Entire batch).
- **Net Amortized Latency per Target:** 65.30 to 76.50 microseconds (µs).
- **End-to-End Pipeline Latency:** 4.2 to 8.8 milliseconds (ms) including host ingestion and formatting.
- **Throughput Capacity:** 13,000 to 15,300 targets / second.
- **Mean Absolute Spectral Error:** 14.08 ppm (Physical floor is ~15 ppm).
- **95th Percentile Error (P95):** 34.93 ppm.
- **Measured Statistical Uncertainty (1σ):** ±17.4 ppm.