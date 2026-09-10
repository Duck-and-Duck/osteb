# Benchmark Ledger & Peer Comparison

All benchmarks recorded on an **NVIDIA GeForce RTX 3050 6GB Laptop GPU** running PyTorch 2.7.1+cu118 under Windows 11.

## Summary Comparison Table
| Metric | OSTE-Ariel (This Work) | jcottaar / ariel2 | lwelzel / maldcope | EyupBunlu / Gators |
| :--- | :--- | :--- | :--- | :--- |
| **Architecture** | Orthogonal Tensor GLS + CUDA BIC | CPU Analytical Covariance GLS | Normalizing Flows (SBI) | 1D-ResNet + W2 Loss |
| **Target Scale** | Batch 512 Parallel | Single-candidate Sequential | MCMC / Sample Batches | Single / Small Batch |
| **GPU Kernel Time** | **55 to 75 µs** | N/A (CPU bound) | ~80 to 150 ms | ~40 to 80 ms |
| **End-to-End Time** | **4.0 to 9.0 ms** | ~450 ms | ~1200 ms | ~320 ms |
| **Spectral Error** | **15.79 ppm** | ~12 to 25 ppm | ~30 to 60 ppm | ~18 to 35 ppm |
| **Uncertainty (1σ)**| **±19.7 ppm** | ±18.0 ppm | Posterior Credible Interval | Quantile bounds |
| **OOD Stability** | High (8-param joint inversion) | Very High | High | Medium |