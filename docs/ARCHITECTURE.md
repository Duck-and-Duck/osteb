# OSTE-Ariel Architectural Specification

## 1. Linear System Formulation
Transit depth extraction in the presence of opto-mechanical jitter is framed as an 8-parameter Generalized Least Squares (GLS) problem:

$$\mathbf{Y}_\lambda = \mathbf{M} \mathbf{W}_\lambda + \mathbf{\epsilon}_\lambda$$

Where the design matrix $\mathbf{M} \in \mathbb{R}^{T \times 8}$ contains:
- Column 0: Constant baseline offset ($c_0$)
- Column 1–5: FGS pointing jitter Taylor basis ($x_t, y_t, x_t^2, y_t^2, x_t y_t$)
- Column 6: Negative Mandel-Agol transit profile ($-T(t)$)
- Column 7: Linear stellar rotational modulation / spot trend ($t / T_{\text{dur}}$)

### Closed-Form Solution:
$$\mathbf{W}_\lambda = (\mathbf{M}^T \mathbf{M} + \lambda_{\text{reg}} \mathbf{I}_8)^{-1} \mathbf{M}^T \mathbf{Y}_\lambda$$

The transit depth $\hat{\delta}_\lambda$ corresponds strictly to index 6:
$$\hat{\delta}_\lambda = \mathbf{W}_{\lambda, 6}$$

### Covariance & Uncertainty:
$$\sigma^2(\hat{\delta}_\lambda) = s^2_\lambda \cdot \big[ (\mathbf{M}^T \mathbf{M})^{-1} \big]_{6,6}$$
Where $s^2_\lambda$ is the residual variance of the fit.

---

## 2. Zero-Host-Bottleneck CUDA BIC Engine
Rather than copying predictions to host memory for Python iterations, hypothesis testing is computed across all 512 batch candidates simultaneously using vectorized tensor reductions:

$$\Delta\text{BIC} = \chi^2_0 - \chi^2_1 - k \ln(N_{\text{channels}})$$

- Threshold: $\Delta\text{BIC} \ge 6.0$ ($>3\sigma$ evidence)
- Fully parallelized across GPU threads without branching divergence.