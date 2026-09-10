import torch
import numpy as np


def evaluate_batch_bic_gpu(pred_mu, pred_sigma, wavelengths_gpu):
    """
    pred_mu:       (B, 52)
    pred_sigma:    (B, 52)
    wavelengths_gpu: (52,)
    """

    B, N = pred_mu.shape

    cont_mask = (
        (wavelengths_gpu >= 2.00)
        & (wavelengths_gpu <= 2.20)
    )

    h2o_mask = (
        (wavelengths_gpu >= 2.55)
        & (wavelengths_gpu <= 2.90)
    )

    ch4_mask = (
        (wavelengths_gpu >= 3.15)
        & (wavelengths_gpu <= 3.45)
    )

    baseline = torch.mean(
        pred_mu[:, cont_mask],
        dim=-1,
        keepdim=True,
    )

    # H2O
    y_h2o = pred_mu[:, h2o_mask]
    s_h2o = torch.clamp(
        pred_sigma[:, h2o_mask],
        min=1e-10,
    )

    chi2_0_h = torch.sum(
        ((y_h2o - baseline) / s_h2o) ** 2,
        dim=-1,
    )

    peak_h2o = torch.max(
        y_h2o,
        dim=-1,
        keepdim=True,
    ).values

    gauss_h2o = torch.exp(
        -(
            (wavelengths_gpu[h2o_mask] - 2.70) ** 2
        ) / 0.030
    ).unsqueeze(0)

    model_h2o = (
        baseline
        + (peak_h2o - baseline) * gauss_h2o
    )

    chi2_1_h = torch.sum(
        ((y_h2o - model_h2o) / s_h2o) ** 2,
        dim=-1,
    )

    n_h = max(int(h2o_mask.sum().item()), 1)

    delta_bic_h2o = (
        chi2_0_h
        - chi2_1_h
        - 2.0 * np.log(n_h)
    )

    # CH4
    y_ch4 = pred_mu[:, ch4_mask]
    s_ch4 = torch.clamp(
        pred_sigma[:, ch4_mask],
        min=1e-10,
    )

    chi2_0_c = torch.sum(
        ((y_ch4 - baseline) / s_ch4) ** 2,
        dim=-1,
    )

    peak_ch4 = torch.max(
        y_ch4,
        dim=-1,
        keepdim=True,
    ).values

    gauss_ch4 = torch.exp(
        -(
            (wavelengths_gpu[ch4_mask] - 3.30) ** 2
        ) / 0.025
    ).unsqueeze(0)

    model_ch4 = (
        baseline
        + (peak_ch4 - baseline) * gauss_ch4
    )

    chi2_1_c = torch.sum(
        ((y_ch4 - model_ch4) / s_ch4) ** 2,
        dim=-1,
    )

    n_c = max(int(ch4_mask.sum().item()), 1)

    delta_bic_ch4 = (
        chi2_0_c
        - chi2_1_c
        - 2.0 * np.log(n_c)
    )

    det_h2o = delta_bic_h2o >= 6.0
    det_ch4 = delta_bic_ch4 >= 6.0

    return (
        det_h2o,
        det_ch4,
        delta_bic_h2o,
        delta_bic_ch4,
    )