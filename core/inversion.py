import torch
import torch.nn as nn


class RobustJointArielInversionEngine(nn.Module):
    """
    FGS Jitter + Transit Profili + Leke Eğim Bileşeni
    toplam 8 parametreli GLS çözücüsü.
    """

    def __init__(self, reg=1e-6):
        super().__init__()
        self.reg = reg

    def forward(self, raw_flux, fgs_basis, transit_template, spot_trend):
        # raw_flux:         (B, 52, T)
        # fgs_basis:        (B, T, 6)
        # transit_template: (B, 1, T)
        # spot_trend:       (B, 1, T)

        with torch.amp.autocast("cuda", enabled=False):
            M = torch.cat(
                [
                    fgs_basis.float(),
                    -transit_template.float().transpose(1, 2),
                    spot_trend.float().transpose(1, 2),
                ],
                dim=-1,
            )  # (B, T, 8)

            Y = raw_flux.float().transpose(1, 2)  # (B, T, 52)

            MTM = torch.bmm(M.transpose(1, 2), M)

            eye = torch.eye(
                8,
                device=raw_flux.device,
                dtype=torch.float32,
            ).unsqueeze(0)

            MTM = MTM + self.reg * eye

            MTY = torch.bmm(M.transpose(1, 2), Y)

            weights = torch.linalg.solve(MTM, MTY)  # (B, 8, 52)

            # Transit katsayısı = 7. sütun (index 6)
            pred_mu = weights[:, 6, :]  # (B, 52)

            model_fit = torch.bmm(M, weights)

            residual = Y - model_fit
            res_var = torch.var(residual, dim=1)

            inv_MTM = torch.linalg.inv(MTM)

            pred_sigma = torch.sqrt(
                torch.clamp(
                    res_var * inv_MTM[:, 6, 6].unsqueeze(-1),
                    min=1e-10,
                )
            )

        return pred_mu, pred_sigma


# Eski kodlarla geriye dönük uyumluluk
JointArielInversionEngine = RobustJointArielInversionEngine