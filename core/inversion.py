import torch
import torch.nn as nn

class JointArielInversionEngine(nn.Module):
    """
    jcottaar/ariel2 ve OSTE-Ariel ortak yaklaşımı:
    Teleskop işaretleme titreşimini (FGS 6 parametre) ve gezegen transit derinliğini
    (1 parametre) 7 sütunlu tek bir tasarım matrisinde M in R^{T x 7} eşzamanlı çözer.
    Transit profili optik jitter uzayına tam ortogonaldir (transit dilution imkansızdır).
    """
    def __init__(self, reg=1e-6):
        super().__init__()
        self.reg = reg

    def forward(self, raw_flux, fgs_basis, transit_template):
        # raw_flux:         (B, 52, T)
        # fgs_basis:        (B, T, 6)
        # transit_template: (B, 1, T) - Mandel-Agol U-şablonu
        B, C, T = raw_flux.shape
        with torch.amp.autocast('cuda', enabled=False):
            M = torch.cat([fgs_basis.float(), -transit_template.float().transpose(1, 2)], dim=-1) # (B, T, 7)
            Y = raw_flux.float().transpose(1, 2) # (B, T, 52)
            
            MTM = torch.bmm(M.transpose(1, 2), M)
            MTM += self.reg * torch.eye(7, device=raw_flux.device, dtype=torch.float32).unsqueeze(0)
            MTY = torch.bmm(M.transpose(1, 2), Y)
            
            weights = torch.linalg.solve(MTM, MTY)
            pred_mu = weights[:, 6, :].transpose(0, 1).transpose(0, 1) # (B, 52)
            
            model_fit = torch.bmm(M, weights)
            residuals = Y - model_fit
            res_var = torch.var(residuals, dim=1)
            
            inv_MTM = torch.linalg.inv(MTM)
            transit_cov_scale = inv_MTM[:, 6, 6].unsqueeze(-1)
            pred_sigma = torch.sqrt(torch.clamp(res_var * transit_cov_scale, min=1e-10))
            
        return pred_mu, pred_sigma
