import torch
import torch.nn as nn
from .inversion import JointArielInversionEngine
from .retrieval import evaluate_single_planet_bic

class OSTEArielPipeline(nn.Module):
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        super().__init__()
        self.device = torch.device(device)
        self.inversion_engine = JointArielInversionEngine().to(self.device)

    def process_spectral_batch(self, raw_flux, fgs_basis, transit_template, wavelengths):
        with torch.no_grad():
            pred_mu, pred_sigma = self.inversion_engine(raw_flux, fgs_basis, transit_template)
            
        mu_np = pred_mu.cpu().numpy()
        sig_np = pred_sigma.cpu().numpy()
        B = mu_np.shape[0]
        
        decisions = []
        for b in range(B):
            d_h2o, d_ch4, bic_h, bic_c = evaluate_single_planet_bic(mu_np[b], sig_np[b], wavelengths)
            decisions.append({
                "H2O_Detected": d_h2o,
                "CH4_Detected": d_ch4,
                "Delta_BIC_H2O": bic_h,
                "Delta_BIC_CH4": bic_c
            })
            
        return pred_mu, pred_sigma, decisions
