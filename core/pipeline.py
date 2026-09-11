import torch
import torch.nn as nn
from .inversion import RobustJointArielInversionEngine
from .retrieval import evaluate_batch_bic_spot_aware_gpu

class OSTEArielPipeline(nn.Module):
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        super().__init__()
        self.device = torch.device(device)
        self.inversion_engine = RobustJointArielInversionEngine().to(self.device)

    def process_spectral_batch(self, raw_flux, fgs_basis, transit_template, spot_trend, wavelengths_gpu):
        pred_mu, pred_sigma = self.inversion_engine(raw_flux, fgs_basis, transit_template, spot_trend)
        det_h2o, det_ch4, bic_h, bic_c = evaluate_batch_bic_spot_aware_gpu(pred_mu, pred_sigma, wavelengths_gpu)
        return pred_mu, pred_sigma, det_h2o, det_ch4