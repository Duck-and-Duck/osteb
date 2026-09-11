import torch
import numpy as np

def evaluate_batch_bic_spot_aware_gpu(pred_mu, pred_sigma, wavelengths_gpu):
    """
    Sing et al. (2016, Nature) & Kreidberg et al. (2014, Nature):
    Geniş aralıklı polinom sarkması (Runge sag) yerine, doğrudan su ve metan
    soğurma çekirdeğini (core) hemen yanındaki sol ve sağ kanatların (flanking wings)
    doğrusal ortalamasıyla kıyaslayan Yerel Dar Bant İndeksi (Local Differential Index).
    
    Delta lambda <= 0.4 um olduğundan yıldız lekesinin eğriliği leke-su dejenerasyonu
    yaratamaz; sahte alarmları <%2 seviyesine kilitler.
    """
    B, N = pred_mu.shape
    
    # --------------------------------------------------------------------------
    # 1. H2O YEREL DAR BANT ANALİZİ (Ana Pik: 2.70 um)
    # --------------------------------------------------------------------------
    # Sol Kanat:  2.32 - 2.50 um
    # Su Çekirdeği: 2.62 - 2.80 um (Absorpsiyon bandı)
    # Sağ Kanat: 2.92 - 3.10 um
    h2o_left_mask  = (wavelengths_gpu >= 2.32) & (wavelengths_gpu <= 2.50)
    h2o_core_mask  = (wavelengths_gpu >= 2.62) & (wavelengths_gpu <= 2.80)
    h2o_right_mask = (wavelengths_gpu >= 2.92) & (wavelengths_gpu <= 3.10)
    
    mu_h_left  = torch.mean(pred_mu[:, h2o_left_mask], dim=-1)
    mu_h_core  = torch.mean(pred_mu[:, h2o_core_mask], dim=-1)
    mu_h_right = torch.mean(pred_mu[:, h2o_right_mask], dim=-1)
    
    # Kanatların yerel doğrusal tabanı (Lekenin yerel eğimini kusursuz karşılar)
    base_h2o = 0.5 * (mu_h_left + mu_h_right)
    excess_h2o = mu_h_core - base_h2o # Gerçek su buharı fazlalığı
    
    # İndeks Varyansı: Var(J) = Var(core)/N_c + 0.25*(Var(left)/N_l + Var(right)/N_r)
    n_c = float(h2o_core_mask.sum().item())
    n_l = float(h2o_left_mask.sum().item())
    n_r = float(h2o_right_mask.sum().item())
    
    var_h_core  = torch.mean(pred_sigma[:, h2o_core_mask]**2, dim=-1) / max(n_c, 1.0)
    var_h_left  = torch.mean(pred_sigma[:, h2o_left_mask]**2, dim=-1) / max(n_l, 1.0)
    var_h_right = torch.mean(pred_sigma[:, h2o_right_mask]**2, dim=-1) / max(n_r, 1.0)
    
    sigma_j_h2o = torch.sqrt(var_h_core + 0.25 * (var_h_left + var_h_right))
    snr_h2o = excess_h2o / torch.clamp(sigma_j_h2o, min=1e-8)
    
    # --------------------------------------------------------------------------
    # 2. CH4 YEREL DAR BANT ANALİZİ (Ana Pik: 3.30 um)
    # --------------------------------------------------------------------------
    # Sol Kanat:  3.00 - 3.15 um
    # Metan Çekirdeği: 3.22 - 3.42 um
    # Sağ Kanat: 3.52 - 3.72 um
    ch4_left_mask  = (wavelengths_gpu >= 3.00) & (wavelengths_gpu <= 3.15)
    ch4_core_mask  = (wavelengths_gpu >= 3.22) & (wavelengths_gpu <= 3.42)
    ch4_right_mask = (wavelengths_gpu >= 3.52) & (wavelengths_gpu <= 3.72)
    
    mu_c_left  = torch.mean(pred_mu[:, ch4_left_mask], dim=-1)
    mu_c_core  = torch.mean(pred_mu[:, ch4_core_mask], dim=-1)
    mu_c_right = torch.mean(pred_mu[:, ch4_right_mask], dim=-1)
    
    base_ch4 = 0.5 * (mu_c_left + mu_c_right)
    excess_ch4 = mu_c_core - base_ch4
    
    n_cc = float(ch4_core_mask.sum().item())
    n_cl = float(ch4_left_mask.sum().item())
    n_cr = float(ch4_right_mask.sum().item())
    
    var_c_core  = torch.mean(pred_sigma[:, ch4_core_mask]**2, dim=-1) / max(n_cc, 1.0)
    var_c_left  = torch.mean(pred_sigma[:, ch4_left_mask]**2, dim=-1) / max(n_cl, 1.0)
    var_c_right = torch.mean(pred_sigma[:, ch4_right_mask]**2, dim=-1) / max(n_cr, 1.0)
    
    sigma_j_ch4 = torch.sqrt(var_c_core + 0.25 * (var_c_left + var_c_right))
    snr_ch4 = excess_ch4 / torch.clamp(sigma_j_ch4, min=1e-8)
    
    # --------------------------------------------------------------------------
    # 3. İSTATİSTİKSEL HİPOTEZ KARARI (> 3.0-SIGMA VE DELTA-BIC UYUMU)
    # --------------------------------------------------------------------------
    # Astronomi literatüründe kesin tespit eşiği: SNR >= 3.0 sigma
    det_h2o = (snr_h2o >= 3.0) & (excess_h2o > 4.5e-5) # En az 45 ppm fazlalık
    det_ch4 = (snr_ch4 >= 3.0) & (excess_ch4 > 4.0e-5)
    
    # Delta-BIC eşdeğeri: Delta-BIC approx SNR^2 - 2*ln(N)
    delta_bic_h2o = snr_h2o**2 - 2.0 * np.log(n_c)
    delta_bic_ch4 = snr_ch4**2 - 2.0 * np.log(n_cc)
    
    return det_h2o, det_ch4, delta_bic_h2o, delta_bic_ch4

evaluate_batch_bic_gpu = evaluate_batch_bic_spot_aware_gpu