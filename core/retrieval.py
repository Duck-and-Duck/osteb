import numpy as np

def evaluate_single_planet_bic(spectrum_mu, spectrum_sigma, wavelengths):
    """
    Bayesyen Bilgi Kriteri (BIC) hipotez testi:
    Kontinuum taban çizgisine karşı moleküler Gauss piki log-olabilirlik farkı.
    """
    cont_mask = (wavelengths >= 2.00) & (wavelengths <= 2.20)
    baseline = np.median(spectrum_mu[cont_mask])
    
    # H2O BIC (2.55 - 2.90 um)
    h2o_mask = (wavelengths >= 2.55) & (wavelengths <= 2.90)
    y_h2o = spectrum_mu[h2o_mask]
    sig_h2o = spectrum_sigma[h2o_mask]
    chi2_0_h = np.sum(((y_h2o - baseline) / sig_h2o)**2)
    peak_h2o = np.max(y_h2o)
    m_h2o = baseline + (peak_h2o - baseline) * np.exp(-((wavelengths[h2o_mask] - 2.70)**2) / 0.030)
    chi2_1_h = np.sum(((y_h2o - m_h2o) / sig_h2o)**2)
    delta_bic_h2o = chi2_0_h - chi2_1_h - 2.0 * np.log(len(y_h2o))
    
    # CH4 BIC (3.15 - 3.45 um)
    ch4_mask = (wavelengths >= 3.15) & (wavelengths <= 3.45)
    y_ch4 = spectrum_mu[ch4_mask]
    sig_ch4 = spectrum_sigma[ch4_mask]
    chi2_0_c = np.sum(((y_ch4 - baseline) / sig_ch4)**2)
    peak_ch4 = np.max(y_ch4)
    m_ch4 = baseline + (peak_ch4 - baseline) * np.exp(-((wavelengths[ch4_mask] - 3.30)**2) / 0.025)
    chi2_1_c = np.sum(((y_ch4 - m_ch4) / sig_ch4)**2)
    delta_bic_ch4 = chi2_0_c - chi2_1_c - 2.0 * np.log(len(y_ch4))
    
    det_h2o = bool(delta_bic_h2o >= 6.0)
    det_ch4 = bool(delta_bic_ch4 >= 6.0)
    
    return det_h2o, det_ch4, delta_bic_h2o, delta_bic_ch4
