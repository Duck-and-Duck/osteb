import numpy as np

class RadiativeTransferSimulator:
    """
    Fiziksel atmosfer transmisyonu + Kromatik Yıldız Lekesi Kirliliği (Stellar Contamination)
    ve Planck Işınımı ile Dalga Boyuna Bağlı Sahte Su Sinyallerini simüle eder.
    """
    def __init__(self, n_channels=52, wl_min=1.95, wl_max=3.90):
        self.wavelengths = np.linspace(wl_min, wl_max, n_channels, dtype=np.float32)
        wl = self.wavelengths
        self.sigma_h2o = 1.2 * np.exp(-((wl - 2.70)**2)/0.035) + 0.4 * np.exp(-((wl - 1.95)**2)/0.02)
        self.sigma_ch4 = 0.9 * np.exp(-((wl - 3.30)**2)/0.025) + 0.3 * np.exp(-((wl - 2.30)**2)/0.03)
        self.sigma_co2 = 1.4 * np.exp(-((wl - 2.75)**2)/0.015) + 0.2 * np.exp(-((wl - 2.02)**2)/0.01)
        self.rayleigh = 0.05 * (wl / 2.0)**(-4.0)

    def planck_flux(self, temp_k):
        # Basitleştirilmiş Planck fonksiyonu bağıl ölçeği
        c2 = 14387.77 # um * K
        return 1.0 / (self.wavelengths**5 * (np.exp(c2 / (self.wavelengths * temp_k)) - 1.0 + 1e-6))

    def forward_spectrum(self, log_h2o, log_ch4, log_co2, teq, log_pcloud, r0=0.0150, f_spot=0.0, t_phot=5000.0, t_spot=4200.0):
        X_h2o = 10.0**log_h2o
        X_ch4 = 10.0**log_ch4
        X_co2 = 10.0**log_co2
        H_factor = (teq / 1200.0) * 0.00015
        
        absorption = (X_h2o * self.sigma_h2o + X_ch4 * self.sigma_ch4 + X_co2 * self.sigma_co2 + self.rayleigh)
        depth = r0 + H_factor * np.log(1.0 + absorption * 1e4)
        
        pcloud = 10.0**log_pcloud
        if pcloud < 0.1:
            max_cutoff = r0 + H_factor * (2.0 + np.log(pcloud / 0.1 + 1e-4))
            depth = np.minimum(depth, max_cutoff)
            
        # Yıldız Lekesi Kirliliği (Stellar Spot Contamination Düzeltmesi)
        if f_spot > 0.001:
            b_phot = self.planck_flux(t_phot)
            b_spot = self.planck_flux(t_spot)
            spot_factor = 1.0 - f_spot * (1.0 - b_spot / b_phot)
            depth = depth / np.maximum(spot_factor, 0.5)
            
        return depth.astype(np.float32)
