import unittest
import numpy as np
import torch
from core.inversion import JointArielInversionEngine
from core.simulator import RadiativeTransferSimulator
from core.retrieval import evaluate_single_planet_bic

class TestOSTEArielEngine(unittest.TestCase):
    def setUp(self):
        self.sim = RadiativeTransferSimulator(n_channels=52)
        self.engine = JointArielInversionEngine()

    def test_simulator_physics(self):
        spec = self.sim.forward_spectrum(log_h2o=-4.0, log_ch4=-4.0, log_co2=-4.5, teq=1000.0, log_pcloud=0.0)
        self.assertEqual(len(spec), 52)
        self.assertTrue(np.all(spec > 0.0100))

    def test_bic_decision_logic(self):
        wl = self.sim.wavelengths
        mock_mu = np.full(52, 0.0150)
        mock_sig = np.full(52, 0.000020)
        # Molekülsüz test
        d_h, d_c, _, _ = evaluate_single_planet_bic(mock_mu, mock_sig, wl)
        self.assertFalse(d_h)
        self.assertFalse(d_c)

if __name__ == "__main__":
    unittest.main()
