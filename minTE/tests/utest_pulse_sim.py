import unittest
from minTE.sim.rf_simulations_for_UTE import *
from minTE.sim.display_simulated_pulses import *
import types
class TestPulseSim(unittest.TestCase):
    def test_simulated_ute_pulses(self):
        usages = [False,True,True]
        indices = [1,1,2]
        for q in range(3):
            signals, bwsim, rf = simulate_ute_pulses(use_half_pulse=usages[q],rf_index=indices[q])
            self.assertIsInstance(signals,np.ndarray)
            self.assertEqual(signals.shape,(200,1030))
            np.testing.assert_almost_equal(np.absolute(bwsim), 8000)
            self.assertIsInstance(rf,types.SimpleNamespace)

    def test_displays(self):
        signals, bwsim, rf = simulate_ute_pulses(use_half_pulse=True, rf_index=1)
        display_rf_pulse(rf,"")
        display_rf_profile(signals,bwsim,"")

if __name__ == "__main__":
    unittest.main()