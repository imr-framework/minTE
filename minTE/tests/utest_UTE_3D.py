# import
import unittest
import numpy as np
from minTE.sequences.write_UTE_3D import *

class TestUTE3D(unittest.TestCase):
    def test_seq_generation(self):
        # Timing is checked within the function
        seq, TE, ktraj = write_UTE_3D_rf_spoiled(N=64, FOV=250e-3, slab_thk=253e-3, FA=10, TR=15e-3, ro_asymmetry=0.97,
                                os_factor=1, rf_type='sinc', rf_dur=1e-3, use_half_pulse=False, save_seq=False)
        # Run test report
        print(seq.test_report())

        # Make sure output types are correct
        self.assertIsInstance(seq, Sequence)
        self.assertIsInstance(TE, float)
        self.assertIsInstance(ktraj, np.ndarray)

if __name__ == "__main__":
    unittest.main()
