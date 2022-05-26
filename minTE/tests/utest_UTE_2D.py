# import
import unittest
import numpy as np
from minTE.sequences.write_UTE_2D import *

class TestUTE2D(unittest.TestCase):
    def test_seq_generation(self):
        print("Testing 2D UTE...")
        # Timing is checked within the function
        seq, TE, ktraj = write_UTE_2D_rf_spoiled(N=256, Nr=804, FOV=253e-3, thk=5e-3, slice_locs=[0],
                                                 FA=10, TR=15e-3, ro_asymmetry=0.97, use_half_pulse=False, rf_dur=1e-3,
                                                 TE_use=None)
        # Run test report
        print(seq.test_report())

        # Make sure output types are correct
        self.assertIsInstance(seq, Sequence)
        self.assertIsInstance(TE, float)
        self.assertIsInstance(ktraj, np.ndarray)

if __name__ == "__main__":
    unittest.main()
