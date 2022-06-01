# import
import unittest
import numpy as np
from minTE.sequences.write_CODE import *

class TestCODE(unittest.TestCase):
    def test_seq_generation(self):
        print("Testing 3D CODE...")
        self.assertEqual(0,0)
        # Timing is checked within the function
        seq, TE, ktraj = make_code_sequence(FOV=253e-3, N=64, TR=15e-3, flip=10, enc_type='3D',
                             os_factor=1, save_seq=False, rf_type='gauss')
        # Run test report
        #print(seq.test_report())
        # Make sure output types are correct
        self.assertIsInstance(seq, Sequence)
        self.assertIsInstance(TE, float)
        self.assertIsInstance(ktraj, np.ndarray)

if __name__ == "__main__":
    unittest.main()
