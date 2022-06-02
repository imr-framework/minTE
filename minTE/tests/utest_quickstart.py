import unittest
from minTE.quickstart.seq_gen_demo import *
from minTE.sequences.write_UTE_2D import write_UTE_2D_rf_spoiled
class TestQuickStart(unittest.TestCase):
    def test_export_waveforms(self):
        seq, _, _ = write_UTE_2D_rf_spoiled(N=16,save_seq=False)
        all_waveforms = export_waveforms(seq)
        self.assertIsInstance(all_waveforms, dict)
        expected_keys = ['t_adc','t_rf','t_rf_centers','t_gx','t_gy','t_gz','adc','rf',
                         'rf_centers','gx','gy','gz','grad_unit','rf_unit','time_unit']
        for expected_key in expected_keys:
            self.assertIn(expected_key, all_waveforms.keys())

    def test_display_functions(self):
        seq, _, ktraj = write_UTE_2D_rf_spoiled(N=16,save_seq=False)
        display_seq_interactive(export_waveforms(seq,time_range=[0,1]))
        display_seq(seq,ktraj,dim=2)

    def test_seq_demos(self):
        seq, ktraj = demo_2D_UTE(use_half_pulse=False)
        self.assertIsInstance(seq,Sequence)
        self.assertIsInstance(ktraj,np.ndarray)
        seq, ktraj = demo_2D_UTE(use_half_pulse=True)
        self.assertIsInstance(seq,Sequence)
        self.assertIsInstance(ktraj,np.ndarray)
        seq, ktraj = demo_3D_UTE(use_half_pulse=False)
        self.assertIsInstance(seq,Sequence)
        self.assertIsInstance(ktraj,np.ndarray)
        seq, ktraj = demo_3D_UTE(use_half_pulse=True)
        self.assertIsInstance(seq,Sequence)
        self.assertIsInstance(ktraj,np.ndarray)
        seq, ktraj = demo_CODE()
        self.assertIsInstance(seq,Sequence)
        self.assertIsInstance(ktraj,np.ndarray)


if __name__ == "__main__":
    unittest.main()

