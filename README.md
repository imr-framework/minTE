# Open-Source Magnetic Resonance Imaging with Minimal Echo Time

<img title="PyPulseq Badge" src="https://img.shields.io/badge/made%20using-pypulseq-brightgreen" width="100">


The minTE repository provides parametrized, radial, short/ultrashort echo time Magnetic Resonance Imaging (MRI) sequence in the Pulseq format [[1]](#references) along with reconstruction functions. Sequence variations include 2D and 3D partially-rewound RF-spoiled GRE and CODE [[2]](#references) generated with the PyPulseq [[3]](#references) library. 

## Quickstart
### Option 1: Local installation
1. Download or clone the repository
2. Set up and activate the virtual environment making sure all packages listed in `requirements.txt` are installed;
3. Note that if virtual-scanner is not automatically installed by the IDE through `requirements.txt`, you need to run the following on the command line with the virtual environment activated:
`pip install git+https://github.com/imr-framework/virtual-scanner.git@2ed303d42038449e28ebe44fa4177541cabe8eb2
`
5. From the root, run `python -m minTE.tests.run_all_unit_tests` to ensure all functions work;
6. From the root, run `python -m minTE.quickstart.seq_gen_demo` to generate and visualize the sequence variations.

### Option 2: Google Colab Notebook 
Open the [demo notebook](https://github.com/imr-framework/minTE/blob/main/minTE_quickstart.ipynb) in Google Colab and follow the instructions to generate and display a sequence file. 

## Sequences and example parameters 
Shared parameters are FOV = 253 mm, 2D slice thickness = 5 mm, flip angle = 10 degrees, and TR = 15 ms. 
| Sequence type  | Dimensions  | Half-pulse option | Matrix size | # spokes | TE (ms) | Acq. time (min:sec)
| -------------  | ----------- | ----------------- | --- | ---| --- | --- |
| UTE / rGRE  | 2D | Yes | 256 | 804 | 0.45 - 0.82 | 00:12, 00:24 |
| UTE / rGRE  | 3D | Yes | 64 | 13056 | 0.12 - 0.66 | 03:01, 06:32 |
| CODE   | 3D | No  | 64 | 13056 | 0.32 | 03:16 | 

## Example Usage 
```python
# Make a 3D half pulse UTE sequence 
seq, TE, ktraj = write_UTE_3D_rf_spoiled(N=64, FOV=250e-3, slab_thk=253e-3, FA=10, TR=15e-3, ro_asymmetry=0.97,
                                         os_factor=1, rf_type='sinc', rf_dur=0.05e-3, use_half_pulse=True)

# Make a 2D full pulse UTE sequence
seq, TE, ktraj = write_UTE_2D_rf_spoiled(N=256, Nr=804, FOV=253e-3, thk=5e-3, slice_locs=[0],
                                         FA=10, TR=15e-3, ro_asymmetry=0.97, use_half_pulse=False, rf_dur=1e-3,
                                         TE_use=None)
                                         
# Make a CODE sequence
seq, TE, ktraj = make_code_sequence(FOV=253e-3, N=64, TR=100e-3, flip=10, enc_type='3D',
                                    os_factor=1, save_seq=False, rf_type='gauss')
```
## Reconstruction
The functions `reconstruct_2d` and `reconstruct_3d` are provided [here](https://github.com/imr-framework/minTE/blob/main/minTE/recon/reconstruct_radial.py) that use PyNUFFT [[4,5]](#references) to generate images from non-cartesian k-spaces. The `'cg'` option is recommended for image quality. 

## Images
### ACR phantom 
<p float="left" align="middle">
  <img src="https://github.com/imr-framework/minTE/blob/main/minTE/figs/ute_2d_full_acr.png" width="500" />
  <img src="https://github.com/imr-framework/minTE/blob/main/minTE/figs/ute_2d_half_acr.png" width="500" />
</p>
<p float="left" align="middle">
  <img src="https://github.com/imr-framework/minTE/blob/main/minTE/figs/ute_3d_full_acr.png" width="300" />
  <img src="https://github.com/imr-framework/minTE/blob/main/minTE/figs/ute_3d_half_acr.png" width="300" />
  <img src="https://github.com/imr-framework/minTE/blob/main/minTE/figs/3D_code.png" width="260" />
</p>


### Ex vivo procine muscle/bone sample
<p float="left" align="middle">
  <img src="https://github.com/imr-framework/minTE/blob/main/minTE/figs/Fig2.png" width="800" />
</p>

## Feedback 
Please communicate all questions and feature suggestions on the Github Issues tab. We appreciate your inputs! 

## References
1. Layton, Kelvin J., et al. "Pulseq: a rapid and hardware‐independent pulse sequence prototyping framework." Magnetic
resonance in medicine 77.4 (2017): 1544-1552.
2. Park, Jang‐Yeon, et al. "Short echo‐time 3D radial gradient‐echo MRI using concurrent dephasing and excitation." Magnetic resonance in medicine 67.2 (2012): 428-436.
3. Ravi, Keerthi, Sairam Geethanath, and John Vaughan. "PyPulseq: A Python Package for MRI Pulse Sequence Design."
Journal of Open Source Software 4.42 (2019): 1725.
4. Lin, Jyh-Miin. “Python Non-Uniform Fast Fourier Transform (PyNUFFT): An Accelerated Non-Cartesian MRI Package on a Heterogeneous Platform (CPU/GPU).” Journal of Imaging 4.3 (2018): 51.
5. J.-M. Lin and H.-W. Chung, Pynufft: python non-uniform fast Fourier transform for MRI Building Bridges in Medical Sciences 2017, St John’s College, CB2 1TP Cambridge, UK

