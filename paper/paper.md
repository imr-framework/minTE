---
title: 'Open-Source Magnetic Resonance Imaging with Minimal Echo Times'

tags:
  - Magnetic Resonance Imaging
  - Pulse Sequence Programming
  - Python
  - Pulseq
  - PyPulseq
  - Reproducible Research

authors:
  - name: Gehua Tong
    orcid: 0000-0001-6263-762X
    affiliation: 1,2
  - name: Sairam Geethanath
    orcid: 0000-0002-3776-4114
    affiliation: 3
  - name: John Thomas Vaughan Jr.
    orcid: 0000-0002-6933-3757
    affiliation: 2

affiliations:
 - name: Department of Biomedical Engineering, Columbia University in the City of New York
   index: 1
 - name: Columbia Magnetic Resonance Research Center, Columbia University in the City of New York
   index: 2
 - name: Biomedical Engineering and Imaging Institute, Dept. of Diagnostic, Molecular and Interventional Radiology, Icahn School of Medicine at Mt. Sinai
   index: 3 

date: 24 May 2022

bibliography: paper.bib
---
# Summary
In the ``minTE`` repository, we present Magnetic Resonance Imaging (MRI) pulse sequences that use minimal Echo Times (TEs). Three customizable sequences were implemented in the Pulseq format [@layton2017pulseq] with the PyPulseq library [@ravi2019pypulseq].

* 2D UTE: a 2D Radiofrequency-spoiled (RF-spoiled) radial Gradient Recalled Echo (rGRE) sequence. The k-space trajectory consists of spokes that 
can be rewound to any point between $k = -k_{max}$ and $k = 0$. Rewinding preserves the $k = 0$ point but increases the minimum TE. The number of spokes follows the Nyquist criterion. A sinc slice-selective RF pulse is used for excitation with two options: full-pulse and half-pulse. The latter uses two excitations per spoke with halved pulse duration and opposite slice selective gradients [@nielsen1999ultra]. This allows it to achieve a lower minimum TE but also doubles the acquisition time. For both versions, RF spoiling allows for shorter Repetition Times (TRs). Example parameters using the half pulse option are TR = 15 ms, TE = 449 us, flip angle (FA) = 10 degrees, Field-Of-View (FOV) = 253 mm, slice thickness = 5 mm, and matrix size (N) = 256.

* 3D UTE: a 3D RF-spoiled rGRE sequence. The k-space trajectory is similar to 2D UTE but with spokes in all directions in 3D space with equal azimuthal and polar angle spacing. Example parameters using the half pulse technique are TR = 15 ms, TE = 119 us, FA = 10 degrees, FOV = 250 mm, slab thickness = 253 mm, and N = 64. 

* CODE: a version of the COcurrent Dephasing and Excitation(CODE) sequence [@park2012short], this implementation uses the same gradient axis for both slab selection and readout. Slab excitation happens at a different 3D orientation for each readout and ideally covers the FOV of interest in all directions. Gaussian RF pulses are used and the k-space sampling is similar to the partially rewound 3D UTE. Example parameters are TR = 15 ms, TE = 320 us, FA = 10 degrees, FOV = slab thickness = 253 mm, and N = 64. 

Each sequence can be generated with a Python function and reconstructed with provided scripts. The reconstruction is a two-step process: first, pre-processing of the raw data into a form that is sorted as a 2D matrix of size (number of ADC samples, number of readouts) and ready for gridding; second, reconstruction using the NUFFT[@fessler2003nonuniform; @lin2018python] library converts 2D or 3D non-Cartesian data to 2D images or 3D volumes. 

# Statement of Need
Short and ultra-short TE sequences visualize tissues with short $T_2$ and $T_2^*$ values. The ability to recover signal from such tissues lends them to musculoskeletal applications such as joint and bone imaging [@holmes2005mrjoint; @jerban2020updatebone]. UTE has also been validated as a non-invasive way of monitoring the lungs of COVID-19 patients [@yang2020clinical]. Complete open-source pipelines of these sequences could help improve the reproducibility of multi-site imaging studies. Sites with Pulseq installed can easily standardize sequences with full transparency of all waveforms. In addition, it contributes to the accessible MR world by providing both customizable sequences and reconstruction code without the need for platform-specific training or access. 
 
# Past and Ongoing Research Projects 
The minTE repository will play a role in the development of MRI methods in inhomogeneous B0 fields which produce short $T_2^*$. Comparsions to state-of-the-art UTE images are needed for sequence families such as Multi-Spectral Imaging (MSI) [@MSIreview] and SWeep Imaging with Fourier Transform (SWIFT) [@swift].

# Acknowledgements
``minTE`` was funded in part by the Seed Grant Program for MR Studies of the Zuckerman Mind Brain Behavior Institute at Columbia University (PI: Geethanath), and was developed at Zuckerman Mind Brain Behavior Institute MRI Platform, a shared resource.

# References



