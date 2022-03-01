from pynufft import NUFFT
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat, savemat
import math
import mat73
# #

def reconstruct_2d(kspace, ktraj, N=256, maxiter=50, half_pulse=True):
    """
    Reconstruct 2D radial UTE data.

    Inputs
    ------
    kspace : np.ndarray
        N_readout x N_channels x N_lines data matrix
    ktraj : np.ndarray
        N_lines x N_readout shaped complex 2D kspace trajectory {kx + 1j*ky}
    N : integer
        Isotropic image matrix size
    maxiter : integer
        Maximum number of iterations to run in conjugate gradient reconstruction
    half_pulse : bool
        Whether half pulse excitation was used

    Returns
    -------
    imspace : np.ndarray
        Complex multi-channel image space of size (N,N,N_channels)
    images : np.ndarray
        Real sum-of-squares combined image of size (N,N)
    """

    # Recon rGRE 3D using PyNUFFT CG method
    ## NUFFT object
    NufftObj = NUFFT()
    Nd = (N, N)  # image size
    print('setting image dimension Nd...', Nd)
    Kd = tuple(2 * x for x in Nd)  # k-space size
    print('setting spectrum dimension Kd...', Kd)
    Jd = tuple(6 for x in Nd)  # interpolation size
    print('setting interpolation size Jd...', Jd)

    # Load ktraj & data
    kt = ktraj
    kk = kspace

    # 3 slices reshaping
    # kk = np.reshape(kk, [512,20,804,3])
    # kk = np.swapaxes(kk,1,3)
    if half_pulse:
        kk = kk[:,:,0::2] + kk[:,:,1::2]
    kk = np.swapaxes(kk, 1, 2)

    print("Data and info loaded.")

    # ------------------------------------------------------------
    imspace = np.zeros((N, N, 20), dtype=complex)
    # Scale kspace trajectory to [-pi, pi]
    kt_sc = math.pi / np.max(np.absolute(kt))
    kt = kt * kt_sc
    # k-space trajectory (3D)
    om = np.zeros((kt.shape[0] * kt.shape[1], 2))
    om[:, 0] = np.real(kt).flatten()
    om[:, 1] = np.imag(kt).flatten()

    NufftObj.plan(om, Nd, Kd, Jd)

    C = kk.shape[-1]
    for c in range(C):  # For each channel
        print(f'begin channel {c + 1}')
        imspace[:, :, c] = NufftObj.solve(y=np.transpose(kk[:, :, c]).flatten(),
                                        solver='cg', maxiter=maxiter)
        print(f'channel {c + 1} reconstructed')

    images = np.sqrt(np.sum(np.absolute(imspace)**2, axis=2))
    return imspace, images


def reconstruct_3d(kspace, ktraj, N=64, maxiter=50, half_pulse=True):
    """
     Reconstruct 3D radial UTE/CODE data.

     Inputs
     ------
     kspace : np.ndarray
         N_readout x N_channels x N_lines data matrix
     ktraj : np.ndarray
         N_lines x N_readout x 3 shaped complex 2D kspace trajectory {(kx,ky,kz)}
     N : integer
         Isotropic image matrix size
     maxiter : integer
         Maximum number of iterations to run in conjugate gradient reconstruction
     half_pulse : bool
         Whether half pulse excitation was used

     Returns
     -------
     imspace : np.ndarray
         Complex multi-channel image space of size (N,N,N,N_channels)
     images : np.ndarray
         Real sum-of-squares combined image of size (N,N,N)
     """

    # Recon rGRE 3D using PyNUFFT CG method
    ## NUFFT object
    NufftObj = NUFFT()
    Nd = (N,N,N)  # image size
    print('setting image dimension Nd...', Nd)
    Kd = tuple(2 * x for x in Nd)  # k-space size
    print('setting spectrum dimension Kd...', Kd)
    Jd = tuple(6 for x in Nd)  # interpolation size
    print('setting interpolation size Jd...', Jd)

    # Load ktraj & data
    kt = ktraj
    kk = kspace

    # 3 slices reshaping
    #kk = np.reshape(kk, [512,20,804,3])
    #kk = np.swapaxes(kk,1,3)
    if half_pulse:
        kk = kk[:,:,0::2] + kk[:,:,1::2]
    kk = np.swapaxes(kk,1,2)

    print("Data and info loaded.")
    #------------------------------------------------------------
    imspace = np.zeros((N, N, N, 20), dtype=complex)
    # Scale [-pi, pi]
    kt_sc = math.pi / np.max(np.absolute(kt))
    kt = kt * kt_sc
    # k-space trajectory (3D)
    om = np.zeros((kt.shape[0] * kt.shape[1], 3))
    om[:, 0] = kt[:,:,0].flatten()
    om[:, 1] = kt[:,:,1].flatten()
    om[:, 2] = kt[:,:,2].flatten()

    NufftObj.plan(om, Nd, Kd, Jd)

    C = kk.shape[-1]
    for c in range(C):  # For each channel
        print(f'begin channel {c+1}')
        imspace[:,:,:,c] = NufftObj.solve(y=np.transpose(kk[:,:,c]).flatten(),
                                                  solver='cg',maxiter=maxiter)
        print(f'channel {c + 1} reconstructed')

    images = np.sqrt(np.sum(np.absolute(imspace)**2, axis=3))
    return imspace, images

