# Test reconstruction for each sequence. (1) Raw data org. (2) 2D or 3D recon.
import unittest
import numpy as np
from minTE.recon.reconstruct_radial import reconstruct_2d, reconstruct_3d
from scipy.io import loadmat, savemat
from pynufft import NUFFT
from minTE.sequences.sequence_helpers import get_radk_params_2D, get_radk_params_3D
import skimage

from pathlib import Path
ROOT_PATH = Path(__file__).parent
PHT_PATH = ROOT_PATH / 'slpht32.mat'

# Helping fx
def normalize_im(im):
    return (im - np.min(im))/(np.max(im)-np.min(im))
def make_radial_ktraj(N, dim=2):
    """
    Makes a 2D radial trajectory
    """
    one_line = np.linspace(0, np.pi, int(N / 2))

    if dim == 2:
        Ns = get_radk_params_2D(N)
        ktraj = np.zeros((Ns, int(N / 2)), dtype=complex)
        q = 0
        for phi in np.linspace(0, 2 * np.pi, Ns, endpoint=False):
            ktraj[q, :] = one_line * np.cos(phi) + 1j * one_line * np.sin(phi)
            q += 1

    elif dim == 3:
        __, Ntheta, Nphi = get_radk_params_3D(1,N)
        ktraj = np.zeros((Ntheta * Nphi, int(N / 2), 3))
        q = 0
        for theta in np.linspace(0, np.pi, Ntheta, endpoint=False):
            for phi in np.linspace(0, 2 * np.pi, Nphi, endpoint=False):
                ktraj[q, :, 0] = one_line * np.sin(theta) * np.cos(phi)
                ktraj[q, :, 1] = one_line * np.sin(theta) * np.sin(phi)
                ktraj[q, :, 2] = one_line * np.cos(theta)
                q += 1

    return ktraj
def load_data_for_recon(N, dim=2):
    ktraj = make_radial_ktraj(N, dim)
    om = np.zeros((ktraj.shape[0] * ktraj.shape[1], dim))
    if dim == 2:
       om[:, 0] = np.real(ktraj).flatten()
       om[:, 1] = np.imag(ktraj).flatten()
    elif dim == 3:
        for p in range(dim):
            om[:,p] = ktraj[:,:,p].flatten()

    pht = loadmat(PHT_PATH)['pht']
    if dim == 3:
        imspace_gt = np.zeros((pht.shape[0],pht.shape[1],N))
        for u in range(N):
            imspace_gt[:,:,u] = pht
    else:
        imspace_gt = pht

    NufftObj = NUFFT()
    Nd = tuple([N]*dim)
    Kd = tuple([2*N]*dim)
    Jd = tuple([6]*dim)
    NufftObj.plan(om, Nd, Kd, Jd)

    y = NufftObj.forward(imspace_gt)
    kspace = np.reshape(y,(ktraj.shape[0],ktraj.shape[1],1))
    kspace = np.swapaxes(kspace,1,2)
    kspace = np.swapaxes(kspace,0,2)

    return kspace, ktraj, imspace_gt

class TestRecon(unittest.TestCase):

    def test_reconstruct_2d(self):
        print("Testing 2D reconstruction...")

        # Make a phantom image and sample it nonuniformly
        kspace, ktraj, imspace_gt = load_data_for_recon(32, 2)
        # Half pulse option
        imspace, images = reconstruct_2d(kspace, ktraj, N=32,
                                         maxiter=50, half_pulse=False)
        self.assertEqual(imspace.shape,(32,32,1))
        self.assertEqual(images.shape,(32,32))

        # Compare to ground truth
        im1 = normalize_im(images)
        im2 = normalize_im(imspace_gt)

        ssim = skimage.metrics.structural_similarity(im1, im2, data_range=1)
        psnr = skimage.metrics.peak_signal_noise_ratio(im1, im2)


        print(ssim)
        print(psnr)


        self.assertTrue(ssim > 0.15)
        self.assertTrue(psnr > 7)

    def test_reconstruct_3d(self):
        print("Testing 3D reconstruction...")

        kspace, ktraj, imspace_gt = load_data_for_recon(32,3)
        imspace, images = reconstruct_3d(kspace, ktraj, N=32,
                                         maxiter=50, half_pulse=False)

        # Compare to ground truth
        im1 = normalize_im(images)
        im2 = normalize_im(imspace_gt)

        ssim = skimage.metrics.structural_similarity(im1, im2, data_range=1)
        psnr = skimage.metrics.peak_signal_noise_ratio(im1, im2)

        print(ssim)
        print(psnr)
        self.assertTrue(psnr > 10)


if __name__ == '__main__':
    unittest.main()
