# Deconvolve the data!
from write_gapped_SWIFT import make_HS_pulse
import scipy.fft as sf
from scipy.io import loadmat, savemat
import numpy as np

if __name__ == '__main__':
    #Load data
    kk = loadmat('swift_info_data.mat')['kspace']
    kk = np.reshape(kk, )
    # Load pulse shape
    num_lines = kk.shape[2]/2
    # Load input function x
    x = loadmat('swift_pulse_shape')['x']
    # Decorrelate data
    for u in range(num_lines):
        kp = kk[] # TODO Combine 2 spokes
        km = kk

        kline_f = sf.fft(kline) # go to freq. domain (R)
        X = sf.fft(x)
        kline_f_deconv = kline_f * np.conj(X) / (np.absolute(X)**2)
        kk_deconv[] = sf.ifft(kline_f_deconv)
    # Save processed data for further NUFFT recon
    savemat('swift_data_decorr.mat',{'kk':kk_deconv})


