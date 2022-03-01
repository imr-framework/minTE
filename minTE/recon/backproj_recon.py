# Recon images using backprojection
#from skimage.transform import radon, rescale
from scipy.io import savemat, loadmat
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import iradon, rescale



def try_reconing_centerout():
    # Load data
    ktraj = loadmat('data/ktraj_ute_minTE_2D_split_grad.mat')['ktraj']
    data = loadmat('data/data_minTE_2d_split_grad.mat')['kspace']
    print(data.shape)
    print(ktraj.shape)
    oneline = np.absolute(ktraj[0,:])

    # Plot ktraj
    for u in range(1):
        plt.plot(np.real(ktraj[u,:]),np.imag(ktraj[u,:]),'*')
   # plt.show()

    # Find thetas
    thetas = np.linspace(0,2*np.pi,805,endpoint=False)
    # Realign to get sinogram

    q = np.round(oneline[0]/(oneline[1]-oneline[0])).astype(int)
    sinogram = np.zeros([805,256*2 + 2*q],dtype=complex)
    ch = 0
    for l in range(805):
        sinogram[l,0:256] = data[-1::-1, ch, l]
        sinogram[l,:] = np.fft.fftshift(np.fft.ifft(sinogram[l,:]))
    outim = iradon(sinogram, theta=thetas, filter_name='ramp')
    print(outim.shape)
    plt.imshow(np.absolute(outim))
    plt.show()

    return outim

if __name__ == '__main__':
    # Try with rewound data! (make this work first!)
    data = loadmat('data/data_rewound_TE10.mat')['kspace']
    print(data.shape)
    #ktraj = loadmat('data/ktraj_ute_rewound_minTE.mat')['ktraj']
    #print(ktraj.shape)


    # Find thetas
    thetas = np.linspace(0,360,data.shape[2],endpoint=False)
    kspace = np.zeros((256,256,20), dtype=complex)
    ch = 0
    imspace = np.zeros((256,256,20),dtype=complex)
    for c in range(1): # For each channel
        for l in range(data.shape[2]):
            sinogram = np.zeros((256,805), dtype=complex)
            sinogram[:,l] = np.fft.fftshift(np.fft.ifft(data[:,c,l]))
        outim = iradon(sinogram, theta=thetas, filter_name='ramp')
        #print(outim.shape)
        imspace[:,:,c] = outim
        kspace[:,:,c] = np.fft.fft2(outim)

    images = np.sqrt(np.sum(np.square(np.absolute(imspace)),axis=-1))
    #plt.imshow(images)
    kspace1 = np.zeros(kspace.shape, dtype=complex)

    plt.imshow()
    plt.show()
    #savemat('rewound_proj_recon_degs_half-402.mat',{'images':images})
