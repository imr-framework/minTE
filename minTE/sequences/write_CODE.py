# CODE sequence for minimal TE imaging
from math import pi
import numpy as np
from pypulseq.Sequence.sequence import Sequence
from pypulseq.calc_duration import calc_duration
from pypulseq.make_adc import make_adc
from pypulseq.make_delay import make_delay
from pypulseq.make_sinc_pulse import make_sinc_pulse
from pypulseq.opts import Opts
from minTE.sequences.sequence_helpers import *
from pypulseq.make_gauss_pulse import make_gauss_pulse
from scipy.io import savemat

def make_code_sequence(FOV=250e-3, N=64, TR=100e-3, flip=15, enc_type='3D',
                       rf_type='gauss', os_factor=1, spoil=False, save_seq=True):
    """
    3D or 2D (projection) CODE sequence

    Parameters
    ----------
    FOV : float, default=0.25
        Isotropic image field-of-view in [meters]
    N : int, default=64
        Isotropic image matrix size.
        Also used for base readout sample number (2x that of Nyquist requirement)
    TR : float, default=0.1
        Repetition time in [seconds]
    flip : float, default=15
        Flip angle in [degrees]
    enc_type : str, default='3D'
        Dimensionality of sequence - '3D' or '2D'
    rf_type : str, default='gauss'
        RF pulse shape - 'gauss' or 'sinc'
    os_factor : float, default=1
        Oversampling factor in readout
        The number of readout samples is the nearest integer from os_factor*N
    spoil : bool
        Whether to add a spoiler gradient.
    save_seq : bool, default=True
        Whether to save this sequence as a .seq file

    Returns
    -------
    seq : Sequence
        PyPulseq CODE sequence object
    TE : float
        Echo time of generated sequence in [seconds]
    ktraj : np.ndarray
        3D k-space trajectory [spoke, readout sample, 3] where
        the last dimension refers to spatial frequency coordinates (kx, ky, kz)
        For 2D projection version, disregard the last dimension.

    """
    # System options (copied from : amri-sos service form)
    system = Opts(max_grad=32, grad_unit='mT/m', max_slew=130, slew_unit='T/m/s',
                  rf_ringdown_time=30e-6, rf_dead_time=100e-6, adc_dead_time=20e-6)
    # Parameters
    # Radial sampling set-up
    dx = FOV / N
    if enc_type == '3D':
        Ns, Ntheta, Nphi = get_radk_params_3D(dx, FOV)
        # Radial encoding details
        thetas = np.linspace(0, np.pi, Ntheta, endpoint=False)
    elif enc_type == "2D":
        Nphi = get_radk_params_2D(N)
        Ntheta = 1
        Ns = Nphi
        thetas = np.array([np.pi/2]) # Zero z-gradient.

    phis = np.linspace(0, 2 * np.pi, Nphi, endpoint=False)
    print(f'{enc_type} acq.: using {Ntheta} thetas and {Nphi} phis - {Ns} spokes in total.')


    # Make sequence components
    # Slice-selective RF pulse: 100 us, 15 deg gauss pulse
    FA = flip * pi / 180

    rf_dur = 100e-6
    thk_slab = FOV

    if rf_type == 'gauss':
        rf, g_pre, __ = make_gauss_pulse(flip_angle=FA, duration=rf_dur, slice_thickness=thk_slab,
                                         system=system, return_gz=True)
    elif rf_type == 'sinc':
        rf, g_pre, __ = make_sinc_pulse(flip_angle=FA, duration=rf_dur, slice_thickness=thk_slab,
                                          apodization=0.5,
                                          time_bw_product=4, system=system, return_gz=True)
    else:
        raise ValueError("RF type can only be sinc or gauss")

    # Round off timing to system requirements
    rf.delay = system.grad_raster_time * round(rf.delay / system.grad_raster_time)
    g_pre.delay = system.grad_raster_time * round (g_pre.delay / system.grad_raster_time)
    dr = FOV/N
    kmax = (1/dr)/2
    Nro = np.round(os_factor * N)

    # Asymmetry! (0 - fully rewound; 1 - half-echo)
    ro_asymmetry = (kmax - g_pre.area/2)/(kmax + g_pre.area/2)
    s = np.round(ro_asymmetry * Nro/2) / (Nro/2)
    ro_duration = 2.5e-3
    dkp = (1/FOV) / (1+s)
    ro_area = N * dkp
    g_ro = make_trapezoid(channel='x', rise_time=20e-6, flat_area=ro_area, flat_time=ro_duration, system=system)
    #g_ro = make_trapezoid(channel='x', flat_area=ro_area, flat_time=ro_duration, system=system)
    adc = make_adc(num_samples=Nro, duration=g_ro.flat_time, delay=g_ro.rise_time, system=system)

    if spoil:
        gro_spoil = make_trapezoid(channel='x', area=0.2 * N / FOV, system=system)
        # Delay
        TRfill = TR - calc_duration(g_pre) - calc_duration(g_ro) - calc_duration(gro_spoil)
    else:
        # Delay
        TRfill = TR - calc_duration(g_pre) - calc_duration(g_ro)

    delayTR = make_delay(TRfill)

    TE = 0.5 * calc_duration(g_pre) + g_ro.rise_time + adc.dwell * Nro / 2 * (1 - s)
    print(f'TE obtained: {TE*1e3} ms')

    # Initiate storage of trajectory
    ktraj = np.zeros([Ns, int(adc.num_samples), 3])

    extra_delay_time = 0

    # Initiate sequence
    seq = Sequence(system)
    u = 0
    # For each direction
    for phi in phis:
        for theta in thetas:
            # Construct oblique gradient
            ug = get_3d_unit_grad(theta, phi)
            g_pre_x, g_pre_y, g_pre_z = make_oblique_gradients(gradient=g_pre,unit_grad=-1*ug)
            g_ro_x, g_ro_y, g_ro_z = make_oblique_gradients(gradient=g_ro, unit_grad=ug)
            # Add components
            seq.add_block(rf, g_pre_x, g_pre_y, g_pre_z)
            seq.add_block(make_delay(extra_delay_time))
            seq.add_block(g_ro_x, g_ro_y, g_ro_z, adc)
            if spoil:
                gs_x, gs_y, gs_z = make_oblique_gradients(gradient=gro_spoil,unit_grad=ug)
                seq.add_block(gs_x,gs_y,gs_z)
            seq.add_block(delayTR)

            # Store trajectory
            gpxh, gpyh, gpzh = make_oblique_gradients(gradient=g_pre,unit_grad=-ug)
            modify_gradient(gpxh,0.5)
            modify_gradient(gpyh,0.5)
            modify_gradient(gpzh,0.5)

            ktraj[u, :, :] = get_ktraj_3d_rew_delay(g_ro_x, gpxh, g_ro_y, gpyh, g_ro_z, gpzh, adc)
            u += 1
    # Save sequence
    if save_seq:
        seq.write(f'seqs/newcode{enc_type}_{rf_type}_TR{TR*1e3:.0f}_TE{TE*1e3:.2f}_FA{flip}_N{N}_delay{extra_delay_time*1e3}ms.seq')
        savemat(f'seqs/ktraj_code{enc_type}_{rf_type}_TR{TR*1e3:.0f}_TE{TE*1e3:.2f}_FA{flip}_N{N}.mat',{'ktraj': ktraj})


    return seq, TE, ktraj

if __name__ == '__main__':
    # 083021
    # seq, TE, ktraj = make_code_sequence(FOV=253e-3, N=64, TR=15e-3, flip=10, enc_type='3D',
    #                          os_factor=1, save_seq=False, rf_type='gauss')
    # seq.write('CODE_64_TR15_FLIP10_FOV253_rise20_092222.seq')
    # savemat('CODE_64_info_rise20_092222.mat',{'TE':TE,'ktraj':ktraj})
    #print(f'TE is {TE*1e3} ms')

    #  With spoiler!
    seq, TE, ktraj = make_code_sequence(FOV=253e-3, N=64, TR=15e-3, flip=10, enc_type='3D',
                                         os_factor=1, save_seq=False, spoil=True, rf_type='gauss')
    print(seq.test_report())
    seq.plot(time_range=[0,45e-3])
    seq.write('CODE_64_TR15_FLIP10_FOV254_SPOILED_100622.seq')
    savemat('CODE_64_info_100622.mat',{'TE':TE,'ktraj':ktraj})
