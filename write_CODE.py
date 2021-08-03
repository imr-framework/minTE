# CODE sequence for minimal TE imaging
from math import pi
import numpy as np
from pypulseq.Sequence.sequence import Sequence
from pypulseq.calc_duration import calc_duration
from pypulseq.make_adc import make_adc
from pypulseq.make_delay import make_delay
from pypulseq.make_sinc_pulse import make_sinc_pulse
from pypulseq.make_trap_pulse import make_trapezoid
from pypulseq.opts import Opts
from write_UTE_3D import get_radk_params_3D
from sequence_helpers import *
from pypulseq.make_gauss_pulse import make_gauss_pulse
from scipy.io import savemat, loadmat

def make_code_sequence(FOV=250e-3, N=64, TR=100e-3, flip=15, enc_type='3D',saveseq=False):
    # System options (copied from : amri-sos service form)
    system = Opts(max_grad=32, grad_unit='mT/m', max_slew=130, slew_unit='T/m/s',
                  rf_ringdown_time=30e-6, rf_dead_time=100e-6, adc_dead_time=20e-6)
    # Parameters
    TR = 100e-3
    FOV = 220e-3
    # Radial scheme set-up
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



    print(f'Using {Ntheta} thetas and {Nphi} phis - {Ns} spokes in total.')


    # Make sequence components
    # Slice-selective RF pulse: 100 us, 15 deg gauss pulse
    flip = 15
    FA = flip * pi / 180
    rf_dur = 100e-6
    thk_slab = 220e-3
    rf, g_pre, __ = make_gauss_pulse(flip_angle=FA, system=system,
                                     duration=rf_dur, slice_thickness=thk_slab, return_gz=True)
    rf.delay = system.grad_raster_time * round(rf.delay / system.grad_raster_time)
    g_pre.delay = system.grad_raster_time * round (g_pre.delay / system.grad_raster_time)
    #g_pre_aligned = make_trapezoid(channel=g_pre.channel, rise_time=4e-5, flat_area=g_pre.flat_area, flat_time=g_pre.flat_time)
    # Readout gradient (5 ms readout time)
    #ro_time = 5e-3
    ro_rise_time = 20e-6
    dk = 1/FOV

    # Readout gradient & ADC
    adc_dwell = 10e-6 # 10 us sampling interval (100 KHz readout bandwidth)

    g_ro = make_trapezoid(channel='x', system=system, amplitude=dk/adc_dwell, flat_time=adc_dwell*int(N/2), rise_time=ro_rise_time)


    flat_delay = (0.5*g_pre.area - 0.5*ro_rise_time*g_ro.amplitude) / g_ro.amplitude
    flat_delay = system.grad_raster_time * round(flat_delay / system.grad_raster_time)
    # First ADC at center of k-space.
    adc = make_adc(system=system, num_samples=int(N/2), dwell = adc_dwell, delay = g_ro.rise_time+flat_delay)

    # Delay
    TRfill = TR - calc_duration(g_pre) - calc_duration(g_ro)
    delayTR = make_delay(TRfill)

    # What is TE?
    TE = 0.5 * calc_duration(g_pre) + adc.delay

    # Initiate storage of trajectory
    ktraj = np.zeros([Ns, int(adc.num_samples), 3])

    # Construct sequence
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
            seq.add_block(g_ro_x, g_ro_y, g_ro_z, adc)
            seq.add_block(delayTR)
            # Store trajectory
            ktraj[u, :, :] = get_ktraj_3d_rew_delay(g_ro_x, g_pre_x, g_ro_y, g_pre_y, g_ro_z, g_pre_z, adc)
            u += 1

    # Display sequence plot
    #seq.plot(time_range=[-TR/2,1.5*TR])
    # Test sequence validity
    out_text = seq.test_report()
    print(out_text)
    # Save sequence
    if saveseq:
        seq.write(f'seqs/code_TR{TR*1e3:.0f}_TE{TE*1e3:.2f}_FA{flip}_N{N}.seq')
        savemat(f'seqs/ktraj_code_TR{TR*1e3:.0f}_TE{TE*1e3:.2f}_FA{flip}_N{N}.mat',{'ktraj': ktraj})

    return seq

if __name__ == '__main__':
    seq = make_code_sequence(FOV=250e-3, N=128, TR=100e-3, flip=15, enc_type='2D',saveseq=True)
    print(seq.test_report())