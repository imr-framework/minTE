# 3D ultra-short TE sequence (radial encoding)
from math import pi
import numpy as np
from pypulseq.Sequence.sequence import Sequence
from pypulseq.calc_duration import calc_duration
from pypulseq.make_adc import make_adc
from pypulseq.make_delay import make_delay
from pypulseq.make_sinc_pulse import make_sinc_pulse
from pypulseq.make_block_pulse import make_block_pulse
from pypulseq.make_trap_pulse import make_trapezoid
from pypulseq.make_extended_trapezoid import make_extended_trapezoid
from pypulseq.opts import Opts
import matplotlib.pyplot as plt
from scipy.io import savemat, loadmat
from sequence_helpers import *

def write_UTE_3D(N, FOV, slab_thk, FA, TR):
    # System info for June (original)
    system = Opts(max_grad=32, grad_unit='mT/m', max_slew=130, slew_unit='T/m/s',
                  rf_ringdown_time=30e-6, rf_dead_time=100e-6, adc_dead_time=20e-6)

    # System info with adc_dead_time adjusted to 10e-6 - still plays!
    #system = Opts(max_grad=32, grad_unit='mT/m', max_slew=130, slew_unit='T/m/s',
    #              rf_ringdown_time=30e-6, rf_dead_time=100e-6, adc_dead_time=10e-6)
    seq = Sequence(system=system)

    # Derived parameters
    dx = FOV/N
    Ns, Ntheta, Nphi = get_radk_params_3D(dx, FOV) # Make this function
    print(f'Using {Ns} spokes, with {Ntheta} thetas and {Nphi} phis')

    # Slab selection
    flip = FA*np.pi/180
    st = 30e-3 # what is this?
    rf_dur = 100e-6

    # Make slice selecting components, for spoiler area
    [rf_ss, gz_ss, gssr] = make_sinc_pulse(flip_angle=flip, duration=rf_dur, system=system, slice_thickness=slab_thk)

    rf, __ = make_block_pulse(flip_angle=flip, duration=rf_dur, system=system)
    #print(gz_ss)
    # Readout gradients
    dk = 1/FOV
    k_width = N*dk
    ro_time = 6.4e-3
    gro = make_trapezoid(channel='x', system=system, flat_area=k_width, flat_time=ro_time, rise_time=20e-6)
    adc = make_adc(N/2, duration=gro.flat_time, delay=gro.rise_time, system=system)

    # Spoiling gradients in all directions
    pre_time = 8e-4
    gx_spoil = make_trapezoid(channel='x', system=system, area=gz_ss.area/2, duration=3*pre_time)
    gy_spoil = make_trapezoid(channel='y', system=system, area=gz_ss.area/2, duration=3*pre_time)
    gz_spoil = make_trapezoid(channel='z', system=system, area=gz_ss.area/2, duration=3*pre_time)

    # Timing
    delayTR = TR - calc_duration(gro) - calc_duration(rf) - calc_duration(gz_spoil)
    #delayTRps = 0.8*delayTR # "to avoid gradient heating"
    delayTRps = delayTR
    delayTRps = delayTRps - np.mod(delayTRps, 1e-5)
    delay = make_delay(delayTRps)

    # Spoke angles
    thetas = np.linspace(0, np.pi, Ntheta, endpoint=False)
    phis = np.linspace(0, 2*np.pi, Nphi, endpoint=False)
    ktraj = np.zeros([Ns, int(adc.num_samples), 3])

    # What TE did we end up getting?

    TE = calc_duration(rf)/2 + gro.rise_time

    print(f"TE = {TE*1e3} ms is achieved.")
    u = 0
    # Add blocks to sequence
    for th in range(Ntheta):
        for ph in range(Nphi):
            # Calculate oblique gradients
            k_width_projx = k_width*np.sin(thetas[th])*np.cos(phis[ph])
            k_width_projy = k_width*np.sin(thetas[th])*np.sin(phis[ph])
            k_width_projz = k_width*np.cos(thetas[th])

            gx = make_trapezoid(channel='x', system=system, flat_area=k_width_projx, flat_time=ro_time, rise_time=gro.rise_time)
            gy = make_trapezoid(channel='y', system=system, flat_area=k_width_projy, flat_time=ro_time, rise_time=gro.rise_time)
            gz = make_trapezoid(channel='z', system=system, flat_area=k_width_projz, flat_time=ro_time, rise_time=gro.rise_time)

            # Slab excitation
            #seq.add_block(rf, gz_ss)
            #seq.add_block(gssr)
            seq.add_block(rf)
            # Readout and ADC sampling
            seq.add_block(gx, gy, gz, adc)
            # Go to end of TR + spoiler
            seq.add_block(delay)
            seq.add_block(gx_spoil, gy_spoil, gz_spoil)

            # Store trajectory
            ktraj[u,:,:] = get_ktraj_3d(gx, gy, gz, adc)
            u += 1

    return seq, ktraj, TE

# TODO partially rewound (s=0 ~ s=1) 3D rGRE/UTE sequence.
# - With choice of readout asymmetry (s=ro_asymmetry value)
# - With RF spoiling (d = 117)
# - With option to turn on or off the "opposite gradient acq." - if on, the acquisition time is doubled.
# - With 0.2*(2Kmax) RO spoiler
# - sampling locations are calculated in the same way as write_UTE_3D()

def write_UTE_3D_rf_spoiled(N, FOV=250e-3, slab_thk=3e-3, FA=10, TR=10e-3, ro_asymmetry=0.97,
                            use_double_acq=True, rf_center = 1, os_factor=1, save_seq=True):
    """
    Parameters
    ----------
    N : int
        Matrix size
    FOV : float
        Field-of-view in [meters]
    thk : float
        Slice thickness in [meters]
    FA : float
        Flip angle in [degrees]
    TR : float
        Repetition time in [seconds]
    ro_asymmetry : float
        The ratio A/B where a A/(A+B) portion of 2*Kmax is omitted and B/(A+B) is acquired.

    """
    # Adapted from pypulseq demo write_ute.py (obtained mid-2021)
    system = Opts(max_grad=32, grad_unit='mT/m', max_slew=130, slew_unit='T/m/s',
                  rf_ringdown_time=30e-6, rf_dead_time=100e-6, adc_dead_time=20e-6)
    seq = Sequence(system=system)

    # Derived parameters
    dx = FOV/N
    Ns, Ntheta, Nphi = get_radk_params_3D(dx, FOV) # Make this function
    print(f'Using {Ns} spokes, with {Ntheta} thetas and {Nphi} phis')

    # Spoke angles
    thetas = np.linspace(0, np.pi, Ntheta, endpoint=False)
    phis = np.linspace(0, 2*np.pi, Nphi, endpoint=False)


    ro_duration = 2.5e-3
    ro_os = os_factor  # Oversampling factor
    # minRF_to_ADC_time = 50e-6 # this was not used.

    rf_spoiling_inc = 117  # RF spoiling increment value.

    # Sequence components
    rf, gz, gz_reph = make_sinc_pulse(flip_angle=FA * np.pi / 180, duration=1e-3, slice_thickness=slab_thk,
                                      apodization=0.5,
                                      time_bw_product=2, center_pos=rf_center, system=system, return_gz=True)

    # Asymmetry! (0 - fully rewound; 1 - hall-echo)
    Nro = np.round(ro_os * N)  # Number of readout points
    s = np.round(ro_asymmetry * Nro / 2) / (Nro / 2)
    dk = (1 / FOV) / (1 + s)
    ro_area = N * dk
    gro = make_trapezoid(channel='x', flat_area=ro_area, flat_time=ro_duration, system=system)
    adc = make_adc(num_samples=Nro, duration=gro.flat_time, delay=gro.rise_time, system=system)
    gro_pre = make_trapezoid(channel='x', area=- (gro.area - ro_area) / 2 - (ro_area / 2) * (1 - s), system=system)

    # Spoilers
    gro_spoil = make_trapezoid(channel='x', area=0.2 * N * dk, system=system)

    # Calculate timing
    TE = gz.fall_time + calc_duration(gro_pre, gz_reph) + gro.rise_time + adc.dwell * Nro / 2 * (1 - s)
    delay_TR = np.ceil((TR - calc_duration(gro_pre, gz_reph) - calc_duration(gz) - calc_duration(
        gro)) / seq.grad_raster_time) * seq.grad_raster_time
    assert np.all(delay_TR >= calc_duration(gro_spoil))  # The TR delay starts at the same time as the spoilers!

    print(f'TE = {TE * 1e6:.0f} us')

    if calc_duration(gz_reph) > calc_duration(gro_pre):
        gro_pre.delay = calc_duration(gz_reph) - calc_duration(gro_pre)

    # Starting RF phase and increments
    rf_phase = 0
    rf_inc = 0

    ktraj = np.zeros([Ns, int(adc.num_samples), 3])
    u = 0
    Nline = Ntheta * Nphi
    for th in range(Ntheta):
        for ph in range(Nphi):
            unit_grad = np.zeros(3)
            unit_grad[0] = np.sin(thetas[th])*np.cos(phis[ph])
            unit_grad[1] = np.sin(thetas[th])*np.sin(phis[ph])
            unit_grad[2] = np.cos(thetas[th])
            for c in range(use_double_acq+1):
                rf.phase_offset = (rf_phase / 180) * np.pi
                adc.phase_offset = (rf_phase / 180) * np.pi
                # (no need to change RF frequency?)
                rf_inc = np.mod(rf_inc + rf_spoiling_inc, 360.0)
                rf_phase = np.mod(rf_phase + rf_inc, 360.0)

                # Reverse slice select amplitude if using double acqs
                gz.amplitude = ((-1)**int(use_double_acq))*gz.amplitude
                gz_reph.amplitude = ((-1)**int(use_double_acq))*gz_reph.amplitude

                gpx, gpy, gpz = make_oblique_gradients(gro_pre, unit_grad)
                grx, gry, grz = make_oblique_gradients(gro, unit_grad)
                gsx, gsy, gsz = make_oblique_gradients(gro_spoil, unit_grad)

                seq.add_block(rf, gz)
                seq.add_block(gpx, gpy, gz_reph)
                seq.add_block(grx, gry, adc)
                seq.add_block(gsx, gsy, make_delay(delay_TR))

                print(f'Spokes: {u+1}/{Nline}')
                ktraj[u, :, :] = get_ktraj_3d(grx, gry, grz, adc, [gpx], [gpy], [gpz])
                u += 1



    ok, error_report = seq.check_timing()  # Check whether the timing of the sequence is correct
    if ok:
        print('Timing check passed successfully')
    else:
        print('Timing check failed. Error listing follows:')
        [print(e) for e in error_report]


    if save_seq:
        seq.write(f'ute_3d_rw_s{s}_N{N}_FOV{FOV}_TR{TR}_TE{TE}_C={use_double_acq+1}_rfcenter{rf_center}.seq')
        savemat(f'ktraj_ute_3d_rw_s{s}_N{N}_FOV{FOV}_TR{TR}_TE{TE}_C={use_double_acq+1}_rfcenter{rf_center}.mat',
                {'ktraj':ktraj})

    return seq, TE


def get_radk_params_3D(dr, fov):
    kmax = 1/(2*dr)
    # Total number of spokes
    Ns = np.round(4*np.pi*((fov*kmax)**2)) # From the criterion that solid surface per spoke = (Nyquist dk)^2
    dtheta = 1/(kmax*fov) # From planar radial Nyquist criterion
    Ntheta = np.ceil(np.pi/dtheta)
    Ntheta = Ntheta + np.mod(Ntheta,2)# Make it even to have 180-deg opposed spokes
    Nphi = np.ceil(Ns/Ntheta) # Fill the 2nd dimension with enough phi resolution so total number of spokes is satisfied
    Nphi = Nphi + np.mod(Nphi,2) # Make it even to have 180-deg opposed spokes
    Ns = Ntheta*Nphi
    return int(Ns), int(Ntheta), int(Nphi)


if __name__ == '__main__':
    # N = 64
    # FA = 15
    # TR = 100e-3
    # seq, ktraj, TE = write_UTE_3D(N=N, FOV=256e-3, slab_thk=30e-3, FA=FA, TR=TR)
    # seq.plot(time_range=[0,40e-3])
    # #print(seq.test_report())
    # name = f'ute_3D_031121_N{N}_FA{FA}_TR{TR}'
    # savemat(f'./seqs/ktraj_{name}.mat', {'ktraj': ktraj, 'TE': TE})
    # seq.write(f'./seqs/{name}.seq')

    # Fully rewound
    seq, TE = write_UTE_3D_rf_spoiled(N=64, FOV=250e-3, slab_thk=200e-3, FA=5, TR=10e-3, ro_asymmetry=0,
                            use_double_acq=False, rf_center=0.5, os_factor=1, save_seq=True)

    seq.plot(time_range=[0,60e-3])
    print(seq.test_report())

    # original UTE style (except for N)
    #write_UTE_3D_rf_spoiled(N=256, FOV=250e-3, slab_thk=200e-3, FA=10, TR=10e-3, ro_asymmetry=0.97,
     #                       use_double_acq=True, rf_center=1, os_factor=2, save_seq=True)