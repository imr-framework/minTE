# 2D ultra-short TE sequence (radial encoding)
from math import pi
import numpy as np
from pypulseq.Sequence.sequence import Sequence
from pypulseq.calc_duration import calc_duration
from pypulseq.make_adc import make_adc
from pypulseq.make_delay import make_delay
from pypulseq.make_sinc_pulse import make_sinc_pulse
from pypulseq.make_trap_pulse import make_trapezoid
from pypulseq.make_extended_trapezoid import make_extended_trapezoid
from pypulseq.opts import Opts
import matplotlib.pyplot as plt
from scipy.io import savemat, loadmat
from pypulseq.points_to_waveform import points_to_waveform
from sequence_helpers import *

# System options (copied from : amri-sos service form)
def write_UTE_2D_original(N, FOV, thk, FA, TE, TR, T_rf=1.5e-3, minTE=False):
    # System info for June
    system = Opts(max_grad=32, grad_unit='mT/m', max_slew=130, slew_unit='T/m/s',
                  rf_ringdown_time=30e-6, rf_dead_time=100e-6, adc_dead_time=20e-6)
    seq = Sequence(system=system)

    # Derived parameters
    flip = FA * np.pi / 180
    N_spoke = int(np.ceil(pi * N))  # Minimum integer number of spokes needed at Nyquist limit
    dtheta = 2 * pi / N_spoke
    thetas = np.arange(0, N_spoke * dtheta, dtheta)
    ktraj = np.zeros([N_spoke, N], dtype=complex)

    # Slice selection RF pulse and gradient
    rf, gz, gzr = make_sinc_pulse(flip_angle=flip, system=system, duration=T_rf,
                                  slice_thickness=thk, apodization=0.5, time_bw_product=4)
    # Readout gradient prototype
    dk = 1 / FOV
    k_width = N * dk
    ro_time = 6.4e-3
    gx = make_trapezoid(channel='x', system=system, flat_area=k_width, flat_time=ro_time)
    adc = make_adc(num_samples=N, duration=gx.flat_time, delay=gx.rise_time)

    # Slice refocusing and spoiler (after readout)
    pre_time = 8e-4
    reph_dur = 1e-3
    gz_reph = make_trapezoid(channel='z', system=system, area=-gz.area / 2, duration=reph_dur)
    gz_spoil = make_trapezoid(channel='z', system=system, area=gz.area * 2, duration=3 * pre_time)

    # Timing
    if minTE:
        TE_fill = 0
        minTE = gx.rise_time + calc_duration(gz_reph) + calc_duration(gz) / 2
        print(f"The minimum TE is {1000 * minTE} ms.")
        TE = minTE
    else:
        TE_fill = TE - (calc_duration(gz_reph) + calc_duration(gz) / 2 + gx.rise_time)

    TR_fill = TR - TE_fill - calc_duration(gz) - calc_duration(gz_reph) - calc_duration(gx) - calc_duration(gz_spoil)

    delay1 = make_delay(TE_fill)
    delay2 = make_delay(TR_fill)

    # Add blocks to seq object
    for ns in range(N_spoke):  # for each spoke
        seq.add_block(rf, gz)  # Slice selection
        seq.add_block(gz_reph)  # Slice refocusing

        # Make oblique readout gradients and record kspace trajectory
        k_width_projx = k_width * np.cos(thetas[ns])
        k_width_projy = k_width * np.sin(thetas[ns])

        # Make oblique readout gradients (x and y)
        gx = make_trapezoid(channel='x', system=system, flat_area=k_width_projx, flat_time=ro_time)
        gy = make_trapezoid(channel='y', system=system, flat_area=k_width_projy, flat_time=ro_time)

        ## TODO make ktraj function!
        ktraj[ns, :] = get_ktraj(gx, gy, adc, display=False)

        seq.add_block(delay1)
        seq.add_block(gx, gy, adc)
        seq.add_block(gz_spoil)
        seq.add_block(delay2)

        # Save k-space trajectory

    # Save sequence
    return seq, ktraj, TE

def write_UTE_2D_original_oblique(N, FOV, thk, FA, TE, TR, T_rf = 1.5e-3, minTE=False, enc='xyz'):
    # System info for June
    system = Opts(max_grad=32, grad_unit='mT/m', max_slew=130, slew_unit='T/m/s',
                  rf_ringdown_time=30e-6, rf_dead_time=100e-6, adc_dead_time=20e-6)
    seq = Sequence(system=system)

    # Encoding orientation
    ug_ro1, ug_ro2, ug_ss = parse_enc(enc)

    # Derived parameters
    flip = FA*np.pi/180
    N_spoke = int(np.ceil(pi*N))# Minimum integer number of spokes needed at Nyquist limit
    dtheta = 2*pi / N_spoke
    thetas = np.arange(0, N_spoke*dtheta, dtheta)
    ktraj = np.zeros([N_spoke, N],dtype=complex)


    # Slice selection RF pulse and gradient
    rf, gz, gzr = make_sinc_pulse(flip_angle=flip, system=system, duration=T_rf,
                             slice_thickness=thk, apodization=0.5, time_bw_product = 4)
    g_ss_x, g_ss_y, g_ss_z = make_oblique_gradients(gz, ug_ss)

    # Readout gradient prototype
    dk = 1/FOV
    k_width = N*dk
    ro_time = 6.4e-3
    g_ro = make_trapezoid(channel='x', system=system, flat_area=k_width, flat_time=ro_time)

    adc = make_adc(num_samples=N, duration=g_ro.flat_time, delay=g_ro.rise_time)

    # Slice refocusing and spoiler (after readout)
    pre_time = 8e-4
    reph_dur = 1e-3
    gz_reph = make_trapezoid(channel='z',system=system, area=-gz.area/2, duration=reph_dur)
    g_ss_reph_x, g_ss_reph_y, g_ss_reph_z = make_oblique_gradients(gz_reph, ug_ss)

    gz_spoil = make_trapezoid(channel='z', system=system, area=gz.area*2, duration=3*pre_time)
    g_ss_spoil_x, g_ss_spoil_y, g_ss_spoil_z = make_oblique_gradients(gz, ug_ss)


    # Timing
    if minTE:
        TE_fill = 0
        minTE = g_ro.rise_time + calc_duration(gz_reph) + calc_duration(gz)/2
        print(f"The minimum TE is {1000*minTE} ms.")
        TE = minTE
    else:
        TE_fill = TE - (calc_duration(gz_reph) + calc_duration(gz)/2 + g_ro.rise_time)

    TR_fill = TR - TE_fill - calc_duration(gz) - calc_duration(gz_reph) - calc_duration(g_ro) - calc_duration(gz_spoil)

    delay1 = make_delay(TE_fill)
    delay2 = make_delay(TR_fill)

    # Add blocks to seq object
    for ns in range(N_spoke): # for each spoke

        # Record k-space trajectory using orthogonal x/y gradients
        k_width_projx = k_width * np.cos(thetas[ns])
        k_width_projy = k_width * np.sin(thetas[ns])
        gx = make_trapezoid(channel='x', system=system, flat_area=k_width_projx, flat_time=ro_time)
        gy = make_trapezoid(channel='y', system=system, flat_area=k_width_projy, flat_time=ro_time)

        # Make oblique readout gradients a
        # Make oblique readout gradients (x and y)
        g_ro_x, g_ro_y, g_ro_z = combine_oblique_radial_readout_2d(g_ro, ug_ro1, ug_ro2, thetas[ns])


        ## TODO make ktraj function!
        ktraj[ns, :] = get_ktraj(gx, gy, adc, display=False)

        seq.add_block(rf, g_ss_x, g_ss_y, g_ss_z) # Slice selection
        seq.add_block(g_ss_reph_x, g_ss_reph_y, g_ss_reph_z) # Slice refocusing
        seq.add_block(delay1)
        seq.add_block(g_ro_x,g_ro_y,g_ro_z, adc)
        seq.add_block(g_ss_spoil_x, g_ss_spoil_y, g_ss_spoil_z)
        seq.add_block(delay2)



    # Save k-space trajectory

    # Save sequence
    return seq, ktraj, TE

def write_UTE_2D_splitgrad_minTE(N, FOV, thk, FA, TR, T_rf = 1.5e-3):
    # System info for June
    system = Opts(max_grad=32, grad_unit='mT/m', max_slew=130, slew_unit='T/m/s',
                  rf_ringdown_time=30e-6, rf_dead_time=100e-6, adc_dead_time=20e-6)
    seq = Sequence(system=system)

    # Derived parameters
    flip = FA*np.pi/180
    N_spoke = int(np.ceil(pi*N))# Minimum integer number of spokes needed at Nyquist limit
    dtheta = 2*pi / N_spoke
    thetas = np.arange(0, N_spoke*dtheta, dtheta)
    ktraj = np.zeros([N_spoke, N],dtype=complex)


    # Readout gradient prototype
    dk = 1/FOV
    k_width = N*dk
    ro_time = 6.4e-3
    g_ro = make_trapezoid(channel='x', system=system, flat_area=k_width, flat_time=ro_time)
    adc = make_adc(num_samples=N, duration=g_ro.flat_time, delay=0)

    # Slice selection RF pulse and gradient
    rf, gz, gzr = make_sinc_pulse(flip_angle=flip, system=system, duration=T_rf,
                             slice_thickness=thk, apodization=0.5, time_bw_product = 4)
    # Slice refocusing and spoiler (after readout)
    pre_time = 8e-4
    reph_dur = 1e-3
    #gz_reph = make_trapezoid(channel='z',system=system, area=-gz.area/2, duration=reph_dur, rise_time=g_ro.rise_time)
    gz_reph = make_trapezoid(channel='z',system=system, area=-gz.area/2, rise_time = g_ro.rise_time)
    gz_spoil = make_trapezoid(channel='z', system=system, area=gz.area*2, duration=3*pre_time)

    # Split slice gradients # TODO
    gz1 = gz
    gz2 = make_extended_trapezoid(channel='z', system=system, times=np.array([0,gz_reph.rise_time, gz_reph.rise_time+gz_reph.flat_time]),
                                               amplitudes = np.array([0,gz_reph.amplitude,gz_reph.amplitude]))

    gz3 = make_extended_trapezoid(channel='z', times=np.array([0, gz_reph.fall_time]), amplitudes = np.array([gz_reph.amplitude,0]))


    # Timing
    minTE = calc_duration(gz_reph) + calc_duration(gz)/2
    print(f"The minimum TE is {1000*minTE} ms.")
    TE = minTE
    TR_fill = TR - TE - (calc_duration(g_ro) - g_ro.rise_time) - calc_duration(gz_spoil) - calc_duration(gz)/2
    delayTR = make_delay(TR_fill)

    for ns in range(N_spoke): # for each spoke

        # Make oblique readout gradients and record kspace trajectory
        k_width_projx = k_width * np.cos(thetas[ns])
        k_width_projy = k_width * np.sin(thetas[ns])

        # Make oblique readout gradients (x and y)
        gx = make_trapezoid(channel='x', system=system, flat_area=k_width_projx, flat_time=ro_time,
                            rise_time=gz_reph.rise_time)
        gy = make_trapezoid(channel='y', system=system, flat_area=k_width_projy, flat_time=ro_time,
                            rise_time=gz_reph.rise_time)
        # Record k-space trajectory
        ktraj[ns, :] = get_ktraj(gx, gy, adc, display=False)
        # Split readout gradients
        if gx.amplitude == 0:
            gx3 = make_trapezoid(channel='x', system=system, amplitude=0, duration=gx.rise_time-system.grad_raster_time)
            gx4 = make_trapezoid(channel='x',system=system, amplitude=0, duration=gx.flat_time)
        else:
            gx3 = make_extended_trapezoid(channel='x', system=system, times=np.array([0, gx.rise_time]), amplitudes=np.array([0, gx.amplitude]))
            gx4 = make_extended_trapezoid(channel='x', system=system, times=np.array([0, gx.flat_time, gx.flat_time + gx.fall_time]),
                                                                   amplitudes=np.array([gx.amplitude, gx.amplitude, 0]))
        if gy.amplitude == 0:
            gy3 = make_trapezoid(channel='y', system=system, amplitude=0, duration=gy.rise_time-system.grad_raster_time)
            gy4 = make_trapezoid(channel='y',system=system, amplitude=0, duration=gy.flat_time)
        else:
            gy3 = make_extended_trapezoid(channel='y', system=system, times=np.array([0, gy.rise_time]), amplitudes=np.array([0, gy.amplitude]))
            gy4 = make_extended_trapezoid(channel='y', system=system, times=np.array([0, gy.flat_time, gy.flat_time + gy.fall_time]),
                                      amplitudes=np.array([gy.amplitude, gy.amplitude, 0]))

        seq.add_block(rf, gz1) # Slice selection
        seq.add_block(gz2) # Slice refocusing
        seq.add_block(gz3, gx3, gy3)
        seq.add_block(gx4,gy4,adc)
        seq.add_block(gz_spoil)
        seq.add_block(delayTR)



    return seq, ktraj, TE

def write_UTE_2D_splitgrad_rewound(N, FOV, thk, FA, TE, TR, T_rf = 1.5e-3, minTE=False):
    # System info for June
    system = Opts(max_grad=32, grad_unit='mT/m', max_slew=130, slew_unit='T/m/s',
                  rf_ringdown_time=30e-6, rf_dead_time=100e-6, adc_dead_time=20e-6)
    seq = Sequence(system=system)

    # Derived parameters
    flip = FA * np.pi/180
    N_spoke = int(np.ceil(pi*N))# Minimum integer number of spokes needed at Nyquist limit
    dtheta = 2*pi / N_spoke
    thetas = np.arange(0, N_spoke*dtheta, dtheta)
    print(thetas)
    ktraj = np.zeros([N_spoke, N],dtype=complex)

    # Slice selection RF pulse and gradient
    rf, gz, gzr = make_sinc_pulse(flip_angle=flip, system=system, duration=T_rf, return_gz=True,
                             slice_thickness=thk, apodization=0.5, time_bw_product = 4)


    # Readout gradient prototype
    dk = 1/FOV
    k_width = N*dk
    ro_time = 6.4e-3
    gro = make_trapezoid(channel='x', system=system, flat_area=k_width, flat_time=ro_time)

    # Readout rewinder prototype
    gro_rew = make_trapezoid(channel='x',system=system, area=-gro.area/2) # Make fastest gradient allowed

    # Slice refocusing and spoiler (after readout)
    pre_time = 8e-4
    gz_reph = gzr
    gz_spoil = make_trapezoid(channel='z', system=system, area=gz.area*2, duration=3*pre_time)

    # Split the gradients
    # Up to end of RF
    #gz1 = make_extended_trapezoid(channel='z', system=system, times=np.array([0,gz.rise_time,gz.rise_time+gz.flat_time]),
    #                              amplitudes=np.array([0,gz.amplitude,gz.amplitude]))

    # Align the gradients between end of RF and beginning of ADC
    tau1 = calc_duration(gzr)
    tau2 = calc_duration(gro_rew) + gro.rise_time
    TE_min = calc_duration(gz)/2 + np.max([tau1, tau2]) + gro.flat_time / 2
    print(f'Minimum TE is {TE_min * 1000} ms.')
    dTE = TE - TE_min

    if dTE < 0:
        minTE = True
        TE = TE_min
        print(f'Specified TE is shorter than minTE. Using minimum TE = {TE_min * 1000} ms. ')

    #gz2_times = np.cumsum([0, gz.fall_time, gz_reph.rise_time, gz_reph.flat_time, gz_reph.fall_time])
    #gz2_amplitudes = np.array([gz.amplitude, 0, gz_reph.amplitude, gz_reph.amplitude, 0])
    #gz2 = make_extended_trapezoid(channel='z',system=system, times=gz2_times, amplitudes=gz2_amplitudes)
    gz2 = gz_reph

    overlapped = dTE <= np.min([tau1,tau2])
    if overlapped:
        print('Overlapped!')
        gz1 = gz
        ro_delay = 0
        if tau1 > tau2:
            ro_delay += tau1 - tau2
        if not minTE:
            ro_delay += dTE
        gro2_times = np.cumsum([0, ro_delay, gro_rew.rise_time, gro_rew.flat_time, gro_rew.fall_time, gro.rise_time])
        gro2_amplitudes = np.array([0, 0, gro_rew.amplitude, gro_rew.amplitude, 0, gro.amplitude])
        adc_no_delay = make_adc(num_samples=N, duration=gro.flat_time, delay=0)

    else:
        print('Not overlapped!')
        TE_fill = (TE - TE_min) - np.min([tau1,tau2])
        adc = make_adc(num_samples=N, duration=gro.flat_time, delay=gro.rise_time)
        delay1 = make_delay(TE_fill)

    # Beginning of ADC to end of RO gradient
    gro3_times = np.cumsum([0, gro.flat_time, gro.fall_time])
    gro3_amplitudes = np.array([gro.amplitude, gro.amplitude, 0])

    # TR delay
    TR_fill = TR - TE - calc_duration(gz) - calc_duration(gz_spoil) - calc_duration(gro)/2
    delay2 = make_delay(TR_fill)


    # Add blocks to seq object
    for ns in range(N_spoke): # for each spoke
        # Scaling constants for oblique readout gradients (x and y)
        cx, cy = np.cos(thetas[ns]), np.sin(thetas[ns])
        if overlapped: #
            gx2 = make_scaled_extended_trapezoid(channel='x', system=system, times=gro2_times,
                                          amplitudes = gro2_amplitudes, scale = cx)
            gy2 = make_scaled_extended_trapezoid(channel='y', system=system, times=gro2_times,
                                          amplitudes = gro2_amplitudes, scale = cy)
            gx3 = make_scaled_extended_trapezoid(channel='x', system=system, times=gro3_times,
                                          amplitudes = gro3_amplitudes, scale = cx)
            gy3 = make_scaled_extended_trapezoid(channel='y', system=system, times=gro3_times,
                                          amplitudes = gro3_amplitudes, scale = cy)
            seq.add_block(rf, gz1)
            seq.add_block(gz2, gx2, gy2)
            seq.add_block(gx3, gy3, adc_no_delay)
        else:
            k_width_projx = k_width * cx
            k_width_projy = k_width * cy
            gx = make_trapezoid(channel='x', system=system, flat_area=k_width_projx,
                                flat_time=ro_time, rise_time=gro.rise_time)
            gx_rew = make_trapezoid(channel='x', system=system, area= -gx.area / 2, duration=calc_duration(gro_rew))
            gy = make_trapezoid(channel='y', system=system, flat_area=k_width_projy,
                                flat_time=ro_time, rise_time = gro.rise_time)
            gy_rew = make_trapezoid(channel='y', system=system, area=-gy.area / 2, duration=calc_duration(gro_rew))

            ktraj[ns, :] = get_ktraj_with_rew(gx, gx_rew, gy, gy_rew, adc, display=False)

            seq.add_block(rf, gz)
            seq.add_block(gz_reph)
            seq.add_block(delay1)
            seq.add_block(gx_rew, gy_rew)
            seq.add_block(gx, gy, adc)

        seq.add_block(gz_spoil)
        seq.add_block(delay2)

    # Save sequence
    return seq, ktraj, TE, thetas

def write_UTE_2D_rf_spoiled(N=250, Nr=128, FOV=250e-3, thk=3e-3, FA=10, TR=10e-3, ro_asymmetry=0.97,
                            use_half_pulse=True, rf_dur=1e-3, TE_use=None, rewinder_size=None):
    """
    Parameters
    ----------
    N : int
        Matrix size
    Nr : int
        Number of radial spokes
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
    rf_dur : float
        RF pulse duration in [seconds]


    """
    # Adapted from pypulseq demo write_ute.py (obtained mid-2021)
    system = Opts(max_grad=32, grad_unit='mT/m', max_slew=130, slew_unit='T/m/s',
                  rf_ringdown_time=30e-6, rf_dead_time=100e-6, adc_dead_time=20e-6)
    seq = Sequence(system=system)

    # Derived parameters
    dphi = 2 * np.pi / Nr #
    ro_duration = 2.5e-3
    ro_os = 2 # Oversampling factor
    #minRF_to_ADC_time = 50e-6 # this was not used.

    rf_spoiling_inc = 117 # RF spoiling increment value.

    # Sequence components
    if use_half_pulse:
        cp = 1
    else:
        cp = 0.5

    rf, gz, gz_reph = make_sinc_pulse(flip_angle=FA*np.pi/180, duration=rf_dur, slice_thickness=thk, apodization=0.5,
                                      time_bw_product=4, center_pos=cp, system=system, return_gz=True)
    gz_ramp_reph = make_trapezoid(channel='z',area=-gz.fall_time*gz.amplitude/2,system=system)


    ##############################
    if rewinder_size is not None:
        if rewinder_size < 0:
            raise ValueError("Rewinder size must be zero or positive")
        else:
            modify_gradient(gz_reph,scale=rewinder_size)
    ###################################


    # Asymmetry! (0 - fully rewound; 1 - hall-echo)
    Nro = np.round(ro_os*N) # Number of readout points
    s = np.round(ro_asymmetry * Nro / 2 ) / (Nro / 2)
    dk = (1/FOV) / (1+s)
    ro_area = N * dk
    gro = make_trapezoid(channel='x', flat_area=ro_area, flat_time=ro_duration, system=system)
    adc = make_adc(num_samples=Nro, duration=gro.flat_time, delay=gro.rise_time, system=system)
    gro_pre = make_trapezoid(channel='x', area= - (gro.area - ro_area)/2 - (ro_area/2) * (1-s), system=system)

    # Spoilers
    gro_spoil = make_trapezoid(channel='x',area=0.2*N*dk, system=system)

    # Calculate timing
    if use_half_pulse and rewinder_size is None:
        TE = gz.fall_time + calc_duration(gro_pre, gz_ramp_reph) + gro.rise_time + adc.dwell * Nro / 2 * (1-s)
        delay_TR = np.ceil((TR - calc_duration(gro_pre, gz_ramp_reph) - calc_duration(gz) - calc_duration(gro)) / seq.grad_raster_time) * seq.grad_raster_time
        if calc_duration(gz_ramp_reph) > calc_duration(gro_pre):
            gro_pre.delay = calc_duration(gz_ramp_reph) - calc_duration(gro_pre)
    else:
        TE = gz.fall_time + calc_duration(gro_pre, gz_reph) + gro.rise_time + adc.dwell * Nro / 2 * (1 - s)
        delay_TR = np.ceil((TR - calc_duration(gro_pre, gz_reph) - calc_duration(gz) - calc_duration(gro)) / seq.grad_raster_time) * seq.grad_raster_time
        if calc_duration(gz_reph) > calc_duration(gro_pre):
            gro_pre.delay = calc_duration(gz_reph) - calc_duration(gro_pre)

    assert np.all(delay_TR >= calc_duration(gro_spoil)) # The TR delay starts at the same time as the spoilers!

    if TE_use is None:
        print(f'TE = {TE*1e6:.0f} us')
    elif TE_use <= TE:
        print(f'Desired TE is lower than minimal TE. Using minimal TE ={TE*1e6:.0f} us')
    else:
        TE_delay = np.ceil((TE_use - TE)/seq.grad_raster_time) * seq.grad_raster_time
        print(f'Minimal TE: {TE*1e6:.0f} us; using TE: {(TE+TE_delay)*1e6:.0f} us')
        TE += TE_delay

    # Starting RF phase and increments
    rf_phase = 0
    rf_inc = 0

    ktraj = np.zeros((2*Nr, int(adc.num_samples)),dtype=complex)
    u = 0

    C = int(use_half_pulse) + 1

    for spoke_ind in range(Nr): # for each spoke
        for c in range(C): # why?
            rf.phase_offset = (rf_phase / 180) * np.pi
            adc.phase_offset = (rf_phase / 180) * np.pi
            # (no need to change RF frequency?)
            rf_inc = np.mod(rf_inc + rf_spoiling_inc, 360.0)
            rf_phase = np.mod(rf_phase + rf_inc, 360.0)

            # Reverse slice select amplitude (always z = 0)
            if use_half_pulse:
                modify_gradient(gz, scale=-1)
                modify_gradient(gz_reph, scale=-1)
                modify_gradient(gz_ramp_reph, scale=-1)

            seq.add_block(rf, gz)
            phi = dphi * spoke_ind

            ug2d = [np.cos(phi),np.sin(phi),0]

            gpx, gpy, __ = make_oblique_gradients(gro_pre, ug2d)
            grx, gry, __ = make_oblique_gradients(gro, ug2d)
            gsx, gsy, __ = make_oblique_gradients(gro_spoil, ug2d)

            if TE_use is not None:
                seq.add_block(make_delay(TE_delay))

            if use_half_pulse and rewinder_size is None:
                seq.add_block(gpx, gpy, gz_ramp_reph)
            else:
                seq.add_block(gpx, gpy, gz_reph)

            seq.add_block(grx, gry, adc)
            seq.add_block(gsx, gsy, make_delay(delay_TR))

            #ktraj[u, :, :] = get_ktraj_3d(grx, gry, grz, adc, [gpx], [gpy], [gpz])
            ktraj[u, :] = get_ktraj_with_rew(grx, gpx, gry, gpy, adc, display=False)
            u += 1

    ok, error_report = seq.check_timing()  # Check whether the timing of the sequence is correct
    if ok:
        print('Timing check passed successfully')
    else:
        print('Timing check failed. Error listing follows:')
        [print(e) for e in error_report]


    return seq, TE, ktraj


def combine_oblique_radial_readout_2d(g, ug1, ug2, theta):
    """
    Inputs
    ------
    g : Pypulseq Gradient object
        Base gradient to be made into oblique gradients
    ug1 : array_like
        Length 3, unit gradient in first direction
    ug2 :
        Length 3, unit gradient in second direction
    theta : float
        Angle in radians of specific radial readout spoke
    Returns
    -------
    gx, gy, gz : Pypulseq Gradient objects in each direction
    """
    # Check that ug1 and ug2 are orthogonal
    if np.dot(ug1,ug2) != 0:
        raise ValueError("The two directions provided must be orthogonal.")
    ug_net = np.array(ug1) * np.cos(theta) + np.array(ug2) * np.sin(theta)
    gx, gy, gz = make_oblique_gradients(g, ug_net)
    return gx, gy, gz

if __name__ == '__main__':
    # Try it!

     seq, TE, ktraj = write_UTE_2D_rf_spoiled(N=256, Nr=804, FOV=250e-3, thk=5e-3, FA=10, TR=10e-3,
                                              ro_asymmetry=0.97, use_half_pulse=True, rf_dur=1e-3,
                                              TE_use=None)
     #print(seq.test_report())

     #seq.write('ute_2d_halfpulse_gz_ramp_reph_082721.seq')
     #savemat('ktraj_ute_2d_halfpulse_gz_ramp_reph_082721.mat',{'ktraj':ktraj})

     #seq.plot(time_range=[0,30e-3])