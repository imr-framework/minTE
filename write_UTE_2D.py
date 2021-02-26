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


# System options (copied from : amri-sos service form)
def write_UTE_2D_original(N, FOV, thk, FA, TE, TR, T_rf = 1.5e-3, minTE=False):
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

    # Slice selection RF pulse and gradient
    rf, gz, gzr = make_sinc_pulse(flip_angle=flip, system=system, duration=T_rf,
                             slice_thickness=thk, apodization=0.5, time_bw_product = 4)
    # Readout gradient prototype
    dk = 1/FOV
    k_width = N*dk
    ro_time = 6.4e-3
    gx = make_trapezoid(channel='x', system=system, flat_area=k_width, flat_time=ro_time)
    adc = make_adc(num_samples=N, duration=gx.flat_time, delay=gx.rise_time)

    # Slice refocusing and spoiler (after readout)
    pre_time = 8e-4
    reph_dur = 1e-3
    gz_reph = make_trapezoid(channel='z',system=system, area=-gz.area/2, duration=reph_dur)
    gz_spoil = make_trapezoid(channel='z', system=system, area=gz.area*2, duration=3*pre_time)

    # Timing
    if minTE:
        TE_fill = 0
        minTE = gx.rise_time + calc_duration(gz_reph) + calc_duration(gz)/2
        print(f"The minimum TE is {1000*minTE} ms.")
        TE = minTE
    else:
        TE_fill = TE - (calc_duration(gz_reph) + calc_duration(gz)/2 + gx.rise_time)

    TR_fill = TR - TE_fill - calc_duration(gz) - calc_duration(gz_reph) - calc_duration(gx) - calc_duration(gz_spoil)

    delay1 = make_delay(TE_fill)
    delay2 = make_delay(TR_fill)

    # Add blocks to seq object
    for ns in range(N_spoke): # for each spoke
        seq.add_block(rf, gz) # Slice selection
        seq.add_block(gz_reph) # Slice refocusing

        # Make oblique readout gradients and record kspace trajectory
        k_width_projx = k_width * np.cos(thetas[ns])
        k_width_projy = k_width * np.sin(thetas[ns])

        # Make oblique readout gradients (x and y)
        gx = make_trapezoid(channel='x', system=system, flat_area=k_width_projx, flat_time=ro_time)
        gy = make_trapezoid(channel='y', system=system, flat_area=k_width_projy, flat_time=ro_time)

        ## TODO make ktraj function!
        ktraj[ns, :] = get_ktraj(gx, gy, adc, display=False)

        seq.add_block(delay1)
        seq.add_block(gx,gy,adc)
        seq.add_block(gz_spoil)
        seq.add_block(delay2)



    # Save k-space trajectory

    # Save sequence
    return seq, ktraj, TE

# TODO
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

# TODO
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
    ktraj = np.zeros([N_spoke, N],dtype=complex)

    # Slice selection RF pulse and gradient
    rf, gz, gzr = make_sinc_pulse(flip_angle=flip, system=system, duration=T_rf,
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
    return seq, ktraj, TE

def write_UTE_3D(N, FOV, FA, TE, TR):

    return seq, ktraj


# TODO: Helpers
def get_ktraj(gx, gy, adc, display=False):
    sampled_times = np.linspace(0, gx.flat_time, adc.num_samples)
    kx_pre = 0.5 * gx.amplitude * gx.rise_time
    ky_pre = 0.5 * gy.amplitude * gy.rise_time

    kx = kx_pre + gx.amplitude * sampled_times
    ky = ky_pre + gy.amplitude * sampled_times

    if display:
        plt.figure(1001)
        plt.plot(kx, ky, 'ro-')
        plt.xlabel('kx')
        plt.ylabel('ky')
        plt.show()

    return (kx + 1j * ky)

def get_ktraj_with_rew(gx, gx_rew, gy, gy_rew, adc, display=False):
    sampled_times = np.linspace(0, gx.flat_time, adc.num_samples)
    kx_pre = 0.5 * gx.amplitude * gx.rise_time + gx_rew.area
    ky_pre = 0.5 * gy.amplitude * gy.rise_time + gy_rew.area

    kx = kx_pre + gx.amplitude * sampled_times
    ky = ky_pre + gy.amplitude * sampled_times

    if display:
        plt.figure(1001)
        plt.plot(kx, ky, 'ro-')
        plt.xlabel('kx')
        plt.ylabel('ky')
        plt.show()

    return (kx + 1j * ky)

def make_scaled_extended_trapezoid(channel, system, times, amplitudes, scale):
    # Remove non-distinct timing
    inds_of_no_change = np.array(np.where(np.diff(times) == 0)) + 1
    for ind in inds_of_no_change:
        times = np.delete(times, ind)
        amplitudes = np.delete(amplitudes, ind)
    if scale == 0:
        return make_trapezoid(channel=channel, system=system, duration=times[-1]-times[0], amplitude = 0)
    else:
        return make_extended_trapezoid(channel=channel, system=system, times=times, amplitudes = scale*amplitudes)



if __name__ == '__main__':
    # Original (TE = 5 ms)
 #   seq, k_traj, TE = write_UTE_2D_original(N=256, FOV=250e-3, thk=5e-3, FA=15, TE=5e-3, TR=20e-3, T_rf = 1.5e-3, minTE=True)
  #  print(seq.test_report())
    #seq.plot(time_range=[0, 40e-3])
  #  name = 'ute_minTE_2D_original_N256-FOV250-thk5-FA15-TR20_022221'
    #seq.write(f'./seqs/{name}.seq')
  #  savemat(f'./seqs/ktraj_{name}.mat',{'ktraj':k_traj, 'TE':TE})


    # Split grad version with minTE
    seq, k_traj, TE = write_UTE_2D_splitgrad_minTE(N=256, FOV=250e-3, thk=5e-3, FA=15, TR=20e-3, T_rf = 1.5e-3)
    # print(seq.test_report())
    # seq.plot(time_range=[0, 40e-3])
    # name = 'ute_minTE_2D_split_grad_N256-FOV250-thk5-FA15-TR20_022221'
    # seq.write(f'./seqs/{name}.seq')
    # savemat(f'./seqs/ktraj_{name}.mat',{'ktraj':k_traj, 'TE':TE})

    # Split grad version with rewinder
    #seq, ktraj, TE = write_UTE_2D_splitgrad_rewound(N=256, FOV=250e-3, thk=5e-3, FA=15, TE=10e-3, TR=100e-3, T_rf=1.5e-3, minTE=False)
   # name = 'ute_rewound_TE10_longTR100_022321'

    #seq, ktraj, TE = write_UTE_2D_splitgrad_rewound(N=256, FOV=250e-3, thk=5e-3, FA=15, TE=0, TR=20e-3, T_rf=1.5e-3, minTE=True)
    #name = 'ute_rewound_minTE_022321'

  #  seq.write(f'./seqs/{name}.seq')
  #  savemat(f'./seqs/ktraj_{name}.mat',{'ktraj':ktraj, 'TE':TE})
 #   seq.plot(time_range=[0,100e-3])
    #print(seq.test_report())