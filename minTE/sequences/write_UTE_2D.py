# 2D ultra-short TE sequence (radial encoding)
import numpy as np
from pypulseq.Sequence.sequence import Sequence
from pypulseq.calc_duration import calc_duration
from pypulseq.make_adc import make_adc
from pypulseq.make_delay import make_delay
from pypulseq.make_sinc_pulse import make_sinc_pulse
from pypulseq.opts import Opts
from minTE.sequences.sequence_helpers import *
from pypulseq.calc_rf_center import calc_rf_center
from scipy.io import savemat

def write_UTE_2D_rf_spoiled(N=250, Nr=128, FOV=250e-3, thk=3e-3, slice_locs=[0], FA=10, TR=10e-3, ro_asymmetry=0.97,
                            use_half_pulse=True, rf_dur=1e-3, enc='xyz',TE_use=None, save_seq=True):
    """
    Single-slice 2D partially rewound UTE with RF spoiling
    (cirte)

    Parameters
    ----------
    N : int, default=250
        Matrix size
    Nr : int, default=128
        Number of radial spokes
    FOV : float, default=250
        Field-of-view in [meters]
    thk : float, default=3
        Slice thickness in [meters]
    slice_locs : float, default=[0]
        Slice location in [meters]
    FA : float, default=10
        Flip angle in [degrees]
    TR : float, default=0.01
        Repetition time in [seconds]
    ro_asymmetry : float, default=0.97
        The ratio A/B where a A/(A+B) portion of 2*Kmax is omitted and B/(A+B) is acquired.
    use_half_pulse: bool, default=True
        Whether to use half pulse excitation for shorter TE;
        This doubles both the number of excitations and acquisition time
    rf_dur : float, default=0.001
        RF pulse duration in [seconds]
    enc : str
        Orthogonal encoding string. 'xyz' means readout in x, phase enc. in y, and slice in z.
        Allowed: 'xyz', 'xzy', 'yzx', 'yxz', 'zxy', 'zyx'
    TE_use : float, default=None
        Desired echo time in [seconds]. If shorter than feasible, minimum TE is used.
    save_seq : bool, default=True
        Whether to save this sequence as a .seq file

    Returns
    -------
    seq : Sequence
        PyPulseq 2D UTE sequence object
    TE : float
        Echo time of generated sequence in [seconds]
    ktraj : np.ndarray
        Complex k-space trajectory (kx + 1j*ky)
        with size [spoke, readout sample]
    """

    # Adapted from pypulseq demo write_ute.py (obtained mid-2021)
    system = Opts(max_grad=32, grad_unit='mT/m', max_slew=130, slew_unit='T/m/s',
                  rf_ringdown_time=30e-6, rf_dead_time=100e-6, adc_dead_time=20e-6)
    seq = Sequence(system=system)

    # Derived parameters
    dphi = 2 * np.pi / Nr #
    ro_duration = 2.5e-3
    ro_os = 2 # Oversampling factor

    rf_spoiling_inc = 117 # RF spoiling increment value.

    # Encoding
    ch_ro1 = enc[0]
    ch_ro2 = enc[1]
    ch_ss = enc[2]

    # Sequence components
    if use_half_pulse:
        cp = 1
    else:
        cp = 0.5

    # RF pulse, slice selecting gradient, and gradient to rephase down-ramp (half pulse only)
    rf, gz, gz_reph = make_sinc_pulse(flip_angle=FA*np.pi/180, duration=rf_dur, slice_thickness=thk, apodization=0.5,
                                      time_bw_product=4, center_pos=cp, system=system, return_gz=True)
    modify_gradient(gz,scale=1,channel=ch_ss)
    modify_gradient(gz_reph,scale=1,channel=ch_ss)
    gz_ramp_reph = make_trapezoid(channel=ch_ss,area=-gz.fall_time*gz.amplitude/2,system=system)

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
    # TR delay (multislice)
    if use_half_pulse:
        TE = gz.fall_time + calc_duration(gro_pre, gz_ramp_reph) + gro.rise_time + adc.dwell * Nro / 2 * (1-s)
        if calc_duration(gz_ramp_reph) > calc_duration(gro_pre):
            gro_pre.delay = calc_duration(gz_ramp_reph) - calc_duration(gro_pre)
        time_per_slice = calc_duration(gz) + calc_duration(gro_pre,gz_ramp_reph) + calc_duration(gro)
        delay_TR_per_slice = (TR - len(slice_locs)*time_per_slice)/len(slice_locs)
    else:
        TE = gz.fall_time + calc_duration(gro_pre, gz_reph) + gro.rise_time + adc.dwell * Nro / 2 * (1 - s)
        if calc_duration(gz_reph) > calc_duration(gro_pre):
            gro_pre.delay = calc_duration(gz_reph) - calc_duration(gro_pre)
        time_per_slice = calc_duration(gz) + calc_duration(gro_pre, gz_reph) + calc_duration(gro)
        delay_TR_per_slice = (TR - len(slice_locs)*time_per_slice)/len(slice_locs)

    if delay_TR_per_slice < 0:
        raise ValueError(f"TR = {TR*1e3}ms is not long enough to accomondate {len(slice_locs)} slices!")

    delay_TR_per_slice = np.ceil(delay_TR_per_slice/seq.grad_raster_time)*seq.grad_raster_time

    # The TR delay starts at the same time as the spoilers!
    assert np.all(delay_TR_per_slice >= calc_duration(gro_spoil))
    # TE delay (if longer than minimal TE is desired)
    TE_delay = 0
    if TE_use is None:
        print(f'TE = {TE*1e6:.0f} us')
    elif TE_use <= TE:
        print(f'Desired TE is lower than minimal TE. Using minimal TE ={TE*1e6:.0f} us')
    else:
        TE_delay = np.ceil((TE_use - TE)/seq.grad_raster_time) * seq.grad_raster_time
        print(f'Minimal TE: {TE*1e6:.0f} us; using TE: {(TE+TE_delay)*1e6:.0f} us')
        TE += TE_delay


    # Starting RF phase and increments for RF spoiling
    rf_phase = 0
    rf_inc = 0

    # Set up k-space trajectory storage
    ktraj = np.zeros((Nr, int(adc.num_samples)),dtype=complex)
    u = 0

    # Full Pulse: C = 1
    # Half pulse: C = 2
    C = int(use_half_pulse) + 1

    ind_ro1 = 'xyz'.find(ch_ro1)
    ind_ro2 = 'xyz'.find(ch_ro2)


    for spoke_ind in range(Nr): # for each spoke
        phi = dphi * spoke_ind
        #ug2d = [np.cos(phi), np.sin(phi), 0]

        ug2d = [0,0,0]
        ug2d[ind_ro1] = np.cos(phi)
        ug2d[ind_ro2] = np.sin(phi)
        gpx, gpy, gpz = make_oblique_gradients(gro_pre, ug2d)
        grx, gry, grz = make_oblique_gradients(gro, ug2d)
        gsx, gsy, gsz = make_oblique_gradients(gro_spoil, ug2d)
        gp = [gpx,gpy,gpz]
        gr = [grx,gry,grz]
        gs = [gsx,gsy,gsz]
        # Extract correct gradients
        gpr1 = gp[ind_ro1]
        gpr2 = gp[ind_ro2]
        grr1 = gr[ind_ro1]
        grr2 = gr[ind_ro2]
        gsr1 = gr[ind_ro1]
        gsr2 = gr[ind_ro2]



        for c in range(C): # half pulse option; C = 2 if using half pulse
            for s in range(len(slice_locs)): # interleaved slices
                # RF spoiling phase calculations
                rf.phase_offset = (rf_phase / 180) * np.pi
                adc.phase_offset = (rf_phase / 180) * np.pi
                # (no need to change RF frequency?)
                rf_inc = np.mod(rf_inc + rf_spoiling_inc, 360.0)
                rf_phase = np.mod(rf_phase + rf_inc, 360.0)
                # RF slice selection freq/phase offsets
                rf.freq_offset = gz.amplitude * slice_locs[s]
                rf.phase_offset = rf.phase_offset - 2 * np.pi * rf.freq_offset * calc_rf_center(rf)[0]

                # Reverse slice select amplitude (always z = 0)
                if use_half_pulse:
                    rf.freq_offset = -1*rf.freq_offset
                    modify_gradient(gz, scale=-1)
                    modify_gradient(gz_reph, scale=-1)
                    modify_gradient(gz_ramp_reph, scale=-1)

                # Add blocks to seq
                seq.add_block(rf, gz)
                if TE_use is not None:
                    seq.add_block(make_delay(TE_delay))

                if use_half_pulse:
                    seq.add_block(gpr1, gpr2, gz_ramp_reph)
                else:
                    seq.add_block(gpr1, gpr2, gz_reph)

                seq.add_block(grr1, grr2, adc)
                seq.add_block(gsr1, gsr2, make_delay(delay_TR_per_slice))

        ktraj[u, :] = get_ktraj_with_rew(grr1, gpr1, grr2, gpr2, adc, display=False)

        u += 1

    ok, error_report = seq.check_timing()  # Check whether the timing of the sequence is correct
    if ok:
        print('Timing check passed successfully')
    else:
        print('Timing check failed. Error listing follows:')
        [print(e) for e in error_report]


    if save_seq:
        seq.write(f'ute_2d_rw_s{s}_N{N}_FOV{FOV}_TR{TR}_TE{TE}_C={use_half_pulse+1}.seq')
        savemat(f'ktraj_ute_2d_rw_s{s}_N{N}_FOV{FOV}_TR{TR}_TE{TE}_C={use_half_pulse+1}.mat',
                {'ktraj':ktraj})

    return seq, TE, ktraj


if __name__ == '__main__':
    # Try it!
    # FOV = 253e-3
    # N = 256
    # Nr = 804
    # thk = 5e-3
    # FA = 10
    #
    # seq, TE, ktraj = write_UTE_2D_rf_spoiled(N=N, Nr=Nr, FOV=FOV, thk=thk, slice_locs=[0],
    #                                       FA=10, TR=15e-3, ro_asymmetry=0.97, use_half_pulse=True, rf_dur=0.6e-3,
    #                                       TE_use=None)
    # seq.plot(time_range=[0,30e-3])
    # print(f'TE is {TE*1e3} ms!')

    #seq.write('ute2d.seq')
    #savemat('ute2d_fov253_half_minRFdur_s097_TR15_FA10_032122.mat',{'ktraj':ktraj,'TE':TE})

    enc = 'zyx'

    # Debug
    seq, TE, ktraj = write_UTE_2D_rf_spoiled(N=256, Nr=804, FOV=253e-3,
                                             thk=5e-3, slice_locs=[0], FA=10, TR=15e-3,
                                             ro_asymmetry=0.97, use_half_pulse=False, rf_dur=1e-3,
                                             enc=enc, TE_use = None)
    seq.write(f'ute2D_{enc}.seq')
    #print(seq.test_report())
    seq.plot(time_range=[0,100e-3])
