# 3D ultra-short TE sequence (radial encoding)

from pypulseq.Sequence.sequence import Sequence
from pypulseq.calc_duration import calc_duration
from pypulseq.make_adc import make_adc
from pypulseq.make_delay import make_delay
from pypulseq.make_sinc_pulse import make_sinc_pulse
from pypulseq.make_block_pulse import make_block_pulse
from pypulseq.make_gauss_pulse import make_gauss_pulse
from pypulseq.opts import Opts
from scipy.io import savemat, loadmat
from minTE.sequences.sequence_helpers import *

def write_UTE_3D_rf_spoiled(N=64, FOV=250e-3, slab_thk=250e-3, FA=10, TR=10e-3, ro_asymmetry=0.97,
                            os_factor=1, rf_type='sinc', rf_dur=1e-3, use_half_pulse=True, save_seq=True):
    """
    Parameters
    ----------
    N : int, default=64
        Matrix size
    FOV : float, default=0.25
        Field-of-view in [meters]
    slab_thk : float, default=0.25
        Slab thickness in [meters]
    FA : float, default=10
        Flip angle in [degrees]
    TR : float, default=0.01
        Repetition time in [seconds]
    ro_asymmetry : float, default=0.97
        The ratio A/B where a A/(A+B) portion of 2*Kmax is omitted and B/(A+B) is acquired.
    os_factor : float, default=1
        Oversampling factor in readout
        The number of readout samples is the nearest integer from os_factor*N
    rf_type : str, default='sinc'
        RF pulse shape - 'sinc', 'gauss', or 'rect'
    rf_dur : float, default=0.001
        RF pulse duration
    use_half_pulse : bool, default=True
        Whether to use half pulse excitation for shorter TE;
        This doubles both the number of excitations and acquisition time
    save_seq : bool, default=True
        Whether to save this sequence as a .seq file

    Returns
    -------
    seq : Sequence
        PyPulseq 2D UTE sequence object
    TE : float
        Echo time of generated sequence in [seconds]
    ktraj : np.ndarray
        3D k-space trajectory [spoke, readout sample, 3] where
        the last dimension refers to spatial frequency coordinates (kx, ky, kz)
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
    rf_spoiling_inc = 117  # RF spoiling increment value.

    if use_half_pulse:
        cp = 1
    else:
        cp = 0.5

    tbw = 2

    # Sequence components
    if rf_type == 'sinc':
        rf, gz, gz_reph = make_sinc_pulse(flip_angle=FA * np.pi / 180, duration=rf_dur, slice_thickness=slab_thk,
                                          apodization=0.5, time_bw_product=tbw, center_pos=cp, system=system, return_gz=True)
        gz_ramp_reph = make_trapezoid(channel='z', area=-gz.fall_time * gz.amplitude / 2, system=system)

    elif rf_type == 'rect':
        rf = make_block_pulse(flip_angle=FA * np.pi/180, duration=rf_dur, slice_thickness=slab_thk, return_gz=False)
    elif rf_type == 'gauss':
        rf, gz, gz_reph = make_gauss_pulse(flip_angle=FA*np.pi/180, duration=rf_dur, slice_thickness=slab_thk,
                                           system=system, return_gz=True)
        gz_ramp_reph = make_trapezoid(channel='z', area=-gz.fall_time * gz.amplitude / 2, system=system)

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
    TE = gro.rise_time + adc.dwell * Nro / 2 * (1 - s)
    if rf_type == 'sinc' or rf_type == 'gauss':
        if use_half_pulse:
            TE += gz.fall_time + calc_duration(gro_pre)
        else:
            TE += calc_duration(gz)/2 + calc_duration(gro_pre)

        delay_TR = np.ceil((TR - calc_duration(gro_pre) - calc_duration(gz) - calc_duration(
            gro)) / seq.grad_raster_time) * seq.grad_raster_time
    elif rf_type == 'rect':
        TE += calc_duration(gro_pre)
        delay_TR = np.ceil((TR - calc_duration(gro_pre) - calc_duration(gro)) / seq.grad_raster_time) * seq.grad_raster_time
    assert np.all(delay_TR >= calc_duration(gro_spoil))  # The TR delay starts at the same time as the spoilers!
    print(f'TE = {TE * 1e6:.0f} us')



    C = int(use_half_pulse) + 1

    # Starting RF phase and increments
    rf_phase = 0
    rf_inc = 0
    Nline = Ntheta * Nphi
    ktraj = np.zeros([Nline, int(adc.num_samples), 3])
    u = 0


    for th in range(Ntheta):
        for ph in range(Nphi):
            unit_grad = np.zeros(3)
            unit_grad[0] = np.sin(thetas[th])*np.cos(phis[ph])
            unit_grad[1] = np.sin(thetas[th])*np.sin(phis[ph])
            unit_grad[2] = np.cos(thetas[th])
            # Two repeats if using half pulse
            for c in range(C):
                # RF spoiling
                rf.phase_offset = (rf_phase / 180) * np.pi
                adc.phase_offset = (rf_phase / 180) * np.pi
                rf_inc = np.mod(rf_inc + rf_spoiling_inc, 360.0)
                rf_phase = np.mod(rf_phase + rf_inc, 360.0)

                # Rewinder and readout gradients, vectorized
                gpx, gpy, gpz = make_oblique_gradients(gro_pre, unit_grad)
                grx, gry, grz = make_oblique_gradients(gro, unit_grad)
                gsx, gsy, gsz = make_oblique_gradients(gro_spoil, unit_grad)

                if rf_type == 'sinc' or rf_type == 'gauss':
                    if use_half_pulse:
                        # Reverse slice select amplitude (always z = 0)
                        modify_gradient(gz, scale=-1)
                        modify_gradient(gz_ramp_reph, scale=-1)
                        gpz_reph = copy.deepcopy(gpz)
                        modify_gradient(gpz_reph, scale=(gpz.area + gz_ramp_reph.area)/gpz.area)
                    else:
                        gpz_reph = copy.deepcopy(gpz)
                        modify_gradient(gpz_reph, scale=(gpz.area + gz_reph.area) / gpz.area)

                    seq.add_block(rf, gz)
                    seq.add_block(gpx, gpy, gpz_reph)

                elif rf_type == 'rect':
                    seq.add_block(rf)
                    seq.add_block(gpx, gpy, gpz)

                seq.add_block(grx, gry, grz, adc)
                seq.add_block(gsx, gsy, gsz, make_delay(delay_TR))

            #print(f'Spokes: {u+1}/{Nline}')
            ktraj[u, :, :] = get_ktraj_3d(grx, gry, grz, adc, [gpx], [gpy], [gpz])
            u += 1



    ok, error_report = seq.check_timing()  # Check whether the timing of the sequence is correct
    if ok:
        print('Timing check passed successfully')
    else:
        print('Timing check failed. Error listing follows:')
        [print(e) for e in error_report]


    if save_seq:
        seq.write(f'ute_3d_rf-{rf_type}_rw_s{s}_N{N}_FOV{FOV}_TR{TR}_TE{TE}_C={use_half_pulse+1}.seq')
        savemat(f'ktraj_ute_3d_rw_s{s}_N{N}_FOV{FOV}_TR{TR}_TE{TE}_C={use_half_pulse+1}.mat',
                {'ktraj':ktraj})

    return seq, TE, ktraj



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
    seq, TE, ktraj = write_UTE_3D_rf_spoiled(N=64, FOV=250e-3, slab_thk=253e-3, FA=10, TR=15e-3, ro_asymmetry=0.97,
                            os_factor=1, rf_type='sinc', rf_dur=0.05e-3, use_half_pulse=True, save_seq=False)
    print(f'TE is {TE*1e3} ms.')
    print(seq.test_report())
    seq.write('ute3d_fov253_64_s097_rfdur50us_halfpulse_TR15_FA10_032422.seq')
    savemat('ute3d_fov253_64_s097_rfdur50us_halfpulse_TR15_FA10_032422.mat', {'TE':TE, 'ktraj':ktraj})

    #seq.plot(time_range=[0,60e-3])

