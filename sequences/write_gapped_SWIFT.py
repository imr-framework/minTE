# Attempt at implementing gapped SWIFT!
from math import pi
from types import SimpleNamespace
from typing import Iterable
from pypulseq.Sequence.sequence import Sequence
from pypulseq.make_adc import make_adc
from pypulseq.opts import Opts
from sequences.write_UTE_3D import get_radk_params_3D
from sequences.sequence_helpers import *
from pypulseq.make_block_pulse import make_block_pulse
from scipy.io import savemat, loadmat
from scipy import integrate
import matplotlib.pyplot as plt
from datetime import date

def write_swift(N, R=128, TR=100e-3, fa_deg=90, L_over=1, enc_type='3D', spoke_os=1, output=False):
    # Let's do SWIFT
    # System
    system = Opts(max_grad=32, grad_unit='mT/m', max_slew=130, slew_unit='T/m/s',
                  rf_ringdown_time=30e-6, rf_dead_time=100e-6, adc_dead_time=20e-6)
    # Parameters
    FOV = 350e-3 # Field of view
    #N_sample = int(N / 2) # Number of points sampled on a radial spoke / number of segments 1 pulse is broken into
    dx = FOV / N # Spatial resolution (isotropic)
    dk = 1 / FOV # Spatial frequency resolution

    # Number of spokes, number of altitude angles, number of azimuthal angles
    if enc_type == '3D':
        Ns, Ntheta, Nphi = get_radk_params_3D(dx, FOV)
        N_line = Ntheta * Nphi
        # Radial encoding details
        thetas = np.linspace(0, np.pi, Ntheta, endpoint=False)
    elif enc_type == "2D":
        Nphi = get_radk_params_2D(N)
        thetas = np.array([np.pi/2]) # Zero z-gradient.
        N_line = Nphi

    phis = np.linspace(0, 2 * np.pi, Nphi*spoke_os, endpoint=False)
    N_line = Nphi*spoke_os

    # Pulse parameters
    FA = fa_deg * pi / 180
    beta = 1
    BW = 2.5e3 # from Siemens implementation paper
    Tp = R / BW # Total pulse duration
    N_seg = int(L_over * R)
    dw = Tp / N_seg # sampling dt

    dc = 0.5 # pulse duty cycle
    T_seg = dw * dc # Duration of 1 segment
    gap = dw - T_seg # time interval where RF is off in each segment
    adc_factor = 0.5 # Where during the gap lies the sample
    g_amp = dk / dw  # Net gradient amplitude
    RF_amp_max = FA / (2*pi*np.power(beta,-1/2)*np.sqrt(R)/BW) # From gapped pulse paper; units: [Hz]

    # Gradient & ADC
    ramp_time = 100e-6
    delay_TR = TR - N_seg * dw
    g_const = make_extended_trapezoid(channel='x', times=np.array([0, dw]), amplitudes=np.array([g_amp,g_amp]), system=system)
    g_delay = make_extended_trapezoid(channel='x', times=np.array([0, delay_TR]),
                                      amplitudes=np.array([g_amp, g_amp]), system=system)
    #dwell = ((1 - adc_factor) * gap - 200e-6) / 2 # This worked for 3D N=16 but not 2D N=128
    #dwell = ((1 - adc_factor) * gap - system.adc_dead_time) / 2
    #dwell = ((1 - adc_factor) * gap) / 2
    dwell = 10e-6

    adc = make_adc(num_samples = 2, delay=T_seg + adc_factor * gap, dwell=dwell)

    # Find RF modulations
    AM, FM = make_HS_pulse(beta=beta, b1=RF_amp_max, bw=BW)
    list_RFs = extract_chopped_pulses(AM, FM, Tp, N_seg, T_seg)
    # Save information
    rf_complex = np.zeros((1,len(list_RFs)),dtype=complex)
    for u in range(len(list_RFs)):
        rf_complex[0,u] = list_RFs[u].signal[0] * np.exp(1j*list_RFs[u].phase_offset)

    amp_gx, amp_gy, amp_gz = 0, 0, 0
    # Construct the sequence
    seq = Sequence(system)
    q = 0
    y = 0

    ktraj = np.zeros((N_seg, N_line, 3))
    nline = 0
    for theta in thetas: # For each altitude angle
        for phi in phis: # For each azimuth angle
            print(f'Adding {q} of {len(thetas)*len(phis)} directions')
            # Calculate gradients
            unit_grad = get_3d_unit_grad(theta,phi)
            g_const_x, g_const_y, g_const_z = make_oblique_gradients(g_const, unit_grad)
            # Ramps from last encode to this oen
            g_ramp_x = make_extended_trapezoid_check_zero(channel='x',times=np.array([0,ramp_time]),
                                               amplitudes=[amp_gx, g_const_x.first], system=system)
            g_ramp_y = make_extended_trapezoid_check_zero(channel='y',times=np.array([0,ramp_time]),
                                               amplitudes=[amp_gy, g_const_y.first], system=system)
            g_ramp_z = make_extended_trapezoid_check_zero(channel='z',times=np.array([0,ramp_time]),
                                               amplitudes=[amp_gz, g_const_z.first], system=system)
            amp_gx, amp_gy, amp_gz = g_const_x.first, g_const_y.first, g_const_z.first
            seq.add_block(g_ramp_x, g_ramp_y, g_ramp_z)
            for n in range(N_seg): # For each RF segment
                # Make ramp-up from the last gradient & this constant gradient
                # Add RF, ADC with delay, and split gradient together
                seq.add_block(list_RFs[n], g_const_x, g_const_y, g_const_z, adc)
                y = y+1
                print('y')

            dk_3d = adc.delay * np.array([g_const_x.waveform[0], g_const_y.waveform[0], g_const_z.waveform[0]])
            ktraj[:, nline, 0] = dk_3d[0]*np.arange(1,N_seg+1)
            ktraj[:, nline, 1] = dk_3d[1]*np.arange(1,N_seg+1)
            ktraj[:, nline, 2] = dk_3d[2]*np.arange(1,N_seg+1)
            nline += 1

            g_delay_x, g_delay_y, g_delay_z = make_oblique_gradients(g_delay, unit_grad)
            seq.add_block(g_delay_x, g_delay_y, g_delay_z)


            q += 1
    #print(seq.test_report())

    #ok, error_report = seq.check_timing()

    today = date.today()
    todayf = today.strftime("%m%d%y")

    savemat(f'swift_info_FOV{FOV*1e3}_FA{fa_deg}_N{N}_{enc_type}_L-over{L_over}_{todayf}_sos{spoke_os}.mat',{'thetas':thetas, 'phis': phis, 'rf_complex': rf_complex, 'ktraj':ktraj})

    if output:
        seq.write(f'ADC2_swift_FOV_{FOV*1e3}_FA{fa_deg}_N{N}_{enc_type}_L-over{L_over}_{todayf}_sos{spoke_os}.seq')
    return seq

# RF pulse creation
def make_HS_pulse(beta, b1, bw):
    def AM(tau):
        return b1 / np.cosh(beta*tau)
    def FM(tau):
        return (bw/2) * np.tanh(beta*tau) / np.tanh(beta)
    return AM, FM

def extract_chopped_pulses(AM, FM, Tp, N_seg, T_seg):
    # Return a list of Pulseq RF objects
    list_RFs = []
    t = np.linspace(0, Tp, N_seg + 1, endpoint=True)
    tau = 2*t/Tp - 1
    phi_c = 0

    total_flip = 0

    for n in range(N_seg):
        # Calculate amplitude and phase at corresponding point (beginning pt.)
        list_RFs.append(make_block_pulse(flip_angle = 2*pi*AM(tau[n])*T_seg, duration=T_seg,
                                         phase_offset = phi_c))
        total_flip += 2*pi*AM(tau[n])*T_seg
        dphi_bar, _ = integrate.quad(FM, a = tau[n],b = tau[n+1])
        phi_c = phi_c + dphi_bar*2*pi
    print(f'Total flip angle from AM: {total_flip}')
    return list_RFs

def make_extended_trapezoid_check_zero(channel: str, amplitudes: Iterable = np.zeros(1), max_grad: float = 0,
                            max_slew: float = 0, system: Opts = Opts(), skip_check: bool = False,
                            times: Iterable = np.zeros(1)) -> SimpleNamespace:

    if np.max(np.absolute(amplitudes)) == 0:
        return make_trapezoid(channel=channel, area=0, duration=times[-1]-times[0])
    else:
        return make_extended_trapezoid(channel, amplitudes, max_grad, max_slew, system, skip_check, times)

if __name__ == '__main__':
    # Inspect HS1 pulse
    #AM, FM = make_HS_pulse(beta=1)
    #tmodel = np.linspace(-1,1,500)
    #plt.plot(tmodel, AM(tmodel),'-k')
    #plt.plot(tmodel, FM(tmodel),'-b')
    #plt.show()
    #seq = Sequence()
    #TR = 100e-3
    ##seq.read('swift_FA5_N16.seq')
    #print(seq.test_report())
    #seq.plot()
    #seq = write_swift(N=16, fa_deg=5, output=True)

    seq = write_swift(N=32, R=128, TR=100e-3, fa_deg=160, L_over=2, enc_type='2D', spoke_os=1, output=True)
    print(seq.test_report())
    seq.plot(time_range=[0,200e-3])