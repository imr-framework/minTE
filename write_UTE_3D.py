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

def write_UTE_3D(N, FOV, slab_thk, FA, TR):
    # System info for June
    system = Opts(max_grad=32, grad_unit='mT/m', max_slew=130, slew_unit='T/m/s',
                  rf_ringdown_time=30e-6, rf_dead_time=100e-6, adc_dead_time=20e-6)
    seq = Sequence(system=system)

    # Derived parameters
    dx = FOV/N
    Ns, Ntheta, Nphi = get_radk_params_3D(dx, FOV) # Make this function
    print(f'Using {Ns} spokes, with {Ntheta} thetas and {Nphi} phis')

    # Slab selection
    flip = FA*np.pi/180
    st = 30e-3 # what is this?
    rf_dur = 100e-6
    [rf, gz_ss, gssr] = make_sinc_pulse(flip_angle=flip, duration=rf_dur, system=system, slice_thickness=slab_thk)
    print(gz_ss)
    # Readout gradients
    dk = 1/FOV
    k_width = N*dk
    ro_time = 6.4e-3
    gro = make_trapezoid(channel='x', system=system, flat_area=k_width, flat_time=ro_time)
    adc = make_adc(N/2, duration=gro.flat_time, delay=gro.rise_time, system=system)

    # Spoiling gradients in all directions
    pre_time = 8e-4
    gx_spoil = make_trapezoid(channel='x', system=system, area=gz_ss.area/2, duration=3*pre_time)
    gy_spoil = make_trapezoid(channel='y', system=system, area=gz_ss.area/2, duration=3*pre_time)
    gz_spoil = make_trapezoid(channel='z', system=system, area=gz_ss.area/2, duration=3*pre_time)

    # Timing
    delayTR = TR - calc_duration(gro) - calc_duration(rf) - calc_duration(gz_spoil)
    delayTRps = 0.8*delayTR # "to avoid gradient heating"
    delayTRps = delayTRps - np.mod(delayTRps, 1e-5)
    delay = make_delay(delayTRps)

    # Spoke angles
    thetas = np.linspace(0, np.pi, Ntheta, endpoint=False)
    phis = np.linspace(0, 2*np.pi, Nphi, endpoint=False)
    ktraj = np.zeros([Ns, int(adc.num_samples), 3])

    # What TE did we end up getting?
    TE = calc_duration(gz_ss)/2 + calc_duration(gssr) + gro.rise_time
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
            seq.add_block(rf, gz_ss)
            seq.add_block(gssr)
            # Readout and ADC sampling
            seq.add_block(gx, gy, gz, adc)
            # Go to end of TR + spoiler
            seq.add_block(delay)
            seq.add_block(gx_spoil, gy_spoil, gz_spoil)
            # Store trajectory
            ktraj[u,:,:] = get_ktraj_3d(gx, gy, gz, adc)
            u += 1

    return seq, ktraj, TE

def write_UTE_3D_rewound():
    # UTE with rewinder!
    seq = None
    ktraj = None
    TE = 0
    return seq, ktraj, TE


def get_radk_params_3D(dr, fov):
    kmax = 1/(2*dr)
    # Total number of spokes
    Ns = np.round(4*np.pi*((fov*kmax)**2)) # From the criterion that solid surface per spoke = (Nyquist dk)^2
    dtheta = 1/(kmax*fov) # From planar radial Nyquist criterion
    Ntheta = np.ceil(np.pi/dtheta)
    Nphi = np.ceil(Ns/Ntheta) # Fill the 2nd dimension with enough phi resolution so total number of spokes is satisfied
    Ns = Ntheta*Nphi
    return int(Ns), int(Ntheta), int(Nphi)

def get_ktraj_3d(gx, gy, gz, adc):
    # Calculate k-space trajectory assuming trapezoidal gx, gy, gz (same shape -> straight readout)
    # and uniform sampling starting at flat part of trapezoid
    N = int(adc.num_samples)
    sampled_times = np.linspace(0, gx.flat_time, N, endpoint=False)
    kx_pre = 0.5 * gx.amplitude * gx.rise_time
    ky_pre = 0.5 * gy.amplitude * gy.rise_time
    kz_pre = 0.5 * gz.amplitude * gz.rise_time

    kx = kx_pre + gx.amplitude * sampled_times
    ky = ky_pre + gy.amplitude * sampled_times
    kz = kz_pre + gz.amplitude * sampled_times

    ktraj = np.zeros([N, 3])
    ktraj[:,0] = kx
    ktraj[:,1] = ky
    ktraj[:,2] = kz


    return ktraj

if __name__ == '__main__':
    N = 64
    seq, ktraj, TE = write_UTE_3D(N=N, FOV=256e-3, slab_thk=30e-3, FA = 6, TR=20e-3)
    seq.plot(time_range=[0,20e-3])
    #print(seq.test_report())
    savemat(f'./seqs/3D_ktraj_{N}.mat', {'ktraj': ktraj, 'TE': TE})
    seq.write(f'./seqs/ute_3D_022421_{N}.seq')