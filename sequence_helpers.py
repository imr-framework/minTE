# Helper functions for UTE sequence construction
import numpy as np
import matplotlib.pyplot as plt
from pypulseq.make_extended_trapezoid import make_extended_trapezoid
from pypulseq.make_trap_pulse import make_trapezoid
import copy

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


def get_ktraj_3d(gx, gy, gz, adc, gx_pre_list=[], gy_pre_list=[], gz_pre_list=[]):
    # Calculate k-space trajectory assuming trapezoidal gx, gy, gz (same shape -> straight readout)
    # and uniform sampling starting at flat part of trapezoid
    N = int(adc.num_samples)
    sampled_times = np.linspace(0, gx.flat_time, N, endpoint=False)

    kx_pre, ky_pre, kz_pre = 0, 0, 0

    for gx_pre in gx_pre_list:
        kx_pre += gx_pre.area
    for gy_pre in gy_pre_list:
        ky_pre += gy_pre.area
    for gz_pre in gz_pre_list:
        kz_pre += gz_pre.area

    kx_pre += 0.5 * gx.amplitude * gx.rise_time
    ky_pre += 0.5 * gy.amplitude * gy.rise_time
    kz_pre += 0.5 * gz.amplitude * gz.rise_time

    kx = kx_pre + gx.amplitude * sampled_times
    ky = ky_pre + gy.amplitude * sampled_times
    kz = kz_pre + gz.amplitude * sampled_times

    ktraj = np.zeros([N, 3])
    ktraj[:,0] = kx
    ktraj[:,1] = ky
    ktraj[:,2] = kz

    return ktraj

def get_ktraj_3d_rew_delay(gx, gx_rew, gy, gy_rew, gz, gz_rew, adc):
    N = int(adc.num_samples)
    sampled_times = np.linspace(0, gx.flat_time, adc.num_samples)
    kx_pre = (0.5 * gx.rise_time + (adc.delay - gx.rise_time)) * gx.amplitude + gx_rew.area
    ky_pre = (0.5 * gy.rise_time + (adc.delay - gy.rise_time)) * gy.amplitude + gy_rew.area
    kz_pre = (0.5 * gz.rise_time + (adc.delay - gz.rise_time)) * gz.amplitude + gz_rew.area

    kx = kx_pre + gx.amplitude * sampled_times
    ky = ky_pre + gy.amplitude * sampled_times
    kz = kz_pre + gz.amplitude * sampled_times

    ktraj = np.zeros([N, 3])
    ktraj[:,0] = kx
    ktraj[:,1] = ky
    ktraj[:,2] = kz

    return ktraj




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


def parse_enc(enc):
    """Helper function for decoding enc parameter
    Parameters
    ----------
    enc : str or array_like
        Inputted encoding scheme to parse
    Returns
    -------
    ug_fe : numpy.ndarray
        Length-3 vector of readout direction
    ug_pe : numpy.ndarray
        Length-3 vector of phase encoding direction
    ug_ss : numpy.ndarray
        Length-3 vector of slice selecting direction

    """
    if isinstance(enc, str):
        xyz_dict = {'x': (1, 0, 0), 'y': (0, 1, 0), 'z': (0, 0, 1)}
        ug_ro = xyz_dict[enc[0]]
        ug_pe = xyz_dict[enc[1]]
        ug_ss = xyz_dict[enc[2]]
    else:
        ug_ro = np.array(enc[0])
        ug_pe = np.array(enc[1])
        ug_ss = np.array(enc[2])

        ug_ro = ug_ro / np.linalg.norm(ug_ro)
        ug_pe = ug_pe / np.linalg.norm(ug_pe)
        ug_ss = ug_ss / np.linalg.norm(ug_ss)

    print('ug_ro: ', ug_ro)
    print('ug_pe: ', ug_pe)
    print('ug_ss: ', ug_ss)

    return ug_ro, ug_pe, ug_ss

def make_oblique_gradients(gradient,unit_grad):
    """Helper function to make oblique gradients

    (Gx, Gy, Gz) are generated from a single orthogonal gradient
    and a direction indicated by unit vector

    Parameters
    ----------
    gradient : Gradient
        Pulseq gradient object
    unit_grad: array_like
        Length-3 unit vector indicating direction of resulting oblique gradient

    Returns
    -------
    ngx, ngy, ngz : Gradient
        Oblique gradients in x, y, and z directions

    """
    ngx = copy.deepcopy(gradient)
    ngy = copy.deepcopy(gradient)
    ngz = copy.deepcopy(gradient)

    modify_gradient(ngx, unit_grad[0],'x')
    modify_gradient(ngy, unit_grad[1],'y')
    modify_gradient(ngz, unit_grad[2],'z')

    return ngx, ngy, ngz

def modify_gradient(gradient,scale,channel=None):
    """Helper function to modify the strength and channel of an existing gradient

    Parameters
    ----------
    gradient : Gradient
        Pulseq gradient object to be modified
    scale : float
        Scalar to multiply the gradient strength by
    channel : str, optional {None, 'x','y','z'}
        Channel to switch gradient into
        Default is None which keeps the original channel

    """
    if gradient.type == 'trap':
        gradient.amplitude *= scale
        gradient.area *= scale
        gradient.flat_area *= scale
    elif gradient.type == 'grad':
        gradient.waveform *= scale
        gradient.first *= scale
        gradient.last *= scale

    if channel != None:
        gradient.channel = channel

def get_3d_unit_grad(theta, phi):
    scale_x = np.sin(theta)*np.cos(phi)
    scale_y = np.sin(theta)*np.sin(phi)
    scale_z = np.cos(theta)
    return np.array([scale_x, scale_y, scale_z])


def get_radk_params_2D(N):
    Ns = np.round(np.pi*N)
    Ns = Ns + np.mod(Ns,2)  # Make it even to have 180-deg opposed spokes
    return int(Ns)