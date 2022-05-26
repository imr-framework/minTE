# Helper functions for UTE sequence construction
import numpy as np
import matplotlib.pyplot as plt
from pypulseq.make_extended_trapezoid import make_extended_trapezoid
from pypulseq.make_trap_pulse import make_trapezoid
import copy


def combine_oblique_radial_readout_2d(g, ug1, ug2, theta):
    """Generates an oblique readout at a given angle using any two orthogonal axes in space

    Parameters
    ----------
    g : Pypulseq Gradient object
        Base gradient to be made into oblique gradients
    ug1 : array_like
        Length 3, unit gradient in first direction
    ug2 : array_like
        Length 3, unit gradient in second direction
    theta : float
        Angle in radians of specific radial readout spoke

    Returns
    -------
    gx, gy, gz : Pypulseq Gradient objects in each direction
    """

    # Check that ug1 and ug2 are orthogonal
    if np.absolute(np.dot(ug1,ug2)) > 1e-12:
        raise ValueError("The two directions provided must be orthogonal.")
    ug_net = np.array(ug1) * np.cos(theta) + np.array(ug2) * np.sin(theta)
    gx, gy, gz = make_oblique_gradients(g, ug_net)
    return gx, gy, gz

def get_ktraj(gx, gy, adc, display=False):
    """Calculate and return one line of k-space trajectory during a 2D readout

    Parameters
    ----------
    gx : SimpleNamespace
        Readout gradient (PyPulseq event) in the x direction
    gy : SimpleNamespace
        Readout gradient (PyPulseq event) in the y direction
    adc : SimpleNamespace
        ADC sampling (Pypulseq event) during readout
    display : bool, default=False
        Whether to display the trajectory in a plot

    Returns
    -------
    ktraj_complex : numpy.ndarray
        Complex representation (kx + 1j*ky) of k-space trajectory for this single readout

    """

    sampled_times = np.linspace(0, gx.flat_time, adc.num_samples, endpoint=False)
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

    ktraj_complex = kx + 1j * ky

    return ktraj_complex

def get_ktraj_with_rew(gx, gx_rew, gy, gy_rew, adc, display=False):
    """
    Calculate and return one line of k-space trajectory
    during a 2D readout, accounting for rewinder gradients
    applied before readout

    Parameters
    ----------
    gx : SimpleNamespace
        Readout gradient (PyPulseq event) in the x direction
    gx_rew : SimpleNamespace
        Rewinder gradient (PyPulseq event) in the x direction
    gy : SimpleNamespace
        Readout gradient (PyPulseq event) in the y direction
    gy_rew : SimpleNamespace
        Rewinder gradient (PyPulseq event) in the y direction
    adc : SimpleNamespace
        ADC sampling (Pypulseq event) during readout
    display : bool, default=False
        Whether to display the trajectory in a plot

    Returns
    -------
    ktraj_complex : numpy.ndarray
        Complex representation (kx + 1j*ky) of k-space trajectory for this single readout

    """
    sampled_times = np.linspace(0, gx.flat_time, adc.num_samples, endpoint=False)
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
    """
    Calculate k-space trajectory assuming trapezoidal gx, gy, gz
    with the same normalized shape and uniform sampling starting
    at the flat part of trapezoid. Optionally, x,y,and z preparation
    gradients before the readout are taken into account.

    Parameters
    ----------
    gx : SimpleNamespace
        Readout gradient (PyPulseq event) in the x direction
    gy : SimpleNamespace
        Readout gradient (PyPulseq event) in the y direction
    gz : SimpleNamespace
        Readout gradients (PyPulseq event) in the z direction
    adc : SimpleNamespace
        ADC sampling (Pypulseq event) for readout
    gx_pre_list : list of SimpleNamespace
        A list of preparation/rewinder x gradients prior to readout
    gy_pre_list : list of SimpleNamespace
        A list of preparation/rewinder y gradients prior to readout
    gz_pre_list : list ofSimpleNamespace
        A list of preparation/rewinder z gradients prior to readout

    Returns
    -------
    ktraj : np.ndarray
        k-space trajectory for this single readout
        Size: (n, 3) where n is the number of readout samples

    """

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
    """
    Calculate and return one line of k-space trajectory
    during a 3D readout, accounting for rewinder gradients
    applied before readout

    Parameters
    ----------
    gx : SimpleNamespace
        Readout gradient (PyPulseq event) in the x direction
    gx_rew : SimpleNamespace
        Rewinder gradient (PyPulseq event) in the x direction
    gy : SimpleNamespace
        Readout gradient (PyPulseq event) in the y direction
    gy_rew : SimpleNamespace
        Rewinder gradient (PyPulseq event) in the y direction
    gz : SimpleNamespace
        Readout gradient (PyPulseq event) in the z direction
    gz_rew : SimpleNamespace
        Rewinder gradient (PyPulseq event) in the z direction
    adc : SimpleNamespace
        ADC sampling (Pypulseq event) during readout

    Returns
    -------
    ktraj : np.ndarray
        k-space trajectory for this single readout
        Size: (n, 3) where n is the number of readout samples

    """
    N = int(adc.num_samples)
    sampled_times = np.linspace(0, gx.flat_time, adc.num_samples, endpoint=False)
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
    """
    A wrapped version of make_extended_trapezoid; removes non-distinct timing,
    scales the gradient, and returns an ordinary zero gradient instead of an
    extended trapezoid when scale = 0

    Parameters
    ----------
    channel : str
        Desired gradient channel; "x", "y", or "z"
    system : Opts, optional, default=Opts()
        System limits
    times : numpy.ndarray
        Time points at which `amplitudes` defines amplitude values.
    amplitudes : numpy.ndarray
        Values defined at `times` time indices
    scale : float
        Real amplitude scale for the output gradient

    Returns
    -------
    grad : SimpleNamespace
        Extended trapezoid gradient event

    """
    inds_of_no_change = np.array(np.where(np.diff(times) == 0)) + 1
    for ind in inds_of_no_change:
        times = np.delete(times, ind)
        amplitudes = np.delete(amplitudes, ind)
    if scale == 0:
        grad = make_trapezoid(channel=channel, system=system, duration=times[-1]-times[0], area=0, amplitude = 0)
    else:
        grad = make_extended_trapezoid(channel=channel, system=system, times=times, amplitudes = scale*amplitudes)
    return grad

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
    """
    Helper function to make oblique gradients
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

    unit_grad = unit_grad / np.linalg.norm(unit_grad)

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
        Channel to switch gradient into; default is None
        which keeps the original channel

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
    """Calculate unit gradient direction from spherical coordinates

    Parameters
    ----------
    theta : float
        Polar angle [radians]
    phi : float
        Azimuthal angle [radians]

    Returns
    -------
    unit_grad : numpy.ndarray
        Unit gradient direction (x, y, z)

    """

    scale_x = np.sin(theta)*np.cos(phi)
    scale_y = np.sin(theta)*np.sin(phi)
    scale_z = np.cos(theta)

    unit_grad = np.array([scale_x, scale_y, scale_z])

    return unit_grad

def get_radk_params_2D(N):
    """Calculate the number of spokes for 2D radial imaging

    Parameters
    ----------
    N : int
        Imaging matrix size (isotropic)

    Returns
    -------
    Ns : int
        Number of spokes needed for the Nyquist criterion

    """

    if not isinstance(N, int): raise TypeError("The input matrix size must be an integer")
    if N <= 0: raise ValueError("Matrix size N must be positive")

    Ns = np.round(np.pi*N)
    Ns = Ns + np.mod(Ns,2)  # Make it even to have 180-deg opposed spokes
    Ns = int(Ns)
    return Ns

def get_radk_params_3D(dr, fov):
    """Calculate the number of polar and azimuthal angles for 3D radial imaging

    Parameters
    ----------
    dr : float
        Spatial resolution
    fov : float
        Field-of-view in [meters]

    Returns
    -------
    Ns : int
        Number of spokes
    Ntheta : int
        Number of polar angles
    Nphi : int
        Number of azimuthal angles
    """

    if dr <= 0: raise ValueError("The resolution must be positive!")
    if fov <= 0: raise ValueError("The field-of-view must be positive!")

    kmax = 1/(2*dr)
    # Total number of spokes
    Ns = np.round(4*np.pi*((fov*kmax)**2)) # From the criterion that solid surface per spoke = (Nyquist dk)^2
    dtheta = 1/(kmax*fov) # From planar radial Nyquist criterion
    Ntheta = np.ceil(np.pi/dtheta)
    Ntheta = Ntheta + np.mod(Ntheta,2)# Make it even to have 180-deg opposed spokes
    Nphi = np.ceil(Ns/Ntheta) # Fill the 2nd dimension with enough phi resolution so total number of spokes is satisfied
    Nphi = Nphi + np.mod(Nphi,2) # Make it even to have 180-deg opposed spokes
    Ns = Ntheta*Nphi

    Ns = int(Ns)
    Ntheta = int(Ntheta)
    Nphi = int(Nphi)

    return Ns, Ntheta, Nphi
