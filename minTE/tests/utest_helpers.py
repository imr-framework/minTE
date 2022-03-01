# Test helper functions
# import
import unittest
import numpy as np
from minTE.sequences.write_CODE import *
from pypulseq.make_trap_pulse import make_trapezoid
from pypulseq.make_extended_trapezoid import make_extended_trapezoid

class TestUTE3D(unittest.TestCase):
    def test_combine_oblique_radial_readout_2d(self):
        g = make_trapezoid(channel='x',amplitude=5000,duration=1e-3,rise_time=1e-4)
        # Generate random unit gradients and random theta

        theta1 = np.random.rand()*np.pi
        phi1 = np.random.rand()*2*np.pi
        ug1 = np.array([np.cos(theta1)*np.cos(phi1),np.cos(theta1)*np.sin(phi1),np.sin(theta1)])

        theta2 = np.random.rand()*np.pi
        phi2 = np.random.rand()*2*np.pi
        ug2 = np.array([np.cos(theta2)*np.cos(phi2),np.cos(theta2)*np.sin(phi2),np.sin(theta2)])

        ug2p = ug2 - ug1 * (np.dot(ug1,ug2) / (np.linalg.norm(ug1)**2))
        ug2pn = ug2p / np.linalg.norm(ug2p)


        alpha = np.random.rand()*2*np.pi

        # Timing is checked within the function
        gx, gy, gz = combine_oblique_radial_readout_2d(g, ug1, ug2pn, alpha)

        # Check amplitude
        np.testing.assert_equal(gx.amplitude, 5000*(ug1[0]*np.cos(alpha) + ug2pn[0]*np.sin(alpha)))
        np.testing.assert_equal(gy.amplitude, 5000*(ug1[1]*np.cos(alpha) + ug2pn[1]*np.sin(alpha)))
        np.testing.assert_equal(gz.amplitude, 5000*(ug1[2]*np.cos(alpha) + ug2pn[2]*np.sin(alpha)))

    def test_get_ktraj(self):
        # Set up gx, gy, adc using known k-space traj - and compare output to it
        fov = 250e-3 # [m]
        dk = 1/fov # [m^-1]
        n = 128
        theta = np.random.rand()*2*np.pi
        gx = make_trapezoid(channel='x', system=Opts(), flat_area=n*dk*np.cos(theta), flat_time=5e-3, rise_time=0.5e-3)
        gy = make_trapezoid(channel='y', system=Opts(), flat_area=n*dk*np.sin(theta), flat_time=5e-3, rise_time=0.5e-3)
        adc = make_adc(num_samples=n, system=Opts(), duration=gx.flat_area, delay=gx.rise_time)
        ktraj_complex = get_ktraj(gx, gy, adc, display=False)
        # Check: shape, type
        np.testing.assert_equal(ktraj_complex.shape, (n,))
        self.assertIsInstance(ktraj_complex, np.ndarray)
        self.assertIs(type(ktraj_complex[0]), np.complex128)
        # Check: values

        kx = np.cos(theta)*np.arange(0,n*dk,dk) + 0.5 * gx.rise_time * gx.amplitude
        ky = np.sin(theta)*np.arange(0,n*dk,dk) + 0.5 * gy.rise_time * gy.amplitude
        np.testing.assert_array_almost_equal(ktraj_complex, kx + 1j*ky)

    def test_get_ktraj_with_rew(self):
        # Set up gx, gy, adc using known k-space traj - and compare output to it
        fov = 250e-3  # [m]
        dk = 1 / fov  # [m^-1]
        n = 128
        theta = np.random.rand() * 2 * np.pi
        gx = make_trapezoid(channel='x', system=Opts(), flat_area=n * dk * np.cos(theta), flat_time=5e-3,
                            rise_time=0.5e-3)
        gy = make_trapezoid(channel='y', system=Opts(), flat_area=n * dk * np.sin(theta), flat_time=5e-3,
                            rise_time=0.5e-3)
        gx_rew = make_trapezoid(channel='x',system=Opts(), area=-gx.area/2, duration=2.5e-3)
        gy_rew = make_trapezoid(channel='y',system=Opts(), area=-gy.area/2, duration=2.5e-3)

        adc = make_adc(num_samples=n, system=Opts(), duration=gx.flat_area, delay=gx.rise_time)


        ktraj_complex = get_ktraj_with_rew(gx, gx_rew, gy, gy_rew, adc, display=False)

        # Check: shape, type
        np.testing.assert_equal(ktraj_complex.shape, (n,))
        self.assertIsInstance(ktraj_complex, np.ndarray)
        self.assertIs(type(ktraj_complex[0]), np.complex128)
        # Check: values

        kx = np.cos(theta) * np.arange(0, n * dk, dk) + 0.5 * gx.rise_time * gx.amplitude + gx_rew.area
        ky = np.sin(theta) * np.arange(0, n * dk, dk) + 0.5 * gy.rise_time * gy.amplitude + gy_rew.area
        np.testing.assert_array_almost_equal(ktraj_complex, kx + 1j * ky)


    def test_get_ktraj_3d(self):
        # Set up gx, gy, adc using known k-space traj - and compare output to it
        fov = 250e-3  # [m]
        dk = 1 / fov  # [m^-1]
        n = 128
        theta = np.random.rand() * np.pi
        phi = np.random.rand() * 2 * np.pi

        gx = make_trapezoid(channel='x', system=Opts(), flat_area=n * dk * np.sin(theta) * np.cos(phi),
                            flat_time=5e-3, rise_time=0.5e-3)
        gy = make_trapezoid(channel='y', system=Opts(), flat_area=n * dk * np.sin(theta) * np.sin(phi),
                            flat_time=5e-3, rise_time=0.5e-3)
        gz = make_trapezoid(channel='z', system=Opts(), flat_area=n * dk * np.cos(theta),
                            flat_time=5e-3, rise_time=0.5e-3)

        gx_rew = make_trapezoid(channel='x',system=Opts(), area=-gx.area/2, duration=2.5e-3)
        gy_rew = make_trapezoid(channel='y',system=Opts(), area=-gy.area/2, duration=2.5e-3)
        gz_rew = make_trapezoid(channel='z',system=Opts(), area=-gz.area/2, duration=2.5e-3)

        adc = make_adc(num_samples=n, system=Opts(), duration=gx.flat_area, delay=gx.rise_time)
        ktraj = get_ktraj_3d(gx, gy, gz, adc, gx_pre_list=[gx_rew], gy_pre_list=[gy_rew], gz_pre_list=[gz_rew])

        # Check: shape, type
        self.assertIsInstance(ktraj, np.ndarray)
        np.testing.assert_equal(ktraj.shape,[n,3])

        # Check: values
        kx = np.sin(theta) * np.cos(phi) * np.arange(0, n * dk, dk) + 0.5 * gx.rise_time * gx.amplitude + gx_rew.area
        ky = np.sin(theta) * np.sin(phi) * np.arange(0, n * dk, dk) + 0.5 * gy.rise_time * gy.amplitude + gy_rew.area
        kz = np.cos(theta) * np.arange(0, n*dk, dk) + 0.5 * gz.rise_time * gz.amplitude + gz_rew.area

        np.testing.assert_array_almost_equal(ktraj[:,0], kx)
        np.testing.assert_array_almost_equal(ktraj[:,1], ky)
        np.testing.assert_array_almost_equal(ktraj[:,2], kz)


    def test_get_ktraj_3d_rew_delay(self):
        # Set up gx, gy, adc using known k-space traj - and compare output to it
        fov = 250e-3  # [m]
        dk = 1 / fov  # [m^-1]
        n = 128
        theta = np.random.rand() * np.pi
        phi = np.random.rand() * 2 * np.pi

        gx = make_trapezoid(channel='x', system=Opts(), flat_area=n * dk * np.sin(theta) * np.cos(phi),
                            flat_time=5e-3, rise_time=0.5e-3)
        gy = make_trapezoid(channel='y', system=Opts(), flat_area=n * dk * np.sin(theta) * np.sin(phi),
                            flat_time=5e-3, rise_time=0.5e-3)
        gz = make_trapezoid(channel='z', system=Opts(), flat_area=n * dk * np.cos(theta),
                            flat_time=5e-3, rise_time=0.5e-3)

        gx_rew = make_trapezoid(channel='x', system=Opts(), area=-gx.area / 2, duration=2.5e-3)
        gy_rew = make_trapezoid(channel='y', system=Opts(), area=-gy.area / 2, duration=2.5e-3)
        gz_rew = make_trapezoid(channel='z', system=Opts(), area=-gz.area / 2, duration=2.5e-3)

        adc_add_delay = 1e-3
        adc = make_adc(num_samples=n, system=Opts(), duration=gx.flat_area, delay=gx.rise_time+adc_add_delay)
        ktraj = get_ktraj_3d_rew_delay(gx, gx_rew, gy, gy_rew, gz, gz_rew, adc)

        # Check: shape, type
        self.assertIsInstance(ktraj, np.ndarray)
        np.testing.assert_equal(ktraj.shape, [n, 3])

        # Check: values
        kx = np.sin(theta) * np.cos(phi) * np.arange(0, n * dk, dk) + \
             0.5 * gx.rise_time * gx.amplitude + gx_rew.area + adc_add_delay * gx.amplitude
        ky = np.sin(theta) * np.sin(phi) * np.arange(0, n * dk, dk) + \
             0.5 * gy.rise_time * gy.amplitude + gy_rew.area + adc_add_delay * gy.amplitude
        kz = np.cos(theta) * np.arange(0, n * dk, dk) + 0.5 * gz.rise_time * gz.amplitude + gz_rew.area + \
             adc_add_delay * gz.amplitude

        np.testing.assert_array_almost_equal(ktraj[:, 0], kx)
        np.testing.assert_array_almost_equal(ktraj[:, 1], ky)
        np.testing.assert_array_almost_equal(ktraj[:, 2], kz)

    def test_make_scaled_extended_trapezoid(self):
        sc = np.random.rand() + 0.5
        g0 = make_extended_trapezoid(channel='x',system=Opts(), times=[0,1e-3,6e-3,7e-3],amplitudes=[4e3,5e3,5e3,6e3])
        g1 = make_scaled_extended_trapezoid(channel='x',system=Opts(), scale=1,
                                            times=[0,1e-3,6e-3,7e-3],amplitudes=[4e3,5e3,5e3,6e3])
        g2 = make_scaled_extended_trapezoid(channel='x',system=Opts(), scale=sc,
                                            times=[0,1e-3,6e-3,7e-3],amplitudes=[4e3,5e3,5e3,6e3])
        g3 = make_scaled_extended_trapezoid(channel='x',system=Opts(), scale=0,
                                            times=[0,1e-3,6e-3,7e-3],amplitudes=[4e3,5e3,5e3,6e3])

        np.testing.assert_equal(g0.channel,g1.channel)
        np.testing.assert_array_equal(g0.waveform,g1.waveform)
        np.testing.assert_equal(g0.type,g1.type)
        np.testing.assert_array_equal(g0.t,g1.t)

        np.testing.assert_array_almost_equal(g0.waveform*sc, g2.waveform)

        np.testing.assert_equal(g3.type,'trap')
        np.testing.assert_equal(g3.amplitude,0)
        np.testing.assert_equal(g3.area,0)

    def test_parse_enc(self):
        ug_ro, ug_pe, ug_ss = parse_enc('xyz')
        np.testing.assert_equal(ug_ro, [1,0,0])
        np.testing.assert_equal(ug_pe, [0,1,0])
        np.testing.assert_equal(ug_ss, [0,0,1])

        ug_ro, ug_pe, ug_ss = parse_enc('zxy')
        np.testing.assert_equal(ug_ro, [0,0,1])
        np.testing.assert_equal(ug_pe, [1,0,0])
        np.testing.assert_equal(ug_ss, [0,1,0])

        ug_ro, ug_pe, ug_ss = parse_enc('yzx')
        np.testing.assert_equal(ug_ro, [0,1,0])
        np.testing.assert_equal(ug_pe, [0,0,1])
        np.testing.assert_equal(ug_ss, [1,0,0])

    def test_make_oblique_gradients(self):
        g = make_trapezoid(channel='x',amplitude=5000,duration=1e-3,rise_time=1e-4)
        theta = np.random.rand()*np.pi
        phi = np.random.rand()*2*np.pi
        ug = [np.cos(theta)*np.cos(phi),np.cos(theta)*np.sin(phi),np.sin(theta)]
        gx, gy, gz = make_oblique_gradients(g, ug)
        # Check channel
        np.testing.assert_equal(gx.channel,'x')
        np.testing.assert_equal(gy.channel,'y')
        np.testing.assert_equal(gz.channel,'z')
        # Check amplitude
        np.testing.assert_equal(gx.amplitude,5000*ug[0])
        np.testing.assert_equal(gy.amplitude,5000*ug[1])
        np.testing.assert_equal(gz.amplitude,5000*ug[2])

    def test_modify_gradient(self):
        g1 = make_trapezoid(channel='x',amplitude=5000,duration=1e-3,rise_time=1e-4)
        g1_area = g1.area
        g1_flat = g1.flat_area
        g2 = make_extended_trapezoid(channel='z',amplitudes=[4000,5000,5000,6000],times=[0,1e-3,6e-3,7e-3])

        # Use a random scale
        sc = np.random.rand()

        # Modify a trapezoidal gradient
        modify_gradient(g1,scale=sc,channel='z')
        np.testing.assert_equal(g1.amplitude, 5000*sc)
        np.testing.assert_equal(g1.channel,'z')
        np.testing.assert_equal(g1.area, g1_area*sc)
        np.testing.assert_equal(g1.flat_area, g1_flat*sc)

        # Modify an arbitrary gradient
        modify_gradient(g2,scale=0,channel='y')
        np.testing.assert_equal(g2.channel,'y')
        np.testing.assert_equal(np.sum(g2.waveform),0)
        np.testing.assert_equal(g2.first, 0)
        np.testing.assert_equal(g2.last, 0)

    def test_get_3d_unit_grad(self):
        ug1 = get_3d_unit_grad(theta=0,phi=0)
        ug2 = get_3d_unit_grad(theta=np.pi/2,phi=0)
        ug3 = get_3d_unit_grad(theta=np.pi/4,phi=np.pi/4)

        np.testing.assert_array_almost_equal(ug1,[0,0,1])
        np.testing.assert_array_almost_equal(ug2,[1,0,0])
        np.testing.assert_array_almost_equal(ug3,[0.5,0.5,1/np.sqrt(2)])

    def test_get_radk_params_2D(self):
        Ns = get_radk_params_2D(N=256)

        np.testing.assert_equal(Ns, 804)

        self.assertRaises(TypeError, get_radk_params_2D, "")
        self.assertRaises(TypeError, get_radk_params_2D, 3.14)
        self.assertRaises(ValueError, get_radk_params_2D, 0)

    def test_get_radk_params_3D(self):

        Ns, Ntheta, Nphi = get_radk_params_3D(dr=1e-3,fov=128e-3)
        np.testing.assert_equal([Ns, Ntheta, Nphi],[51712,202,256])

        self.assertRaises(ValueError, get_radk_params_3D, -5e-3, 0.5)
        self.assertRaises(ValueError, get_radk_params_3D, 5e-3, -0.5)


if __name__ == "__main__":
    unittest.main()
