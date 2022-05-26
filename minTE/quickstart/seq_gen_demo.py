from minTE.sequences.write_UTE_2D import *
from minTE.sequences.write_UTE_3D import *
from minTE.sequences.write_CODE import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import math
from scipy.io import savemat
from tkinter import *
from tkinter.filedialog import askdirectory

def export_waveforms(seq, time_range=(0, np.inf)):
    """
    Exports all waveforms from a seq object with associated timings for easy plotting

    Parameters
    ----------
    time_range : iterable, default=(0, np.inf)
        Time range (x-axis limits) for all waveforms. Default is 0 to infinity (entire sequence).
    Returns
    -------
    all_waveforms: dict
        Dictionary containing all sequence waveforms and time array(s)
        The keys are listed here:
        't_adc' - ADC timing array [seconds]
        't_rf' - RF timing array [seconds]
        ''
        'adc' - ADC complex signal (amplitude=1, phase=adc phase)
        'rf' - RF complex signal
        'gx' - x gradient
        'gy' - y gradient
        'gz' - z gradient
    """
    # Check time range validity
    if not all([isinstance(x, (int, float)) for x in time_range]) or len(time_range) != 2:
        raise ValueError('Invalid time range')

    t0 = 0
    adc_t_all = np.array([])
    adc_signal_all = np.array([],dtype=complex)
    rf_t_all =np.array([])
    rf_signal_all = np.array([],dtype=complex)
    rf_t_centers = np.array([])
    rf_signal_centers = np.array([],dtype=complex)
    gx_t_all = np.array([])
    gy_t_all = np.array([])
    gz_t_all = np.array([])
    gx_all = np.array([])
    gy_all = np.array([])
    gz_all = np.array([])


    for block_counter in range(len(seq.dict_block_events)): # For each block
        block = seq.get_block(block_counter + 1)  # Retrieve it
        is_valid = time_range[0] <= t0 <= time_range[1] # Check if "current time" is within requested range.
        if is_valid:
            # Case 1: ADC
            if hasattr(block, 'adc'):
                adc = block.adc # Get adc info
                # From Pulseq: According to the information from Klaus Scheffler and indirectly from Siemens this
                # is the present convention - the samples are shifted by 0.5 dwell # OK

                t = adc.delay + (np.arange(int(adc.num_samples)) + 0.5) * adc.dwell
                adc_t = t0 + t
                adc_signal = np.exp(1j * adc.phase_offset) * np.exp(1j * 2 * np.pi * t * adc.freq_offset)
                adc_t_all = np.append(adc_t_all, adc_t)
                adc_signal_all = np.append(adc_signal_all, adc_signal)

            if hasattr(block, 'rf'):
                rf = block.rf
                tc, ic = calc_rf_center(rf)
                t = rf.t + rf.delay
                tc = tc + rf.delay
                #
                # sp12.plot(t_factor * (t0 + t), np.abs(rf.signal))
                # sp13.plot(t_factor * (t0 + t), np.angle(rf.signal * np.exp(1j * rf.phase_offset)
                #                                         * np.exp(1j * 2 * math.pi * rf.t * rf.freq_offset)),
                #           t_factor * (t0 + tc), np.angle(rf.signal[ic] * np.exp(1j * rf.phase_offset)
                #                                          * np.exp(1j * 2 * math.pi * rf.t[ic] * rf.freq_offset)),
                #           'xb')

                rf_t = t0 + t
                rf = rf.signal * np.exp(1j * rf.phase_offset) \
                                                        * np.exp(1j * 2 * math.pi * rf.t * rf.freq_offset)
                rf_t_all = np.append(rf_t_all, rf_t)
                rf_signal_all = np.append(rf_signal_all, rf)
                rf_t_centers = np.append(rf_t_centers, rf_t[ic])
                rf_signal_centers = np.append(rf_signal_centers, rf[ic])

            grad_channels = ['gx', 'gy', 'gz']
            for x in range(len(grad_channels)): # Check each gradient channel: x, y, and z
                if hasattr(block, grad_channels[x]): # If this channel is on in current block
                    grad = getattr(block, grad_channels[x])
                    if grad.type == 'grad':# Arbitrary gradient option
                        # In place unpacking of grad.t with the starred expression
                        g_t = t0 +  grad.delay + [0, *(grad.t + (grad.t[1] - grad.t[0]) / 2),
                                              grad.t[-1] + grad.t[1] - grad.t[0]]
                        g = 1e-3 * np.array((grad.first, *grad.waveform, grad.last))
                    else: # Trapezoid gradient option
                        g_t = t0 + np.cumsum([0, grad.delay, grad.rise_time, grad.flat_time, grad.fall_time])
                        g = 1e-3 * grad.amplitude * np.array([0, 0, 1, 1, 0])

                    if grad.channel == 'x':
                        gx_t_all = np.append(gx_t_all, g_t)
                        gx_all = np.append(gx_all,g)
                    elif grad.channel == 'y':
                        gy_t_all = np.append(gy_t_all, g_t)
                        gy_all = np.append(gy_all,g)
                    elif grad.channel == 'z':
                        gz_t_all = np.append(gz_t_all, g_t)
                        gz_all = np.append(gz_all,g)


        t0 += seq.arr_block_durations[block_counter] # "Current time" gets updated to end of block just examined

    all_waveforms = {'t_adc': adc_t_all, 't_rf': rf_t_all, 't_rf_centers': rf_t_centers,
                     't_gx': gx_t_all, 't_gy':gy_t_all, 't_gz':gz_t_all,
                     'adc': adc_signal_all, 'rf': rf_signal_all, 'rf_centers': rf_signal_centers,'gx':gx_all, 'gy':gy_all, 'gz':gz_all,
                     'grad_unit': '[kHz/m]', 'rf_unit': '[Hz]', 'time_unit':'[seconds]'}

    return all_waveforms

def display_seq_interactive(all_waveforms,time_range=(0,np.inf)):
  """
  Displays timing diagram of any Pulseq seq object

  Parameters
  ----------
  all_waveforms : dict
    Output from export_waveforms
  time_range : tuple
    Display range; (start_time, end_time) in seconds within the sequence
  """
  fig = make_subplots(rows=3, cols=2,
      subplot_titles=("RF magnitude", "Gx", "RF phase", "Gy", "ADC", "Gz"),shared_xaxes='all',row_heights=[10,10,10])

  fig.add_trace(go.Scatter(x=all_waveforms['t_rf'],y=np.absolute(all_waveforms['rf']),mode='lines',name='RF magnitude',line=dict(color='blue',width=2)),
                row=1,col=1)
  fig.add_trace(go.Scatter(x=all_waveforms['t_rf'],y=np.angle(all_waveforms['rf']),mode='lines',name='RF phase',line=dict(color='gray',width=2)),
                row=2,col=1)
  fig.add_trace(go.Scatter(x=all_waveforms['t_adc'],y=np.angle(all_waveforms['adc']),mode='markers',name='ADC with phase',line=dict(color='red',width=2)),
                row=3,col=1)
  fig.add_trace(go.Scatter(x=all_waveforms['t_gx'],y=all_waveforms['gx'],mode='lines',name='Gx',line=dict(color='green', width=2)),
                row=1,col=2)
  fig.add_trace(go.Scatter(x=all_waveforms['t_gy'],y=all_waveforms['gy'],mode='lines',name='Gy',line=dict(color='orange', width=2)),
                row=2,col=2)
  fig.add_trace(go.Scatter(x=all_waveforms['t_gz'],y=all_waveforms['gz'],mode='lines',name='Gz',line=dict(color='purple', width=2)),
                row=3,col=2)

  fig.update_xaxes(title_text="Time (seconds)", row=3, col=1,range=time_range)
  fig.update_xaxes(title_text="Time (seconds)", row=3, col=2,range=time_range)
  fig.update_yaxes(title_text=all_waveforms['rf_unit'],row=1,col=1)
  fig.update_yaxes(title_text='[rads]',row=2,col=1)
  fig.update_yaxes(title_text='[rads]',row=3,col=1)


  fig.show()

def display_ktraj(ktraj,dim):
  """Displays k-space trajectory

  Parameters
  ----------
  ktraj : np.ndarray
    Output kspace trajectory from any of the write_X functions
  dim : int
    Dimensionality of k-space: 2 or 3

  """

  fig = go.Figure()
  if dim == 2:
    fig.add_trace(go.Scatter(x=np.real(ktraj.flatten()),y=np.imag(ktraj.flatten()),marker=dict(size=1,color="red")))

  elif dim == 3:
    fig.add_trace(go.Scatter3d(x=ktraj[:,:,0].flatten(),y=ktraj[:,:,1].flatten(),z=ktraj[:,:,2].flatten(),marker=dict(size=1,color="red")))

  fig.update_layout(width=800, height=800, title="K-space trajectory")


  fig.show()

def demo_2D_UTE(use_half_pulse=True):
    """Demonstrate 2D UTE sequence generation

    Parameters
    ----------
    use_half_pulse : bool
        Whether to use a half pulse. If False, a full pulse is used.

    Returns
    -------
    seq : Sequence
        PyPulseq sequence object
    ktraj : np.ndarray
        K-space trajectory of the radial sequence
    """
    seq, TE, ktraj = write_UTE_2D_rf_spoiled(N=64, Nr=202, FOV=253e-3, thk=5e-3, slice_locs=[0],
                                              FA=10, TR=15e-3, ro_asymmetry=0.97, use_half_pulse=use_half_pulse, rf_dur=1e-3,
                                              TE_use=None,save_seq=False)
    return seq, ktraj

def demo_3D_UTE(use_half_pulse):
    """Demonstrate 3D UTE sequence generation

    Parameters
    ----------
    use_half_pulse : bool
        Whether to use a half pulse. If False, a full pulse is used.

    Returns
    -------
    seq : Sequence
        PyPulseq sequence object
    ktraj : np.ndarray
        K-space trajectory of the radial sequence
    """
    seq, TE, ktraj = write_UTE_3D_rf_spoiled(N=16, FOV=250e-3, slab_thk=253e-3, FA=10, TR=15e-3, ro_asymmetry=0.97,
                            os_factor=1, rf_type='sinc', rf_dur=1e-3, use_half_pulse=use_half_pulse, save_seq=False)
    return seq, ktraj

def demo_CODE():
    """Demonstrate CODE sequence generation

    Returns
    -------
    seq : Sequence
        PyPulseq sequence object
    ktraj : np.ndarray
        K-space trajectory of the radial sequence
    """

    seq, TE, ktraj = make_code_sequence(FOV=253e-3, N=16, TR=15e-3, flip=10, enc_type='3D',
                      os_factor=1, rf_type='gauss',save_seq=False)
    return seq, ktraj

def display_seq(seq,dim):
    """Helper function; exports and displays a demo sequence
    """
    all_waveforms = export_waveforms(seq, time_range=(0, 32e-3))
    display_seq_interactive(all_waveforms, time_range=(0, 32e-3))
    display_ktraj(ktraj,dim)

def dialog_save_files(seq, ktraj, seq_type):
    """Helper function for saving files locally
    """
    root = Tk()
    # Get paths from dialogue
    root.geometry('200x150')
    f = askdirectory()
    if f is None: return
    # Save files as requested
    seq.write(f'{f}/{seq_type}.seq')
    savemat(f'{f}/ktraj_{seq_type}.mat', {'ktraj':ktraj})

if __name__ == "__main__":
    # Ask user which sequence to generate
    dim = 3
    while True:
        seq_type = input("Enter sequence type (options: ute_2d_half, ute_2d_full, ute_3d_half, ute_3d_full, code): ")
        # Generate seq object
        if seq_type == 'ute_2d_half':
            seq, ktraj = demo_2D_UTE(use_half_pulse=True)
            dim = 2
            break
        elif seq_type == 'ute_2d_full':
            seq, ktraj = demo_2D_UTE(use_half_pulse=False)
            dim = 2
            break
        elif seq_type == 'ute_3d_half':
            seq, ktraj = demo_3D_UTE(use_half_pulse=True)
            break
        elif seq_type == 'ute_3d_full':
            seq, ktraj = demo_3D_UTE(use_half_pulse=False)
            break
        elif seq_type == 'code':
            seq, ktraj = demo_CODE()
            break
        else:
            print("Please choose one of the five available sequences.")
    # Test and display
    print("Generating test report...")
    print(seq.test_report())
    print("Displaying sequence...")
    display_seq(seq,dim)

    # Ask if user wants to save files
    save_bool = True
    save_ans = input("Would you like to save the sequence files (y/n)? ")
    if save_ans.upper() in ['N','NO']:
        print("No files saved.")
        save_bool = False
    elif save_ans.upper() in ['Y','YES']:
        print("Saving the files")
    else:
        print("Saving the files anyway")

    # Open dialog to save the files
    if save_bool:
        dialog_save_files(seq, ktraj,seq_type)


