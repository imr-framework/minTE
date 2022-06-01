from math import pi
from virtualscanner.server.simulation.rf_sim.rf_helpers import *
from virtualscanner.server.simulation.rf_sim.rf_simulations import simulate_rf
from minTE.sequences.write_UTE_2D import write_UTE_2D_rf_spoiled
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def simulate_ute_pulses(use_half_pulse=False, rf_index=1):
    """
    Simulate pulses from a 2D UTE sequence

    Parameters
    ----------
    use_half_pulse : bool
        Whether to simulate a half pulse. If False, a full pulse is simulated.
    rf_index : int
        Which RF pulse to use. rf_index = 2 means the second RF pulse of the seq is simulated.

    """
    # Make a UTE sequence and extract the first RF pulse
    thk = 5e-3
    fa = 90
    seq, __, __ = write_UTE_2D_rf_spoiled(thk=thk, FA=fa, use_half_pulse=use_half_pulse, save_seq=False)
    rf = seq.get_block((rf_index-1)*4+1).rf
    gss = seq.get_block((rf_index-1)*4+1).gz

    GAMMA = 42.58e6 * 2 * pi
    GAMMA_BAR = GAMMA / (2 * np.pi)
    rf_dt = rf.t[1] - rf.t[0]
    print(f'Slice bw : {thk * gss.amplitude} Hz')
    bwsim = 2 * thk * gss.amplitude
    signals, m = simulate_rf(bw_spins=bwsim, n_spins=200, pdt1t2=(1,0,0), flip_angle=90, dt=rf_dt,
                             solver="RK45", pulse_type='custom', pulse_shape=rf.signal / GAMMA_BAR, display=False)
    # Visualize signals
    return signals, bwsim, rf


def display_rf_pulse(rf,title):
    """
    Displays a PyPulseq RF pulse

    Parameters
    ----------
    rf : SimpleNamespace
        PyPulseq rf pulse (obtained with seq.get_block(n).rf)
    title : str
        Plot title
    """

    fig = make_subplots(rows=2,cols=1,subplot_titles=("RF magnitude","RF phase"))
    fig.update_layout(title=title)
    fig.add_trace(go.Scatter(x=rf.t, y=np.absolute(rf.signal), mode='lines',
                   line=dict(color='blue', width=2)),row=1, col=1)
    fig.add_trace(go.Scatter(x=rf.t, y=np.angle(rf.signal), mode='lines',
                   line=dict(color='black', width=2)),row=2, col=1)
    fig.update_xaxes(title_text="Time (s)",row=2,col=1)
    fig.update_yaxes(title_text="|B1| (Hz)",row=1,col=1)
    fig.update_yaxes(title_text="B1 phase (rads)",row=2,col=1)

    fig.show()
    return

def display_rf_profile(signals,bw,title):
    """
    Displays a simulated RF profile (transverse magnetization only)

    Parameters
    ----------
    signals : numpy.ndarray
        n_spins x n_timepoints simulated signal; only the last time point is displayed
    bw : float
        Simulation bandwidth
    title : str
        Plot title
    """

    signal = signals[:,-1]
    dfs = np.linspace(-bw/2,bw/2,len(signals))
    fig = make_subplots(rows=2,cols=1,subplot_titles=("Signal amplitude","Signal phase"))
    fig.update_layout(title=title)

    fig.add_trace(go.Scatter(x=dfs,y=np.absolute(signal), mode='lines',line=dict(color='red',width=2)),row=1,col=1)
    fig.add_trace(go.Scatter(x=dfs,y=np.angle(signal), mode='lines',line=dict(color='blue',width=2)),row=2,col=1)

    fig.update_xaxes(title_text="df (Hz)",row=2,col=1)
    fig.update_yaxes(title_text="|Mxy| (a.u.)",row=1,col=1)
    fig.update_yaxes(title_text="Mxy phase (rads)",row=2,col=1)

    fig.show()
    return

if __name__ == "__main__":
    signals_full, bwsim, rf_full = simulate_ute_pulses(use_half_pulse=False,rf_index=1)
    signals_half_1, _, rf_half = simulate_ute_pulses(use_half_pulse=True,rf_index=1)
    signals_half_2, _, _ = simulate_ute_pulses(use_half_pulse=True,rf_index=2)


    display_rf_pulse(rf_full,f"Full pulse ")
    display_rf_pulse(rf_half,f"Half pulse")

    display_rf_profile(signals_full,bwsim,f'Full pulse (slice BW = {bwsim/2} Hz)')
    display_rf_profile(signals_half_1,bwsim,f'Half pulse (1st) (slice BW = {bwsim/2} Hz)')
    display_rf_profile(signals_half_2,bwsim,f'Half pulse (2nd) (slice BW = {bwsim/2} Hz)')
    display_rf_profile(signals_half_1+signals_half_2,bwsim,f'Half pulse (sum of 1st and 2nd) (slice BW = {bwsim/2} Hz)')
