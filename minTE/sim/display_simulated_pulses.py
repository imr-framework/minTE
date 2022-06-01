# Display slice profiles of half-pulse vs. full-pulse excitation in UTE
# Simulated with Virtual Scanner 2.0
from scipy.io import loadmat
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

def display_simulated_pulses(data_path,rev=False):
    a = loadmat(data_path)
    thk_sim = a['thk_sim'][0][0]
    m = a['m']

    if rev:
        m = m[-1::-1,:]

    thk = 5e-3
    fig = make_subplots(rows=2,cols=1,subplot_titles=("Slice profile (magnitude)","Slice profile (phase)",))
    fig.update_annotations(font_size=30)
    #fig.update_layout(xaxis_title="Location along slice direction (mm)",
    #                  yaxis_title="Magnetization component (a.u.)")

    fig['layout']['xaxis2']['title'] = 'Location along slice direction (mm)'
    fig['layout']['xaxis']['title'] = ''
    fig['layout']['yaxis']['title'] = 'Magnetization (a.u.)'
    fig['layout']['yaxis2']['title'] = 'Phase (rad)'
    fig.update_yaxes()
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')

    zlocs = 1e3*np.linspace(-thk_sim/2,thk_sim/2,m.shape[0],endpoint=True)
    fig.add_trace(go.Scatter(x=zlocs,y=np.absolute(m[:,0]+1j*m[:,1]),line=dict(width=5)),row=1,col=1)
    fig.add_trace(go.Scatter(x=zlocs,y=np.angle(m[:,0]+1j*m[:,1]),line=dict(width=5)),row=2,col=1)


    fig.add_vrect(x0=-1e3*thk/2, x1=1e3*thk/2, row="all", col=1, annotation_text="slice", fillcolor="green",
                  opacity=0.2,line_width=0)

    fig.update_layout(font=dict(size=24))
    fig.update_layout(showlegend=False)

    fig.show()

    return

if __name__ == "__main__":
    display_simulated_pulses("2d_UTE_fa90_fullpulse.mat",False)
    display_simulated_pulses("2d_UTE_fa90_halfpulse.mat",False)
    display_simulated_pulses("2d_UTE_fa90_halfpulse.mat",True)
    display_simulated_pulses("2d_UTE_90_halfpulse_combined.mat",False)