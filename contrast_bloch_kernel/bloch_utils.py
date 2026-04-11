import numpy as np
import matplotlib.pyplot as plt


def B1(t, pos=0, **params):
    """Sinc pulse of duration T_pulse and amplitude Bmax at time pos"""
    B = np.zeros(t.shape)
    pulse_time = np.abs(t-pos)<params['T_pulse']/2
    B[pulse_time] = params['Bmax'] * np.sinc(2*params['n_lobes']*(t[pulse_time]-pos)/params['T_pulse'])
    return B


def solve_bloch_implicit(tvalues, pulse, **params_bloch):
    """Finite difference solution of Bloch equations:"""
    dt = (np.max(tvalues) - np.min(tvalues))/(len(tvalues-1))
    Mx = np.zeros(tvalues.shape)
    My = np.zeros(tvalues.shape)
    Mz = np.zeros(tvalues.shape)
    Mz[0] = params_bloch['M0z']
    for i,t in enumerate(tvalues):
        w = params_bloch['dOmega']*dt/2
        b = params_bloch['gamma']*pulse[i]*np.cos(params_bloch['phi'])*dt/2
        beta1 = dt/(2*params_bloch['T1'])
        beta2 = dt/(2*params_bloch['T2'])
        s = params_bloch['gamma']*pulse[i]*np.sin(params_bloch['phi'])*dt/2
        S = np.array([[beta2, w, s],
                     [-w, beta2, -b],
                     [-s, b, beta1]])
        if i<len(tvalues)-1:
            # Bloch equations:
            rhs = (np.eye(3)-S)@np.array([Mx[i], My[i], Mz[i]]) + np.array([0, 0, 2*beta1*params_bloch['M0z']])
            M = np.linalg.solve(np.eye(3)+S,rhs)
            Mx[i+1] = M[0]
            My[i+1] = M[1]
            Mz[i+1] = M[2]
    return Mx, My, Mz


def plot_results(tvalues, pulse, Mx,My,Mz, block=True, export=False):
    i=0
    offs = int(0.05*len(tvalues))
    if np.max(np.abs(pulse)>0):
        while pulse[i]==0:
            i += 1
    fig, ax = plt.subplots(1, 3, constrained_layout=True, figsize=(15, 4))
    ax[0].plot(tvalues, pulse, linewidth=0.5, color='tab:blue')
    ax[0].legend(['RF pulse'], loc='upper right')
    ax[0].set_xlabel('t [ms]')
    ax[0].set_ylabel('B1 [T]')
    # ax[0].set_xlim((tvalues[max((i-offs,0))],np.max(tvalues)))
    ax[1].plot(tvalues, Mx, tvalues, My, tvalues, Mz, linewidth=0.6)
    ax[1].legend(['Mx', 'My', 'Mz'], loc='upper right')
    # ax[1].set_xlim((tvalues[max((i-offs,0))], np.max(tvalues)))
    ax[1].set_xlabel('t [ms]')
    ax[1].set_ylabel('M (normalized)')
    ax[2].plot(tvalues, np.sqrt(Mx**2 + My**2), tvalues, Mx**2 + My**2 + Mz**2, linewidth=0.6)
    ax[2].legend(['|Mx+iMy|', 'Mx^2+My^2+Mz^2'], loc='upper right')
    # ax[2].set_xlim((tvalues[max((i-offs,0))], np.max(tvalues)))
    ax[2].set_xlabel('t [ms]')
    ax[2].set_ylabel('M (normalized)')
    plt.show(block=block)

