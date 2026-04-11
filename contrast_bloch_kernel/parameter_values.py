# Parameter values for RF pulse:
params_pulse = {
    'T_pulse': 1,       # [ms] duration of RF pulse
    'Bmax':    3.55e-4, # [T] amplitude of RF pulse (3.55e-4 = 90º pulse)
    'n_lobes': 5,       # number of sinc lobes
}

# Parameter values for Bloch equation:
params_bloch = {
    'dOmega':  0,       # [kHz] difference from Larmor frequency
    'gamma':   4.2e+4,  # [kHz/T] gyromagnetic ratio
    'M0z':     1,       # initial value of magnetization (normalized to 1)
    'T2':      70,      # [ms] T2 relaxation time
    'T1':      250,     # [ms] T1 relaxation time
    'phi':     0.0,     # phase of RF pulse
}
T_obs = 1500    # [ms] observation time
t_res = 1e-2 # [ms] time resolution of t-axis

