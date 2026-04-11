from bloch_utils import *
from parameter_values import *

# --------------------------------------------
# Solve Bloch equations on [0,T_obs]:
# --------------------------------------------

tvalues = np.arange(0,T_obs, t_res)

TR     = 450  # repetition time
T_init = 200  # tine of first pulse

# Several identical Sinc pulses separated by time TR:
pulse = B1(tvalues-T_init,**params_pulse) + B1(tvalues-TR-T_init,**params_pulse) + B1(tvalues-2*TR-T_init,**params_pulse)

Mx, My, Mz = solve_bloch_implicit(tvalues, pulse, **params_bloch)

# Plot results:
plot_results(tvalues, pulse, Mx,My,Mz, block=True, export=True)



















