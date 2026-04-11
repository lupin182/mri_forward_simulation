# Bloch Solver
The Bloch Equations provide a mathematical model for the response of nuclear spins to an external RF pulse, as used in MR imaging. The equations are given by  
$$\mathsf{\frac{d M_x(t)}{dt} = -\Delta\omega M_y(t) - \gamma B_{1,y}(t)M_z(t) - \frac{M_x(t)}{T_2}  }$$  
$$\mathsf{\frac{d M_y(t)}{dt} = \Delta\omega M_x(t) + \gamma B_{1,x}(t)M_z(t) - \frac{M_y(t)}{T_2}  }$$  
$$\mathsf{\frac{d M_z(t)}{dt} = \gamma B_{1,y}(t)M_x(t) - \gamma B_{1,x}(t)M_y(t) - \frac{M_z(t) - M_z^0}{T_1}  },$$  
where $\gamma$ is a physical constant known as the [gyromagnetic ratio](https://en.wikipedia.org/wiki/Gyromagnetic_ratio) and $B_1$ is the magnetic field corresponding to the applied RF pulse. Moreover, $\mathsf{\Delta\omega}$ denotes the deviation in frequency from the [Larmor frequency](https://en.wikipedia.org/wiki/Larmor_precession#Larmor_frequency) of the tissue and $\mathsf{T_1}$, $\mathsf{T_2}$ are tissue-specific constants governing the time decay of $\mathsf M$.

This repository contains a small collection of Python Codes to solve the above equations numerically. An example script `main.py` is provided, which illustrates the usage.

![a plot of the algorithm's output](https://github.com/frank-roesler/bloch_solver/blob/main/figure2.png)

Any comments or queries are welcome at https://frank-roesler.github.io/contact/
