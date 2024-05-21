import numpy as np
from scipy.integrate import solve_ivp
from SALib.sample import sobol as sobol_sample
from SALib.analyze import sobol
from matplotlib import pyplot as plt
import time

# Define time circle
T_cycle = 1.0

# Define the elastance function
def elastance(T, E_max, E_min, T_es, T_ed):
    t = T % T_cycle
    if t < T_es:
        return (E_max - E_min) * 0.5 * (1 - np.cos(np.pi * t / T_es)) + E_min
    elif t < T_ed:
        return (E_max - E_min) * 0.5 * (1 + np.cos(np.pi * (t - T_es) / (T_ed - T_es))) + E_min
    else:
        return E_min

# Define the valve flow functions
def valve(P_in, P_out, R):
    if P_in > P_out:
        q = (P_in - P_out) / R
    else:
        q = 0.0
    return q

# Define the system of ODEs
def model(t, y, E_max, E_min, T_es, T_ed, Z_ao, R_mv, R_s, C_sa, C_sv):
    V_lv = y[0] # Left ventricular volume
    P_sa = y[1] # Systemic arterial pressure
    P_sv = y[2] # Systemic venous pressure

    P_lv = elastance(t, E_max, E_min, T_es, T_ed) * V_lv # Left ventricular pressure

    Q_lv = valve(P_sv, P_lv, R_mv) # Left ventricular (MV) flow
    Q_sa = valve(P_lv, P_sa, Z_ao) # Systemic arterial (AO) flow
    Q_sv = valve(P_sa, P_sv, R_s) # Systemic venous (Sys) flow

    
    # Equations for the left ventricular volume, systemic arterial pressure, and systemic venous pressure           
    dV_lv_dt = Q_lv - Q_sa
    dP_sa_dt = (Q_sa - Q_sv) / C_sa
    dP_sv_dt = (Q_sv - Q_lv) / C_sv
    
    return [dV_lv_dt, dP_sa_dt, dP_sv_dt]

# Initial conditions
V_lv_0 = 233.3 # Initial left ventricular volume
P_sa_0 = 7.0 # Initial systemic arterial pressure
P_sv_0 = 7.0 # Initial systemic venous pressure

y0 = [V_lv_0, P_sa_0, P_sv_0]

# Time span
t_span = (0* T_cycle, 5 * T_cycle)
t_eval = np.linspace(*t_span, 1500)


# Define the problem for the sensitivity analysis
problem = {
    'num_vars': 9,
    'names': ['E_max', 'E_min', 'T_es', 'T_ed', 'Z_ao', 'R_mv', 'R_s', 'C_sa', 'C_sv'],
    'bounds': [[1.35, 1.65], [0.027, 0.033], [0.27, 0.33], [0.405, 0.495], [0.0297, 0.0363], [0.054, 0.066], [0.999, 1.221], [1.017, 1.243], [9.9, 12.1]]
} #10% variation

start_time = time.time()

# Generate samples
param_values = sobol_sample.sample(problem, 5000, calc_second_order=False)

# Run the model for each sample
output_vlv = []
output_psa = []
# For this example, outputs is the value of Systemic arterial pressure at time index 50
for params in param_values:
    sol = solve_ivp(model, t_span, y0, t_eval = t_eval, method= 'RK45', args = (params[0], params[1], params[2], params[3], params[4], params[5], params[6], params[7], params[8]), rtol=1e-8, atol=1e-8)
    # append max value of P_sa
    output_vlv.append(np.max(sol.y[0]))
    output_psa.append(np.max(sol.y[1]))



# Perform the sensitivity analysis
Si_vlv = sobol.analyze(problem, np.array(output_vlv), calc_second_order=False)
Si_psa = sobol.analyze(problem, np.array(output_psa), calc_second_order=False)


# Print the sensitivity indices
print('First-order indices for V_lv:')
print(Si_vlv['S1'])
print('Total-order indices for V_lv:')
print(Si_vlv['ST'])
print('First-order indices for P_sa:')
print(Si_psa['S1'])
print('Total-order indices for P_sa:')
print(Si_psa['ST'])

end_time = time.time()
print(f"Running time: {end_time - start_time} seconds")   