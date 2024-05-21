import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from SALib.sample import sobol as sobol_sample
from SALib.analyze import sobol
import time


T_cycle = 1.0 # Cardiac cycle time

# Define the elastance function for ventricle
def elastance_v(T, E_max, E_min, T_es_v, T_ep_v):
    t = T % T_cycle
    if t < T_es_v:
        return (E_max - E_min) * 0.5 * (1 - np.cos(np.pi * t / T_es_v)) + E_min
    elif t < T_ep_v:
        return (E_max - E_min) * 0.5 * (1 + np.cos(np.pi * (t - T_es_v) / (T_ep_v - T_es_v))) + E_min
    else:
        return E_min


# Define the elastance function for atrium
def elastance_a(T, E_max, E_min, T_ep_a, T_bs_a, T_es_a):
    t = T % T_cycle
    if t < T_ep_a:
        return (E_max - E_min) * 0.5 * (1 - np.cos(np.pi * (t - T_ep_a) / (T_cycle - T_es_a + T_ep_a))) + E_min
    elif t < T_bs_a:
        return E_min
    elif t < T_es_a:
        return (E_max - E_min) * 0.5 * (1 - np.cos(np.pi * (t - T_bs_a) / (T_es_a - T_bs_a))) + E_min
    else:
        return (E_max - E_min) * 0.5 * (1 + np.cos(np.pi * (t - T_es_a) / (T_cycle - T_es_a + T_ep_a))) + E_min




# Define the valve flow functions
def valve(P_in, P_out, R):
    if P_in > P_out:
        q = (P_in - P_out) / R
    else:
        q = 0.0
    return q

# Define the system of ODEs
def model(t, y, E_max_lv, E_min_lv, E_max_rv, E_min_rv, E_max_la, E_min_la, E_max_ra, E_min_ra, T_es_v, T_ep_v, T_ep_a, T_bs_a, T_es_a, Z_ao, R_mv, R_pu, R_ts, R_s, R_sv, R_p, R_pv, C_sa, C_sv, C_pa, C_pv):
    V_lv = y[0] # stressed Left ventricular volume
    V_rv = y[1] # stressed Right ventricular volume
    V_la = y[2] # stressed Left atrial volume
    V_ra = y[3] # stressed Right atrial volume
    V_sa = y[4] # stressed Systemic arterial volume
    V_sv = y[5] # stressed Systemic venous volume
    V_pa = y[6] # stressed Pulmonary arterial volume
    V_pv = y[7] # stressed Pulmonary venous volume


    # Pressures
    P_lv = elastance_v(t, E_max_lv, E_min_lv, T_es_v, T_ep_v) * V_lv # Left ventricular pressure
    P_rv = elastance_v(t, E_max_rv, E_min_rv, T_es_v, T_ep_v) * V_rv # Right ventricular pressure
    P_la = elastance_a(t, E_max_la, E_min_la, T_ep_a, T_bs_a, T_es_a) * V_la # Left atrial pressure
    P_ra = elastance_a(t, E_max_ra, E_min_ra, T_ep_a, T_bs_a, T_es_a) * V_ra # Right atrial pressure
    P_sa = V_sa / C_sa # Systemic arterial pressure
    P_sv = V_sv / C_sv # Systemic venous pressure
    P_pa = V_pa / C_pa # Pulmonary arterial pressure
    P_pv = V_pv / C_pv # Pulmonary venous pressure

    # Flow rates
    Q_sa = valve(P_lv, P_sa, Z_ao) # Systemic arterial (AO) flow
    Q_sv = (P_sa - P_sv) / R_s # Systemic venous flow
    Q_ra = (P_sv -P_ra) / R_sv # Right atrial flow
    Q_rv = valve(P_ra, P_rv, R_ts) # Right ventricular (TS) flow
    Q_pa = valve(P_rv, P_pa, R_pu) # Pulmonary arterial (PU) flow
    Q_pv = (P_pa - P_pv) / R_p # Pulmonary venous flow
    Q_la = (P_pv - P_la) / R_pv # Left atrial flow
    Q_lv = valve(P_la, P_lv, R_mv) # Left ventricular (MV) flow
    
    # Equations for variables         
    dV_lv_dt = Q_lv - Q_sa
    dV_rv_dt = Q_rv - Q_pa
    dV_la_dt = Q_la - Q_lv
    dV_ra_dt = Q_ra - Q_rv
    dV_sa_dt = Q_sa - Q_sv
    dV_sv_dt = Q_sv - Q_ra
    dV_pa_dt = Q_pa - Q_pv
    dV_pv_dt = Q_pv - Q_la
    

    return [dV_lv_dt, dV_rv_dt, dV_la_dt, dV_ra_dt, dV_sa_dt, dV_sv_dt, dV_pa_dt, dV_pv_dt]


# Initial conditions
y0 = [146, 146, 48.1, 48.1, 167.2, 241.1, 83.0, 57.6]

# Time span
t_span = (0* T_cycle, 5 * T_cycle)
t_eval = np.linspace(*t_span, 1500)

# Define the problem for the sensitivity analysis
problem = {
    'num_vars': 25,
    'names': ['E_max_lv', 'E_min_lv', 'E_max_rv', 'E_min_rv', 'E_max_la', 'E_min_la', 'E_max_ra', 'E_min_ra',
              'T_es_v', 'T_ep_v', 'T_ep_a', 'T_bs_a', 'T_es_a',
              'Z_ao', 'R_mv', 'R_pu', 'R_ts', 'R_s', 'R_sv', 'R_p', 'R_pv', 'C_sa', 'C_sv', 'C_pa', 'C_pv'],
    'bounds': [[1.8939, 1.9713], [0.0537, 0.0559], [0.3314, 0.3450], [0.0201, 0.0209], [0.8396, 0.8738], [0.0408, 0.0424], [0.7750, 0.8066], [0.0408, 0.0424],
            [0.294, 0.306], [0.49, 0.51], [0.098, 0.102], [0.441, 0.459], [0.882, 0.918],
            [0.0141, 0.0147], [0.00294, 0.00306], [0.00245, 0.00255], [0.00294, 0.00306], [0.8467, 0.8813], [0.0706, 0.0734], [0.03528, 0.03672], 
            [0.03528, 0.03672], [1.7626, 1.8346], [29.5271, 30.7322], [6.7720, 7.0484], [11.3021, 11.7635]] # 2% variation


}

start_time = time.time()



# Generate the Sobol samples
param_values = sobol_sample.sample(problem, 4096, calc_second_order= False)

output_vlv = []
output_vrv = []
output_vla = []
output_vra = []
output_psa = []

# Run the model for each sample
for params in param_values:
    sol = solve_ivp(model, t_span, y0, t_eval=t_eval, method='RK45', args= params, rtol=1e-8, atol=1e-8)
    # Append the varivaces to the output
    output_vlv.append(np.var(sol.y[0]))
    output_vrv.append(np.var(sol.y[1]))
    output_vla.append(np.var(sol.y[2]))
    output_vra.append(np.var(sol.y[3]))
    output_psa.append(np.var(sol.y[4] / params[21]))
    

    
# Perform the sensitivity analysis
Si_vlv = sobol.analyze(problem, np.array(output_vlv), calc_second_order=False)
Si_vrv = sobol.analyze(problem, np.array(output_vrv), calc_second_order=False)
Si_vla = sobol.analyze(problem, np.array(output_vla), calc_second_order=False)
Si_vra = sobol.analyze(problem, np.array(output_vra), calc_second_order=False)
Si_psa = sobol.analyze(problem, np.array(output_psa), calc_second_order=False)

# Print the sensitivity indices
print('S1 vlv:', Si_vlv['S1'])
print('ST vlv:', Si_vlv['ST'])
print('S1 vrv:', Si_vrv['S1'])
print('ST vrv:', Si_vrv['ST'])
print('S1 vla:', Si_vla['S1'])
print('ST vla:', Si_vla['ST'])
print('S1 vra:', Si_vra['S1'])
print('ST vra:', Si_vra['ST'])
print('S1 psa:', Si_psa['S1'])
print('ST psa:', Si_psa['ST'])

end_time = time.time()
print(f"Running time: {end_time - start_time} seconds")    





