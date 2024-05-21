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
t_span = (0, 5 * T_cycle)
t_eval = np.linspace(0, 5 * T_cycle, 1500)



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
param_names = problem['names']
output_vlv = []
output_vrv = []
output_vla = []
output_vra = []
output_psa = []

# Run the model for each sample
for params in param_values:
    sol = solve_ivp(model, t_span, y0, t_eval=t_eval, method='RK45', args=tuple(params))
    output_vlv.append(np.max(sol.y[0]))
    output_vrv.append(np.max(sol.y[1]))
    output_vla.append(np.max(sol.y[2]))
    output_vra.append(np.max(sol.y[3]))
    output_psa.append(np.max(sol.y[4] / params[21]))



# Perform the sensitivity analysis
Si_vlv = sobol.analyze(problem, np.array(output_vlv), calc_second_order=False)
Si_vrv = sobol.analyze(problem, np.array(output_vrv), calc_second_order=False)
Si_vla = sobol.analyze(problem, np.array(output_vla), calc_second_order=False)
Si_vra = sobol.analyze(problem, np.array(output_vra), calc_second_order=False)
Si_psa = sobol.analyze(problem, np.array(output_psa), calc_second_order=False)



end_time = time.time()
print(f"Running time: {end_time - start_time} seconds")


outputs = ['V_lv', 'V_rv', 'V_la', 'V_ra', 'P_sa']


s1_data = np.array([Si_vlv['S1'], Si_vrv['S1'], Si_vla['S1'], Si_vra['S1'], Si_psa['S1']])
st_data = np.array([Si_vlv['ST'], Si_vrv['ST'], Si_vla['ST'], Si_vra['ST'], Si_psa['ST']])

import matplotlib.pyplot as plt
import seaborn as sns
import os
save_dir = os.path.expanduser('~/team/')


fig, ax = plt.subplots(2, 1, figsize=(18, 12))
sns.heatmap(s1_data, ax=ax[0], cmap='viridis', annot=True, fmt=".2f",
            xticklabels=param_names, yticklabels=outputs,
            cbar_kws={'label': 'S1 Score'},
            annot_kws={"size": 10})  
ax[0].set_title('First Order Sensitivity Scores (S1)')

sns.heatmap(st_data, ax=ax[1], cmap='viridis', annot=True, fmt=".2f",
            xticklabels=param_names, yticklabels=outputs,
            cbar_kws={'label': 'ST Score'},
            annot_kws={"size": 10}) 
ax[1].set_title('Total Order Sensitivity Scores (ST)')

plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(save_dir + 'sensitivity_heatmaps.png')  
plt.close(fig)


output_indices = {'vlv': 0, 'psa': 4}  


average_s1 = np.mean(s1_data[[output_indices['vlv'], output_indices['psa']], :], axis=0)
average_st = np.mean(st_data[[output_indices['vlv'], output_indices['psa']], :], axis=0)


s1_sorted_indices = np.argsort(-average_s1)  
st_sorted_indices = np.argsort(-average_st)


fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 16))


ax1.bar(range(len(param_names)), average_s1[s1_sorted_indices], color='blue')
ax1.set_xticks(range(len(param_names)))
ax1.set_xticklabels(np.array(param_names)[s1_sorted_indices], rotation=45, ha='right')
ax1.set_title('Ranking of the parameters based on the average S1')
ax1.set_ylabel('Average S1')


ax2.bar(range(len(param_names)), average_st[st_sorted_indices], color='blue')
ax2.set_xticks(range(len(param_names)))
ax2.set_xticklabels(np.array(param_names)[st_sorted_indices], rotation=45, ha='right')
ax2.set_title('Ranking of the parameters based on the average ST')
ax2.set_ylabel('Average ST')

plt.tight_layout()
plt.savefig(save_dir + 'parameter_ranking_bar_charts.png')  
plt.close(fig)

print("Charts saved to:", save_dir + 'parameter_ranking_bar_charts.png')
