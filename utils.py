import numba
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert
from scipy.integrate import solve_ivp
import pywt
import nolds
from tqdm import tqdm

from joblib import Parallel, delayed

def coupled_fhn(t, state, k1=1, k2=1, c=1, a=0.7, b=0.8, alpha=3, w=1):
    x1, y1, x2, y2 = state
    dx1 = alpha*(y1 + x1 - (x1**3)/3 + (k1+c*x2))
    dy1 = -(1/alpha)*(w**2*x1 - a + b*y1)
    dx2 = alpha*(y2 + x2 - (x2**3)/3 + (k2+c*x1))
    dy2 = -(1/alpha)*(w**2*x2 - a + b*y2)
    return np.array([dx1, dy1, dx2, dy2])

def run_experiment_floop(transform_method='hilbert'):

    k1 = -1.4 
    c_values = np.linspace(0, 1, 1000)              
    delta_ratio_values = np.linspace(0, 1.05, 1000)    

    # Arrays to store metrics.
    # freq_diff_matrix = [0]*100 #np.zeros((len(delta_ratio_values), len(c_values)))
    # R_matrix = [0]*100 #np.zeros((len(delta_ratio_values), len(c_values)))
    freq_diff_matrix = []
    R_matrix = []
    
    # Simulation settings.
    t_span = (0, 40)                                  # Total simulation time.
    t_eval = np.linspace(t_span[0], t_span[1], 100)    # Evaluation times.
    initial_state = np.array([0.001, 0.001, 0.001, 0.001])                         # Initial conditions for both oscillators.

    # --- Wavelet parameters (used if transform_method == 'wavelet') ---
    scale = 10        # You may need to adjust this scale.
    wavelet_name = 'cmor1.0-0.1'

    # Loop over the grid.
    for i, delta_ratio in tqdm(enumerate(delta_ratio_values)):
        # Compute k2
        k2 = k1 + delta_ratio
        temp_f = []
        temp_r = []
        for j, c in enumerate(c_values):
            sol = solve_ivp(coupled_fhn, t_span, initial_state, t_eval=t_eval, args=(k1, k2, c), method='LSODA', vectorized=True)
            
            # Discard transients (use only second half of simulation).
            mask = sol.t > (t_span[1] / 2)
            t_trans = sol.t[mask]
            x1_trans = sol.y[0][mask]
            x2_trans = sol.y[2][mask]
            dt = t_trans[1] - t_trans[0]
            
            # Obtain the instantaneous phase using the chosen method.
            if transform_method.lower() == 'hilbert':
                analytic1 = hilbert(x1_trans)
                analytic2 = hilbert(x2_trans)
                phase1 = np.unwrap(np.angle(analytic1))
                phase2 = np.unwrap(np.angle(analytic2))

            elif transform_method.lower() == 'wavelet':
                # Use PyWavelets to perform the continuous wavelet transform.
                # The cwt function returns coefficients for each scale.
                scales = np.array([scale])  # Use a single scale.
                coeffs1, freqs1 = pywt.cwt(x1_trans, scales, wavelet_name, sampling_period=dt)
                coeffs2, freqs2 = pywt.cwt(x2_trans, scales, wavelet_name, sampling_period=dt)
                phase1 = np.unwrap(np.angle(coeffs1[0]))
                phase2 = np.unwrap(np.angle(coeffs2[0]))
            else:
                raise ValueError("Unknown transform method. Use 'hilbert' or 'wavelet'.")
            
            # --- Compute instantaneous frequencies (for reference) ---
            dt = t_trans[1] - t_trans[0]
            inst_freq1 = np.diff(phase1) / (2 * np.pi * dt)
            inst_freq2 = np.diff(phase2) / (2 * np.pi * dt)
            avg_freq1 = np.mean(inst_freq1)
            avg_freq2 = np.mean(inst_freq2)
            # freq_diff_matrix[i, j] = np.abs(avg_freq1 - avg_freq2)
            # temp_f[j] = (np.abs(avg_freq1 - avg_freq2))
            temp_f.append((np.abs(avg_freq1 - avg_freq2)))

            
            # --- Compute the synchronicity index R ---
            # R = |< exp(i*(phase1 - phase2)) >|
            phase_diff = phase1 - phase2
            R = np.abs(np.mean(np.exp(1j * phase_diff)))
            # R_matrix[i, j] = R
            # temp_r[j] = R
            temp_r.append(R)
        # freq_diff_matrix[i] = temp_f
        # R_matrix[i] = temp_r
        freq_diff_matrix.append(temp_f)
        R_matrix.append(temp_r)

    return c_values, delta_ratio_values, freq_diff_matrix, R_matrix

def hilbert_freq_and_R(t_span, initial_state, t_eval, k1, k2, c):
    sol = solve_ivp(coupled_fhn, t_span, initial_state, t_eval=t_eval, args=(k1, k2, c), method='LSODA', vectorized=True)

    mask = sol.t > (t_span[1] / 2)
    t_trans = sol.t[mask]
    x1_trans = sol.y[0][mask]
    x2_trans = sol.y[2][mask]
    dt = t_trans[1] - t_trans[0]

    analytic1 = hilbert(x1_trans)
    analytic2 = hilbert(x2_trans)
    phase1 = np.unwrap(np.angle(analytic1))
    phase2 = np.unwrap(np.angle(analytic2))

    dt = t_trans[1] - t_trans[0]
    inst_freq1 = np.diff(phase1) / (2 * np.pi * dt)
    inst_freq2 = np.diff(phase2) / (2 * np.pi * dt)
    avg_freq1 = np.mean(inst_freq1)
    avg_freq2 = np.mean(inst_freq2)
    freq_diff = (np.abs(avg_freq1 - avg_freq2))

    phase_diff = phase1 - phase2
    R = np.abs(np.mean(np.exp(1j * phase_diff)))
    return R, freq_diff

def wavelet_freq_and_R(t_span, initial_state, t_eval, k1, k2, c, wavelet='cmor1.0-0.1', scale=10):
    sol = solve_ivp(coupled_fhn, t_span, initial_state, t_eval=t_eval, args=(k1, k2, c), method='LSODA', vectorized=True)
    
    # Discard transients (use only second half of simulation).
    mask = sol.t > (t_span[1] / 2)
    t_trans = sol.t[mask]
    x1_trans = sol.y[0][mask]
    x2_trans = sol.y[2][mask]
    dt = t_trans[1] - t_trans[0]

    scales = np.array([scale])  # Use a single scale.
    coeffs1, freqs1 = pywt.cwt(x1_trans, scales, wavelet, sampling_period=dt)
    coeffs2, freqs2 = pywt.cwt(x2_trans, scales, wavelet, sampling_period=dt)
    phase1 = np.unwrap(np.angle(coeffs1[0]))
    phase2 = np.unwrap(np.angle(coeffs2[0]))

    dt = t_trans[1] - t_trans[0]
    inst_freq1 = np.diff(phase1) / (2 * np.pi * dt) 
    inst_freq2 = np.diff(phase2) / (2 * np.pi * dt)
    avg_freq1 = np.mean(inst_freq1)
    avg_freq2 = np.mean(inst_freq2)
    freq_diff = (np.abs(avg_freq1 - avg_freq2))

    phase_diff = phase1 - phase2
    R = np.abs(np.mean(np.exp(1j * phase_diff)))
    return R, freq_diff


def run_experiment_parallel(resolution=100, transformation="hilbert", n_jobs=4, wavelet="cmor1.0-0.1", scale=10):

    k1 = -1.4 
    c_values = np.linspace(0, 1, resolution)              
    delta_ratio_values = np.linspace(0, 1.05, resolution)    

    # Arrays to store metrics.
    freq_diff_matrix = np.zeros((len(delta_ratio_values), len(c_values)))
    R_matrix = np.zeros((len(delta_ratio_values), len(c_values)))
    # freq_diff_matrix = []
    # R_matrix = []
    
    # Simulation settings.
    t_span = (0, 40)                                  # Total simulation time.
    t_eval = np.linspace(t_span[0], t_span[1], 100)    # Evaluation times.
    initial_state = np.array([0.001, 0.001, 0.001, 0.001])  

    # Loop over the grid.
    for i, delta_ratio in tqdm(enumerate(delta_ratio_values)):
        # Compute k2
        k2 = k1 + delta_ratio
        if transformation == "hilbert":
            results = np.array(Parallel(n_jobs=n_jobs)(delayed(hilbert_freq_and_R)(t_span, initial_state, t_eval, k1, k2, c) for c in c_values))
        elif transformation == "wavelet":
            results = np.array(Parallel(n_jobs=n_jobs)(delayed(wavelet_freq_and_R)(t_span, initial_state, t_eval, k1, k2, c, wavelet, scale) for c in c_values))
        else:
            raise ValueError("Unknown transform method. Use 'hilbert' or 'wavelet'.")         

        freq_diff_matrix[i,:] = results[:,1]
        R_matrix[i,:] = results[:,0] 
        # freq_diff_matrix.append(temp_f)
        # R_matrix.append(temp_r)

    return c_values, delta_ratio_values, freq_diff_matrix, R_matrix

def get_solns_cfhn(t_span, initial_state, t_eval, k1, k2, c):
    sol = solve_ivp(coupled_fhn, t_span, initial_state, t_eval=t_eval, args=(k1, k2, c), method='LSODA', vectorized=True)
    return sol

def get_lyapunov_exp(t_span, initial_state, t_eval, k1, k2, c):
    sol = solve_ivp(coupled_fhn, t_span, initial_state, t_eval=t_eval, args=(k1, k2, c), method='LSODA', vectorized=True)

    mask = sol.t > (t_span[1] / 2)
    t_trans = sol.t[mask]
    x1_trans = sol.y[0][mask]
    x2_trans = sol.y[2][mask]
    x1_trans = np.real(x1_trans)
    x2_trans = np.real(x2_trans)

    lambda_x1 = nolds.lyap_r(x1_trans,
                             lag=1,
                             min_tsep=1,
                             trajectory_len=30,
                             emb_dim=4,
                             fit='poly',
                             )
    lambda_x2 = nolds.lyap_r(x2_trans,
                             lag=1,
                             min_tsep=1,
                             trajectory_len=30,
                             emb_dim=4,
                             fit='poly',
                             )

    return lambda_x1, lambda_x2

def get_correlation_dim(t_span, initial_state, t_eval, k1, k2, c):
    sol = solve_ivp(coupled_fhn, t_span, initial_state, t_eval=t_eval, args=(k1, k2, c), method='LSODA', vectorized=True)

    mask = sol.t > (t_span[1] / 2)
    t_trans = sol.t[mask]
    x1_trans = sol.y[0][mask]
    x2_trans = sol.y[2][mask]
    x1_trans = np.real(x1_trans)
    x2_trans = np.real(x2_trans)

    lambda_x1 = nolds.corr_dim(x1_trans,
                               emb_dim=4,
                               lag=1,
                               fit='poly',
                             )
    lambda_x2 = nolds.corr_dim(x2_trans,
                               emb_dim=4,
                               lag=1,
                               fit='poly',
                             )

    return lambda_x1, lambda_x2

def run_complexity(resolution=100, measure='lyapunov', n_jobs=4):
    k1 = -1.4 
    c_values = np.linspace(0, 1, resolution)              
    delta_ratio_values = np.linspace(0, 1.05, resolution)    

    #preallocate
    x1_grid = np.zeros((len(delta_ratio_values), len(c_values)))
    x2_grid = np.zeros((len(delta_ratio_values), len(c_values)))
    
    # Simulation settings.
    t_span = (0, 40)
    t_eval = np.linspace(t_span[0], t_span[1], 100)
    initial_state = np.array([0.001, 0.001, 0.001, 0.001])  

    for i, delta_ratio in tqdm(enumerate(delta_ratio_values)):
        k2 = k1 + delta_ratio
        if measure == "lyapunov":
            results = np.array(Parallel(n_jobs=n_jobs)(delayed(get_lyapunov_exp)(t_span, initial_state, t_eval, k1, k2, c) for c in c_values))
        elif measure == "correlation":
            results = np.array(Parallel(n_jobs=n_jobs)(delayed(get_correlation_dim)(t_span, initial_state, t_eval, k1, k2, c) for c in c_values))
        else:
            raise ValueError("no valid measure, use 'lyapunov' or 'correlation'.")         
        x1_grid[i,:] = results[:,0]
        x2_grid[i,:] = results[:,1]

    return c_values, delta_ratio_values, x1_grid, x2_grid

def sinusoid_ode(t, y, w=1.):
    x, v = y
    dxdt = v
    dvdt = -w**2 * x
    return [dxdt, dvdt]


def calibrate_lyapexp(freq=1):
    y0 = [1.0, 0.0]
    t_span = (0, 5)
    t_eval = np.linspace(t_span[0], t_span[1], 100)
    soln = solve_ivp(sinusoid_ode, t_span, y0, t_eval=t_eval, method='LSODA', vectorized=True)
    exponent = nolds.lyap_r(soln.y[0],
                            lag=1,
                            min_tsep=1,
                            trajectory_len=30,
                            emb_dim=4,
                            fit='poly',
                            )

    return exponent

def plot_complexity(c_values, delta_ratio_values, measure_x1, measure_x2, measure='lyapunov'):
    """
    Plot the results of the grid search experiment.
    
    Parameters:
        c_values (np.ndarray): Array of coupling factors.
        delta_ratio_values (np.ndarray): Array of dk/k₁ values.

    """
    yaxis = (delta_ratio_values)/(1.4)
    X, Y = np.meshgrid(c_values, yaxis)
    
    label = "complexity measure"
    if measure == 'lyapunov':
        label = r"$\lambda_{max}$"
        title = "Max Lyapunov Exponent"
        zero = calibrate_lyapexp()
        print(zero)
        plot_x1 = np.clip(measure_x1, zero, None)
        plot_x2 = np.clip(measure_x2, zero, None)
        
    elif measure == "correlation":
        label = r"$D_c$"
        title = "Correlation Dimension"
        plot_x1 = measure_x1
        plot_x2 = measure_x2

    
    plt.figure(figsize=(12, 5))
    
    # x2
    plt.subplot(1, 2, 1)
    contour1 = plt.contourf(X, Y, plot_x1, levels=40, cmap='viridis')
    plt.colorbar(contour1, label='')
    plt.xlabel('Coupling Factor (c)')
    plt.ylabel(r'$\Delta k/k_1$')
    plt.title(fr'{title} $x_1$')
    
    # x2
    plt.subplot(1, 2, 2)
    contour2 = plt.contourf(X, Y, plot_x2, levels=40, cmap='viridis')
    plt.colorbar(contour2, label=label)
    plt.xlabel('Coupling Factor (c)')
    plt.ylabel(r'$\Delta k/k_1$')
    plt.title(fr'{title} $x_2$')
    
    plt.tight_layout()
    plt.show()