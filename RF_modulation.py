import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy.interpolate import interp1d
import os

# =============================================================================
# 1. Global Plotting Style Settings
# =============================================================================
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 18

# =============================================================================
# 2. Physical Parameters
# =============================================================================
L = 4e-3          # Cavity length (unit: meters)
n_refr = 3.3      # Refractive index of the medium inside the cavity (dimensionless)
M = 58e-9         # Phase modulation depth (dimensionless)
c = 3e8           # Speed of light in vacuum (unit: m/s)
beta_norm = -2e-16  # Group Velocity Dispersion (GVD) parameter beta2 (unit: s^2/m), negative for anomalous dispersion

# =============================================================================
# 3. Derived Parameters
# =============================================================================
c_n = c / n_refr    # Speed of light in the medium
T = L / c_n         # Cavity round-trip time
f_rep = 1.0 / T     # Repetition rate (also the Free Spectral Range, FSR)
print(f"Repetition rate f_rep = {f_rep:.3e} Hz")

# =============================================================================
# 4. Helper Functions
# =============================================================================
def calculate_delta_f_min(c_n, M, beta_phys):
    """Calculates the cosine modulation amplitude of the instantaneous frequency."""
    if M / beta_phys < 0:
        return np.nan # Avoid square root of a negative number
    return (c_n / np.pi) * np.sqrt(M / beta_phys)

def calculate_delta_f0(detuning_omega, c_n, L, beta_phys):
    """Calculates the DC offset of the frequency based on angular frequency detuning."""
    return (detuning_omega * L * c_n) / ((2 * np.pi)**2 * beta_phys)

def calculate_t0(detuning_omega, L, c_n, M, beta_phys):
    """Calculates the time offset induced by the detuning."""
    if beta_phys * M < 0:
        return np.nan # Avoid square root of a negative number
    
    # The argument for arcsin must be within [-1, 1]
    arg = (detuning_omega * L) / (8.0 * np.sqrt(beta_phys * M))
    if np.abs(arg) > 1.0:
        return np.nan
    return (L / (2.0 * np.pi * c_n)) * np.arcsin(arg)

# =============================================================================
# 5. Grid and Sampling Setup
# =============================================================================
# The range of mode numbers to analyze (from -60 to +60)
n_modes_max = 60

mode_number_array = np.arange(-n_modes_max, n_modes_max + 1)
frequencies_hz_array = mode_number_array * f_rep

# The physical dispersion parameter has the opposite sign of beta_norm
beta_phys = -beta_norm
# Calculate the theoretical maximum angular frequency detuning
Delta_Omega_max = (8.0 * np.sqrt(beta_phys * M)) / L
# print(f"Maximum angular detuning Delta_Omega_max = {Delta_Omega_max:.3e} rad/s")

# Number of steps for the detuning scan
num_detuning_steps = 201
detuning_abs_array = np.linspace(-Delta_Omega_max, Delta_Omega_max, num_detuning_steps)

# Number of sampling points in the time domain (within one period T)
n_samples_per_period = 4096
# Time resolution
time_resolution = T / n_samples_per_period
# Time array for one round trip
t = np.linspace(0, T, n_samples_per_period, endpoint=False)

# Zero-padding factor to increase spectral resolution
zero_padding_factor = 16
# Total number of points for FFT
N_fft = n_samples_per_period * zero_padding_factor
# Frequency axis for the FFT result
fft_freqs = np.fft.fftshift(np.fft.fftfreq(N_fft, d=time_resolution))

# =============================================================================
# 6. Main Loop: Scan Detuning and Calculate Spectra
# =============================================================================
# Initialize a 2D array to store the power spectrum for each detuning value
power_map_linear = np.zeros((len(frequencies_hz_array), len(detuning_abs_array)))

# Initialize variables to store diagnostic data at specific detuning points
# Zero detuning
zero_detuning_index = np.argmin(np.abs(detuning_abs_array))
inst_freq_t_zero_detuning = None
power_spectrum_zero_detuning = None
# Negative maximum detuning
inst_freq_t_neg_max_detuning = None
power_spectrum_neg_max_detuning = None
# Positive maximum detuning
inst_freq_t_pos_max_detuning = None
power_spectrum_pos_max_detuning = None

# Calculate the instantaneous frequency modulation amplitude
delta_f_min_val = calculate_delta_f_min(c_n, M, beta_phys)
print(f"Instantaneous frequency modulation amplitude delta_f_min = {delta_f_min_val:.3e} Hz")

# Loop over each detuning value
for i, d_omega in enumerate(detuning_abs_array):
    # Calculate the time offset t0
    t0 = calculate_t0(d_omega, L, c_n, M, beta_phys)
    
    # If t0 is not a number (NaN), it means no stable solution exists for this detuning.
    # Set the power to zero and continue to the next iteration.
    if np.isnan(t0):
        power_map_linear[:, i] = 0.0
        continue

    # Calculate the frequency DC offset delta_f0
    delta_f0 = calculate_delta_f0(d_omega, c_n, L, beta_phys)
    # Calculate the instantaneous frequency over time
    inst_freq_t = -delta_f0 + delta_f_min_val * np.cos((2 * np.pi / T) * (t/2.0 - t0))
    
    # Integrate the instantaneous frequency to get the phase over time
    phase_t = -2.0*np.pi * delta_f0 * t + 2*T*delta_f_min_val * np.sin((2*np.pi)/T*(t/2.0-t0))
    # Construct the complex electric field E(t)
    E_complex_t_single = np.exp(1j * phase_t)

    # Zero-pad the time-domain signal to improve spectral resolution
    E_padded = np.zeros(N_fft, dtype=np.complex128)
    E_padded[:n_samples_per_period] = E_complex_t_single
    
    # Perform Fast Fourier Transform (FFT) and shift the zero-frequency component to the center
    spectrum_fft = np.fft.fftshift(np.fft.fft(E_padded))
    # Calculate the power spectrum (square of the magnitude)
    power_spectrum_col = np.abs(spectrum_fft)**2

    # Use linear interpolation to find the power at the exact mode frequencies
    interp_func = interp1d(fft_freqs, power_spectrum_col, bounds_error=False, fill_value=0.0)
    interpolated_power = interp_func(frequencies_hz_array)
    # Store the resulting spectrum in the power map
    power_map_linear[:, i] = interpolated_power
    
    # Store the instantaneous frequency and spectrum for specific diagnostic points
    if i == zero_detuning_index:
        inst_freq_t_zero_detuning = inst_freq_t.copy()
        power_spectrum_zero_detuning = interpolated_power.copy()
    if i == 0: # First point corresponds to -Delta_Omega_max
        inst_freq_t_neg_max_detuning = inst_freq_t.copy()
        power_spectrum_neg_max_detuning = interpolated_power.copy()
    if i == len(detuning_abs_array) - 1: # Last point corresponds to +Delta_Omega_max
        inst_freq_t_pos_max_detuning = inst_freq_t.copy()
        power_spectrum_pos_max_detuning = interpolated_power.copy()

# Normalize the entire power map globally
max_all = np.nanmax(power_map_linear)
if max_all > 0:
    power_map_linear /= max_all
# Convert the linear power map to a decibel (dB) scale, adding a small epsilon to avoid log10(0)
power_map_db = 10.0 * np.log10(power_map_linear + 1e-12)

# =============================================================================
# 7. Plot 1: Tuning-Power Map (Heatmap)
# =============================================================================
fig1, ax = plt.subplots(figsize=(9, 6))
# Normalize the detuning axis to the range [-1, 1]
detuning_normalized_array = detuning_abs_array / Delta_Omega_max
# Create the heatmap using pcolormesh
cax = ax.pcolormesh(detuning_normalized_array, mode_number_array, power_map_db,
                    cmap='viridis', shading='auto', vmin=-40, vmax=0)

# Set labels and ticks
ax.set_xlabel(r'Modulation Detuning $\Delta\Omega / \Delta\Omega_{max}$')
ax.set_ylabel('Mode Number')
ax.set_xticks([-1, 0, 1])
ax.set_xticklabels([r'$-\Delta\Omega_{max}$', '0', r'$\Delta\Omega_{max}$'])

# Add a color bar
cbar = fig1.colorbar(cax, ax=ax)
cbar.set_label('Optical Power [dB]')
ax.set_ylim(-n_modes_max, n_modes_max)

# Force the plot's aspect ratio to be square for better visualization
x_range = ax.get_xlim()[1] - ax.get_xlim()[0]
y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
ax.set_aspect(abs(x_range / y_range), adjustable='box')

plt.tight_layout()

# =============================================================================
# 8. Diagnostic Plots for Specific Detuning Points
# =============================================================================

# --- Figure 2: Zero Detuning ---
fig2, axes = plt.subplots(2, 1, figsize=(9, 7))
if inst_freq_t_zero_detuning is not None:
    # Top subplot: Plot instantaneous frequency vs. time
    axes[0].plot(t * 1e9, inst_freq_t_zero_detuning, linewidth=2)
    axes[0].set_xlabel('Time (ns)')
    axes[0].set_ylabel('Instantaneous Freq.')
    axes[0].set_title('Instantaneous Frequency (Zero Detuning)')
    axes[0].grid(True)
    axes[0].yaxis.set_major_formatter(mticker.EngFormatter(unit='Hz'))
    
    # Bottom subplot: Plot the corresponding power spectrum
    if np.max(power_spectrum_zero_detuning) > 0:
        power_spectrum_zero_detuning /= np.max(power_spectrum_zero_detuning)
    axes[1].stem(mode_number_array, power_spectrum_zero_detuning, basefmt=" ")
    axes[1].set_xlabel('Mode Number')
    axes[1].set_ylabel('Normalized Power')
    axes[1].set_title('Spectrum at Zero Detuning')
    axes[1].set_xlim(-n_modes_max, n_modes_max)
    axes[1].grid(True)
plt.tight_layout()

# --- Figure 3: Negative Maximum Detuning (-Delta Omega_max) ---
fig3, axes = plt.subplots(2, 1, figsize=(9, 7))
if inst_freq_t_neg_max_detuning is not None:
    # Top subplot: Plot instantaneous frequency
    axes[0].plot(t * 1e9, inst_freq_t_neg_max_detuning, linewidth=2, color='coral')
    axes[0].set_xlabel('Time (ns)')
    axes[0].set_ylabel('Instantaneous Freq.')
    axes[0].set_title(r'Instantaneous Frequency (Detuning = $-\Delta\Omega_{max}$)')
    axes[0].yaxis.set_major_formatter(mticker.EngFormatter(unit='Hz'))
    axes[0].grid(True)
    
    # Bottom subplot: Plot the spectrum
    if np.max(power_spectrum_neg_max_detuning) > 0:
        power_spectrum_neg_max_detuning /= np.max(power_spectrum_neg_max_detuning)
    # The markerfmt argument only accepts marker style and color, not a combined string.
    axes[1].stem(mode_number_array, power_spectrum_neg_max_detuning, basefmt=" ", linefmt='coral', markerfmt='D')
    axes[1].set_xlabel('Mode Number')
    axes[1].set_ylabel('Normalized Power')
    axes[1].set_title(r'Spectrum at Detuning = $-\Delta\Omega_{max}$')
    axes[1].set_xlim(-n_modes_max, n_modes_max)
    axes[1].grid(True)
plt.tight_layout()

# --- Figure 4: Positive Maximum Detuning (+Delta Omega_max) ---
fig4, axes = plt.subplots(2, 1, figsize=(9, 7))
if inst_freq_t_pos_max_detuning is not None:
    # Top subplot: Plot instantaneous frequency
    axes[0].plot(t * 1e9, inst_freq_t_pos_max_detuning, linewidth=2, color='mediumseagreen')
    axes[0].set_xlabel('Time (ns)')
    axes[0].set_ylabel('Instantaneous Freq.')
    axes[0].set_title(r'Instantaneous Frequency (Detuning = $+\Delta\Omega_{max}$)')
    axes[0].grid(True)
    
    # Bottom subplot: Plot the spectrum
    if np.max(power_spectrum_pos_max_detuning) > 0:
        power_spectrum_pos_max_detuning /= np.max(power_spectrum_pos_max_detuning)
    # Correcting the markerfmt argument
    axes[1].stem(mode_number_array, power_spectrum_pos_max_detuning, basefmt=" ", linefmt='mediumseagreen', markerfmt='D')
    axes[1].set_xlabel('Mode Number')
    axes[1].set_ylabel('Normalized Power')
    axes[1].set_title(r'Spectrum at Detuning = $+\Delta\Omega_{max}$')
    axes[1].set_xlim(-n_modes_max, n_modes_max)
    axes[1].grid(True)
plt.tight_layout()

# =============================================================================
# 9. Save or Show Figures
# =============================================================================
# Set to True to save figures to disk, False to display them interactively
isSave = False
output_dir = "output_figures"

# Create the output directory if it doesn't exist
if isSave and not os.path.exists(output_dir):
    os.makedirs(output_dir)

if isSave:
    # Save each figure to the output directory
    fig1.savefig(os.path.join(output_dir, 'fig1_tuning_map.png'), dpi=600)
    fig2.savefig(os.path.join(output_dir, 'fig2_diag_zero_detuning.png'), dpi=600)
    fig3.savefig(os.path.join(output_dir, 'fig3_diag_neg_max_detuning.png'), dpi=600)
    fig4.savefig(os.path.join(output_dir, 'fig4_diag_pos_max_detuning.png'), dpi=600)
    print(f"Figures saved to '{output_dir}' directory.")
    # Close all plot windows to prevent them from popping up in save mode
    plt.close('all')
else:
    # Show the plots on the screen
    plt.show()