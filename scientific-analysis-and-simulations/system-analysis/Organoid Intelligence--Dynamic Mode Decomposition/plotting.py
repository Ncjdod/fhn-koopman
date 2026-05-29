import numpy as np
import matplotlib.pyplot as plt

def plot_results(t_span, ys, a, b, tau, I_type, I_val, y0, u_data=None, fitted_data=None, noisy_target=None, save_path=None, show_plot=True):
    """Plots FHN activation time series and the phase space portrait with nullclines."""
    v = ys[:, 0]
    w = ys[:, 1]
    
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig = plt.figure(figsize=(14, 6))
    
    c_v = '#1f77b4'
    c_w = '#ff7f0e'
    c_v_null = '#2ca02c'
    c_w_null = '#d62728'
    
    ax1 = fig.add_subplot(121)
    ax1.plot(t_span, v, label=r'Membrane Potential $v(t)$', color=c_v, linewidth=2.0)
    ax1.plot(t_span, w, label=r'Recovery Variable $w(t)$', color=c_w, linewidth=2.0)
    
    if u_data is not None:
        ax1.plot(t_span, u_data, label=r'Stimulus Current $I_{ext}(t)$', color='#d62728', linewidth=1.5, linestyle=':', alpha=0.9)
    
    if noisy_target is not None:
        ax1.scatter(t_span[::5], noisy_target[::5, 0], color='black', alpha=0.3, s=8, label='Noisy Target $v_{meas}$')
    if fitted_data is not None:
        ax1.plot(t_span, fitted_data[:, 0], '--', color='#9467bd', linewidth=1.5, label='Fitted $v_{opt}$')
        
    ax1.set_title("Neural Activation Time Series", fontsize=14, fontweight='bold', pad=12)
    ax1.set_xlabel("Time (dimensionless)", fontsize=12)
    ax1.set_ylabel("State Magnitude", fontsize=12)
    ax1.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.9)
    ax1.set_xlim(t_span[0], t_span[-1])
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    ax2 = fig.add_subplot(122)
    
    v_vals = np.linspace(np.min(v) - 0.5, np.max(v) + 0.5, 400)
    v_nullcline = v_vals - (v_vals**3) / 3.0 + I_val
    w_nullcline = (v_vals + a) / b
    
    ax2.plot(v_vals, v_nullcline, '--', color=c_v_null, alpha=0.8, linewidth=1.5, label=r'$v$-nullcline ($dv/dt=0$)')
    ax2.plot(v_vals, w_nullcline, '--', color=c_w_null, alpha=0.8, linewidth=1.5, label=r'$w$-nullcline ($dw/dt=0$)')
    
    ax2.plot(v, w, color='#9467bd', linewidth=2.5, label='System Trajectory')
    ax2.scatter(y0[0], y0[1], color='red', s=50, zorder=5, label=r'Initial Condition $(v_0, w_0)$')
    
    from scipy.optimize import fsolve
    fp_func = lambda x: x - (x**3)/3.0 - (x + a)/b + I_val
    fp_v = float(fsolve(fp_func, 0.0)[0])
    fp_w = (fp_v + a) / b
    ax2.scatter(fp_v, fp_w, color='black', marker='*', s=120, zorder=6, label=f'Fixed Point ({fp_v:.2f}, {fp_w:.2f})')
    
    ax2.set_title("Phase Portrait & Nullclines", fontsize=14, fontweight='bold', pad=12)
    ax2.set_xlabel(r"Membrane Potential $v$", fontsize=12)
    ax2.set_ylabel(r"Recovery Variable $w$", fontsize=12)
    ax2.set_ylim(np.min(w) - 0.3, np.max(w) + 0.3)
    ax2.set_xlim(np.min(v) - 0.3, np.max(v) + 0.3)
    ax2.legend(loc='lower right', frameon=True, facecolor='white', framealpha=0.9)
    ax2.grid(True, linestyle='--', alpha=0.6)
    
    plt.suptitle(f"FitzHugh-Nagumo Model Dynamics\n(a={a:.2f}, b={b:.2f}, \u03c4={tau:.2f}, Current={I_type} ({I_val:.2f}))", 
                 fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Saved visualization plot to {save_path}")
        
    if show_plot:
        plt.show()
    else:
        plt.close()

def plot_dmdc_results(s_x, s_p, eigenvalues, B_tilde, r, H, p, save_path=None, show_plot=True):
    """Plots SVD decay spectra, Koopman spectrum, and control sensitivities for DMDc."""
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig = plt.figure(figsize=(18, 5.5))
    
    ax1 = fig.add_subplot(131)
    ax1.semilogy(s_x, 'o-', color='#1f77b4', markersize=4, label='State X Spectrum')
    ax1.semilogy(s_p, 's--', color='#9467bd', markersize=4, label=r'Augmented $\Omega$ Spectrum')
    ax1.axvline(x=r-1, color='#d62728', linestyle=':', label=f'State Truncation r={r}')
    ax1.axvline(x=p-1, color='#2ca02c', linestyle='-.', label=f'Augmented Truncation p={p}')
    ax1.set_title("SVD Energy Spectra Decay", fontsize=12, fontweight='bold')
    ax1.set_xlabel("Singular Value Index", fontsize=11)
    ax1.set_ylabel("Singular Value Magnitude", fontsize=11)
    ax1.legend(frameon=True, facecolor='white', framealpha=0.9)
    ax1.grid(True, which="both", alpha=0.5)
    
    ax2 = fig.add_subplot(132)
    theta = np.linspace(0, 2*np.pi, 200)
    ax2.plot(np.cos(theta), np.sin(theta), color='gray', linestyle='--', alpha=0.7, label='Unit Circle')
    ax2.scatter(eigenvalues.real, eigenvalues.imag, color='#2ca02c', edgecolor='black', s=70, zorder=5, label='Autonomous modes')
    ax2.set_title(f"Intrinsic Koopman Spectrum (r={r})", fontsize=12, fontweight='bold')
    ax2.set_xlabel(r"Real Part $\Re(\lambda)$", fontsize=11)
    ax2.set_ylabel(r"Imaginary Part $\Im(\lambda)$", fontsize=11)
    ax2.grid(True, alpha=0.5)
    ax2.axhline(0, color='black', linewidth=0.5)
    ax2.axvline(0, color='black', linewidth=0.5)
    ax2.set_aspect('equal')
    ax2.legend(frameon=True, loc='upper right')
    ax2.set_xlim(-1.4, 1.4)
    ax2.set_ylim(-1.4, 1.4)
    
    ax3 = fig.add_subplot(133)
    b_magnitudes = np.abs(np.squeeze(B_tilde))
    indices = np.arange(len(b_magnitudes))
    ax3.bar(indices, b_magnitudes, color='#ff7f0e', edgecolor='black', alpha=0.85, width=0.6)
    ax3.set_title("Control Input Sensitivity (|B|)", fontsize=12, fontweight='bold')
    ax3.set_xlabel("Subspace Mode Index", fontsize=11)
    ax3.set_ylabel("Influence Magnitude", fontsize=11)
    ax3.set_xticks(indices)
    ax3.grid(True, linestyle='--', alpha=0.5)
    
    plt.suptitle(f"Dynamic Mode Decomposition with Control (DMDc) Analysis\n(Base Dynamics A_tilde: {r}x{r} | Input Coupling B_tilde: {r}x1 | Delays H={H})", 
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    if save_path:
        dmdc_save_path = save_path.replace(".png", "_dmdc.png")
        plt.savefig(dmdc_save_path, dpi=300)
        print(f"Saved DMDc visualization plot to {dmdc_save_path}")
        
    if show_plot:
        plt.show()
    else:
        plt.close()

def plot_dmd_results(s, eigenvalues, r, H, save_path=None, show_plot=True):
    """Plots SVD singular values spectrum decay and complex eigenvalues on the unit circle for DMD."""
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    ax1.semilogy(s, 'o-', color='#1f77b4', linewidth=2.0, markersize=5, label='Singular Values')
    ax1.axvline(x=r-1, color='#d62728', linestyle='--', linewidth=1.5, label=f'Truncation Rank r={r}')
    ax1.set_title("SVD Singular Value Spectrum (Energy Decay)", fontsize=13, fontweight='bold', pad=10)
    ax1.set_xlabel("Singular Value Index", fontsize=11)
    ax1.set_ylabel("Singular Value Magnitude (log scale)", fontsize=11)
    ax1.legend(frameon=True, facecolor='white', framealpha=0.9)
    ax1.grid(True, which="both", linestyle='--', alpha=0.5)
    
    theta = np.linspace(0, 2 * np.pi, 200)
    ax2.plot(np.cos(theta), np.sin(theta), color='gray', linestyle='--', alpha=0.7, label='Unit Circle')
    ax2.scatter(eigenvalues.real, eigenvalues.imag, color='#2ca02c', edgecolor='black', s=80, zorder=5, label='DMD Eigenvalues')
    
    ax2.set_title(f"DMD Koopman Spectrum (Complex Plane, r={r})", fontsize=13, fontweight='bold', pad=10)
    ax2.set_xlabel(r"Real Part $\Re(\lambda)$", fontsize=11)
    ax2.set_ylabel(r"Imaginary Part $\Im(\lambda)$", fontsize=11)
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.axhline(y=0, color='black', linewidth=0.8, alpha=0.5)
    ax2.axvline(x=0, color='black', linewidth=0.8, alpha=0.5)
    ax2.set_aspect('equal')
    ax2.legend(frameon=True, facecolor='white', framealpha=0.9, loc='upper right')
    
    ax2.set_xlim(-1.5, 1.5)
    ax2.set_ylim(-1.5, 1.5)
    
    plt.suptitle(f"Hankel Dynamic Mode Decomposition (Hankel-DMD) Analysis\n(Delay coordinates H={H}, Truncated state space r={r})", 
                 fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    if save_path:
        dmd_save_path = save_path.replace(".png", "_dmd.png")
        plt.savefig(dmd_save_path, dpi=300)
        print(f"Saved DMD visualization plot to {dmd_save_path}")
        
    if show_plot:
        plt.show()
    else:
        plt.close()
