"""
FitzHugh-Nagumo neural simulation and DMD/DMDc analysis entrypoint.
"""

import os
import argparse
import csv
import numpy as np
import jax.numpy as jnp

from dynamics import get_external_current, run_hankel_dmd, run_dmdc
from simulation import simulate_fhn, fit_fhn_parameters
from plotting import plot_results, plot_dmd_results, plot_dmdc_results

def main():
    """Main orchestration entrypoint for FHN simulation, parameter fitting, and DMD/DMDc."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, 'data')
    plots_dir = os.path.join(script_dir, 'plots')
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    
    parser = argparse.ArgumentParser(description="FitzHugh-Nagumo Neural Model Simulator and Parameter Estimator")
    
    parser.add_argument('--v0', type=float, default=-1.5, help="Initial membrane potential (default: -1.5)")
    parser.add_argument('--w0', type=float, default=-0.5, help="Initial recovery variable (default: -0.5)")
    parser.add_argument('--a', type=float, default=0.7, help="Parameter a (default: 0.7)")
    parser.add_argument('--b', type=float, default=0.8, help="Parameter b (default: 0.8)")
    parser.add_argument('--tau', type=float, default=12.5, help="Time constant tau (default: 12.5)")
    parser.add_argument('--I', type=float, default=0.5, help="Constant external current amplitude/value I (default: 0.5)")
    parser.add_argument('--I-type', type=str, default='constant', choices=['constant', 'step', 'sine', 'pulse'],
                        help="Type of dynamic external current (constant, step, sine, pulse) (default: constant)")
    
    parser.add_argument('--dmd', action='store_true', help="Run Hankel Dynamic Mode Decomposition (Hankel-DMD)")
    parser.add_argument('--dmdc', action='store_true', help="Run Dynamic Mode Decomposition with Control (DMDc)")
    parser.add_argument('--dmd-H', type=int, default=50, help="Delay embedding dimension H for Hankel matrix (default: 50)")
    parser.add_argument('--dmd-r', type=int, default=10, help="Truncation rank r for state projection subspace (default: 10)")
    parser.add_argument('--dmd-p', type=int, default=15, help="Truncation rank p for DMDc augmented state-control matrix (default: 15)")
    
    parser.add_argument('--t-max', type=float, default=100.0, help="Total simulation time (default: 100.0)")
    parser.add_argument('--dt', type=float, default=0.1, help="Sampling time step (default: 0.1)")
    
    parser.add_argument('--output', type=str, default=os.path.join(data_dir, 'fhn_data.csv'), 
                        help="Output CSV filename to save time series")
    parser.add_argument('--no-plot', action='store_true', help="Disable matplotlib plotting")
    parser.add_argument('--save-plot', type=str, default=os.path.join(plots_dir, 'fhn_plot.png'), 
                        help="Save the plot as a PNG image file")
    parser.add_argument('--fit-demo', action='store_true', help="Run parameters fitting demonstration using Optax")
    
    args = parser.parse_args()
    
    y0 = [args.v0, args.w0]
    n_steps = int(args.t_max / args.dt) + 1
    t_span = jnp.linspace(0.0, args.t_max, n_steps)
    
    u_data = jnp.array([get_external_current(t, args.I_type, args.I) for t in t_span])
    
    print(f"Simulating FitzHugh-Nagumo model...")
    print(f"Parameters: a={args.a}, b={args.b}, tau={args.tau}, Current={args.I_type} (amplitude={args.I})")
    print(f"Time span: [0, {args.t_max}] with dt={args.dt} ({n_steps} points)")
    
    ys = simulate_fhn(
        y0, t_span, 
        a=args.a, b=args.b, tau=args.tau, I_type=args.I_type, I_val=args.I
    )
    
    fitted_trajectory = None
    noisy_target = None
    true_a, true_b, true_tau = args.a, args.b, args.tau
    
    if args.fit_demo:
        import jax
        key = jax.random.PRNGKey(42)
        noise = jax.random.normal(key, ys.shape) * 0.08
        noisy_target = ys + noise
        
        fitted_params, loss_history, fitted_trajectory = fit_fhn_parameters(
            y0, t_span, noisy_target, args.I_type, args.I, lr=0.03, steps=150
        )
        
        print("\nOptimization Complete!")
        print(f"Target Values: a={true_a:.4f}, b={true_b:.4f}, tau={true_tau:.4f}")
        print(f"Fitted Values: a={fitted_params['a']:.4f}, b={fitted_params['b']:.4f}, tau={fitted_params['tau']:.4f}")
        print(f"Absolute error: a_err={abs(fitted_params['a']-true_a):.4f}, b_err={abs(fitted_params['b']-true_b):.4f}, tau_err={abs(fitted_params['tau']-true_tau):.4f}")
        
    if args.output:
        print(f"\nSaving time series data to {args.output}...")
        try:
            with open(args.output, mode='w', newline='') as f:
                writer = csv.writer(f)
                header = ['time', 'v_potential', 'w_recovery', 'I_ext']
                if args.fit_demo:
                    header += ['v_measured', 'w_measured', 'v_fitted', 'w_fitted']
                writer.writerow(header)
                
                for i in range(len(t_span)):
                    row = [float(t_span[i]), float(ys[i, 0]), float(ys[i, 1]), float(u_data[i])]
                    if args.fit_demo:
                        row += [float(noisy_target[i, 0]), float(noisy_target[i, 1]),
                                float(fitted_trajectory[i, 0]), float(fitted_trajectory[i, 1])]
                    writer.writerow(row)
            print(f"Successfully wrote {len(t_span)} steps of time series data to {args.output}")
        except Exception as e:
            print(f"Error saving CSV: {e}")
            
    if args.dmd:
        print(f"\nRunning Hankel-DMD Analysis on potential v...")
        print(f"Hankel Matrix parameters: H={args.dmd_H}, Truncation Rank r={args.dmd_r}")
        
        try:
            A_matrix, dmd_eigenvalues, s_vals, dmd_X, dmd_Y = run_hankel_dmd(
                ys[:, 0], H=args.dmd_H, r=args.dmd_r
            )
            
            print("Hankel-DMD Complete!")
            print(f"Shifted Hankel X shape: {dmd_X.shape}")
            print(f"Shifted Hankel Y shape: {dmd_Y.shape}")
            print(f"Truncated Dynamic Matrix A shape: {A_matrix.shape}")
            print(f"Top 5 Singular Values: {s_vals[:5]}")
            print(f"Koopman Eigenvalues (first 5):\n{dmd_eigenvalues[:5]}")
            
            if args.output:
                dmd_output_path = args.output.replace(".csv", "_dmd_A.csv")
                np.savetxt(dmd_output_path, A_matrix, delimiter=",")
                print(f"Saved truncated transition matrix A to {dmd_output_path}")
                
            if not args.no_plot or args.save_plot:
                print("Generating DMD matplotlib spectrum plots...")
                plot_dmd_results(
                    s_vals, dmd_eigenvalues, r=args.dmd_r, H=args.dmd_H,
                    save_path=args.save_plot, show_plot=not args.no_plot
                )
        except Exception as e:
            print(f"Error running DMD: {e}")

    if args.dmdc:
        print(f"\nRunning Dynamic Mode Decomposition with Control (DMDc)...")
        print(f"Hankel parameters: H={args.dmd_H} | Truncation Ranks: state r={args.dmd_r}, augmented p={args.dmd_p}")
        
        try:
            A_tilde, B_tilde, dmdc_eigenvalues, s_x, s_p, dmdc_X, dmdc_Y, dmdc_U = run_dmdc(
                ys[:, 0], u_data, H=args.dmd_H, r=args.dmd_r, p=args.dmd_p
            )
            
            print("DMDc Complete!")
            print(f"Augmented state-input matrix Omega shape: ({dmdc_X.shape[0] + dmdc_U.shape[0]}, {dmdc_X.shape[1]})")
            print(f"Autonomous transition A_tilde shape: {A_tilde.shape}")
            print(f"Control coupling B_tilde shape: {B_tilde.shape}")
            print(f"Top 5 Intrinsic Singular Values: {s_x[:5]}")
            print(f"Intrinsic Koopman Eigenvalues (first 5):\n{dmdc_eigenvalues[:5]}")
            
            if args.output:
                dmdc_A_path = args.output.replace(".csv", "_dmdc_A.csv")
                dmdc_B_path = args.output.replace(".csv", "_dmdc_B.csv")
                np.savetxt(dmdc_A_path, A_tilde, delimiter=",")
                np.savetxt(dmdc_B_path, B_tilde, delimiter=",")
                print(f"Saved autonomous operator A_tilde to {dmdc_A_path}")
                print(f"Saved control operator B_tilde to {dmdc_B_path}")
                
            if not args.no_plot or args.save_plot:
                print("Generating DMDc matplotlib spectra and control plots...")
                plot_dmdc_results(
                    s_x, s_p, dmdc_eigenvalues, B_tilde, r=args.dmd_r, H=args.dmd_H, p=args.dmd_p,
                    save_path=args.save_plot, show_plot=not args.no_plot
                )
        except Exception as e:
            print(f"Error running DMDc: {e}")

    if not args.no_plot or args.save_plot:
        print("\nGenerating matplotlib visualization...")
        plot_results(
            t_span, ys, 
            a=args.a, b=args.b, tau=args.tau, I_type=args.I_type, I_val=args.I, 
            y0=y0, 
            u_data=u_data,
            fitted_data=fitted_trajectory, 
            noisy_target=noisy_target,
            save_path=args.save_plot,
            show_plot=not args.no_plot
        )

if __name__ == '__main__':
    main()
