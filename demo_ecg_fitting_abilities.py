"""
Demonstration script showing PyHEARTS ability to fit both normal and abnormal ECG waves.

This script generates synthetic ECG beats with various abnormalities and visualizes
how PyHEARTS fits Gaussian curves to them, demonstrating robustness to:
- Missing peaks (P, Q, S, T)
- Inverted R waves
- Multiple abnormalities
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd

from pyhearts import PyHEARTS
from pyhearts.processing.gaussian import gaussian_function


def create_synthetic_beat(
    sampling_rate=1000,
    include_p=True,
    include_q=True,
    include_r=True,
    include_s=True,
    include_t=True,
    invert_r=False,
    noise_level=0.02,
):
    """
    Create a synthetic ECG beat with configurable waves.
    
    Parameters
    ----------
    sampling_rate : float
        Sampling rate in Hz
    include_p, include_q, include_r, include_s, include_t : bool
        Whether to include each wave component
    invert_r : bool
        If True, R wave is inverted (negative)
    noise_level : float
        Standard deviation of Gaussian noise to add
        
    Returns
    -------
    signal : np.ndarray
        Synthetic ECG signal
    time : np.ndarray
        Time array in seconds
    """
    duration = 1.0  # 1 second beat
    n_samples = int(duration * sampling_rate)
    time = np.linspace(0, duration, n_samples)
    signal = np.zeros(n_samples)
    
    # R-peak at center (0.5 seconds)
    r_time = 0.5
    r_idx = int(r_time * sampling_rate)
    
    # Standard wave timings (relative to R)
    p_time = r_time - 0.16  # P wave ~160ms before R
    q_time = r_time - 0.04  # Q wave ~40ms before R
    s_time = r_time + 0.04  # S wave ~40ms after R
    t_time = r_time + 0.25  # T wave ~250ms after R
    
    # Add waves if requested
    if include_p:
        p_amplitude = 0.15
        p_width = 0.02
        signal += p_amplitude * np.exp(-((time - p_time) ** 2) / (2 * p_width ** 2))
    
    if include_q:
        q_amplitude = -0.1
        q_width = 0.01
        signal += q_amplitude * np.exp(-((time - q_time) ** 2) / (2 * q_width ** 2))
    
    if include_r:
        r_amplitude = 1.0 if not invert_r else -1.0
        r_width = 0.01
        signal += r_amplitude * np.exp(-((time - r_time) ** 2) / (2 * r_width ** 2))
    
    if include_s:
        s_amplitude = -0.2
        s_width = 0.01
        signal += s_amplitude * np.exp(-((time - s_time) ** 2) / (2 * s_width ** 2))
    
    if include_t:
        t_amplitude = 0.3
        t_width = 0.04
        signal += t_amplitude * np.exp(-((time - t_time) ** 2) / (2 * t_width ** 2))
    
    # Add small amount of noise
    if noise_level > 0:
        signal += np.random.normal(0, noise_level, n_samples)
    
    return signal, time


def extract_fitted_gaussians(output_dict, cycle_idx, sampling_rate):
    """
    Extract fitted Gaussian parameters from output_dict for a given cycle.
    
    Returns
    -------
    fitted_gaussians : dict
        Dictionary mapping wave labels to (center, height, std) tuples
    """
    fitted = {}
    components = ["P", "Q", "R", "S", "T"]
    
    # Handle both dict and DataFrame formats
    if isinstance(output_dict, pd.DataFrame):
        df = output_dict
    else:
        # Convert list-based dict to DataFrame for easier indexing
        df = pd.DataFrame(output_dict)
    
    for comp in components:
        center_key = f"{comp}_gauss_center"
        height_key = f"{comp}_gauss_height"
        std_key = f"{comp}_gauss_stdev_samples"
        
        if center_key in df.columns and height_key in df.columns and std_key in df.columns:
            if cycle_idx < len(df):
                center = df[center_key].iloc[cycle_idx]
                height = df[height_key].iloc[cycle_idx]
                std = df[std_key].iloc[cycle_idx]
                
                if not (pd.isna(center) or pd.isna(height) or pd.isna(std)):
                    fitted[comp] = (float(center), float(height), float(std))
    
    return fitted


def plot_beat_with_fit(
    signal,
    time,
    output_dict,
    epochs_df,
    cycle_label,
    cycle_idx_in_df,
    title,
    save_path=None,
):
    """
    Plot a single ECG beat with detected peaks and Gaussian fits.
    
    Parameters
    ----------
    signal : np.ndarray
        Full ECG signal
    time : np.ndarray
        Time array
    output_dict : pd.DataFrame
        PyHEARTS output DataFrame
    epochs_df : pd.DataFrame
        Epochs dataframe
    cycle_label : int
        Cycle label to plot
    cycle_idx_in_df : int
        Index in output_df for accessing fitted parameters
    title : str
        Plot title
    save_path : str, optional
        Path to save figure
    """
    # Get the cycle data
    cycle_data = epochs_df[epochs_df["cycle"] == cycle_label].sort_values("index")
    if len(cycle_data) == 0:
        print(f"Warning: No data for cycle label {cycle_label}")
        return None
    
    # Extract cycle signal and indices
    cycle_signal = cycle_data["signal_y"].values
    cycle_indices = cycle_data["index"].values
    cycle_time = cycle_data["time"].values if "time" in cycle_data.columns else cycle_indices / 1000.0
    
    # Create relative indices for plotting
    xs = np.arange(len(cycle_signal))
    
    # Extract fitted Gaussians
    fitted_gaussians = extract_fitted_gaussians(output_dict, cycle_idx_in_df, sampling_rate=1000)
    
    # Reconstruct the full Gaussian fit
    # Note: gauss_center is already in cycle-relative indices
    fit_components = []
    for comp in ["P", "Q", "R", "S", "T"]:
        if comp in fitted_gaussians:
            center, height, std = fitted_gaussians[comp]
            # gauss_center is already in cycle-relative indices
            if center is not None and not pd.isna(center):
                center_idx_rel = int(np.round(center))
                if 0 <= center_idx_rel < len(cycle_signal):
                    fit_components.append((center_idx_rel, height, std))
    
    # Generate fit curve
    fit_signal = np.zeros_like(cycle_signal)
    if fit_components:
        fit_params = []
        for center_idx, height, std in fit_components:
            fit_params.extend([center_idx, height, std])
        fit_signal = gaussian_function(xs, *fit_params)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot original signal
    ax.plot(xs, cycle_signal, "b-", linewidth=2, label="ECG Signal", alpha=0.8)
    
    # Plot fitted Gaussians
    if len(fit_signal) > 0 and np.any(np.isfinite(fit_signal)):
        ax.plot(xs, fit_signal, "r--", linewidth=2, label="Gaussian Fit", alpha=0.8)
    
    # Plot individual Gaussian components
    colors = {"P": "orange", "Q": "green", "R": "red", "S": "purple", "T": "magenta"}
    for comp in ["P", "Q", "R", "S", "T"]:
        if comp in fitted_gaussians:
            center, height, std = fitted_gaussians[comp]
            center_idx = int(np.round(center))
            if 0 <= center_idx < len(cycle_signal):
                # Plot individual Gaussian component
                comp_fit = height * np.exp(-((xs - center_idx) ** 2) / (2 * std ** 2))
                ax.plot(xs, comp_fit, "--", color=colors[comp], linewidth=1.5, 
                       alpha=0.6, label=f"{comp} Gaussian")
                
                # Mark peak location
                peak_val = cycle_signal[center_idx] if center_idx < len(cycle_signal) else 0
                ax.scatter(center_idx, peak_val, color=colors[comp], s=100, 
                          zorder=5, edgecolor="white", linewidth=1.5)
                ax.text(center_idx, peak_val + 0.1 * np.ptp(cycle_signal), comp,
                       color=colors[comp], fontsize=12, fontweight="bold", ha="center")
    
    # Mark detected peaks from output_dict
    # Handle both dict and DataFrame formats
    if isinstance(output_dict, pd.DataFrame):
        df = output_dict
    else:
        df = pd.DataFrame(output_dict)
    
    for comp in ["P", "Q", "R", "S", "T"]:
        center_key = f"{comp}_center_idx"
        if center_key in df.columns and cycle_idx_in_df < len(df):
            center_idx = df[center_key].iloc[cycle_idx_in_df]
            if center_idx is not None and not pd.isna(center_idx):
                # Convert global index to relative index within cycle
                cycle_start_idx = cycle_data["index"].iloc[0]
                center_idx_rel = int(center_idx - cycle_start_idx)
                if 0 <= center_idx_rel < len(cycle_signal):
                    peak_val = cycle_signal[center_idx_rel]
                    ax.scatter(center_idx_rel, peak_val, color=colors[comp], s=150,
                             marker="x", zorder=6, linewidth=2.5)
    
    ax.set_xlabel("Sample Index (within cycle)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Amplitude (mV)", fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    
    return (fig, ax)


def create_demonstration_plots():
    """Create a series of plots demonstrating PyHEARTS fitting capabilities."""
    
    # Create output directory
    output_dir = Path("demo_fitting_plots")
    output_dir.mkdir(exist_ok=True)
    
    # Define test cases
    test_cases = [
        {
            "name": "Normal PQRST",
            "description": "Complete normal ECG beat with all waves",
            "params": {
                "include_p": True,
                "include_q": True,
                "include_r": True,
                "include_s": True,
                "include_t": True,
                "invert_r": False,
            }
        },
        {
            "name": "Missing P Wave",
            "description": "ECG beat without P wave (atrial fibrillation, junctional rhythm)",
            "params": {
                "include_p": False,
                "include_q": True,
                "include_r": True,
                "include_s": True,
                "include_t": True,
                "invert_r": False,
            }
        },
        {
            "name": "Missing Q Wave",
            "description": "ECG beat without Q wave (common variant)",
            "params": {
                "include_p": True,
                "include_q": False,
                "include_r": True,
                "include_s": True,
                "include_t": True,
                "invert_r": False,
            }
        },
        {
            "name": "Missing S Wave",
            "description": "ECG beat without S wave (rare variant)",
            "params": {
                "include_p": True,
                "include_q": True,
                "include_r": True,
                "include_s": False,
                "include_t": True,
                "invert_r": False,
            }
        },
        {
            "name": "Missing T Wave",
            "description": "ECG beat without T wave (hyperkalemia, ischemia)",
            "params": {
                "include_p": True,
                "include_q": True,
                "include_r": True,
                "include_s": True,
                "include_t": False,
                "invert_r": False,
            }
        },
        {
            "name": "Inverted R Wave",
            "description": "ECG beat with inverted R wave (dextrocardia, lead reversal)",
            "params": {
                "include_p": True,
                "include_q": True,
                "include_r": True,
                "include_s": True,
                "include_t": True,
                "invert_r": True,
            }
        },
        {
            "name": "Missing P and T Waves",
            "description": "ECG beat missing both P and T waves (multiple abnormalities)",
            "params": {
                "include_p": False,
                "include_q": True,
                "include_r": True,
                "include_s": True,
                "include_t": False,
                "invert_r": False,
            }
        },
        {
            "name": "QRS Only (No P or T)",
            "description": "ECG beat with only QRS complex (severe abnormality)",
            "params": {
                "include_p": False,
                "include_q": True,
                "include_r": True,
                "include_s": True,
                "include_t": False,
                "invert_r": False,
            }
        },
    ]
    
    print("=" * 80)
    print("PyHEARTS ECG Fitting Demonstration")
    print("=" * 80)
    print(f"\nGenerating {len(test_cases)} demonstration plots...\n")
    
    all_figures = []
    
    for i, test_case in enumerate(test_cases):
        print(f"\n[{i+1}/{len(test_cases)}] Processing: {test_case['name']}")
        print(f"  Description: {test_case['description']}")
        
        # Create synthetic beat
        signal, time = create_synthetic_beat(
            sampling_rate=1000,
            noise_level=0.02,
            **test_case["params"]
        )
        
        # Create a longer signal with multiple beats for better R-peak detection
        # Repeat the beat 3 times with spacing
        n_beats = 3
        beat_length = len(signal)
        spacing = int(0.3 * 1000)  # 300ms spacing between beats
        full_signal = np.zeros(beat_length * n_beats + spacing * (n_beats - 1))
        
        for j in range(n_beats):
            start_idx = j * (beat_length + spacing)
            end_idx = start_idx + beat_length
            full_signal[start_idx:end_idx] = signal
        
        # Analyze with PyHEARTS
        analyzer = PyHEARTS(sampling_rate=1000, verbose=False, plot=False)
        output_df, epochs_df = analyzer.analyze_ecg(full_signal)
        
        # Find the middle cycle (most representative)
        if len(epochs_df) > 0 and len(output_df) > 0:
            cycles = sorted(epochs_df["cycle"].unique())
            if len(cycles) > 0:
                cycle_label = cycles[len(cycles) // 2]  # Use middle cycle label
                
                # Find the corresponding index in output_df
                # The output_df index should match the cycle index
                cycle_idx_in_df = None
                for idx in output_df.index:
                    if idx < len(cycles) and cycles[idx] == cycle_label:
                        cycle_idx_in_df = idx
                        break
                
                if cycle_idx_in_df is None:
                    # Fallback: use first available cycle
                    cycle_label = cycles[0]
                    cycle_idx_in_df = 0
                
                # Use DataFrame directly (more reliable than converting to dict)
                output_dict = output_df
                
                # Create plot
                title = f"{test_case['name']}\n{test_case['description']}"
                save_path = output_dir / f"demo_{i+1:02d}_{test_case['name'].lower().replace(' ', '_')}.png"
                
                result = plot_beat_with_fit(
                    full_signal,
                    time,
                    output_dict,
                    epochs_df,
                    cycle_label,  # Pass cycle label, not index
                    cycle_idx_in_df,  # Also pass df index for output_dict access
                    title,
                    save_path=save_path,
                )
                
                if result is not None:
                    fig, ax = result
                    all_figures.append((fig, test_case['name']))
                    print(f"  ✓ Plot created and saved")
                else:
                    print(f"  ✗ Failed to create plot")
            else:
                print(f"  ✗ No cycles detected")
        else:
            print(f"  ✗ No epochs created (epochs_df len={len(epochs_df)}, output_df len={len(output_df)})")
    
    # Create summary figure with all beats
    print(f"\nCreating summary figure...")
    create_summary_figure(test_cases, output_dir)
    
    print("\n" + "=" * 80)
    print("Demonstration complete!")
    print(f"All plots saved to: {output_dir}")
    print("=" * 80)
    
    return all_figures


def create_summary_figure(test_cases, output_dir):
    """Create a summary figure showing all test cases side by side."""
    n_cases = len(test_cases)
    n_cols = 4
    n_rows = (n_cases + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))
    axes = axes.flatten() if n_cases > 1 else [axes]
    
    for i, test_case in enumerate(test_cases):
        ax = axes[i]
        
        # Create synthetic beat
        signal, time = create_synthetic_beat(
            sampling_rate=1000,
            noise_level=0.01,  # Less noise for cleaner summary
            **test_case["params"]
        )
        
        # Plot the beat
        ax.plot(time * 1000, signal, "b-", linewidth=2, alpha=0.8)
        ax.set_title(test_case["name"], fontsize=11, fontweight="bold")
        ax.set_xlabel("Time (ms)", fontsize=9)
        ax.set_ylabel("Amplitude (mV)", fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # Mark expected wave locations
        colors = {"P": "orange", "Q": "green", "R": "red", "S": "purple", "T": "magenta"}
        r_time_ms = 500
        
        if test_case["params"]["include_p"]:
            ax.axvline((r_time_ms - 160), color=colors["P"], linestyle="--", alpha=0.5, linewidth=1)
            ax.text((r_time_ms - 160), ax.get_ylim()[1] * 0.9, "P", color=colors["P"], 
                   fontsize=8, ha="center", fontweight="bold")
        if test_case["params"]["include_q"]:
            ax.axvline((r_time_ms - 40), color=colors["Q"], linestyle="--", alpha=0.5, linewidth=1)
            ax.text((r_time_ms - 40), ax.get_ylim()[0] * 0.9, "Q", color=colors["Q"], 
                   fontsize=8, ha="center", fontweight="bold")
        if test_case["params"]["include_r"]:
            ax.axvline(r_time_ms, color=colors["R"], linestyle="--", alpha=0.5, linewidth=1)
            ax.text(r_time_ms, ax.get_ylim()[1] * 0.9 if not test_case["params"]["invert_r"] 
                   else ax.get_ylim()[0] * 0.9, "R", color=colors["R"], 
                   fontsize=8, ha="center", fontweight="bold")
        if test_case["params"]["include_s"]:
            ax.axvline((r_time_ms + 40), color=colors["S"], linestyle="--", alpha=0.5, linewidth=1)
            ax.text((r_time_ms + 40), ax.get_ylim()[0] * 0.9, "S", color=colors["S"], 
                   fontsize=8, ha="center", fontweight="bold")
        if test_case["params"]["include_t"]:
            ax.axvline((r_time_ms + 250), color=colors["T"], linestyle="--", alpha=0.5, linewidth=1)
            ax.text((r_time_ms + 250), ax.get_ylim()[1] * 0.9, "T", color=colors["T"], 
                   fontsize=8, ha="center", fontweight="bold")
    
    # Hide unused subplots
    for i in range(n_cases, len(axes)):
        axes[i].axis("off")
    
    plt.suptitle("PyHEARTS ECG Fitting Demonstration - All Test Cases", 
                fontsize=16, fontweight="bold", y=0.995)
    plt.tight_layout()
    
    summary_path = output_dir / "summary_all_cases.png"
    plt.savefig(summary_path, dpi=150, bbox_inches="tight")
    print(f"  ✓ Summary figure saved: {summary_path}")


if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Create all demonstration plots
    figures = create_demonstration_plots()
    
    # Show plots (optional - comment out if running in headless mode)
    # plt.show()

