import matplotlib.pyplot as plt


def plot_epochs(all_cycles, x_vals):
    """
    Plots all ECG cycles aligned to R peaks, with the number of cycles displayed in the legend.

    Parameters:
    - all_cycles: list of np.array, each array represents a single ECG cycle
    - x_vals: np.array, time values corresponding to each point in the cycle
    - save_path: str or None, file path to save the plot; if None, the plot is shown instead
    """
    
    # Set up the figure size for the plot
    plt.figure(figsize=(8, 5))
    
    # Plot each cycle in the list of all cycles with transparency
    for cycle in all_cycles:
        plt.plot(x_vals, cycle, alpha=0.7, linewidth=1)  # alpha for transparency
    
    # Add legend to show the number of cycles
    num_cycles = len(all_cycles)
    plt.legend([f'{num_cycles} Cycles'], fontsize=11, loc='upper right', frameon=True)
    
    # Set the title and axis labels
    plt.title('All Cycles Aligned to R Peaks', fontsize=14, weight='bold')
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('ECG Signal (mV)', fontsize=12)
    plt.show()
    
    
       
