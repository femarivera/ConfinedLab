##########################################################################################
#  modtransient6.py - Modular Transient Utilities for MODFLOW 6 Model Time-Series Analysis
##########################################################################################
#
#  Author: MARIN RIVERA Carlos Felipe
#  Organization: Bordeaux INP, Lab EPOC, Université de Bordeaux
#  Project: Funded by the OneWater PEPR DEESAC Project
#
#  DESCRIPTION:
#  ------------
#  As part of the ConfinedLab project, this module provides utilities for analyzing,
#  processing, and visualizing transient (time-dependent) results from MODFLOW 6 groundwater models.
#
#  MAIN FEATURES:
#  --------------
#  - Extract and process time-series data (heads, flows, budgets) from MODFLOW 6 outputs.
#  - Visualize temporal evolution of model heads and budget components.
#  - Computes the proportions of water flow to wells from storage release and capture rates.
#  - If zones are defined, plots and analyses water budgets for each zone.

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Import local modules
import sys
sys.path.append('..')
from mlibs import modplot6 # type: ignore


def simplify_name(name):
    """
    Simplifies a column or component name for plotting and display.
    If the input string contains parentheses, extracts and returns the text inside them.
    Otherwise, returns the stripped original string.

    Args:
        name (str): The input string to simplify (e.g., a column name).

    Returns:
        str: Simplified name for display or legend.
    """
    # Extract content inside parentheses, if present
    if '(' in name and ')' in name:
        simplified = name.split('(')[1].split(')')[0].strip()  # Extract text inside parentheses
    else:
        simplified = name.strip()  # Fallback if no parentheses are found
    return simplified
   
def plot_head_time_series(head_file_path, 
                          gwf, 
                          output_path, 
                          show=False, 
                          save=False,
                          figsize=(14, 12), 
                          fontsize=14, 
                          tau=None, 
                          time_units='days'):
    """
    Plot MODFLOW 6 simulated groundwater head time series for one or more observation points.

    This function reads head observation data from a CSV file (as exported by Flopy or MODFLOW 6),
    and generates a time series plot for each observation location. It supports optional display
    and saving of the plot, automatic axis scaling, and marking equilibrium times based on a 
    provided time constant (tau).

    Args:
        head_file_path (str): Path to the head observation CSV file.
        gwf (flopy.modflow.ModflowGwf): Flopy groundwater flow model object.
        output_path (str): Path to save the plot if save is True.
        show (bool, optional): Whether to display the plot interactively. Defaults to False.
        save (bool, optional): Whether to save the plot to disk. Defaults to False.
        figsize (tuple, optional): Figure size in inches. Defaults to (14, 12).
        fontsize (int, optional): Font size for plot labels and titles. Defaults to 14.
        tau (float or None, optional): Time constant for equilibrium analysis. If provided, 
            vertical lines are drawn at 3*tau (95% equilibrium) and 5*tau (99% equilibrium).
        time_units (str, optional): "days" or "years". Units for time axis label. Defaults to 'days'.
                                    Assumes model inputs in days by default.

    Returns:
        None. Displays and/or saves a plot showing groundwater head values over time for each observation.
    """
    import matplotlib.pyplot as plt
    import os
    
    # Retrieve head observation data using Flopy
    #csv = gwf.head_obs.output.obs(f=head_file_path).get_data()
    csv = pd.read_csv(head_file_path)

    fig = plt.figure(figsize=figsize)

    # Plot head values over time
    if time_units == 'days':
        time_axis = csv["time"]  # Assuming input time is in days
        time_axis_label = 'Time [days]'
    if time_units == 'years':
        time_axis = csv["time"] / 360  # Convert days to years
        time_axis_label = 'Time [years]'
    else:
        raise ValueError("time_units must be 'days' or 'years'")

    for name in csv.columns[1:]:  # Skip the first column (time)
        plt.plot(time_axis, csv[name], label=name)

    # Automatically adapt the y-axis limits based on the data range
    plt.xlabel(time_axis_label, fontsize=fontsize/1.2)
    plt.ylabel('Head [m]', fontsize=fontsize/1.2)
    plt.title('HEAD TIME SERIES', fontsize=fontsize)
    plt.legend(fontsize=fontsize/1.2)
    plt.grid(True)

    # Plot equilibrium lines if tau is provided
    if tau is not None:
        eq_95 = 3 * tau
        eq_99 = 5 * tau
        plt.axvline(eq_95, color='red', linestyle='--', label=f'95% Equilibrium (3τ) at {eq_95} {time_axis_label}')
        plt.axvline(eq_99, color='blue', linestyle='--', label=f'99% Equilibrium (5τ) at {eq_99} {time_axis_label}')
        plt.legend(fontsize=fontsize/1.2, loc='upper right')

    # Adjust layout and show plot
    plt.tight_layout(rect=[0, 0, 1, 0.96])  

    if show:
        plt.tight_layout()
        plt.show()

    # Save plot
    if save:
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        fig.savefig(output_path, dpi=300)
        plt.close(fig)

def time_step_length(stress_period_data):
    """
    Compute the time step lengths for each stress period in a MODFLOW 6 simulation.

    This function calculates the sequence of time step durations for each stress period,
    accounting for both uniform and variable (TSMULT) time stepping. It supports MODFLOW's
    time step multiplier logic, returning a list of lists where each inner list contains
    the time step lengths for one stress period.

    Parameters:
        stress_period_data (list of tuples): Each tuple is (PERLEN, NSTP, TSMULT), where
            PERLEN (float): Length of the stress period.
            NSTP (int): Number of time steps in the stress period.
            TSMULT (float): Time step multiplier (1 for uniform, >1 for variable).

    Returns:
        time_steps (list of lists): Each inner list contains the time step lengths for one stress period.
    """
    time_steps = []
    
    for period_data in stress_period_data:
        perlen, nstp, tsmult = period_data
        period_time_steps = []
        
        if tsmult == 1:
            # If TSMULT is 1, each time step is equal (PERLEN / NSTP)
            delta_t = perlen / nstp
            period_time_steps = [delta_t] * nstp
        else:
            # Calculate the first time step using the formula
            delta_t1 = (perlen * (tsmult - 1)) / ((tsmult ** nstp) - 1)
            period_time_steps = [delta_t1]
            
            # Generate successive time steps by multiplying previous time step by TSMULT
            for i in range(1, nstp):
                delta_t_next = period_time_steps[-1] * tsmult
                period_time_steps.append(delta_t_next)
        
        # Add the period time steps to the main time_steps list
        time_steps.append(period_time_steps)
    
    return time_steps

def generate_cumulative_time(stress_period_data):
    """
    Generate the cumulative simulation time at each time step for a MODFLOW 6 transient model.

    This function computes the cummulative elapsed time for all time steps across all stress periods,
    using the MODFLOW 6 stress period definitions and time step multipliers. The result is a list of
    cumulative times, useful for plotting or indexing time-dependent results.

    Parameters:
        stress_period_data (list of tuples): Each tuple is (PERLEN, NSTP, TSMULT), where
            PERLEN (float): Length of the stress period.
            NSTP (int): Number of time steps in the stress period.
            TSMULT (float): Time step multiplier (1 for uniform, >1 for variable).

    Returns:
        cumulative_time (list of float): Cumulative simulation time at each time step across all stress periods.
    """
    time_steps = time_step_length(stress_period_data)  # Generate the time steps first
    
    cumulative_time = []
    total_time = 0  # Initialize total cumulative time
    
    for period_time_steps in time_steps:
        for t in period_time_steps:
            total_time += t
            cumulative_time.append(total_time)
    
    return cumulative_time

def elapsed_time(stress_period_data, sp_num, ts_num):
    """
    Calculate the total elapsed simulation time up to a specific stress period and time step in MODFLOW 6.

    Parameters:
        stress_period_data (list of tuples): Each tuple is (PERLEN, NSTP, TSMULT), where
            PERLEN (float): Length of the stress period.
            NSTP (int): Number of time steps in the stress period.
            TSMULT (float): Time step multiplier (1 for uniform, >1 for variable).
        sp_num (int): Stress period index (0-based).
        ts_num (int): Time step index within the stress period (0-based).

    Returns:
        elapsed_time (float): Total elapsed simulation time up to the specified stress period and time step.
    """
    # Generate time steps and cumulative time for all stress periods
    time_steps = time_step_length(stress_period_data)
    cumulative_time = generate_cumulative_time(stress_period_data)
    
    elapsed_time = 0
    # Sum the elapsed time for all previous stress periods
    for i in range(sp_num):
        elapsed_time += sum(time_steps[i])
    
    # Add the time steps for the current stress period up to the requested time step
    elapsed_time += sum(time_steps[sp_num][:ts_num+1])
    
    return elapsed_time

def total_sim_time(stress_period_data):
    """
    Returns the total simulation time (total cumulative time at the end of the last stress period).

    Parameters:
    - stress_period_data (list): List of tuples, where each tuple is (PERLEN, NSTP, TSMULT)

    Returns:
    - total_time (float): Total simulation time
    """
    cumulative_time = generate_cumulative_time(stress_period_data)
    return cumulative_time[-1]  # Return the total elapsed time after the last stress period

def process_csv_budget(csv_path):
    """
    Processes a MODFLOW 6 budget CSV to compute water balance components and percentages. Recommended use before
    plotting time series.

    Args:
        csv_path (str): Path to the budget CSV file.

    Outputs:
        Updates the CSV with new columns for induced recharge, captured discharge, storage release, capture rates, 
        percentages, and net flows for each inflow/outflow pair.
    """
    # Load the CSV file
    data = pd.read_csv(csv_path)
    # Prepare data for time series
    time_data = data["time"]

    # Extract the reference inflow and outflow from the first time step (first row)
    reference_inflow = data["TOTAL_IN"].iloc[0]
    reference_outflow = data["TOTAL_OUT"].iloc[0]

    # Identify columns
    columns_in = [col for col in data.columns if col.endswith("_IN") and "STO" not in col and col != "TOTAL_IN"]
    columns_well = [col for col in data.columns if "WEL" in col and "OUT" in col]
    columns_out = [col for col in data.columns if col.endswith("_OUT") and "WEL" not in col and "STO" not in col and col != "TOTAL_OUT"]

    # Storage components
    columns_storage_in = [col for col in data.columns if "STO" in col and "IN" in col]
    columns_storage_out = [col for col in data.columns if "STO" in col and "OUT" in col]

    # Compute components
    induced_recharge = data[columns_in].sum(axis=1) - reference_inflow
    decreased_discharge = data[columns_out].sum(axis=1)
    captured_discharge = reference_outflow - decreased_discharge
    total_pumped = data[columns_well].sum(axis=1)
    storage_in = data[columns_storage_in].sum(axis=1)
    storage_out = data[columns_storage_out].sum(axis=1)
    from_storage = storage_in - storage_out
    storage_change_rate = storage_out - storage_in
    capture = induced_recharge + captured_discharge
    storage_change_integrals = np.array([np.trapz(storage_change_rate[:i+1], time_data[:i+1]) - 
                                         np.trapz(storage_change_rate[:i], time_data[:i]) 
                                         if i > 0 else 0 for i in range(len(storage_change_rate))])
    storage_change = np.cumsum(storage_change_integrals)

    # Compute percentages (handle division by zero)
    induced_recharge_pct = (induced_recharge * 100 / total_pumped).where(total_pumped != 0, 0)
    captured_discharge_pct = (captured_discharge * 100 / total_pumped).where(total_pumped != 0, 0)
    from_storage_pct = (from_storage * 100 / total_pumped).where(total_pumped != 0, 0)
    capture_pct = (capture * 100 / total_pumped).where(total_pumped != 0, 0)

    # Add computed components and percentages to the DataFrame
    data["Induced_Recharge"] = induced_recharge
    data["Captured_Discharge"] = captured_discharge
    data["Storage_Release"] = from_storage
    data["Capture"] = capture
    data["Storage_Change_rate"] = storage_change_rate
    data["Storage_Change"] = storage_change
    data["Induced_Recharge_Pct"] = induced_recharge_pct
    data["Captured_Discharge_Pct"] = captured_discharge_pct
    data["Storage_Release_Pct"] = from_storage_pct
    data["Capture_Pct"] = capture_pct

    # Compute net flow for each inflow/outflow pair
    net_flow_columns = []
    for col_in in columns_in:
        base_name = col_in[:-3]  # Remove the "_IN" suffix
        matching_out_col = base_name + "_OUT"
        if matching_out_col in columns_out:
            net_flow_col_name = base_name + "_Net_Flow"
            data[net_flow_col_name] = data[col_in] - data[matching_out_col]
            net_flow_columns.append(net_flow_col_name)

    # Overwrite the original file with the updated DataFrame
    data.to_csv(csv_path, index=False)

def process_csv_zonebudget(csv_path):
    """
    Processes a MODFLOW 6 zonebudget CSV to compute water balance components and percentages.Recommended use before
    plotting time series.

    Args:
        csv_path (str): Path to the zone budget CSV file.

    Outputs:
        Updates the CSV with new columns for induced recharge, captured discharge, storage release, capture rates, 
        percentages, and net flows for each inflow/outflow pair, for each zone.
    """
    # Load the CSV file
    df = pd.read_csv(csv_path)

    # Filter inflow, outflow, and storage columns
    inflow_columns = [
        col for col in df.columns if 
        ("IN" in col or "FROM" in col) and 
        "STO" not in col and "DATA" not in col and "ZONE 0" not in col
    ]
    outflow_columns = [
        col for col in df.columns if 
        ("OUT" in col or "TO" in col) and 
        "STO" not in col and "DATA" not in col and "ZONE 0" not in col and "WEL" not in col
    ]
    storage_out_columns = [
        col for col in df.columns if "STO" in col and "OUT" in col
    ]
    storage_in_columns = [
        col for col in df.columns if "STO" in col and "IN" in col
    ]
    pumped_columns = [
        col for col in df.columns if "WEL" in col and "OUT" in col
    ]

    # Calculate reference inflow and outflow at time zero (reference state)
    reference_inflow = df.loc[df['totim'] == 0, inflow_columns].sum(axis=1).values[0]
    reference_outflow = df.loc[df['totim'] == 0, outflow_columns].sum(axis=1).values[0]

    # Compute components using vectorized operations
    induced_recharge = df[inflow_columns].sum(axis=1) - reference_inflow
    captured_discharge = reference_outflow - df[outflow_columns].sum(axis=1)
    storage_in = df[storage_in_columns].sum(axis=1)
    storage_out = df[storage_out_columns].sum(axis=1)
    from_storage = storage_in - storage_out
    total_pumped = df[pumped_columns].sum(axis=1)
    capture = induced_recharge + captured_discharge

    # Compute percentages (handle division by zero)
    induced_recharge_pct = (induced_recharge * 100 / total_pumped).where(total_pumped != 0, 0)
    captured_discharge_pct = (captured_discharge * 100 / total_pumped).where(total_pumped != 0, 0)
    from_storage_pct = (from_storage * 100 / total_pumped).where(total_pumped != 0, 0)
    capture_pct = (capture * 100 / total_pumped).where(total_pumped != 0, 0)

    # Add computed components and percentages to the DataFrame
    df["Induced_Recharge"] = induced_recharge
    df["Captured_Discharge"] = captured_discharge
    df["From_Storage"] = from_storage
    df["Capture"] = capture
    df["Induced_Recharge_Pct"] = induced_recharge_pct
    df["Captured_Discharge_Pct"] = captured_discharge_pct
    df["From_Storage_Pct"] = from_storage_pct
    df["Capture_Pct"] = capture_pct

    # Overwrite the CSV file
    df.to_csv(csv_path, index=False)

def plot_bud_time_series(file_path, 
                         output_path, 
                         show=False, 
                         save=False,
                         figsize=(14, 12), 
                         fontsize=14,
                         time_units='days'):
    """
    Creates time series plots for inflow, outflow, storage components, and change in storage over time
    based on a budget summary CSV output of a MODFLOW 6 Transient simulation.

    Args:
        file_path (str): Path to the budget CSV file. The file should have a column called 'time' and 
                         columns ending in _IN, _OUT, containing TOTAL_IN, TOTAL_OUT, and columns with STO.
        output_path (str): Path to save the plot if save is True.
        show (bool): Display the plot interactively.
        save (bool): Save the plot to disk.
        figsize (tuple): Figure size in inches.
        fontsize (int): Font size for plot labels.
        time_units (str): "days" or "years". Units for time axis label. Defaults to 'days'. 
                        Assumes model inputs in days by default.

    Outputs:
        A figure with four subplots showing:
        1. Inflow components over time.
        2. Outflow components over time.
        3. Total Inflows and Total Outflows over time.
        4. Cumulative change in Storage over time.
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    import os

    # Load the CSV file
    data = pd.read_csv(file_path)
    
    # Identify columns for inflow, outflow, and total
    columns_in = [col for col in data.columns if col.endswith("_IN") and "STO" not in col and col != "TOTAL_IN"]
    columns_out = [col for col in data.columns if col.endswith("_OUT") and "STO" not in col and col != "TOTAL_OUT"]
    columns_total = ["TOTAL_IN", "TOTAL_OUT"]
    data_total_in = data[columns_in].sum(axis=1)
    data_total_out = data[columns_out].sum(axis=1)

    # Identify columns for storage components
    columns_storage_in = [col for col in data.columns if "STO" in col and "IN" in col]
    columns_storage_out = [col for col in data.columns if "STO" in col and "OUT" in col]

    # Prepare data for time series
    if time_units == 'days':
        time_data = data["time"]  # Assuming input time is in days
        time_axis_label = 'Time [days]'
    elif time_units == 'years':
        time_data = data["time"] / 360  # Convert days to years
        time_axis_label = 'Time [years]'
    else:
        raise ValueError("time_units must be 'days' or 'years'")
    
    # Find the global max across all relevant columns
    ymax = max(
    data[columns_in].to_numpy().max(),
    data[columns_out].to_numpy().max(),
    data_total_in.max(),
    data_total_out.max())

    # Create a figure with subplots
    fig, axs = plt.subplots(2, 2, figsize=figsize)

    # Plot inflow components (excluding TOTAL_IN and STO columns)
    for col in columns_in:
        axs[0, 0].plot(time_data, data[col], label=simplify_name(col))
    axs[0, 0].set_title("INFLOW COMPONENTS", fontsize=fontsize)
    axs[0, 0].set_xlabel(time_axis_label, fontsize=fontsize/1.2)
    axs[0, 0].set_ylabel("m³/day", fontsize=fontsize/1.2)
    axs[0, 0].legend(fontsize=fontsize/1.2)
    axs[0, 0].set_ylim(0, ymax*1.1)  # Set y-axis limit based on global max
    axs[0, 0].grid()

    # Plot outflow components (excluding TOTAL_OUT and STO columns)
    for col in columns_out:
        axs[0, 1].plot(time_data, data[col], label=simplify_name(col))
    axs[0, 1].set_title("OUTFLOW COMPONENTS", fontsize=fontsize)
    axs[0, 1].set_xlabel(time_axis_label, fontsize=fontsize/1.2)
    axs[0, 1].set_ylabel("m³/day", fontsize=fontsize/1.2)
    axs[0, 1].legend(fontsize=fontsize/1.2)
    axs[0, 1].set_ylim(0, ymax*1.1)  # Set y-axis limit based on global max
    axs[0, 1].grid()

    # TOTAL IN and TOTAL OUT
    axs[1, 0].plot(time_data, data_total_in, label="TOTAL INFLOW", color="blue")
    axs[1, 0].plot(time_data, data_total_out, label="TOTAL OUTFLOW", color="red")
    axs[1, 0].set_title("TOTAL FLOWS", fontsize=fontsize)
    axs[1, 0].set_xlabel(time_axis_label, fontsize=fontsize/1.2)
    axs[1, 0].set_ylabel("m³/day", fontsize=fontsize/1.2)
    axs[1, 0].legend(fontsize=fontsize/1.2)
    axs[1, 0].set_ylim(0, ymax*1.1)  # Set y-axis limit based on global max
    axs[1, 0].grid()

    # Compute and plot CHANGE IN STORAGE (STORAGE OUT - STORAGE IN)
    storage_in = data[columns_storage_in].sum(axis=1)
    storage_out = data[columns_storage_out].sum(axis=1)
    storage_change_rate = storage_out - storage_in
    storage_change_integrals = np.array([np.trapz(storage_change_rate[:i+1], time_data[:i+1]) - 
                                         np.trapz(storage_change_rate[:i], time_data[:i]) 
                                         if i > 0 else 0 for i in range(len(storage_change_rate))])
    storage_change = np.cumsum(storage_change_integrals)
    axs[1, 1].plot(time_data, storage_change, label="STORAGE CHANGE", color="green")
    axs[1, 1].set_title("CHANGE IN STORAGE", fontsize=fontsize)
    axs[1, 1].set_xlabel(time_axis_label, fontsize=fontsize/1.2)
    axs[1, 1].set_ylabel("m³", fontsize=fontsize/1.2)
    axs[1, 1].legend(fontsize=fontsize/1.2)
    axs[1, 1].grid()

    # Adjust layout and show plot
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    if show:
        plt.show()
    
    # Save plot
    if save:
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        fig.savefig(output_path, dpi=300)
        plt.close(fig)

def plot_net_flow_time_series(file_path, 
                              output_path, 
                              show=False, 
                              save=False, 
                              figsize=(14, 12), 
                              fontsize=16, 
                              tau=None, 
                              time_units='days'):
    """
    Creates time series plots for the difference between inflow and outflow components.
    Positive values represent net inflows, and negative values represent net outflows.

    Args:
        file_path (str): Path to the budget CSV file. The file should have a column called 'time' and 
                         columns ending in _IN, _OUT.
        output_path (str): Path to save the plot if save is True.
        show (bool): Whether to display the plot. Defaults to False.
        save (bool): Whether to save the plot. Defaults to False.
        figsize (tuple): Size of the figure. Defaults to (14, 12).
        fontsize (int): Font size for plot labels and titles.
        tau (float or None): Time constant. If provided, vertical lines will be drawn at 3*tau and 5*tau. Defaults to None.
        time_units (str): "days" or "years". Units for time axis label. Defaults to 'days'.
                         Assumes model inputs in days by default.

    Outputs:
        A figure with time series plots showing the difference between inflow and outflow
        for each component.
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    import os

    # Load the CSV file
    data = pd.read_csv(file_path)

    # Identify matching inflow and outflow columns
    columns_in = [col for col in data.columns if col.endswith("_IN") and "STO" not in col and col != "TOTAL_IN"]
    columns_out = [col.replace("_IN", "_OUT") for col in columns_in if col.replace("_IN", "_OUT") in data.columns]

    # Prepare time data
    if time_units == 'days':
        time_data = data["time"]  # Assuming input time is in days
        time_axis_label = 'Time [days]'
    elif time_units == 'years':
        time_data = data["time"] / 360  # Convert days to years
        time_axis_label = 'Time [years]'
    else:
        raise ValueError("time_units must be 'days' or 'years'")

    # Create a figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot net flux for each component
    legend_labels = []
    for col_in, col_out in zip(columns_in, columns_out):
        net_flux = data[col_in] - data[col_out]
        label = simplify_name(col_in)
        ax.plot(time_data, net_flux, label=label)
        legend_labels.append(label)

    # Sort legend labels alphabetically
    handles, labels = ax.get_legend_handles_labels()
    sorted_handles_labels = sorted(zip(handles, labels), key=lambda x: x[1])
    handles, labels = zip(*sorted_handles_labels)
    ax.legend(handles, labels, fontsize=fontsize / 1.2)

    # Horizontal line at zero for the X-axis
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')

    ax.set_title("Net Flow (Inflow - Outflow) Components", fontsize=fontsize)
    ax.set_xlabel(time_axis_label, fontsize=fontsize / 1.2)
    ax.set_ylabel("Net Flow [m³/day]", fontsize=fontsize / 1.2)
    ax.grid()

    # Plot equilibrium lines if tau is provided
    if tau is not None:
        eq_95 = 3 * tau
        eq_99 = 5 * tau
        ax.axvline(eq_95, color='red', linestyle='--', label=f'95% Equilibrium (3τ) at {eq_95} {time_axis_label}')
        ax.axvline(eq_99, color='blue', linestyle='--', label=f'99% Equilibrium (5τ) at {eq_99} {time_axis_label}')
        ax.legend(fontsize=fontsize / 1.2, loc='upper right')

    # Adjust layout and show plot
    plt.tight_layout()

    if show:
        plt.show()

    # Save plot
    if save:
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        fig.savefig(output_path, dpi=300)
        plt.close(fig)

def plot_water_to_wells(file_path, 
                        output_path, 
                        show = False,
                        save = False,
                        figsize = (14, 12),
                        fontsize = 14, 
                        time_units='days'):
    """
    Plots various water budget components related to well abstraction:
    
    The first time step in the transient simulation should be a steady state stress period
    to evaluate the effects of pumping starting from natural/baseline/reference conditions.

    - Induced Recharge: Inflow components (excluding recharge) 
      (corresponds to the induced inflows from flow boundaries like RIV, GHB, etc.).
    - Decreased Discharge: Outflow components (excluding well abstractions)
      (corresponds to the captured/intercepted discharge).
    - From Storage: Storage release sourcing well abstraction.
    - Capture: Sum of Induced Recharge and Decreased Discharge.
    - Total Pumped: Sum of all well abstractions.
    
    Args:
        file_path (str): Path to the budget CSV file containing water budget data.
        output_path (str): Path to save the plot if save is True.
        show (bool): Whether to display the plot. Defaults to False.
        save (bool): Whether to save the plot. Defaults to False.  
        figsize (tuple): Size of the figure. Defaults to (14, 12).
        fontsize (int): Font size for plot labels and titles. Defaults to 14.
        time_units (str): "days" or "years". Units for time axis label. Defaults to 'days'.
                         Assumes model inputs in days by default.
    
    Outputs:
        A figure with four subplots:
        1. Induced Recharge, Decreased Discharge, and From Storage over time.
        2. Capture and From Storage over time.
        3. Percentages of Induced Recharge, Decreased Discharge, and From Storage with respect to Total Pumped.
        4. Percentages of Capture and From Storage with respect to Total Pumped.
    """
    # Load the CSV file
    data = pd.read_csv(file_path)

    # Extract the reference inflow and outflow from the first time step (first row)
    reference_inflow = data["TOTAL_IN"].iloc[0]  # value from the first row in "TOTAL_INFLOW"
    reference_outflow = data["TOTAL_OUT"].iloc[0]  # value from the first row in "TOTAL_OUTFLOW"

    # Identify columns
    columns_in = [col for col in data.columns if col.endswith("_IN") and "STO" not in col and col != "TOTAL_IN"]
    columns_well = [col for col in data.columns if "WEL" in col and "OUT" in col]
    columns_out = [col for col in data.columns if col.endswith("_OUT") and "WEL" not in col and "STO" not in col and col != "TOTAL_OUT"]
    
        
    # Storage components
    columns_storage_in = [col for col in data.columns if "STO" in col and "IN" in col]
    columns_storage_out = [col for col in data.columns if "STO" in col and "OUT" in col]

    # Prepare time data
    if time_units == 'days':
        time_data = data["time"]  # Assuming input time is in days
        time_axis_label = 'Time [days]'
    elif time_units == 'years':
        time_data = data["time"] / 360  # Convert days to years
        time_axis_label = 'Time [years]'
    else:
        raise ValueError("time_units must be 'days' or 'years'")

    # Compute components
    induced_recharge = data[columns_in].sum(axis=1) - reference_inflow
    decreased_discharge = data[columns_out].sum(axis=1)
    captured_discharge = reference_outflow - decreased_discharge
    total_pumped = data[columns_well].sum(axis=1)
    storage_in = data[columns_storage_in].sum(axis=1)
    storage_out = data[columns_storage_out].sum(axis=1)
    from_storage = storage_in - storage_out
    capture = induced_recharge + captured_discharge

    # Compute percentages (handle division by zero)
    induced_recharge_pct = (induced_recharge * 100 / total_pumped).where(total_pumped != 0, 0)
    captured_discharge_pct = (captured_discharge * 100 / total_pumped).where(total_pumped != 0, 0)
    from_storage_pct = (from_storage * 100 / total_pumped).where(total_pumped != 0, 0)
    capture_pct = (capture * 100 / total_pumped).where(total_pumped != 0, 0)

    # Create a figure with subplots
    fig, axs = plt.subplots(2, 2, figsize=figsize)

    # Plot 1: Induced Recharge, Decreased Discharge, and From Storage
    axs[0, 0].plot(time_data, induced_recharge, label="Induced inflows", color = "blue")
    axs[0, 0].plot(time_data, captured_discharge, label="Captured outflows", color = "red")
    axs[0, 0].plot(time_data, from_storage, label="Storage release", color = "green")
    axs[0, 0].set_title("WATER TO WELLS", fontsize=fontsize)
    axs[0, 0].set_xlabel(time_axis_label, fontsize=fontsize/1.2)
    axs[0, 0].set_ylabel("m³/day", fontsize=fontsize/1.2)
    axs[0, 0].legend(fontsize=fontsize/1.2)
    axs[0, 0].grid()

    # Plot 2: Capture and From Storage
    axs[0, 1].plot(time_data, capture, label="Capture", color="purple")
    axs[0, 1].plot(time_data, from_storage, label="Storage release", color="green")
    axs[0, 1].set_title("CAPTURE AND STORAGE", fontsize=fontsize)
    axs[0, 1].set_xlabel(time_axis_label, fontsize=fontsize/1.2)
    axs[0, 1].set_ylabel("m³/day",fontsize=fontsize/1.2)
    axs[0, 1].legend(fontsize=fontsize/1.2)
    axs[0, 1].grid()

    # Plot 3: Percentages of Induced Recharge, Decreased Discharge, and From Storage
    axs[1, 0].plot(time_data, induced_recharge_pct, label="Induced inflows %", color="blue")
    axs[1, 0].plot(time_data, captured_discharge_pct, label="Captured outflows %", color="red")
    axs[1, 0].plot(time_data, from_storage_pct, label="Storage release %", color="green")
    axs[1, 0].set_title("WATER TO WELLS PERCENTAGE", fontsize=fontsize)
    axs[1, 0].set_xlabel(time_axis_label, fontsize=fontsize/1.2)
    axs[1, 0].set_ylabel("Percentage (%)",fontsize=fontsize/1.2)
    axs[1, 0].legend(fontsize=fontsize/1.2)
    axs[1, 0].grid()

    # Plot 4: Percentages of Capture and From Storage
    axs[1, 1].plot(time_data, capture_pct, label="Capture %", color="purple")
    axs[1, 1].plot(time_data, from_storage_pct, label="Storage release %", color="green")
    axs[1, 1].set_title("CAPTURE AND STORAGE PERCENTAGE", fontsize=fontsize)
    axs[1, 1].set_xlabel(time_axis_label, fontsize=fontsize/1.2)
    axs[1, 1].set_ylabel("Percentage (%)",fontsize=fontsize/1.2)
    axs[1, 1].legend(fontsize=fontsize/1.2)
    axs[1, 1].grid()

    # Adjust layout and show plot
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    # Adjust layout and show plot
    plt.ioff()
    if show:
        plt.tight_layout()
        plt.show()

    # Save plot
    if save:
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        fig.savefig(output_path, dpi=300)
        plt.close(fig) 

def plot_bud_sum_transient(file_path, 
                           time, 
                           output_path, 
                           show = False, 
                           save = False):
    """
    Creates bar plots for inflow, outflow, and total flows from a MODFLOW 6 budget summary CSV at a specified time.

    Args:
        file_path (str): Path to the budget CSV file (one row per time step).
        time (float): Simulation time to plot. Corresponds to the elapsed time in model units.
        output_path (str): Path to save the figure if save is True.
        show (bool): Display the plot interactively.
        save (bool): Save the plot to disk.

    Outputs:
        A figure with subplots for inflow, outflow, total flows, and change in storage.
    """
    import pandas as pd
    import matplotlib.pyplot as plt

    # Load the CSV file
    data = pd.read_csv(file_path)

    # Identify columns
    columns_in = [col for col in data.columns if col.endswith("_IN") and "STO" not in col and col != "TOTAL_IN"]
    columns_out = [col for col in data.columns if col.endswith("_OUT") and "STO" not in col and col != "TOTAL_OUT"]
    columns_total = ["TOTAL_IN", "TOTAL_OUT"]
    columns_storage_in = [col for col in data.columns if col.endswith("_IN") and "STO" in col and col != "TOTAL_IN"]
    columns_storage_out = [col for col in data.columns if col.endswith("_OUT") and "STO" in col and col != "TOTAL_OUT"]

    # Filter the data for the specified time
    data_time = data[data['time'] == time]

    # Check if data for the specified time exists
    if data_time.empty:
        print(f"No data found for time: {time}")
        return

    # Prepare data for plots (we assume the time column has only one row for each time step)
    data_in = data_time[columns_in].iloc[0]
    data_out = data_time[columns_out].iloc[0]
    data_total = data_time[columns_total].iloc[0]
    data_total_in = data_time[columns_in].sum(axis=1).iloc[0]
    data_total_out = data_time[columns_out].sum(axis=1).iloc[0]
    sum_storage_in = data_time[columns_storage_in].sum(axis=1).iloc[0]  # Sum along the rows
    sum_storage_out = data_time[columns_storage_out].sum(axis=1).iloc[0]  # Sum along the rows
    data_storage = sum_storage_out - sum_storage_in

    # Simplify column names for plotting
    columns_in_simplified = [simplify_name(col) for col in columns_in]
    columns_out_simplified = [simplify_name(col) for col in columns_out]

    # Create a figure with subplots
    fig, axs = plt.subplots(1, 4, figsize=(19, 5))

    # Determine the common y-axis range based on the "Total Inflow and Outflow" plot
    common_ylim_max = max(max(data_in.values), max(data_out.values), max(data_total.values), data_storage) * 1.1 # Add 10% padding

    # Plot inflow components
    axs[0].bar(columns_in_simplified, data_in.values, color="blue")
    axs[0].set_title("Inflow Components")
    axs[0].set_xlabel("Component")
    axs[0].set_ylabel("m³/day")
    axs[0].set_ylim(0,common_ylim_max) 
    for i, val in enumerate(data_in.values):
        axs[0].text(i, val, f"{val:.2f}", ha="center", va="bottom")

    # Plot outflow components
    axs[1].bar(columns_out_simplified, data_out.values, color="red")
    axs[1].set_title("Outflow Components")
    axs[1].set_xlabel("Component")
    axs[1].set_ylabel("m³/day")
    axs[1].set_ylim(0,common_ylim_max) 
    for i, val in enumerate(data_out.values):
        axs[1].text(i, val, f"{val:.2f}", ha="center", va="bottom")

    # Plot total inflow and outflow
    axs[2].bar(["Total Inflows", "Total Outflows"], [data_total_in, data_total_out], color="green")
    axs[2].set_title("Total Inflow and Outflow")
    axs[2].set_xlabel("Component")
    axs[2].set_ylabel("m³/day")
    axs[2].set_ylim(0,common_ylim_max) 
    for i, val in enumerate([data_total_in, data_total_out]):
        axs[2].text(i, val, f"{val:.2f}", ha="center", va="bottom")

    # Plot change in storage
    axs[3].bar(["Change in storage"], [data_storage], color="purple")  # Wrap label and value in lists for single bar
    axs[3].set_title("Change in storage")
    axs[3].set_xlabel("Component")
    axs[3].set_ylabel("m³/day")
    axs[3].set_ylim(-common_ylim_max, common_ylim_max )
    # Add text label for the single bar
    axs[3].text(0, data_storage, f"{data_storage:.2f}", ha="center", va="bottom")


    # Adjust layout and show plot
    plt.ioff()
    if show:
        plt.tight_layout()
        plt.show()

    # Save plot
    if save:
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        fig.savefig(output_path, dpi=300)
        plt.close(fig) 

def plot_zone_budget(csv_path, 
                     output_dir, 
                     show = False, 
                     save = False, 
                     figsize = (14, 12),
                     fontsize = 14,
                     zone_descriptions = None,
                     time_units = 'days'):
    """
    Plots time series of inflows, outflows, total flows, and storage change for each zone from a budget CSV.

    Args:
        csv_path (str): Path to the zone budget CSV file.
        output_dir (str): Directory to save figures if save is True.
        show (bool): Display plots interactively.
        save (bool): Save plots to disk.
        figsize (tuple): Figure size in inches.
        fontsize (int): Font size for plot labels.
        zone_descriptions (dict or None): Optional mapping of zone numbers to descriptions.
        time_units (str): "days" or "years". Units for time axis label. Defaults to 'days'.
                            Assumes model inputs in days by default.

    Outputs:
        Figures for each zone showing inflow, outflow, total flows, storage change, and inter-zone transfers.
    """
    # Load the CSV file
    df = pd.read_csv(csv_path)
    
    # Prepare data for time series
    if time_units == 'days':
        time_data = df["totim"]  # Assuming input time is in days
        time_axis_label = 'Time [days]'
    elif time_units == 'years':
        time_data = df["totim"] / 360  # Convert days to years
        time_axis_label = 'Time [years]'
    else:
        raise ValueError("time_units must be 'days' or 'years'")


    # Identify unique zones
    zones = df['zone'].unique()
    zone_descriptions = zone_descriptions

    # Filter columns for inflows and outflows
    inflow_columns = [
        col for col in df.columns if 
        ("IN" in col or "FROM" in col) and 
        "STO" not in col and "DATA" not in col and "ZONE 0" not in col
    ]
    outflow_columns = [
        col for col in df.columns if 
        ("OUT" in col or "TO" in col) and 
        "STO" not in col and "DATA" not in col and "ZONE 0" not in col
    ]
    storage_out_columns = [
        col for col in df.columns if "STO" in col and "OUT" in col
    ]
    storage_in_columns = [
        col for col in df.columns if "STO" in col and "IN" in col
    ]
    
    # Create plots for each zone
    for zone in zones:
        # Exclude "FROM/TO" columns containing the zone's own number
        zone_specific_exclude = f"ZONE {int(zone)}"
        zone_inflow_columns = [col for col in inflow_columns if zone_specific_exclude not in col]
        zone_outflow_columns = [col for col in outflow_columns if zone_specific_exclude not in col]
        
        zone_data = df[df['zone'] == zone]
        
        # Prepare data for time series
        if time_units == 'days':
            time_data = zone_data["totim"]  # Assuming input time is in days
            time_axis_label = 'Time [days]'
        elif time_units == 'years':
            time_data = zone_data["totim"] / 360  # Convert days to years
            time_axis_label = 'Time [years]'
        else:
            raise ValueError("time_units must be 'days' or 'years'")

        storage_in = zone_data[storage_in_columns].sum(axis=1)
        storage_out = zone_data[storage_out_columns].sum(axis=1)
        storage_change_rate = storage_out - storage_in
        storage_change_integrals = np.array([np.trapz(storage_change_rate[:i+1], time_data[:i+1]) - 
                                         np.trapz(storage_change_rate[:i], time_data[:i]) 
                                         if i > 0 else 0 for i in range(len(storage_change_rate))])
        storage_change = np.cumsum(storage_change_integrals)
        data_in = zone_data[zone_inflow_columns].sum(axis=1)
        data_out = zone_data[zone_outflow_columns].sum(axis=1)

        ymax = max(
        zone_data[zone_inflow_columns].to_numpy().max(),
        zone_data[zone_outflow_columns].to_numpy().max(),
        data_in.max(),
        data_out.max()
)

        # Create a subplot for inflows and outflows
        fig, ax = plt.subplots(2, 2, figsize=figsize, constrained_layout=True)
        
        # Plot inflows
        for col in zone_inflow_columns:
            ax[0,0].plot(zone_data['totim'], zone_data[col], label=simplify_name(col))
        ax[0,0].set_title(f'ZONE {zone} INFLOW COMPONENTS', fontsize=fontsize)
        ax[0,0].set_xlabel(time_axis_label, fontsize=fontsize/1.2)
        ax[0,0].set_ylabel('Flow [m³/day]', fontsize=fontsize/1.2)
        ax[0,0].legend(fontsize=fontsize/1.2)
        ax[0,0].set_ylim(0, ymax*1.1)  # Set y-axis limit based on global max
        ax[0,0].grid()
        
        # Plot outflows
        for col in zone_outflow_columns:
            ax[0,1].plot(zone_data['totim'], zone_data[col], label=simplify_name(col))
        ax[0,1].set_title(f'ZONE {zone} OUTFLOWS', fontsize=fontsize)
        ax[0,1].set_xlabel(time_axis_label, fontsize=fontsize/1.2)
        ax[0,1].set_ylabel('Flow [m³/day]', fontsize=fontsize/1.2)
        ax[0,1].legend(fontsize=fontsize/1.2)
        ax[0,1].set_ylim(0, ymax*1.1)  # Set y-axis limit based on global max
        ax[0,1].grid()

        # Plot TOTAL IN TOTAL OUT
        ax[1,0].plot(zone_data['totim'], data_in, label="TOTAL INFLOWS")
        ax[1,0].plot(zone_data['totim'], data_out, label="TOTAL OUTFLOWS")
        ax[1,0].set_title(f'ZONE {zone} TOTAL FLOWS', fontsize=fontsize)
        ax[1,0].set_xlabel(time_axis_label, fontsize=fontsize/1.2)
        ax[1,0].set_ylabel('Flow [m³/day]', fontsize=fontsize/1.2)
        ax[1,0].legend(fontsize=fontsize/1.2)
        ax[1,0].set_ylim(0, ymax*1.1)  # Set y-axis limit based on global max
        ax[1,0].grid()
        
        # Plot change in storage
        ax[1,1].plot(zone_data['totim'], storage_change, label="CHANGE IN STORAGE")
        ax[1,1].set_title(f'ZONE {zone} CHANGE IN STORAGE', fontsize=fontsize)
        ax[1,1].set_xlabel(time_axis_label, fontsize=fontsize/1.2)
        ax[1,1].set_ylabel('Flow [m³/day]', fontsize=fontsize/1.2)
        ax[1,1].legend(fontsize=fontsize/1.2)
        ax[1,1].grid()
        
        # Adjust layout and show plot
        plt.ioff()
        if show:
            plt.tight_layout()
            plt.show()

         #Save plot
        if save:
            image_path = os.path.join(output_dir, f'Zone {zone} budget.png')
            fig.savefig(image_path, dpi=300)
            plt.close(fig) 
    
    # Plot the difference "FROM ZONE x - TO ZONE x" for all other zones
    fig2 = plt.figure(figsize = figsize)
    for zone in zones:
        other_zones = [f"ZONE {int(z)}" for z in zones if z != zone]
        from_columns = [f"FROM {oz}" for oz in other_zones]
        to_columns = [f"TO {oz}" for oz in other_zones]
        
        from_to_difference = (
            df.loc[df['zone'] == zone, from_columns].sum(axis=1) -
            df.loc[df['zone'] == zone, to_columns].sum(axis=1)
        )
        description = zone_descriptions.get(zone, f"ZONE {zone}")
        plt.plot(df.loc[df['zone'] == zone, 'totim'], from_to_difference, label=f'ZONE {zone} - {description}')
    
    plt.title('WATER TRANSFERS VIA VERTICAL LEAKAGE', fontsize=fontsize)
    plt.xlabel('Time [days]', fontsize=fontsize)
    plt.ylabel('Flow Balance (Inflows - Outflows) [m³/day]', fontsize=fontsize)
    plt.legend(fontsize=fontsize/1.2)
    plt.grid()

    # Adjust layout and show plot
    plt.ioff()
    if show:
        plt.tight_layout()
        plt.show()

    #Save plot
    if save:
        image_path = os.path.join(output_dir, "zonebudget_summary_t.png")
        fig2.savefig(image_path, dpi=300)
        plt.close(fig2) 

def plot_water_to_wells_zonebud(csv_path, 
                                output_dir, 
                                show = False, 
                                save = False,
                                fontsize = 14, 
                                time_units = 'days'):
    """
    Plots water budget components and sources of water to wells for each zone from a zone budget CSV.

    Args:
        csv_path (str): Path to the zone budget CSV file.
        output_dir (str): Directory to save figures if save is True.
        show (bool): Display plots interactively.
        save (bool): Save plots to disk.
        fontsize (int): Font size for plot labels.
        time_units (str): "days" or "years". Units for time axis label. Defaults to 'days'.
                         Assumes model inputs in days by default.

    Outputs:
        Figures for each zone showing storage release, induced recharge, captured discharge, capture, and their percentages.
    """
    # Load the CSV file
    df = pd.read_csv(csv_path)

    # Identify unique zones
    zones = df['zone'].unique()

    # Filter inflow, outflow, and storage columns
    inflow_columns = [
        col for col in df.columns if 
        ("IN" in col or "FROM" in col) and 
        "STO" not in col and "DATA" not in col and "ZONE 0" not in col
    ]
    outflow_columns = [
        col for col in df.columns if 
        ("OUT" in col or "TO" in col) and 
        "STO" not in col and "DATA" not in col and "ZONE 0" not in col and "WEL" not in col
    ]
    storage_out_columns = [
        col for col in df.columns if "STO" in col and "OUT" in col
    ]
    storage_in_columns = [
        col for col in df.columns if "STO" in col and "IN" in col
    ]
    pumped_columns = [
        col for col in df.columns if "WEL" in col and "OUT" in col
    ]

    # Process each zone
    for zone in zones:
        zone_data = df[df['zone'] == zone]

        if time_units == 'days':
            time_data = zone_data["totim"]  # Assuming input time is in days
            time_axis_label = 'Time [days]'
        elif time_units == 'years':
            time_data = zone_data["totim"] / 360  # Convert days to years
            time_axis_label = 'Time [years]'
        else:
            raise ValueError("time_units must be 'days' or 'years'")

        # Calculate reference inflow and outflow at time zero (reference state)
        #reference_inflow = zone_data.loc[zone_data['totim'] == 0, inflow_columns].sum(axis=1).values[0]
        #reference_outflow = zone_data.loc[zone_data['totim'] == 0, outflow_columns].sum(axis=1).values[0]
        reference_inflow = zone_data[inflow_columns].iloc[0].sum()
        reference_outflow = zone_data[outflow_columns].iloc[0].sum()

        # Compute components using vectorized operations
        induced_recharge = zone_data[inflow_columns].sum(axis=1) - reference_inflow
        captured_discharge = reference_outflow - zone_data[outflow_columns].sum(axis=1)
        storage_in = zone_data[storage_in_columns].sum(axis=1)
        storage_out = zone_data[storage_out_columns].sum(axis=1)
        from_storage = storage_in - storage_out
        total_pumped = zone_data[pumped_columns].sum(axis=1)
        capture = induced_recharge + captured_discharge

        # Compute percentages (handle division by zero)
        induced_recharge_pct = (induced_recharge * 100 / total_pumped).where(total_pumped != 0, 0)
        captured_discharge_pct = (captured_discharge * 100 / total_pumped).where(total_pumped != 0, 0)
        from_storage_pct = (from_storage * 100 / total_pumped).where(total_pumped != 0, 0)
        capture_pct = (capture * 100 / total_pumped).where(total_pumped != 0, 0)

        # Create plots for the current zone
        fig, axs = plt.subplots(2, 2, figsize=(14, 12))
        fig.suptitle(f'Zone {zone} Analysis', fontsize=16)

        # Subplot 1: From Storage, Induced Recharge, Captured Discharge
        axs[0, 0].plot(time_data, from_storage, label='Storage release', color='green')
        axs[0, 0].plot(time_data, induced_recharge, label='Induced Inflows', color='blue')
        axs[0, 0].plot(time_data, captured_discharge, label='Captured Outflows', color='red')
        axs[0, 0].set_title('WATER TO WELLS', fontsize=fontsize)
        axs[0, 0].set_xlabel(time_axis_label, fontsize=fontsize/1.2)
        axs[0, 0].set_ylabel('Flow (m³/day)', fontsize=fontsize/1.2)
        axs[0, 0].legend(fontsize=fontsize/1.2)
        axs[0, 0].grid()

        # Subplot 2: From Storage and Capture
        axs[0, 1].plot(time_data, from_storage, label='Storage release', color='green')
        axs[0, 1].plot(time_data, capture, label='Capture', color='purple')
        axs[0, 1].set_title('CAPTURE AND STORAGE', fontsize=fontsize)
        axs[0, 1].set_xlabel(time_axis_label, fontsize=fontsize/1.2)
        axs[0, 1].set_ylabel('Flow (m³/day)', fontsize=fontsize/1.2)
        axs[0, 1].legend(fontsize=fontsize/1.2)
        axs[0, 1].grid()

        # Subplot 3: Percentages of Total Pumped (Flows)
        axs[1, 0].plot(time_data, from_storage_pct, label='From Storage (%)', color='purple')
        axs[1, 0].plot(time_data, induced_recharge_pct, label='Induced Recharge (%)', color='blue')
        axs[1, 0].plot(time_data, captured_discharge_pct, label='Captured Discharge (%)', color='green')
        axs[1, 0].set_title('WATER TO WELLS PERCENTAGE', fontsize=fontsize)
        axs[1, 0].set_xlabel(time_axis_label, fontsize=fontsize/1.2)
        axs[1, 0].set_ylabel('Percent (%)', fontsize=fontsize/1.2)
        axs[1, 0].legend(fontsize=fontsize/1.2)
        axs[1, 0].grid()

        # Subplot 4: Percentages of Total Pumped (From Storage and Capture)
        axs[1, 1].plot(time_data, from_storage_pct, label='From Storage (%)', color='purple')
        axs[1, 1].plot(time_data, capture_pct, label='Capture (%)', color='orange')
        axs[1, 1].set_title('CAPTURE AND STORAGE PERCENTAGE', fontsize=fontsize)
        axs[1, 1].set_xlabel(time_axis_label, fontsize=fontsize/1.2)
        axs[1, 1].set_ylabel('Percent (%)', fontsize=fontsize/1.2)
        axs[1, 1].legend(fontsize=fontsize/1.2)
        axs[1, 1].grid()

        # Adjust layout and show plot
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        # Adjust layout and show plot
        plt.ioff()
        if show:
            plt.tight_layout()
            plt.show()

        #Save plot
        if save:
            image_path = os.path.join(output_dir, f'Zone {zone} water to wells.png')
            fig.savefig(image_path, dpi=300)
            plt.close(fig) 

def plot_storage_change_rate(file_path, 
                            output_path, 
                            show=False, 
                            save=False, 
                            figsize=(14, 12), 
                            fontsize=14,
                            xlim=None,  # Tuple for x-axis limits
                            ylim=None,
                            time_units="days"):  # Tuple for y-axis limits
    """
    Creates a time series plot of the storage change rate.

    Args:
        file_path (str): Path to the budget CSV file. The file should have a column called 'time' and 
                         columns for storage components after being processed with process_csv_budget.
        output_path (str): Path to save the plot if save is True.
        show (bool): Whether to display the plot. Defaults to False.
        save (bool): Whether to save the plot. Defaults to False.
        figsize (tuple): Size of the figure. Defaults to (14, 12).
        fontsize (int): Font size for plot labels and titles.
        xlim (tuple or None): Limits for x-axis (e.g., (0, 500)). If None, default matplotlib behavior.
        ylim (tuple or None): Limits for y-axis (e.g., (-10, 10)). If None, default matplotlib behavior.
        time_units (str): "days" or "years". Units for time axis label. Defaults to 'days'. Assumes model inputs in days by default.

    Outputs:
        A figure with the storage change rate time series.
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    import os

    # Load the CSV file
    data = pd.read_csv(file_path)

    # Identify columns for storage components
    columns_storage_in = [col for col in data.columns if "STO" in col and "IN" in col]
    columns_storage_out = [col for col in data.columns if "STO" in col and "OUT" in col]

    # Prepare data
    if time_units == 'days':
        time_data = data["time"]  # Assuming input time is in days
        time_axis_label = 'Time [days]'
    elif time_units == 'years':
        time_data = data["time"] / 360  # Convert days to years
        time_axis_label = 'Time [years]'
    else:
        raise ValueError("time_units must be 'days' or 'years'")
    
    storage_in = data[columns_storage_in].sum(axis=1)
    storage_out = data[columns_storage_out].sum(axis=1)
    storage_change_rate = storage_out - storage_in

    # Plotting
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(time_data, storage_change_rate, label="STORAGE CHANGE RATE", color="green")

    ax.set_title("Storage change rate", fontsize=fontsize)
    ax.set_xlabel(time_axis_label, fontsize=fontsize / 1.2)
    ax.set_ylabel("m³/day", fontsize=fontsize / 1.2)
    ax.legend(fontsize=fontsize / 1.2)
    ax.grid()

    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)

    plt.tight_layout()

    if show:
        plt.show()

    if save:
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        fig.savefig(output_path, dpi=300)
        plt.close(fig)

def plot_storage_change(file_path, 
                        output_path, 
                        show=False, 
                        save=False, 
                        figsize=(14, 12), 
                        fontsize=14, 
                        xlim=None,
                        ylim=None,
                        time_units="days"):
    """
    Creates a time series plot for change in storage (cumulative).

    Args:
        file_path (str): Path to the budget CSV file. The file should have a column called 'time' and 
                         columns for storage components after being processed with process_csv_budget.
        output_path (str): Path to save the plot if save is True.
        show (bool): Whether to display the plot. Defaults to False.
        save (bool): Whether to save the plot. Defaults to False.
        figsize (tuple): Size of the figure. Defaults to (14, 12).
        fontsize (int): Font size for plot labels and titles.
        xlim (tuple or None): x-axis limits (min, max).
        ylim (tuple or None): y-axis limits (min, max).
        time_units (str): "days" or "years". Units for time axis label. Defaults to 'days'. Assumes model inputs in days by default.

    Outputs:
        A figure with the cumulative change in storage time series.
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    import os

    # Load the CSV file
    data = pd.read_csv(file_path)

    # Identify columns for storage components
    columns_storage_in = [col for col in data.columns if "STO" in col and "IN" in col]
    columns_storage_out = [col for col in data.columns if "STO" in col and "OUT" in col]

    # Prepare data for time series
    if time_units == 'days':
        time_data = data["time"]  # Assuming input time is in days
        time_axis_label = 'Time [days]'
    elif time_units == 'years':
        time_data = data["time"] / 360  # Convert days to years
        time_axis_label = 'Time [years]'
    else:
        raise ValueError("time_units must be 'days' or 'years'")

    # Compute the storage change rate (STORAGE OUT - STORAGE IN)
    storage_in = data[columns_storage_in].sum(axis=1)
    storage_out = data[columns_storage_out].sum(axis=1)
    storage_change_rate = storage_out - storage_in

    # Compute the cumulative storage change
    storage_change = np.cumsum(storage_change_rate)

    # Plot cumulative storage change
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(time_data, storage_change, label="CUMULATIVE STORAGE CHANGE", color="green")

    ax.set_title("Cumulative Change in Storage", fontsize=fontsize)
    ax.set_xlabel(time_axis_label, fontsize=fontsize / 1.2)
    ax.set_ylabel("m³", fontsize=fontsize / 1.2)
    ax.legend(fontsize=fontsize / 1.2)
    ax.grid()

    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)

    plt.tight_layout()

    if show:
        plt.show()

    if save:
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        fig.savefig(output_path, dpi=300)
        plt.close(fig)

def compute_time_steps(stress_period_data):
    """
    Return a list of lists with the time-step lengths for each stress period.

    Parameters
    ----------
    stress_period_data : list of tuples
        [(PERLEN, NSTP, TSMULT), ...] for each stress period.

    Returns
    -------
    time_steps : list of list of floats
        time_steps[sp][ts] = length of that time step
    """
    time_steps = []
    for perlen, nstp, tsmult in stress_period_data:
        if perlen == 0:
            # steady-state or zero-length period: produce zero-length steps
            time_steps.append([0.0] * nstp)
            continue

        if abs(tsmult - 1.0) < 1e-12:
            # uniform steps
            dt = float(perlen) / float(nstp)
            time_steps.append([dt] * nstp)
        else:
            # geometric progression: dt_i = dt0 * tsmult**i, sum_i dt_i = perlen
            r = float(tsmult)
            n = int(nstp)
            dt0 = float(perlen) * (r - 1.0) / (r**n - 1.0)
            steps = [dt0 * (r**i) for i in range(n)]
            time_steps.append(steps)

    return time_steps

def timestep_index_from_totim(stress_period_data, totim, tol=1e-9):
    """
    Map a MODFLOW totim (time at the END of a time step) to cumulative time-step index.

    Parameters
    ----------
    stress_period_data : list of (PERLEN, NSTP, TSMULT)
        As used in the TDIS block.
    totim : float
        Cumulative simulation time (MODFLOW totim) — expected to be the time at the end of some step.
    tol : float
        Relative tolerance for matching totim to an end-of-step time (default 1e-9).

    Returns
    -------
    ts_global : int
        Cumulative time-step index (0-based, counts every step including any zero-length steady steps).
    sp_num : int
        Stress-period index (0-based).
    ts_num : int
        Time-step index within the stress period (0-based).

    Raises
    ------
    ValueError
        If totim does not correspond (within tolerance) to an end-of-step time, or totim is outside simulation time.
    """
    time_steps = compute_time_steps(stress_period_data)

    elapsed = 0.0
    ts_global = 0
    eps = max(1e-12, abs(totim) * tol)  # combined small absolute + relative tolerance

    for sp_num, steps in enumerate(time_steps):
        for ts_num, dt in enumerate(steps):
            elapsed += float(dt)
            # exact/near match
            if abs(elapsed - totim) <= eps:
                return ts_global, sp_num, ts_num
            # if elapsed passed totim (and wasn't close), totim falls inside the step => error
            if elapsed > totim + eps:
                raise ValueError(
                    f"totim {totim} falls inside a time step (end-of-step = {elapsed:.12g}). "
                    "MODFLOW totim should be the end-of-step time."
                )
            ts_global += 1

    # Finished loop; totim beyond final elapsed time?
    if abs(elapsed - totim) <= eps:
        # return last step
        # compute last indices
        last_sp = len(time_steps) - 1
        last_ts = len(time_steps[-1]) - 1
        return ts_global - 1, last_sp, last_ts

    raise ValueError(f"totim {totim} is beyond the end of the simulation (final totim = {elapsed}).")

def plot_residual_diffusion(
    gwf,
    start_time: float,
    time: float,
    perioddata,
    nrow: int,
    transient_heads: np.ndarray,
    steady_state_heads: np.ndarray,
    title: str = "Absolute residual diffusion cross section",
    label: str = "Absolute residual diffusion (m)",
    vmin: float = None,
    vmax: float = None,
    save: bool = True,
    output_folder: str = None,
    plot_name : str = "residual_diffusion.png",):

    """
    Plots the absolute residual diffusion between transient and steady-state heads
    for analyzing transient response after a step change in stress.

    Parameters
    ----------
    gwf : flopy GroundwaterFlowModel
        The FloPy groundwater model object.
    start_time : float
        Start time in seconds.
    perioddata : list
        MODFLOW period data.
    nrow : int
        Row index to plot the cross section.
    transient_heads : np.ndarray
        Transient head array.
    steady_state_heads : np.ndarray
        Steady-state head array.
    hobj : flopy.utils.HeadFile
        FloPy headfile object.
    title : str
        Plot title.
    label : str
        Colorbar label.
    vmin : float or None
        Minimum value for colormap.
    vmax : float or None
        Maximum value for colormap.
    save : bool
        Whether to save the figure.
    output_folder : str or None
        Path to save the figure. Required if save=True.
    plot_name : str
        Name of the plot file (e.g., "residual_diffusion.png").
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import flopy

    # Determine start and end steps
    start_step = timestep_index_from_totim(perioddata, start_time)[0]
    step = timestep_index_from_totim(perioddata, time)[0]
    
    # Compute residual
    array = np.abs(transient_heads[step] - steady_state_heads)
    
    # Set up figure
    fig = plt.figure(figsize=(19, 5))
    ax = fig.add_subplot(1, 1, 1)
    mx = flopy.plot.PlotCrossSection(ax=ax, model=gwf, line={"row": nrow})
    
    # Plot array
    pa = mx.plot_array(array, alpha=1, masked_values=[1.0e30], cmap="viridis", vmin=vmin, vmax=vmax)
    
    # Plot grid and colorbar
    mx.plot_grid(color="0.5", alpha=0.2)
    cb = plt.colorbar(pa, ax=ax)
    cb.set_label(label)
    
    # Title
    ax.set_title(title)
    
    # Layout
    plt.tight_layout()
    
    # Save or show
    if save:
        if output_folder is None:
            raise ValueError("output_folder must be provided if save=True")
        os.makedirs(output_folder, exist_ok=True)
        fig.savefig(f"{output_folder}/{plot_name}", dpi=300)
        plt.close(fig)
    else:
        plt.show()

def animate_sto_cb_cross_section(
        gwf,
        cb,                  # CellBudgetFile object
        nrow,                # Row index for cross-section
        cs_output_folder,    # Folder to save individual plots
        gif_output_path,     # Path to save GIF
        boundary_keywords=None,
        show=False, save=True,
        figsize=(19, 6), fontsize=14,
        gif_start=0, gif_step=1, duration=0.5,
        vmin=None, vmax=None):
    """
    Create a cross-section animation of storage change (STO-SS + STO-SY) 
    using modplot6.plot_cross_section_array.

    Args:
        gwf (flopy.mf6.ModflowGwf): Groundwater flow model object.
        cb (flopy.utils.CellBudgetFile): Cell budget file object.
        nrow (int): Row index for cross-section.
        cs_output_folder (str): Directory to save cross-section images.
        gif_output_path (str): Path to save the generated animation GIF.
        boundary_keywords (list, optional): Boundary condition keywords.
        show (bool): Show plots interactively.
        save (bool): Save plots to files.
        figsize (tuple): Figure size.
        fontsize (int/float): Font size for labels.
        gif_start (int): First time step to include.
        gif_step (int): Step between frames.
        duration (float): Frame duration in seconds.
        vmin, vmax (float): Color scale limits for consistent animation.
    """
    import os
    import numpy as np
    import imageio

    os.makedirs(cs_output_folder, exist_ok=True)

    steps = cb.get_kstpkper()
    num_timesteps = len(steps)
    image_paths = []

    for t_index in range(gif_start, num_timesteps, gif_step):
        kstpkper = steps[t_index]

        try:
            sto_ss = cb.get_data(kstpkper=kstpkper, text="STO-SS", full3D=True)[0]
            sto_sy = cb.get_data(kstpkper=kstpkper, text="STO-SY", full3D=True)[0]
            total_sto = -(sto_ss + sto_sy)  # net storage change
        except:
            print(f"Skipping {kstpkper}, storage data not available")
            continue

        output_path = os.path.join(cs_output_folder, f"cross_section_storage_{t_index}.png")

        # Use your prebuilt cross-section plotter
        modplot6.plot_cross_section_array(
            gwf,
            nrow,
            output_path,
            boundary_keywords=boundary_keywords,
            show=show,
            save=save,
            figsize=figsize,
            fontsize=fontsize,
            array=total_sto,
            title=f"Storage change cross-section, step {t_index} ({kstpkper})",
            colorbar=True,
            log=False,
            vmin=vmin,
            vmax=vmax,
        )

        if save:
            image_paths.append(output_path)

        print(f"Saved cross-section plot for step {t_index} ({kstpkper}) at {output_path}")

    # Build GIF
    if save and len(image_paths) > 0:
        with imageio.get_writer(gif_output_path, mode="I", duration=duration) as writer:
            for img in image_paths:
                writer.append_data(imageio.imread(img))

        print(f"Animation saved at {gif_output_path}")

# Experimental: Plotting functions with stabilization analysis

def plot_storage_change_rate_with_stabilization(file_path, 
                                                output_path, 
                                                show=False, 
                                                save=False, 
                                                figsize=(14, 12), 
                                                fontsize=14, 
                                                tstart=0,  # Time after which the stabilization analysis starts
                                                epsilon=None,  # Threshold for stabilization (None means skip)
                                                xlim=None,  # Tuple for x-axis limits
                                                ylim=None, 
                                                time_units=None):  # Tuple for y-axis limits
    """
    Creates a time series plot for change in storage and marks the time step where the curve stabilizes.

    Args:
        file_path (str): Path to the budget CSV file. The file should have a column called 'time' and 
                         columns for storage components.
        output_path (str): Path to save the plot if save is True.
        show (bool): Whether to display the plot. Defaults to False.
        save (bool): Whether to save the plot. Defaults to False.
        figsize (tuple): Size of the figure. Defaults to (14, 12).
        fontsize (int): Font size for plot labels and titles.
        tstart (int or float): The time step after which the stabilization analysis starts.
        epsilon (float or None): The stabilization threshold. If None, no stabilization analysis is done.
        xlim (tuple or None): Limits for x-axis (e.g., (0, 500)). If None, default matplotlib behavior.
        ylim (tuple or None): Limits for y-axis (e.g., (-10, 10)). If None, default matplotlib behavior.
        time_units (str or None): Units for time axis label. If None, defaults to 'days'. If "years", converts days to years.
                                 Assumes model inputs in days by default.
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    import os

    # Load the CSV file
    data = pd.read_csv(file_path)

    # Identify columns for storage components
    columns_storage_in = [col for col in data.columns if "STO" in col and "IN" in col]
    columns_storage_out = [col for col in data.columns if "STO" in col and "OUT" in col]

    # Prepare data
    time_data = data["time"]
    storage_in = data[columns_storage_in].sum(axis=1)
    storage_out = data[columns_storage_out].sum(axis=1)
    storage_change_rate = storage_out - storage_in

    # Initialize stabilization index only if epsilon is set
    stabilization_index = None
    stable_idx = None

    if epsilon is not None:
        stabilization_index = np.zeros_like(storage_change_rate, dtype=float)
        stabilization_index[:tstart + 1] = 100.0  # Arbitrary high value before tstart

        for i in range(tstart + 1, len(storage_change_rate)):
            delta_flux = storage_change_rate[i] * 100 / storage_change_rate[tstart]
            stabilization_index[i] = np.abs(delta_flux)

        stable_indices = np.where(stabilization_index <= epsilon)[0]
        if len(stable_indices) > 0:
            stable_idx = stable_indices[0]

    # Plotting
    fig, ax = plt.subplots(figsize=figsize)
    if time_units == 'years':
        time_data = time_data / 360  # Convert days to years
        time_axis_label = "Time [years]"
    else:
        time_data = time_data  # Keep in days
        time_axis_label = "Time [days]"

    ax.plot(time_data, storage_change_rate, label="STORAGE CHANGE RATE", color="green")

    if stable_idx is not None:
        ax.axvline(time_data.iloc[stable_idx], color="green", linestyle='dotted')
        ax.text(time_data.iloc[stable_idx], storage_change_rate.iloc[stable_idx] + 0.2,
                f'Near equilibrium at t={round(time_data.iloc[stable_idx], 1)} days',
                fontsize=fontsize / 1.2, color="green")

    ax.set_title("Storage change rate", fontsize=fontsize)
    ax.set_xlabel(time_axis_label, fontsize=fontsize / 1.2)
    ax.set_ylabel("m³/day", fontsize=fontsize / 1.2)
    ax.legend(fontsize=fontsize / 1.2)
    ax.grid()

    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)

    plt.tight_layout()

    if show:
        plt.show()

    if save:
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        fig.savefig(output_path, dpi=300)
        plt.close(fig)

def plot_storage_change_with_stabilization(file_path, 
                                           output_path, 
                                           show=False, 
                                           save=False, 
                                           figsize=(14, 12), 
                                           fontsize=14, 
                                           tstart=0,  # Time after which the stabilization analysis starts
                                           epsilon=None,  # Threshold for stabilization or None to skip
                                           xlim=None,
                                           ylim=None):
    """
    Creates a time series plot for change in storage (cumulative) and marks the time step where the curve stabilizes.

    Args:
        file_path (str): Path to the budget CSV file. The file should have a column called 'time' and 
                         columns for storage components.
        output_path (str): Path to save the plot if save is True.
        show (bool): Whether to display the plot. Defaults to False.
        save (bool): Whether to save the plot. Defaults to False.
        figsize (tuple): Size of the figure. Defaults to (14, 12).
        fontsize (int): Font size for plot labels and titles.
        tstart (int or float): The time step after which the stabilization analysis starts.
        epsilon (float or None): Stabilization threshold (e.g., 0.01), or None to skip analysis.
        xlim (tuple or None): x-axis limits (min, max).
        ylim (tuple or None): y-axis limits (min, max).

    Outputs:
        A figure with the cumulative change in storage time series, with a marker for the stabilization point if applicable.
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    import os

    # Load the CSV file
    data = pd.read_csv(file_path)

    # Identify columns for storage components
    columns_storage_in = [col for col in data.columns if "STO" in col and "IN" in col]
    columns_storage_out = [col for col in data.columns if "STO" in col and "OUT" in col]

    # Prepare data for time series
    time_data = data["time"]

    # Compute the storage change rate (STORAGE OUT - STORAGE IN)
    storage_in = data[columns_storage_in].sum(axis=1)
    storage_out = data[columns_storage_out].sum(axis=1)
    storage_change_rate = storage_out - storage_in

    # Compute the cumulative storage change
    storage_change = np.cumsum(storage_change_rate)

    # Optional: stabilization analysis
    stable_idx = None
    if epsilon is not None:
        stabilization_index = np.zeros_like(storage_change, dtype=float)
        stabilization_index[:tstart+1] = 100  # Arbitrary high value before tstart

        for i in range(tstart+1, len(storage_change)):
            delta_flux = storage_change[i] - storage_change[i - 1]
            delta_time = time_data.iloc[i] - time_data.iloc[i - 1]
            stabilization_index[i] = np.abs(delta_flux) * 100 / delta_time

        stable_indices = np.where(stabilization_index <= epsilon)[0]
        if len(stable_indices) > 0:
            stable_idx = stable_indices[0]

    # Plot cumulative storage change
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(time_data, storage_change, label="CUMULATIVE STORAGE CHANGE", color="green")

    if stable_idx is not None:
        ax.axvline(time_data.iloc[stable_idx], color="green", linestyle='dotted')  # Vertical line at stabilization
        ax.text(time_data.iloc[stable_idx], storage_change[stable_idx] + 0.2,
                f'Near equilibrium at t={round(time_data.iloc[stable_idx], 1)} days',
                fontsize=fontsize / 1.2, color="green")

    ax.set_title("Cumulative Change in Storage", fontsize=fontsize)
    ax.set_xlabel("Time [days]", fontsize=fontsize / 1.2)
    ax.set_ylabel("m³", fontsize=fontsize / 1.2)
    ax.legend(fontsize=fontsize / 1.2)
    ax.grid()

    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)

    plt.tight_layout()

    if show:
        plt.show()

    if save:
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        fig.savefig(output_path, dpi=300)
        plt.close(fig)

def fit_time_series(csv_path, columns, tau_guess, tstart=0, output_path=None, show=False, save=False, fig_size=(14, 12)):
    import numpy as np
    import pandas as pd
    from scipy.optimize import curve_fit
    from sklearn.metrics import r2_score
    import matplotlib.pyplot as plt
    import os

    # Load CSV
    df = pd.read_csv(csv_path)

    # Define the model function
    def model(t, h0, h1, tau):
        return h1 - (h1 - h0) * np.exp(-t / tau)

    # Extract time values
    t_data = df['time'].values

    # Filter the data based on tstart
    valid_indices = t_data >= tstart
    t_data = t_data[valid_indices]

    # Store results for each column
    fit_results = {}

    # Iterate over each column in the list
    for col in columns:
        h_data = df[col].values[valid_indices]

        # Dynamic initial guesses
        h0_guess = h_data[0]
        h1_guess = h_data[-1]

        # Perform nonlinear regression
        initial_guesses = [h0_guess, h1_guess, tau_guess]
        popt, _ = curve_fit(model, t_data, h_data, p0=initial_guesses)

        # Extract fitted parameters
        h0_fit, h1_fit, tau_fit = popt

        # Calculate 3*tau and 5*tau for equilibrium points
        t_95_eq = 3 * tau_fit  # Approximate time for 95% equilibrium
        t_99_eq = 5 * tau_fit  # Approximate time for 99% equilibrium

        # Calculate R²
        h_data_fit = model(t_data, *popt)
        r2 = r2_score(h_data, h_data_fit)

        # Create the plot
        plt.figure(figsize=fig_size)
        plt.scatter(t_data, h_data, label=f'Observed Data ({col})', color='blue', s=10)
        plt.plot(t_data, h_data_fit, label=f'Fitted Curve (R² = {r2:.3f})', color='red', linewidth=2)
        plt.axvline(x=t_95_eq, color='green', linestyle='--', label=f'95% Equilibrium (3τ = {t_95_eq:.2f} days)')
        plt.axvline(x=t_99_eq, color='purple', linestyle='--', label=f'99% Equilibrium (5τ = {t_99_eq:.2f} days)')
        plt.xlabel('Time')
        plt.ylabel(col)
        plt.title(f'Nonlinear Fit for {col}\nh_0={h0_fit:.2f}, h_1={h1_fit:.2f}, τ={tau_fit:.2f}')
        plt.legend()

        # Save the plot if required
        if save:
            if output_path is None:
                raise ValueError("output_path must be specified if save=True")
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            plot_file = os.path.join(output_path, f'{col}_fit_plot.png')
            plt.savefig(plot_file, bbox_inches='tight', dpi=300)
            print(f"Plot saved for {col} at: {plot_file}")

        # Show the plot if required
        if show:
            plt.show()
        else:
            plt.close()

        # Print the parameters for reference
        print(f"{col} - Fitted parameters:\n h0 = {h0_fit:.2f}, h1 = {h1_fit:.2f}, τ = {tau_fit:.2f}, R² = {r2:.3f}\n")

        # Store the fitted parameters for this column
        fit_results[col] = {'h0': h0_fit, 'h1': h1_fit, 'tau': tau_fit, 'r2': r2}

    # Return the results for all columns
    return fit_results
          
def fit_time_series2(csv_path, columns, tau_guess, tstart=0, output_path=None, show=False, save=False, fig_size=(14, 12)):
    import numpy as np
    import pandas as pd
    from scipy.optimize import curve_fit
    from sklearn.metrics import r2_score
    import matplotlib.pyplot as plt
    import os

    # Load CSV
    df = pd.read_csv(csv_path)

    # Define the model function, only fitting tau
    def model(t, tau, h0, h1):
        return h1 - (h1 - h0) * np.exp(-t / tau)

    # Extract time values
    t_data = df['time'].values

    # Filter the data based on tstart
    valid_indices = t_data >= tstart
    t_data = t_data[valid_indices]

    # Store results for each column
    tau_results = {}

    # Iterate over each column in the list
    for col in columns:
        h_data = df[col].values[valid_indices]

        # Extract h1 as the last element of the time series and h0 as the value at tstart+1
        h1 = h_data[-1]
        h0 = h_data[np.argmax(t_data >= tstart)]  # Get the value at tstart + 1

        # Perform nonlinear regression to fit only tau
        initial_guesses = [tau_guess]
        popt, _ = curve_fit(lambda t, tau: model(t, tau, h0, h1), t_data, h_data, p0=initial_guesses)

        # Extract fitted tau
        tau_fit = popt[0]

        # Calculate 3*tau and 5*tau for equilibrium points
        t_95_eq = 3 * tau_fit  # Approximate time for 95% equilibrium
        t_99_eq = 5 * tau_fit  # Approximate time for 99% equilibrium

        # Calculate R²
        h_data_fit = model(t_data, tau_fit, h0, h1)
        r2 = r2_score(h_data, h_data_fit)

        # Create the plot
        plt.figure(figsize=fig_size)
        plt.scatter(t_data, h_data, label=f'Observed Data ({col})', color='blue', s=10)
        plt.plot(t_data, h_data_fit, label=f'Fitted Curve (R² = {r2:.3f})', color='red', linewidth=2)
        plt.axvline(x=t_95_eq, color='green', linestyle='--', label=f'95% Equilibrium (3τ = {t_95_eq:.2f} days)')
        plt.axvline(x=t_99_eq, color='purple', linestyle='--', label=f'99% Equilibrium (5τ = {t_99_eq:.2f} days)')
        plt.xlabel('Time')
        plt.ylabel(col)
        plt.title(f'Nonlinear Fit for {col}\nh_0={h0:.2f}, h_1={h1:.2f}, τ={tau_fit:.2f}')
        plt.legend()

        # Save the plot if required
        if save:
            if output_path is None:
                raise ValueError("output_path must be specified if save=True")
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            plot_file = os.path.join(output_path, f'{col}_fit_plot.png')
            plt.savefig(plot_file, bbox_inches='tight', dpi=300)
            print(f"Plot saved for {col} at: {plot_file}")

        # Show the plot if required
        if show:
            plt.show()
        else:
            plt.close()

        # Print the parameters for reference
        print(f"{col} - Fitted parameters:\n h0 = {h0:.2f}, h1 = {h1:.2f}, τ = {tau_fit:.2f}, R² = {r2:.3f}\n")

        # Store the fitted tau value for this column
        tau_results[col] = tau_fit

    # Return the results for all columns
    return tau_results       

def plot_net_flow_time_series_with_equilibrium_markers(
        file_path,
        output_path,
        show=False,
        save=False,
        figsize=(14, 12),
        fontsize=16,
        tstart=0,  # Time after which the stabilization analysis starts
        equilibrium_percentage=99,  # Percentage for equilibrium threshold
        tau=1,  # Time interval for delta computation
        boundary_keywords=None):  # Filter columns by keywords
    """
    Creates time series plots for the difference between inflow and outflow components,
    and marks the time step where the stabilization criteria is met.

    Args:
        file_path (str): Path to the budget CSV file. The file should have a column called 'time' and 
                         columns ending in _IN, _OUT.
        output_path (str): Path to save the plot if save is True.
        show (bool): Whether to display the plot. Defaults to False.
        save (bool): Whether to save the plot. Defaults to False.
        figsize (tuple): Size of the figure. Defaults to (14, 12).
        fontsize (int): Font size for plot labels and titles.
        tstart (int or float): The time step index after which the stabilization analysis starts.
        equilibrium_percentage (float): Equilibrium percentage threshold (e.g., 99%).
        tau (int or float): Time interval for computing delta stabilization criteria.
        boundary_keywords (list of str or None): List of keywords to filter columns to be analyzed. If None, the analysis is performed on all curves.
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    import os

    # Load the CSV file
    data = pd.read_csv(file_path)

    # Identify matching inflow and outflow columns
    columns_in = [col for col in data.columns if col.endswith("_IN") and "STO" not in col and col != "TOTAL_IN"]
    columns_out = [col.replace("_IN", "_OUT") for col in columns_in if col.replace("_IN", "_OUT") in data.columns]

    # Prepare time data
    time_data = data["time"]

    # Create a figure
    fig, ax1 = plt.subplots(figsize=figsize)

    # Plot net flux for each component
    legend_labels = []
    lines = []  # To store the line objects for later use in vertical line and annotation
    for col_in, col_out in zip(columns_in, columns_out):
        net_flux = data[col_in] - data[col_out]
        label = simplify_name(col_in)
        line, = ax1.plot(time_data, net_flux, label=label)
        legend_labels.append(label)
        lines.append(line)  # Store the line object for later use

    # Perform stabilization analysis for components matching the boundary_keywords
    if boundary_keywords is None:
        # If boundary_keywords is None, analyze all components
        components_to_analyze = zip(columns_in, columns_out, lines)
    else:
        # If boundary_keywords is provided, only analyze those components that match the keywords
        components_to_analyze = [(col_in, col_out, line) for col_in, col_out, line in zip(columns_in, columns_out, lines)
                                 if any(keyword in col_in for keyword in boundary_keywords)]

    for col_in, col_out, line in components_to_analyze:
        net_flux = data[col_in] - data[col_out]

        # Compute stabilization index
        stabilization_index = np.zeros_like(net_flux, dtype=float)
        
        # Set initial values for stabilization analysis
        h0 = net_flux.iloc[tstart + 1]  # Initial head value at tstart + 1
        h1 = net_flux.iloc[-1]  # Final head value
        #delta = (100 - equilibrium_percentage) * abs((h1 - h0)) / tau
        delta = h0 + (equilibrium_percentage*(h1 - h0)/100)

        # Before and at tstart, set stabilization index to 100 (no effect on analysis)
        #stabilization_index[:tstart + 1] = 100  # Or any standard value you want
        
        # Compute slope percent for tstart and after
        for i in range(0, len(net_flux)):
            #delta_flux = net_flux[i] - net_flux[i - 1]
            #delta_time = time_data.iloc[i] - time_data.iloc[i - 1]
            #stabilization_index[i] = np.abs(delta_flux) * 100 / delta_time
            stabilization_index[i] = net_flux[i]

        # Find the first time step where the stabilization index is less than delta
        if h1 <= h0 :
            stable_idx = np.where(stabilization_index <= delta)[0]
        else:
            stable_idx = np.where(stabilization_index >= delta)[0]

        if len(stable_idx) > 0:
            stable_idx = stable_idx[0]
            ax1.axvline(time_data.iloc[stable_idx], color=line.get_color(), linestyle='dotted')  # Vertical line in the same color as the curve
            # Annotate the plot with the same color as the curve, rounding to one decimal place
            ax1.text(time_data.iloc[stable_idx], net_flux.iloc[stable_idx] + 0.2,
                     f'Stable at t={round(time_data.iloc[stable_idx], 1)} days',
                     fontsize=fontsize / 1.2, color=line.get_color())

    # Sort legend labels alphabetically
    handles, labels = ax1.get_legend_handles_labels()
    sorted_handles_labels = sorted(zip(handles, labels), key=lambda x: x[1])
    handles, labels = zip(*sorted_handles_labels)
    ax1.legend(handles, labels, fontsize=fontsize / 1.2)

    # Horizontal line at zero for the X-axis
    ax1.axhline(0, color='black', linewidth=0.8, linestyle='--')

    ax1.set_title("Net Flow (Inflow - Outflow) Components", fontsize=fontsize)
    ax1.set_xlabel("Time [days]", fontsize=fontsize / 1.2)
    ax1.set_ylabel("Net Flow [m³/day]", fontsize=fontsize / 1.2)
    ax1.grid()

    # Adjust layout and show plot
    plt.tight_layout()

    if show:
        plt.show()

    # Save plot
    if save:
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        fig.savefig(output_path, dpi=300)
        plt.close(fig)

def plot_head_time_series_with_equilibrium_markers(
        head_file_path, 
        gwf, 
        output_path, 
        show=False, 
        save=False,
        figsize=(14, 12), 
        fontsize=14, 
        tstart=0,  # Time after which the stabilization analysis starts
        equilibrium_percentage=99,  # Percentage for equilibrium threshold
        tau=1):  # Time interval for delta computation
    """
    Plots the head time series from MODFLOW output, performs stabilization analysis based on delta criteria,
    and generates the plot.
    
    Args:
        head_file_path (str): Path to the head observation CSV file.
        gwf (flopy.modflow.ModflowGwf): Flopy groundwater flow model object.
        tstart (int or float): The time step after which the stabilization analysis starts.
        equilibrium_percentage (float): Equilibrium percentage threshold (e.g., 99%).
        tau (int or float): Time interval for computing delta stabilization criteria.
    
    Outputs:
        A plot showing the head values over time with stabilization markers.
    """
    # Retrieve head observation data using Flopy
    csv = gwf.head_obs.output.obs(f=head_file_path).get_data()
    
    fig = plt.figure(figsize=figsize)

    # Plot head values over time and perform stabilization analysis
    for name in csv.dtype.names[1:]:  # Skip the first column (totim) as it's time
        head_values = csv[name]
        
        # Plot the head time series
        plt.plot(csv["totim"], head_values, label=name)
        
        # Compute stabilization index
        stabilization_index = np.zeros_like(head_values, dtype=float)
        
        # Set initial values for stabilization analysis
        h0 = head_values[tstart + 1]  # Initial head value at tstart + 1
        h1 = head_values[-1]  # Final head value
        #delta = (100 - equilibrium_percentage) * abs((h1 - h0)) / tau
        delta = h0 + (equilibrium_percentage*(h1 - h0)/100)

        # Before and at tstart, set stabilization index to 100 (no effect on analysis)
        #stabilization_index[:tstart + 1] = 100  # Or any standard value you want
        
        # Compute slope percent for tstart and after
        for i in range(0, len(head_values)):
            #delta_head = head_values[i] - head_values[i - 1]
            #delta_time = csv["totim"][i] - csv["totim"][i - 1]
            #stabilization_index[i] = np.abs(delta_head) * 100 / delta_time
            stabilization_index[i] = head_values[i]

        # Find the first time step where the stabilization index is less than delta
        if h1 <= h0 :
            stable_idx = np.where(stabilization_index <= delta)[0]
        else:
            stable_idx = np.where(stabilization_index >= delta)[0]

        if len(stable_idx) > 0:
            stable_idx = stable_idx[0]
            plt.axvline(csv["totim"][stable_idx], color=plt.gca().lines[-1].get_color(), linestyle='dotted')  # Vertical line in the same color as the curve
            # Annotate the plot with the same color as the curve, rounding to one decimal place
            plt.text(csv["totim"][stable_idx], head_values[stable_idx] + 0.2,
                     f'Stable at t={round(csv["totim"][stable_idx], 1)} days',
                     fontsize=fontsize / 1.2, color=plt.gca().lines[-1].get_color())

    plt.xlabel('Time [days]', fontsize=fontsize / 1.2)
    plt.ylabel('Head [m]', fontsize=fontsize / 1.2)
    plt.title('HEAD TIME SERIES', fontsize=fontsize)
    plt.legend(fontsize=fontsize / 1.2)
    plt.grid(True)

    # Adjust layout and show plot
    plt.tight_layout(rect=[0, 0, 1, 0.96])  

    if show:
        plt.tight_layout()
        plt.show()

    # Save plot
    if save:
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        fig.savefig(output_path, dpi=300)
        plt.close(fig)

