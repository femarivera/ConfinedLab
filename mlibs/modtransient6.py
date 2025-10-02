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
import sys
import imageio
import flopy

from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

# ===== Global style settings =====
plt.rcParams['font.family'] = 'Calibri'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['figure.dpi'] = 300 

# Import local modules
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

    # Storage
    columns_storage_in = [col for col in data.columns if "STO" in col and "IN" in col]
    columns_storage_out = [col for col in data.columns if "STO" in col and "OUT" in col]

    # Storage componenets
    columns_storage_ss_in = [col for col in data.columns if "STO" in col and "IN" in col and "SS" in col]
    columns_storage_ss_out = [col for col in data.columns if "STO" in col and "OUT" in col and "SS" in col]
    columns_storage_sy_in = [col for col in data.columns if "STO" in col and "IN" in col and "SY" in col]
    columns_storage_sy_out = [col for col in data.columns if "STO" in col and "OUT" in col and "SY" in col]

    # Compute components
    induced_recharge = data[columns_in].sum(axis=1) - reference_inflow
    discharge = data[columns_out].sum(axis=1)
    captured_discharge = reference_outflow - discharge
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

    # Compute storage change rates per drainance, compressibility, and total
    sto_ss = data[columns_storage_ss_out].sum(axis=1) - data[columns_storage_ss_in].sum(axis=1)
    sto_sy = data[columns_storage_sy_out].sum(axis=1) - data[columns_storage_sy_in].sum(axis=1)
    sto_total = sto_ss + sto_sy

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
    data["STO-SS"] = sto_ss
    data["STO-SY"] = sto_sy
    data["STO-TOTAL"] = sto_total

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
    storage_out_columns = [col for col in df.columns if "STO" in col and "OUT" in col]
    storage_in_columns = [col for col in df.columns if "STO" in col and "IN" in col]
    pumped_columns = [col for col in df.columns if "WEL" in col and "OUT" in col]

    # Storage componenets
    columns_storage_ss_in = [col for col in df.columns if "STO" in col and "IN" in col and "SS" in col]
    columns_storage_ss_out = [col for col in df.columns if "STO" in col and "OUT" in col and "SS" in col]
    columns_storage_sy_in = [col for col in df.columns if "STO" in col and "IN" in col and "SY" in col]
    columns_storage_sy_out = [col for col in df.columns if "STO" in col and "OUT" in col and "SY" in col]

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

    # Compute storage change rates per drainance, compressibility, and total
    sto_ss = df[columns_storage_ss_out].sum(axis=1) - df[columns_storage_ss_in].sum(axis=1)
    sto_sy = df[columns_storage_sy_out].sum(axis=1) - df[columns_storage_sy_in].sum(axis=1)
    sto_total = sto_ss + sto_sy

    # Add computed components and percentages to the DataFrame
    df["Induced_Recharge"] = induced_recharge
    df["Captured_Discharge"] = captured_discharge
    df["From_Storage"] = from_storage
    df["Capture"] = capture
    df["Induced_Recharge_Pct"] = induced_recharge_pct
    df["Captured_Discharge_Pct"] = captured_discharge_pct
    df["From_Storage_Pct"] = from_storage_pct
    df["Capture_Pct"] = capture_pct
    df["STO-SS"] = sto_ss
    df["STO-SY"] = sto_sy
    df["STO-TOTAL"] = sto_total

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
                     csv_output_dir,
                     fig_output_dir, 
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
        csv_output_dir (str): Directory to save vertical leakage data.
        fig_output_dir (str): Directory to save the figures if save is True.
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
            image_path = os.path.join(fig_output_dir, f'Zone {zone} budget.png')
            fig.savefig(image_path, dpi=300)
            plt.close(fig) 

    # Plot the difference "FROM ZONE x - TO ZONE x" for all other zones
    # Prepare an empty dict to collect vertical leakage data
    vertical_leakage_dict = {}

    # Use unique timesteps
    timesteps = sorted(df['totim'].unique())
    vertical_leakage_dict['totim'] = timesteps

    # Compute FROM-TO difference for each zone
    for zone in zones:
        other_zones = [f"ZONE {int(z)}" for z in zones if z != zone]
        from_columns = [f"FROM {oz}" for oz in other_zones]
        to_columns = [f"TO {oz}" for oz in other_zones]
        
        # Sum over all rows for this zone at each timestep
        differences = []
        for t in timesteps:
            mask = (df['zone'] == zone) & (df['totim'] == t)
            diff = df.loc[mask, from_columns].sum(axis=1).values - df.loc[mask, to_columns].sum(axis=1).values
            # If empty (no row for this timestep), set to 0
            differences.append(diff[0] if len(diff) > 0 else 0)
        
        vertical_leakage_dict[f'ZONE_{zone}'] = differences

    # Create DataFrame
    vertical_leakage_df = pd.DataFrame(vertical_leakage_dict)

    # Save to CSV
    csv_path = os.path.join(csv_output_dir, "vertical_leakage.csv")
    vertical_leakage_df.to_csv(csv_path, index=False)
    print(f"Vertical leakage data saved to {csv_path}")

    # Optional: Plotting
    fig2 = plt.figure(figsize=figsize)
    for zone in zones:
        description = zone_descriptions.get(zone, f"ZONE {zone}")
        plt.plot(vertical_leakage_df['totim'], vertical_leakage_df[f'ZONE_{zone}'], label=f'ZONE {zone} - {description}')

    plt.title('WATER TRANSFERS VIA VERTICAL LEAKAGE', fontsize=fontsize)
    plt.xlabel('Time [days]', fontsize=fontsize)
    plt.ylabel('Flow Balance (Inflows - Outflows) [m³/day]', fontsize=fontsize)
    plt.legend(fontsize=fontsize/1.2)
    plt.grid()
    plt.tight_layout()
    if show:
        plt.show()
    if save:
        image_path = os.path.join(fig_output_dir, "zonebudget_summary_t.png")
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
                # Compare distance to current elapsed and to previous elapsed
                prev_elapsed = elapsed - float(dt)
                dist_prev = abs(totim - prev_elapsed)
                dist_curr = abs(totim - elapsed)

                if dist_curr < dist_prev:
                    return ts_global, sp_num, ts_num
                else:
                    return ts_global - 1, sp_num, ts_num - 1
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

def tr_storage_change_rate(zonebudfile, csv_output_folder, fig_output_folder,
                                   show=False, save_csv=True, save_fig=True,
                                   figsize=(14, 8), fontsize=14,
                                   xlim=None, ylim=None, threshold=None, start_time=0.0):
    """
    Plot the total (summed across all zones) storage change rate 
    from a zone budget file.

    Parameters
    ----------
    zonebudfile : str
        Path to the zone budget CSV file. Must contain columns: 'zone', 'totim', 'STO-TOTAL'.
    csv_output_folder : str
        Folder to save processed CSV file.
    fig_output_folder : str
        Folder to save figure.
    show : bool, default=False
        If True, display the plot.
    save_csv : bool, default=True
        If True, save the processed CSV file.
    save_fig : bool, default=True
        If True, save the plot figure.
    figsize : tuple, default=(14, 8)
        Figure size for the plot.
    fontsize : int, default=14
        Font size for labels and legend.
    xlim : tuple or None, default=None
        Limits for x-axis (years).
    ylim : tuple or None, default=None
        Limits for y-axis.
    threshold : float or None, default=None
        If provided, marks the first time abs(total) < threshold (after start_time).
    start_time : float, default=0.0
        Starting time (in years). Plot begins here, with x-axis reset so this = 0.
    """

    # Read file
    df = pd.read_csv(zonebudfile)

    # Check required columns
    required_cols = {'zone', 'totim', 'STO-TOTAL'}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Input file must contain columns: {required_cols}")

    # Convert time to years
    df['time_years'] = df['totim'] / 360.0

    # Subset from start_time onward
    df = df[df['time_years'] >= start_time].copy()

    # Shift so that start_time becomes 0
    df['time_since_start'] = df['time_years'] - start_time

    # Aggregate by time (sum across all zones)
    df_total = df.groupby('time_since_start')['STO-TOTAL'].sum()

    # Plot
    plt.figure(figsize=figsize)
    ax = plt.gca()

    line, = ax.plot(df_total.index, df_total.values, color="blue", label=None)

    legend_label = "Total"

    if threshold is not None:
        crossing = df_total[df_total.abs() < threshold]
        if not crossing.empty:
            t_cross = crossing.index[0]
            v_cross = crossing.iloc[0]

            # Marker + vertical line
            ax.plot(t_cross, v_cross, 'o', color=line.get_color(), markersize=8, label=None)
            ax.axvline(t_cross, linestyle="--", color=line.get_color(), alpha=0.6, label=None)

            legend_label = f"Total, tr = {t_cross:.0f} years"
        else:
            legend_label = "Total, tr = none"

    ax.set_xlabel("Time since step change (years)", fontsize=fontsize)
    ax.set_ylabel("Storage Change Rate", fontsize=fontsize)
    ax.set_title("Storage Change Rate full system", fontsize=fontsize+2)
    ax.grid(True)

    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    ax.legend([line], [legend_label], fontsize=fontsize-2)

    # Save outputs
    if save_csv:
        os.makedirs(csv_output_folder, exist_ok=True)
        csv_path = os.path.join(csv_output_folder, "total_storage_change_rate.csv")
        df_total.to_csv(csv_path, header=["STO-TOTAL"])
    
    if save_fig:
        os.makedirs(fig_output_folder, exist_ok=True)
        fig_path = os.path.join(fig_output_folder, "total_storage_change_rate.png")
        plt.savefig(fig_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()

def tr_storage_change_rate_zones(zonebudfile, csv_output_folder, fig_output_folder, 
                                   show=False, save_csv=True, save_fig=True, 
                                   figsize=(14, 12), fontsize=14,
                                   xlim=None, ylim=None, threshold=None, start_time=0.0):
    """
    Plot storage change rate for each zone from a zone budget file.

    Parameters
    ----------
    zonebudfile : str
        Path to the zone budget CSV file. Must contain columns: 'zone', 'totim', 'STO-TOTAL'.
    csv_output_folder : str
        Folder to save processed CSV file.
    fig_output_folder : str
        Folder to save figure.
    show : bool, default=False
        If True, display the plot.
    save_csv : bool, default=True
        If True, save the processed CSV file.
    save_fig : bool, default=True
        If True, save the plot figure.
    figsize : tuple, default=(14, 12)
        Figure size for the plot.
    fontsize : int, default=14
        Font size for labels and legend.
    xlim : tuple or None, default=None
        Limits for x-axis (years).
    ylim : tuple or None, default=None
        Limits for y-axis.
    threshold : float or None, default=None
        If provided, marks the first time abs(STO-TOTAL) < threshold (after start_time).
        Adds a point + vertical dashed line, and annotates legend as tr=xxx years.
    start_time : float, default=0.0
        Starting time (in years). Plot begins here, with x-axis reset so this = 0.
    """
    
    # Read file
    df = pd.read_csv(zonebudfile)
    
    # Check required columns
    required_cols = {'zone', 'totim', 'STO-TOTAL'}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Input file must contain columns: {required_cols}")
    
    # Convert time to years
    df['time_years'] = df['totim'] / 360.0
    
    # Subset from start_time onward
    df = df[df['time_years'] >= start_time].copy()
    
    # Shift so that start_time becomes 0
    df['time_since_start'] = df['time_years'] - start_time
    
    # Pivot data: rows = shifted time, columns = zone, values = STO-TOTAL
    df_pivot = df.pivot_table(index='time_since_start', columns='zone', values='STO-TOTAL')
    
    # Plot
    plt.figure(figsize=figsize)
    ax = plt.gca()
    
    legend_labels = []
    legend_handles = []
    
    for zone in df_pivot.columns:
        y = df_pivot[zone].dropna()
        x = df_pivot.index[:len(y)]
        
        # Plot main line
        line, = ax.plot(x, y, label=None)
        
        tr_label = f"Zone {zone}"
        
        if threshold is not None:
            crossing = y[y.abs() < threshold]
            if not crossing.empty:
                t_cross = crossing.index[0]  # already relative to start_time
                v_cross = crossing.iloc[0]
                
                # Add marker + vertical line
                ax.plot(t_cross, v_cross, 'o', color=line.get_color(), markersize=8, label=None)
                ax.axvline(t_cross, linestyle="--", color=line.get_color(), alpha=0.6, label=None)
                
                tr_label = f"Zone {zone}, tr = {t_cross:.0f} years"
            else:
                tr_label = f"Zone {zone}, tr = none"
        
        legend_handles.append(line)
        legend_labels.append(tr_label)
    
    ax.set_xlabel("Time since step change (years)", fontsize=fontsize)
    ax.set_ylabel("Storage Change Rate", fontsize=fontsize)
    ax.set_title("Storage Change Rate per Zone", fontsize=fontsize+2)
    ax.grid(True)
    
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    
    ax.legend(legend_handles, legend_labels, fontsize=fontsize-2)
    
    # Save outputs
    if save_csv:
        os.makedirs(csv_output_folder, exist_ok=True)
        csv_path = os.path.join(csv_output_folder, "storage_change_rate_per_zone.csv")
        df_pivot.to_csv(csv_path)
    
    if save_fig:
        os.makedirs(fig_output_folder, exist_ok=True)
        fig_path = os.path.join(fig_output_folder, "storage_change_rate_per_zone.png")
        plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    
    if show:
        plt.show()
    else:
        plt.close()

def absolute_head_diffusion_zones(transient_heads, steady_state_heads, times, zone_array,
                             start_step=0, threshold_absolute=0.01, stability_threshold=0.01,
                             array_output_folder=".", fig_output_folder=".",
                             save_fig=True, show_fig=False,
                             zone_descriptions=None,
                             bounds="95p"):
    """
    Plots per-zone head differences (transient - steady state) with mean, median, and either
    95% interval or mean ± std as shaded bounds. Annotates the time when mean drops below threshold_value.

    Parameters:
    -----------
    transient_heads : np.ndarray
        4D array (n_time, n_layer, n_row, n_col)
    steady_state_heads : np.ndarray
        3D array (n_layer, n_row, n_col)
    times : np.ndarray or list
        Simulation times corresponding to transient_heads
    zone_array : np.ndarray
        3D array identifying zones (n_layer, n_row, n_col)
    start_step : int
        Time step index to start analysis (default=0)
    threshold_absolute : float
        Absolute value at which the mean is considered "relaxed" (default=0.01)
    stability_threshold : float
        Threshold to exclude cells nearly steady at start_step (default=0.01)
    array_output_folder : str
        Folder path to save diff_array.npy
    fig_output_folder : str
        Folder path to save figure
    save_fig : bool
        Whether to save the figure
    show_fig : bool
        Whether to display the figure
    zone_descriptions : dict, optional
        Dictionary mapping zone numbers to descriptive names
    bounds : str, default "95p"
        Method for shaded bounds: "95p" = 2.5th–97.5th percentiles,
        "stdev" = mean ± standard deviation
    """

    # Ensure output folders exist
    os.makedirs(array_output_folder, exist_ok=True)
    os.makedirs(fig_output_folder, exist_ok=True)

    # Compute differences
    diff_array = np.abs(transient_heads - steady_state_heads)  # (ntsp, nlay, nrow, ncol)

    # Unique zones (exclude background if needed)
    zones = np.unique(zone_array)
    zones = zones[zones > 0]

    # X-axis: time since start_step
    time_since_start = times[start_step:] - times[start_step]  # zero at start_step
    time_in_years = time_since_start / 360  # convert to years

    # Prepare subplots
    n_zones = len(zones)
    fig, axes = plt.subplots(n_zones, 1, figsize=(14, 4 * n_zones), sharex=True)
    if n_zones == 1:
        axes = [axes]

    for ax, zone in zip(axes, zones):
        # Mask: zone selection
        zone_mask = (zone_array == zone)

        # Exclude cells nearly steady at start_step
        exclude_mask = np.abs(diff_array[start_step]) < stability_threshold
        combined_mask = np.logical_or(~zone_mask, exclude_mask)

        # Apply mask
        diff_zone = np.where(combined_mask, np.nan, diff_array)

        # Flatten spatial dimensions
        diff_zone_flat = diff_zone.reshape(diff_zone.shape[0], -1)

        # Select only times after start_step
        selected_diff_zone = diff_zone_flat[start_step:]

        # Compute stats per timestep
        means = np.array([np.nanmean(selected_diff_zone[t]) for t in range(selected_diff_zone.shape[0])])
        #medians = [np.nanmedian(selected_diff_zone[t]) for t in range(selected_diff_zone.shape[0])]

        if bounds == "95p":
            lower = [np.nanpercentile(selected_diff_zone[t], 2.5) for t in range(selected_diff_zone.shape[0])]
            upper = [np.nanpercentile(selected_diff_zone[t], 97.5) for t in range(selected_diff_zone.shape[0])]
        elif bounds == "stdev":
            lower = [max(means[t] - np.nanstd(selected_diff_zone[t]), 0) for t in range(selected_diff_zone.shape[0])]
            upper = [means[t] + np.nanstd(selected_diff_zone[t]) for t in range(selected_diff_zone.shape[0])]
        else:
            raise ValueError("bounds must be '95p' or 'stdev'")

        # Plot lines
        #ax.plot(time_in_years, medians, color="darkblue", linestyle="-", label="Median", linewidth=1.5)
        ax.plot(time_in_years, means, color="blue", linestyle="--", label="Mean", linewidth=1.5)
        ax.fill_between(time_in_years, lower, upper, color="lightblue", alpha=0.3,
                        label="95% interval" if bounds=="95p" else "Standard Deviation")
        # Add borders
        ax.plot(time_in_years, lower, color="black", linestyle="--", linewidth=1, alpha=0.7)
        ax.plot(time_in_years, upper, color="black", linestyle="--", linewidth=1, alpha=0.7)

        # Annotate first time mean < threshold_absolute
        below_threshold_idx = np.where(means < threshold_absolute)[0]
        if below_threshold_idx.size > 0:
            idx = below_threshold_idx[0]
            t_cross = time_in_years[idx]
            mean_value = means[idx]
            ax.scatter(t_cross, mean_value, color="green", s=50, zorder=5)
            ax.axvline(t_cross, color="green", linestyle=":", linewidth=1.5)
            ax.text(t_cross + 500, ax.get_ylim()[1]*0.9, f"tr={int(round(t_cross))} yr",
                    color="green", rotation=0, va='top', fontweight='bold')

        # Labels
        ax.set_ylabel("Head difference: transient - steady state (m)")
        if zone_descriptions and zone in zone_descriptions:
            ax.set_title(f"Zone {zone}: {zone_descriptions[zone]}")
        else:
            ax.set_title(f"Zone {zone}")

    axes[-1].set_xlabel("Time since step change (years)")
    ax.set_ylim(bottom=0)
    axes[0].legend()
    plt.tight_layout()

    # Save diff_array
    np.save(os.path.join(array_output_folder, "diff_array_absolute.npy"), diff_array)
    
    # Save figure
    if save_fig:
        fig_path = os.path.join(fig_output_folder, "diff_absolute_zones.png")
        plt.savefig(fig_path, dpi=300)

    if show_fig:
        plt.show()
    else:
        plt.close(fig)

    return diff_array

def absolute_head_diffusion(transient_heads, steady_state_heads, times,
                            start_step=0, threshold_absolute=0.01, stability_threshold=0.01,
                            fig_output_folder=".", save_fig=True, show_fig=False,
                            bounds="95p"):
    """
    Plots overall head differences (transient - steady state) with mean, and either
    95% interval or mean ± std as shaded bounds. NaN values are ignored in stats
    and plotting. Can save and/or display the figure.

    Parameters:
    -----------
    transient_heads : np.ndarray
        4D array (n_time, n_layer, n_row, n_col)
    steady_state_heads : np.ndarray
        3D array (n_layer, n_row, n_col)
    times : np.ndarray or list
        Simulation times corresponding to transient_heads
    start_step : int
        Time step index to start analysis (default=0)
    threshold_absolute : float
        Value below which cells are considered already steady state (default=0.01)
    stability_threshold : float
        Threshold to exclude cells nearly steady at start_step (default=0.01)
    fig_output_folder : str
        Folder path to save figure
    save_fig : bool
        Whether to save the figure
    show_fig : bool
        Whether to display the figure
    bounds : str, default "95p"
        Method for shaded bounds: "95p" = 2.5th–97.5th percentiles,
        "stdev" = mean ± standard deviation
    """

    # Ensure output folder exists
    os.makedirs(fig_output_folder, exist_ok=True)

    # Compute differences
    diff_array = np.abs(transient_heads - steady_state_heads)  # (ntsp, nlay, nrow, ncol)

    # Mask: exclude cells nearly steady at start_step
    exclude_mask = np.abs(diff_array[start_step]) < stability_threshold
    diff_array[:, exclude_mask] = np.nan  # apply mask across all timesteps

    # Flatten spatial dimensions
    diff_flat = diff_array.reshape(diff_array.shape[0], -1)
    selected_diff = diff_flat[start_step:]

    # X-axis: time since start_step
    time_since_start = times[start_step:] - times[start_step]
    time_in_years = time_since_start / 360  # convert to years

    # Compute statistics ignoring NaNs
    means = np.nanmean(selected_diff, axis=1)
    #medians = np.nanmedian(selected_diff, axis=1)

    if bounds == "95p":
        lower = np.nanpercentile(selected_diff, 2.5, axis=1)
        upper = np.nanpercentile(selected_diff, 97.5, axis=1)
    elif bounds == "stdev":
        std = np.nanstd(selected_diff, axis=1)
        lower = np.maximum(means - std, 0)
        upper = means + std
    else:
        raise ValueError("bounds must be '95p' or 'stdev'")

    # Plot
    fig, ax = plt.subplots(figsize=(12, 5))
    #.plot(time_in_years, medians, color="darkblue", linestyle="-", label="Median", linewidth=1.5)
    ax.plot(time_in_years, means, color="blue", linestyle="--", label="Mean", linewidth=1.5)
    
    # Use np.where to replace NaNs with nan-safe arrays for plotting
    ax.fill_between(time_in_years, np.where(np.isnan(lower), np.nan, lower),
                    np.where(np.isnan(upper), np.nan, upper),
                    color="lightblue", alpha=0.3,
                    label="95% interval" if bounds=="95p" else "Standard Deviation")
    ax.plot(time_in_years, lower, color="black", linestyle="--", linewidth=1, alpha=0.7)
    ax.plot(time_in_years, upper, color="black", linestyle="--", linewidth=1, alpha=0.7)

    # Annotate first time mean < threshold_absolute
    below_threshold_idx = np.where(means < threshold_absolute)[0]
    if below_threshold_idx.size > 0:
        idx = below_threshold_idx[0]
        t_cross = time_in_years[idx]
        mean_value = means[idx]
        ax.scatter(t_cross, mean_value, color="green", s=50, zorder=5)
        ax.axvline(t_cross, color="green", linestyle=":", linewidth=1.5)
        ax.text(t_cross + 500, ax.get_ylim()[1]*0.9, f"tr={int(round(t_cross))} yr",
                color="green", rotation=0, va='top', fontweight='bold')

    # Labels
    ax.set_xlabel("Time since step change (years)")
    ax.set_ylabel("Head difference (m)")
    ax.set_title("Absolute head difference: transient - steady state")
    ax.set_ylim(bottom=0)
    ax.legend()
    plt.tight_layout()

    # Save figure
    if save_fig:
        fig_path = os.path.join(fig_output_folder, "diff_absolute_total.png")
        plt.savefig(fig_path, dpi=300)

    # Show figure
    if show_fig:
        plt.show()
    else:
        plt.close(fig)

    return diff_array

def relative_head_diffusion_zones(transient_heads, steady_state_heads, times, zone_array,
                                  start_step=0, threshold_percent= 5, stability_threshold=0.01,
                                  array_output_folder=".", fig_output_folder=".",
                                  save_fig=True, show_fig=False,
                                  zone_descriptions=None,
                                  bounds="95p"):
    """
    Plots per-zone relative head differences ((transient - steady)/initial_diff) with mean,
    median, and either 95% interval or mean ± std as shaded bounds. Annotates the time
    when mean drops below threshold.

    Parameters:
    -----------
    transient_heads : np.ndarray
        4D array (n_time, n_layer, n_row, n_col)
    steady_state_heads : np.ndarray
        3D array (n_layer, n_row, n_col)
    times : np.ndarray or list
        Simulation times corresponding to transient_heads
    zone_array : np.ndarray
        3D array identifying zones (n_layer, n_row, n_col)
    start_step : int
        Time step index to start analysis (default=0)
    threshold_percent : float
        Value at which the mean is considered "relaxed" (default=5)
    stability_threshold : float
        Threshold to exclude cells nearly steady at start_step (default=0.01)
    array_output_folder : str
        Folder path to save diff_array.npy
    fig_output_folder : str
        Folder path to save figure
    save_fig : bool
        Whether to save the figure
    show_fig : bool
        Whether to display the figure
    zone_descriptions : dict, optional
        Dictionary mapping zone numbers to descriptive names
    bounds : str, default "95p"
        Method for shaded bounds: "95p" = 2.5th–97.5th percentiles,
        "stdev" = mean ± standard deviation
    """

    # Ensure output folders exist
    os.makedirs(array_output_folder, exist_ok=True)
    os.makedirs(fig_output_folder, exist_ok=True)

    # Compute initial absolute differences at start_step
    initial_diff = np.abs(transient_heads[start_step] - steady_state_heads)

    # Compute differences at all times
    diff_array = np.abs(transient_heads - steady_state_heads)

    # Normalize by initial_diff, avoiding division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        relative_diff = np.where(initial_diff > stability_threshold,
                                 diff_array * 100/ initial_diff,
                                 np.nan)

    # Unique zones (exclude background if needed)
    zones = np.unique(zone_array)
    zones = zones[zones > 0]

    # X-axis: time since start_step
    time_since_start = times[start_step:] - times[start_step]  # zero at start_step
    time_in_years = time_since_start / 360  # convert to years

    # Prepare subplots
    n_zones = len(zones)
    fig, axes = plt.subplots(n_zones, 1, figsize=(14, 4 * n_zones), sharex=True)
    if n_zones == 1:
        axes = [axes]

    for ax, zone in zip(axes, zones):
        # Mask: zone selection
        zone_mask = (zone_array == zone)

        # Exclude cells with small initial difference
        exclude_mask = initial_diff <= stability_threshold 
        combined_mask = np.logical_or(~zone_mask, exclude_mask)

        # Apply mask
        diff_zone = np.where(combined_mask, np.nan, relative_diff)

        # Flatten spatial dimensions
        diff_zone_flat = diff_zone.reshape(diff_zone.shape[0], -1)

        # Select only times after start_step
        selected_diff_zone = diff_zone_flat[start_step:]

        # Compute stats per timestep
        means = np.array([np.nanmean(selected_diff_zone[t]) for t in range(selected_diff_zone.shape[0])])

        if bounds == "95p":
            lower = [np.nanpercentile(selected_diff_zone[t], 2.5) for t in range(selected_diff_zone.shape[0])]
            upper = [np.nanpercentile(selected_diff_zone[t], 97.5) for t in range(selected_diff_zone.shape[0])]
        elif bounds == "stdev":
            lower = [max(means[t] - np.nanstd(selected_diff_zone[t]), 0) for t in range(selected_diff_zone.shape[0])]
            upper = [min(means[t] + np.nanstd(selected_diff_zone[t]), 100) for t in range(selected_diff_zone.shape[0])]
        else:
            raise ValueError("bounds must be '95p' or 'stdev'")

        # Plot lines
        ax.plot(time_in_years, means, color="blue", linestyle="--", label="Mean", linewidth=1.5)
        ax.fill_between(time_in_years, lower, upper, color="lightblue", alpha=0.3,
                        label="95% interval" if bounds=="95p" else "Standard Deviation")
        ax.plot(time_in_years, lower, color="black", linestyle="--", linewidth=1, alpha=0.7)
        ax.plot(time_in_years, upper, color="black", linestyle="--", linewidth=1, alpha=0.7)

        # Annotate first time mean < threshold_percent
        below_threshold_idx = np.where(means < threshold_percent)[0]
        if below_threshold_idx.size > 0:
            idx = below_threshold_idx[0]
            t_cross = time_in_years[idx]
            mean_value = means[idx]
            ax.scatter(t_cross, mean_value, color="green", s=50, zorder=5)
            ax.axvline(t_cross, color="green", linestyle=":", linewidth=1.5)
            ax.text(t_cross + 0.5, ax.get_ylim()[1]*0.9, f"tr={t_cross:.1f} yr",
                    color="green", rotation=0, va='top', fontweight='bold')

        # Labels
        ax.set_ylabel("Relative head difference")
        if zone_descriptions and zone in zone_descriptions:
            ax.set_title(f"Zone {zone}: {zone_descriptions[zone]}")
        else:
            ax.set_title(f"Zone {zone}")

    axes[-1].set_xlabel("Time since step change (years)")
    ax.set_ylim(0, 100)
    axes[0].legend()
    plt.tight_layout()

    # Save diff_array
    np.save(os.path.join(array_output_folder, "diff_array_relative.npy"), relative_diff)
    
    # Save figure
    if save_fig:
        fig_path = os.path.join(fig_output_folder, "diff_relative_zones.png")
        plt.savefig(fig_path, dpi=300)

    if show_fig:
        plt.show()
    else:
        plt.close(fig)

    return relative_diff

def relative_head_diffusion(transient_heads, steady_state_heads, times,
                       start_step=0, threshold_percent=5, stability_threshold=0.01,
                       fig_output_folder=".", save_fig=True, show_fig=False,
                       bounds="95p"):
    """
    Plots residual head differences (transient - steady state) normalized by initial difference.
    Small initial differences below threshold_value are excluded to avoid division by zero.
    Can save and/or display the figure.

    Parameters
    ----------
    transient_heads : np.ndarray
        4D array (n_time, n_layer, n_row, n_col)
    steady_state_heads : np.ndarray
        3D array (n_layer, n_row, n_col)
    times : np.ndarray or list
        Simulation times corresponding to transient_heads
    start_step : int
        Time step index to start analysis (default=0)
    threshold_percent : float
        Threshold to calculate response time (default=5)
    stability_threshold : float
        Percent threshold for stability used to mask initial differences near steady state (default=0.01).
    fig_output_folder : str
        Folder path to save figure
    save_fig : bool
        Whether to save the figure
    show_fig : bool
        Whether to display the figure
    bounds : str, default "95p"
        Method for shaded bounds: "95p" = 2.5th–97.5th percentiles,
        "stdev" = mean ± standard deviation
    """

    os.makedirs(fig_output_folder, exist_ok=True)

    # Compute initial difference
    initial_diff = np.abs(transient_heads[start_step] - steady_state_heads)
    mask = initial_diff > stability_threshold  # boolean mask to avoid division by very small numbers

    # Compute normalized difference (residual)
    diff_array = np.abs(transient_heads - steady_state_heads)
    # Mask: exclude cells nearly steady at start_step
    exclude_mask = np.abs(diff_array[start_step]) < stability_threshold
    diff_array = np.where(exclude_mask, np.nan, diff_array)

    relative_array = np.zeros_like(diff_array)
    relative_array[:, mask] = diff_array[:, mask] * 100 / initial_diff[mask]

    # Flatten spatial dimensions
    relative_flat = relative_array.reshape(relative_array.shape[0], -1)
    selected_relative = relative_flat[start_step:]

    # X-axis: time since start_step
    time_since_start = times[start_step:] - times[start_step]
    time_in_years = time_since_start / 360  # assuming time in days

    # Compute stats per timestep
    means = np.array([np.nanmean(selected_relative[t]) for t in range(selected_relative.shape[0])])
    #medians = np.array([np.nanmedian(selected_relative[t]) for t in range(selected_relative.shape[0])])

    if bounds == "95p":
        lower = [np.nanpercentile(selected_relative[t], 2.5) for t in range(selected_relative.shape[0])]
        upper = [np.nanpercentile(selected_relative[t], 97.5) for t in range(selected_relative.shape[0])]
    elif bounds == "stdev":
        lower = [max(means[t] - np.nanstd(selected_relative[t]), 0) for t in range(selected_relative.shape[0])]
        upper = [min(means[t] + np.nanstd(selected_relative[t]), 100) for t in range(selected_relative.shape[0])]
    else:
        raise ValueError("bounds must be '95p' or 'stdev'")

    # Plot
    fig, ax = plt.subplots(figsize=(12, 5))
    #ax.plot(time_in_years, medians, color="darkblue", linestyle="-", label="Median", linewidth=1.5)
    ax.plot(time_in_years, means, color="blue", linestyle="--", label="Mean", linewidth=1.5)
    ax.fill_between(time_in_years, lower, upper, color="lightblue", alpha=0.3,
                    label="95% interval" if bounds=="95p" else "Standard Deviation")
    ax.plot(time_in_years, lower, color="black", linestyle="--", linewidth=1, alpha=0.7)
    ax.plot(time_in_years, upper, color="black", linestyle="--", linewidth=1, alpha=0.7)

    # Annotate first time mean < threshold_percent
    below_threshold_idx = np.where(means < threshold_percent)[0]
    if below_threshold_idx.size > 0:
        idx = below_threshold_idx[0]
        t_cross = time_in_years[idx]
        mean_value = means[idx]
        ax.scatter(t_cross, mean_value, color="green", s=50, zorder=5)
        ax.axvline(t_cross, color="green", linestyle=":", linewidth=1.5)
        ax.text(t_cross + 100, ax.get_ylim()[1]*0.9, f"tr={t_cross:.1f} yr",
                color="green", rotation=0, va='top', fontweight='bold')

    # Labels
    ax.set_xlabel("Time since step change (years)")
    ax.set_ylabel("Relative head difference (%)")
    ax.set_title("Relative head difference relative to initial difference")
    ax.set_ylim(0, 100)
    ax.legend()
    plt.tight_layout()

    if save_fig:
        fig_path = os.path.join(fig_output_folder, "diff_relative_total.png")
        plt.savefig(fig_path, dpi=300)

    if show_fig:
        plt.show()
    else:
        plt.close(fig)

    return relative_array

def response_time_array_absolute(gwf,
    steady_state_heads,
    transient_heads,
    times_list,
    threshold_absolute=0.01,
    stability_threshold=0.01,
    array_output_folder=None,
    fig_output_folder=None,
    save_array=True,
    save_plot=True,
    show_plot=False,
    start_step=30
):
    """
    Compute the absolute response time of transient heads to steady state.

    Parameters
    ----------
    gwf : flopy.mf6.ModflowGwf
        The groundwater flow model object (for plotting purposes)
    steady_state_heads : ndarray
        3D array of steady-state heads (nlay, nrow, ncol)
    transient_heads : ndarray
        3D array of transient heads (ntime, nlay, nrow, ncol)
    times_list : array-like
        Simulation times corresponding to transient_heads
    threshold_absolute : float
        Absolute threshold for relaxation
    stability_threshold : float, optional
        Threshold to treat zero initial differences as NaN
    array_output_folder : str, optional
        Folder to save the response time array as .npy
    fig_output_folder : str, optional
        Folder to save the plot
    save_array : bool, optional
        Whether to save the response time array
    save_plot : bool, optional
        Whether to save the plot
    show_plot : bool, optional
        Whether to display the plot
    start_step : int, optional
        Step to start computing response time

    Returns
    -------
    response_time_array : ndarray
        Array of response times in same shape as steady_state_heads
    """

    end_step = transient_heads.shape[0] - 1
    times = np.array(times_list)

    # ------------------------------ Compute absolute response time ----------------------------------- #
    nlay, nrow, ncol = steady_state_heads.shape

    # Precompute initial difference to detect zeros
    initial_diff = np.abs(transient_heads[start_step] - steady_state_heads)
    zero_diff_mask = initial_diff <= stability_threshold
    initial_diff = initial_diff.astype(float)
    initial_diff[zero_diff_mask] = np.nan

    # Initialize response time array with end_time as default
    response_time_array = np.full((nlay, nrow, ncol), times[end_step])

    # Boolean array to track assigned cells
    assigned = np.zeros((nlay, nrow, ncol), dtype=bool)

    # Loop through transient times and compute absolute relaxation
    for t in range(start_step, end_step):
        relaxation = np.abs(transient_heads[t] - steady_state_heads)
        mask = (relaxation <= threshold_absolute) & (~assigned)
        response_time_array[mask] = times[t]
        assigned[mask] = True

    # Set response time to start time where initial difference was zero
    response_time_array[np.isnan(initial_diff)] = times[start_step]

    # Save response time array if requested
    if save_array and array_output_folder:
        np.save(f"{array_output_folder}/response_time_absolute.npy", response_time_array)

    # Plotting
    fig = plt.figure(figsize=(19, 5))
    ax = fig.add_subplot(1, 1, 1)

    # Use middle row for cross-section
    mx = flopy.plot.PlotCrossSection(ax=ax, model=gwf, line={"row": nrow // 2})
    pa = mx.plot_array((response_time_array - times[start_step]) / 360, alpha=1, cmap="viridis", vmin=0)
    mx.plot_grid(color="0.5", alpha=0.2)
    cb = plt.colorbar(pa, ax=ax)
    cb.set_label("Response time (years)")
    ax.set_title(f"Response Time to Absolute Threshold {threshold_absolute}")
    plt.tight_layout()

    if save_plot and fig_output_folder:
        fig.savefig(f"{fig_output_folder}/Response_time_absolute.png", dpi=300)

    if show_plot:
        plt.show()
    else:
        plt.close(fig)

    return response_time_array

def response_time_array_relative(gwf,
    steady_state_heads,
    transient_heads,
    times_list,
    threshold_percent=5,
    stability_threshold=0.01,
    array_output_folder=None,
    fig_output_folder=None,
    save_array=True,
    save_plot=True,
    show_plot=False,
    start_step=30
):
    """
    Compute the relative response time of transient heads to steady state.

    Parameters
    ----------
    gwf : flopy.mf6.ModflowGwf
        The groundwater flow model object (for plotting purposes)
    steady_state_heads : ndarray
        3D array of steady-state heads (nlay, nrow, ncol)
    transient_heads : ndarray
        3D array of transient heads (ntime, nlay, nrow, ncol)
    times_list : array-like
        Simulation times corresponding to transient_heads
    threshold_percent : float
        Percent threshold for relaxation
    stability_threshold : float, optional
        Threshold to treat zero initial differences as NaN
    array_output_folder : str, optional
        Folder to save the response time array as .npy
    fig_output_folder : str, optional
        Folder to save the plot
    save_array : bool, optional
        Whether to save the response time array
    save_plot : bool, optional
        Whether to save the plot
    show_plot : bool, optional
        Whether to display the plot
    start_step : int, optional
        Step to start computing response time

    Returns
    -------
    response_time_array : ndarray
        Array of response times in same shape as steady_state_heads
    """

    end_step = transient_heads.shape[0] - 1
    times = np.array(times_list)

    # ------------------------------ Compute relative response time ----------------------------------- #
    nlay, nrow, ncol = steady_state_heads.shape

    # Precompute initial difference (denominator)
    initial_diff = np.abs(transient_heads[start_step] - steady_state_heads)

    # Mask for zero initial differences
    zero_diff_mask = initial_diff <= stability_threshold
    initial_diff = initial_diff.astype(float)
    initial_diff[zero_diff_mask] = np.nan

    # Initialize response time array with end_time as default
    response_time_array = np.full((nlay, nrow, ncol), times[end_step])

    # Boolean array to track assigned cells
    assigned = np.zeros((nlay, nrow, ncol), dtype=bool)

    # Loop through transient times and compute relaxation
    for t in range(start_step, end_step):
        relaxation = np.abs(transient_heads[t] - steady_state_heads) * 100.0 / initial_diff
        mask = (relaxation <= threshold_percent) & (~assigned)
        response_time_array[mask] = times[t]
        assigned[mask] = True

    # Set response time to start time where initial difference was zero
    response_time_array[np.isnan(initial_diff)] = times[start_step]

    # Save response time array if requested
    if save_array and array_output_folder:
        np.save(f"{array_output_folder}/response_time_relative.npy", response_time_array)

    # Plotting
    fig = plt.figure(figsize=(19, 5))
    ax = fig.add_subplot(1, 1, 1)

    # Use middle row for cross-section
    mx = flopy.plot.PlotCrossSection(ax=ax, model=gwf, line={"row": nrow // 2})
    pa = mx.plot_array((response_time_array - times[start_step]) / 360, alpha=1, cmap="viridis", vmin=0)
    mx.plot_grid(color="0.5", alpha=0.2)
    cb = plt.colorbar(pa, ax=ax)
    cb.set_label("Response time (years)")
    ax.set_title(f"Response Time to {threshold_percent}% Relaxation")
    plt.tight_layout()

    if save_plot and fig_output_folder:
        fig.savefig(f"{fig_output_folder}/Response_time_relative.png", dpi=300)

    if show_plot:
        plt.show()
    else:
        plt.close(fig)

    return response_time_array
