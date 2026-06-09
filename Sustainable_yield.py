import time
start_time = time.time()

import sys
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Import local modules
# Set path to parent dir containing mlibs modules (absolute path)
mlibs_path = "/srv/common/deesac/ConfinedLab"
sys.path.append(mlibs_path)
from mlibs import modpump6, modgeom6 # type: ignore

# --------------------------------------------------------------------------------------- #
# ------------------------------- RUN CONTROL ------------------------------------------ #
# --------------------------------------------------------------------------------------- #

ITERATE = True  # Set to True to run the iteration process
EFFICIENCY = True  # Set to True to run the efficiency-based iteration process

ESTIMATE = True  # Set to True to run the sustainable yield estimation

planning_horizons = [ 5, 10, 25, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 30000, 40000, 50000] # In years
pump_start = 3600000 # In model totim units (days)

# --------------------------------------------------------------------------------------- #
# ------------------------------- INPUTS ITERATION -------------------------------------- #
# --------------------------------------------------------------------------------------- #

# Set paths to setup add model files
setup_file = "setup.xlsx" 
model_file = "Model.py"

# Define a general output folder for all results of sustainable yield estimation
output_folder = "sust_yield_results"

# Define output directories for all yield iteration runs, for a summary of the yield iterations, and for generated plots (absolute paths)
iterations_output_dir = os.path.join(output_folder, "yield_iterations")
summary_dir = os.path.join(output_folder, "summary_iterations")
plot_folder = os.path.join(output_folder, "plots")

# Define model workspace name subscript (used to taylor output paths for each yield iteration)
model_ws_name = "mf" 

# Set output file basenames as written by the modflow model output
model_name = 'DEESACt'
budget_file_name = f"{model_name}_budget.csv"
zonebud_file_name = "zonebud.csv"
head_file_name = "head_obs_t.csv"

# --------------------------------------------------------------------------------------- #
# ------------------------------- INPUTS ESTIMATION ------------------------------------- #
# --------------------------------------------------------------------------------------- #

# Define paths to inputs of the sustainable yield estimation
input_folder = summary_dir # Takes as input the path to the summary of the yield iterations

# Define start time of pumping, planning horizons and constraints
constraints = [
    { 
    'label': "Spring discharge all zones", 
    'id': "drn_all", 
    'constrain': "DRN(DRN1)",
    'flow': "NET", 
    'zone': "ALL", 
    'threshold_type': "RELATIVE", 
    'threshold': 0.9,
    'reference': None, 
    'neighbour_zones': None, 
    'color' : "Purple" 
    },

    # { 'label': "River discharge zone 1", 'id': "riv_1", 'constrain': "RIV",
    #  'flow': "NET", 'zone': 1, 'threshold_type': "RELATIVE", 'threshold': 0.9,
    #  'reference': None, 'neighbour_zones': None, 'color' : "Blue" },

    # { 'label': "Leakage zone 3", 'id': "leak_3", 'constrain': "LEAKAGE",
    # 'flow': "NET", 'zone': 3, 'threshold_type': "ABSOLUTE", 'threshold': -40,
    # 'reference': None, 'neighbour_zones': [2,4], 'color' : "orange" },

    { 
    'label': "Lateral outflow zone 3",
    'id': "ghb_3", 
    'constrain': "GHB(GHB2)",
    'flow': "NET", 
    'zone': "ALL", 
    'threshold_type': "RELATIVE", 
    'threshold': 0.9,
    'reference': None, 
    'neighbour_zones': None, 
    'color' : "red" 
    },

    {
    "label": "Head at pumping well",
    "id": "head_aqf",
    "constrain": "HEAD",
    "flow": "NET",              # ignored for HEAD
    "zone": "ALL",              # ignored for HEAD
    "threshold_type": "ABSOLUTE",
    "threshold": 0,
    "reference": None,
    "neighbour_zones": None,    # ignored for HEAD
    "color": "orange",
    "head_obs": "CAQ1_12_400_100"         # REQUIRED for HEAD
}
    ]

# --------------------------------------------------------------------------------------- #
# ------------------------------- YIELD ITERATION --------------------------------------- #
# --------------------------------------------------------------------------------------- #

if ITERATE: 
    if EFFICIENCY:
        print("Running efficiency-based pumping rate iteration...")
        modpump6.iterate_pumping_rate_transient_eff(setup_file, model_file, model_ws_name, model_name, 
                                            iterations_output_dir, summary_dir, 
                                            budget_file_name, zonebud_file_name, head_file_name)
    else:
        print("Running full-based pumping rate iteration...")
        modpump6.iterate_pumping_rate_transient(setup_file, model_file, mlibs_path, 
                                                iterations_output_dir, summary_dir, 
                                                model_ws_name, budget_file_name, 
                                                zonebud_file_name, head_file_name)
    

# --------------------------------------------------------------------------------------- #
# -------------------------- SUSTAINABLE YIELD ESTIMATION ------------------------------- #
# --------------------------------------------------------------------------------------- #

if ESTIMATE:
    # Convert years to totim units (days) for the sustainable yield function
    planning_horizons_totim = [val * 360 for val in planning_horizons]
    # Loop over planning horizons and estimate sustainable yield 
    qs_values = []
    for tp in planning_horizons_totim:
        qs, df = modpump6.estimate_sustainable_yield(
            input_folder,
            output_folder,
            plot_folder,
            pump_start,
            pump_zone="ALL",
            planning_horizon=tp,
            constraints=constraints,
            model_name=model_name,
            csv_filename= f"Q_vs_flow_{int(tp/360)}.csv",
            plot_filename=f"Q_vs_flow_{int(tp/360)}.png",
            plot_units="years",
            conversion_factor=360)
        qs_values.append(qs)
        print(f"Sustainable yield: {qs}")
        print(df.head())

    # Save summary of sustainable yield values vs planning horizon
    qs_df = pd.DataFrame({"Planning_Horizon": planning_horizons, "Sustainable_Yield": qs_values})
    qs_df.to_csv(os.path.join(output_folder, "sustainable_yield_summary.csv"), index=False)

    # Plot sustainable yield vs planning horizon
    def plot_df(
        qs,
        title="Sustainable Yield Plot",
        xlabel="Pumping Rate [m³/day]",
        ylabel="Flow Rate [m³/day]",
        figsize=(8,6),
        save_path=None):

        """
        Plot a 2-column DataFrame: first column = x, second column = y.
        """
        # Extract columns
        x = qs.iloc[:, 0]
        y = qs.iloc[:, 1]

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        # Plot with scientific styling
        ax.plot(x, y, marker='o', linestyle='-', color='navy')

        # Titles and labels
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)

        # Grid and legend
        ax.grid(True, which="both", linestyle="--", linewidth=0.7, alpha=0.7)

        # Scientific notation for axes if values are large/small
        #ax.ticklabel_format(style="sci", axis="both", scilimits=(0,0))

        # Tight layout
        plt.tight_layout()

        # Save or show
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        else:
            plt.show()
    plot_df(
        qs_df,
        title="Sustainable yield vs Planning horizon",
        xlabel="Planning Horizon [years]",
        ylabel="Sustainable yield [m³/day]",
        save_path=plot_folder + "/sustainable_yield_vs_horizon.png")

end_time = time.time()
print(f"Total execution time: {end_time - start_time:.2f} seconds")
