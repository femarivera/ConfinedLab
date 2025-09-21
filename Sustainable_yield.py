import time
start_time = time.time()

import sys
import os
import matplotlib.pyplot as plt
import pandas as pd

# Import local modules
sys.path.append('..')
from mlibs import modpump6 # type: ignore

ITERATE = True  # Set to True to run the iteration process
ESTIMATE = True  # Set to True to run the sustainable yield estimation

# --------------------------------------------------------------------------------------- #
# ------------------------------- YIELD ITERATION --------------------------------------- #
# --------------------------------------------------------------------------------------- #
# Set paths to setup add model files (absolute paths)
setup_file = "C:/Users/cmarinriver/Projects/ConfinedLab/setup.xlsx" 
model_file = "C:/Users/cmarinriver/Projects/ConfinedLab/Model.py" 

# Set path to parent dir containing mlibs (absolute path)
mlibs_path = "C:/Users/cmarinriver/Projects/ConfinedLab"

# Define output directories for yield iteration runs and for a summary of the iterations (absolute paths)
output_dir = "C:/Users/cmarinriver/Projects/ConfinedLab/sust_yield_results/yield_iterations"
summary_dir = os.path.join(output_dir, "summary_iterations")

if ITERATE: 

    # Define model workspace name subscript (used to taylor output paths for each model run)
    model_ws_name = "mf" 

    # Set output file basenames as written by the model output (not full paths, just basenames)
    model_name = 'DEESACt'
    budget_file_name = f"{model_name}_budget.csv"
    zonebud_file_name = "zonebud.csv"
    head_file_name = "head_obs_t.csv"

    modpump6.iterate_pumping_rate_transient(setup_file, model_file, mlibs_path, 
                                            output_dir, summary_dir, model_ws_name, 
                                            budget_file_name, zonebud_file_name, head_file_name)

# --------------------------------------------------------------------------------------- #
# -------------------------- SUSTAINABLE YIELD ESTIMATION ------------------------------- #
# --------------------------------------------------------------------------------------- #

# Define paths to input/output folders (absolute paths)
input_folder = summary_dir # Takes as input the path to the summary of the iterations
output_folder = "C:/Users/cmarinriver/Projects/ConfinedLab/sust_yield_results"
plot_folder = os.path.join(output_folder, "plots")

if ESTIMATE:

    # Define planning horizons and constraints
    planning_horizons = [ 10, 25, 50, 75, 100, 200, 500, 1000] # In years
    pump_start = 7560000 # In model totim units (days)
    constraints = [
        { 'label': "Spring discharge all zones", 'id': "drn_all", 'constrain': "DRN",
        'flow': "NET", 'zone': "ALL", 'threshold_type': "RELATIVE", 'threshold': 0.9,
        'reference': None, 'neighbour_zones': None, 'color' : "Purple" },

        #{ 'label': "River discharge zone 1", 'id': "riv_1", 'constrain': "RIV",
        #  'flow': "NET", 'zone': 1, 'threshold_type': "RELATIVE", 'threshold': 0.9,
        #  'reference': None, 'neighbour_zones': None, 'color' : "Blue" },

        { 'label': "River discharge zone 3", 'id': "riv_3", 'constrain': "RIV",
        'flow': "NET", 'zone': 3, 'threshold_type': "RELATIVE", 'threshold': 0.9,
        'reference': None, 'neighbour_zones': None, 'color' : "lightblue" },

        #{ 'label': "Leakage zone 3", 'id': "leak_3", 'constrain': "LEAKAGE",
        #'flow': "NET", 'zone': 3, 'threshold_type': "ABSOLUTE", 'threshold': -40,
        #'reference': None, 'neighbour_zones': [2,4], 'color' : "orange" },

        { 'label': "Lateral outflow zone 3", 'id': "ghb_3", 'constrain': "GHB",
        'flow': "NET", 'zone': 3, 'threshold_type': "ABSOLUTE", 'threshold': 0,
        'reference': None, 'neighbour_zones': None, 'color' : "red" },
    ]

    # Convert years to totim units (days) for the sustainable yield function
    planning_horizons_totim = [val * 365 for val in planning_horizons]

    # Loop over planning horizons and estimate sustainable yield 
    qs_values = []
    for tp in planning_horizons_totim:
        qs, df = modpump6.estimate_sustainable_yield(
            input_folder=input_folder,
            output_folder=output_folder,
            plot_folder=plot_folder,
            pump_start=pump_start,
            pump_zone="ALL",
            planning_horizon=tp,
            constraints=constraints,
            csv_filename= f"Q_vs_flow_{int(tp/365)}.csv",
            plot_filename=f"Q_vs_flow_{int(tp/365)}.png",
            plot_units="years")
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
