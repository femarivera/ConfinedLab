
import time
start_time = time.time()
import sys
import os

# Import local modules
sys.path.append('..')
from mlibs import modpump6 # type: ignore

# --------------------------------------------------------------------------------------- #
# -------------------------- SUSTAINABLE YIELD ESTIMATION ------------------------------- #
# --------------------------------------------------------------------------------------- #

# Define paths
input_folder = "C:/Users/cmarinriver/Projects/ConfinedLab/sust_yield_results/Summary_iterations"
output_folder = "C:/Users/cmarinriver/Projects/ConfinedLab/sust_yield_results/Outputs"
plot_folder = os.path.join(output_folder, "plots")
os.makedirs(output_folder, exist_ok=True)
os.makedirs(plot_folder, exist_ok=True)

planning_horizons = [5, 10, 25, 50, 75, 100, 200, 500, 1000, 2000] # In years
pump_start = 7550000 # In model totim units (days)
constraints = [
    { 'label': "Spring discharge all zones", 'id': "drn_all", 'constrain': "DRN",
      'flow': "NET", 'zone': "ALL", 'threshold_type': "RELATIVE", 'threshold': 0.9,
      'reference': None, 'neighbour_zones': None, 'color' : "Purple" },

    { 'label': "River discharge zone 1", 'id': "riv_1", 'constrain': "RIV",
      'flow': "NET", 'zone': 1, 'threshold_type': "RELATIVE", 'threshold': 0.9,
      'reference': None, 'neighbour_zones': None, 'color' : "Blue" },

    { 'label': "River discharge zone 3", 'id': "riv_3", 'constrain': "RIV",
      'flow': "NET", 'zone': 3, 'threshold_type': "RELATIVE", 'threshold': 0.9,
      'reference': None, 'neighbour_zones': None, 'color' : "lightblue" },

    { 'label': "Leakage zone 3", 'id': "leak_3", 'constrain': "LEAKAGE",
      'flow': "NET", 'zone': 3, 'threshold_type': "ABSOLUTE", 'threshold': -2.5,
      'reference': None, 'neighbour_zones': [2,4], 'color' : "orange" },

    { 'label': "Lateral outflow zone 3", 'id': "ghb_3", 'constrain': "GHB",
      'flow': "NET", 'zone': 3, 'threshold_type': "ABSOLUTE", 'threshold': 0,
      'reference': None, 'neighbour_zones': None, 'color' : "red" }
]

planning_horizons_totim = [val * 365 for val in planning_horizons] # Convert years to days
qs_values = []
for tp in planning_horizons_totim:
    qs, df, plot_file = modpump6.estimate_sustainable_yield(
        input_folder=input_folder,
        output_folder=output_folder,
        plot_folder=plot_folder,
        pump_start=pump_start,
        planning_horizon=tp,
        constraints=constraints,
        csv_filename= f"flow_summary_{tp}.csv",
        plot_filename=f"yield_plot_{tp}.png"
    )
    print(f"Sustainable yield: {qs}")
    print(df.head())
    qs_values.append(qs)

# Save summary of qs values
import pandas as pd

qs_df = pd.DataFrame({"Planning_Horizon": planning_horizons, "Sustainable_Yield": qs_values})
qs_df.to_csv(os.path.join(output_folder, "sustainable_yield_summary.csv"), index=False)

import matplotlib.pyplot as plt

def plot_df(
    qs,
    title="Sustainable Yield Plot",
    xlabel="Pumping Rate [m³/day]",
    ylabel="Flow Rate [m³/day]",
    legend_label="Qs curve",
    figsize=(8,6),
    save_path=None
):
    """
    Plot a 2-column DataFrame: first column = x, second column = y.
    """
    # Extract columns
    x = qs.iloc[:, 0]
    y = qs.iloc[:, 1]

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot with scientific styling
    ax.plot(x, y, marker='o', linestyle='-', color='navy', label=legend_label)

    # Titles and labels
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)

    # Grid and legend
    ax.grid(True, which="both", linestyle="--", linewidth=0.7, alpha=0.7)
    ax.legend(fontsize=10)

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
    title="Sustainable Yield vs Planning Horizon",
    xlabel="Planning Horizon [years]",
    ylabel="Sustainable yield [m³/day]",
    save_path=plot_folder + "/sustainable_yield_vs_horizon.png")
