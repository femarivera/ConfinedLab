import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import flopy
import os
import sys
from matplotlib.animation import FuncAnimation, PillowWriter

# Import local modules
import sys
sys.path.append('..')
from mlibs import modplot6 # type: ignore

def create_cross_section_animation(image_paths, 
                                   output_file):
    """
    Creates an animation from a list of image file paths.

    Parameters:
        image_paths (list of str): List of file paths to images.
        output_file (str): Path to save the animation GIF.

    Returns:
        None (saves the animation as a GIF).
    """
    plt.ioff()

    frames = [plt.imread(img) for img in image_paths]

    fig, ax = plt.subplots(figsize=(15, 6))
    img = ax.imshow(frames[0])
    ax.axis("off")

    def update(frame):
        img.set_data(frame)
        return [img]

    ani = FuncAnimation(fig, update, frames=frames, interval=300, blit=True)

    # Save animation as a GIF
    ani.save(output_file, writer=PillowWriter(fps=1))
    print(f"Animation saved as {output_file}")
    plt.close(fig)        

def update_well_pumping_rate_steady(gwf, 
                                    wel_spd, 
                                    wel, 
                                    q):
    """
    Updates the pumping rate (q) for all wells in wel_spd[0] and modifies the corresponding well package (wel).
    Used for STEADY STATE SIMULATIONS.
    
    Parameters:
        wel_spd (dict): The dictionary containing well stress period data.
        wel (flopy.mf6.ModflowGwfwel): The well package object.
        q (float): The new pumping rate to update.
    
    Returns:
        None (modifies wel_spd in place and updates the wel object).
    """
    # Iterate through each well in stress period 0 and update the pumping rate
    for i in range(len(wel_spd[0])):
        # Update the tuple in wel_spd
        wel_spd[0][i] = (wel_spd[0][i][0], wel_spd[0][i][1], wel_spd[0][i][2], q)
    
    # Update the well package (wel) with the new pumping rate for stress period 0
    wel = flopy.mf6.ModflowGwfwel(gwf, 
                                 pname = "wel",
                                 save_flows = True,
                                 stress_period_data = wel_spd)

def iterate_pumping_rate_steady(sim, 
                                gwf, 
                                wel_spd, 
                                wel, 
                                q_values, 
                                model_ws, 
                                cross_section_dir, 
                                output_dir, 
                                budget_csv_file_name, 
                                row,
                                boundary_keywords = None,
                                animate=False, 
                                save_budget = False, 
                                save_wells = False):
    """
    Function to iterate through different pumping rates, run simulations, and generate plots.
    Used for STEADY STATE SIMULATIONS.

    Parameters:
        sim (flopy.mf6.MFSimulation): The simulation object.
        wel (flopy.mf6.ModflowGwfwel): The well package object.
        q_values (np.ndarray): Array of pumping rates to iterate through.
        budget_csv_file_name (str): Name of the CSV file generated after each simulation
        AS DEFINED IN THE OUTPUT CONTROL PACKAGE.

    Returns:
        Plot of induced recharge, natural discharge, and captured discharge vs pumping rates
    """

    # --------------------------------------------------------------------- #
    # ----------------------------- NO PUMPING ---------------------------- #
    # --------------------------------------------------------------------- #

    # Read initial simulation with q=0 to get the natural inflow and natural outflow
    update_well_pumping_rate_steady(gwf, wel_spd, wel, 0)  # Set pumping rate to 0 for natural conditions
    sim.write_simulation()
    success, buff = sim.run_simulation()
    if not success:
        print(f"Simulation failed for pumping rate 0")
        return

    # Path to the initial CSV file (natural conditions)
    csv_file_path = os.path.join(model_ws, budget_csv_file_name)
    data = pd.read_csv(csv_file_path)

    # Get natural inflow and natural outflow from TOTAL_IN and TOTAL_OUT
    natural_inflow = data['TOTAL_IN'].iloc[-1]
    natural_outflow = data['TOTAL_OUT'].iloc[-1]

    # --------------------------------------------------------------------- #
    # ----------------------------- PUMPING RATES ------------------------- #
    # --------------------------------------------------------------------- #
    
    # Initialize lists and paths to output directories
   
    induced_recharge_results = []
    natural_discharge_results = []
    captured_discharge_results = []
    pumping_rates = []
    image_paths = []

    # Identify relevant columns (excluding first and last columns)
    relevant_columns = data.columns[1:-1]  # Exclude first and last columns
    # Initialize a list to store the results for each column in relevant_columns
    column_results = {col: [] for col in relevant_columns}

    for idx, q in enumerate(q_values):
        print(f"Running simulation with pumping rate: {q} m³/day")
        
        # Update the pumping rate for all wells
        update_well_pumping_rate_steady(gwf, wel_spd, wel, q)
        
        # Write and run the simulation
        sim.write_simulation()
        success, buff = sim.run_simulation()
        if not success:
            print(f"Simulation failed for pumping rate {q}")
            continue
        
        # ---------------------------- ANIMATION --------------------------------- #
        if animate:
            # Create directory if it does not exist
            if cross_section_dir and not os.path.exists(cross_section_dir):
                os.makedirs(cross_section_dir)

            # Load outputs
            head = gwf.output.head().get_data()
            bud = gwf.output.budget()
            spdis = bud.get_data(text='DATA-SPDIS')[0]
            qx, qy, qz = flopy.utils.postprocessing.get_specific_discharge(spdis, gwf)
            
            #Plot cross section
            plt.ioff()
            fig, ax = plt.subplots(figsize=(19, 4))
            modplot6.plot_cross_section_row(gwf, head, qx, qy, qz, row, 
                                            model_ws,
                                            boundary_keywords = boundary_keywords,
                                            flow_dir = False, surface = True, layers=True,
                                            show = False, save = False, ax=ax)
            plt.title(f"Cross-Section for Pumping Rate: {abs(q):.1f} m³/day")

            # Save the plot as an image                 
            # List to store paths of cross-section images
            image_path = os.path.join(cross_section_dir, f"cross_section_q_{idx}.png")
            fig.savefig(image_path, dpi=300)
            image_paths.append(image_path)
            plt.close(fig)
        
            # Create animation from saved images
            if image_paths:
                create_cross_section_animation(image_paths, os.path.join(model_ws, "cross_section_animation.gif"))
            else:
                print("No successful simulations to animate.")

        # ------------------- DATA FOR FLOW BUDGET PLOTS ----------------------- #

        # Path to the current CSV file
        csv_file_path = os.path.join(model_ws, budget_csv_file_name)

        # Load the CSV file generated by the current simulation
        data = pd.read_csv(csv_file_path)

        for col in relevant_columns:
            column_results[col].append(data[col].iloc[-1])

        # Compute induced recharge, natural discharge, and capture
        total_in = data['TOTAL_IN'].iloc[-1]
        induced_recharge = total_in - natural_inflow  # Induced recharge = total_in - natural_inflow
        
        
        columns_other_out = [col for col in data.columns if col.endswith("_OUT") and "WEL" not in col and col != "TOTAL_OUT"]
        natural_discharge = data[columns_other_out].sum(axis=1).iloc[-1]
        
        # Compute captured discharge
        captured_discharge = natural_outflow - natural_discharge

        # Store results
        pumping_rates.append(abs(q))
        induced_recharge_results.append(induced_recharge)
        natural_discharge_results.append(natural_discharge)
        captured_discharge_results.append(captured_discharge)

    # --------------------------------------------------------------------- #
    # -------------------------- PLOTS FLOW BUDGET ------------------------ #
    # --------------------------------------------------------------------- #

    def simplify_name(name):
        # If parentheses are present
        if '(' in name and ')' in name:
            # Extract text inside parentheses
            simplified = name.split('(')[1].split(')')[0].strip()
             # Extract text after the underscore and strip spaces
            after_underscore = name.split(')')[1].split('_')[1].strip()
            # Combine both parts with a space
            simplified = simplified + " " + after_underscore
            return simplified
        else:
        # If no parentheses, replace the underscore with a space
            simplified = name.replace('_', ' ')  
            return simplified

    # Split relevant columns into two groups based on "_IN" and "_OUT"
    columns_in = [col for col in relevant_columns if "_IN" in col]
    columns_out = [col for col in relevant_columns if "_OUT" in col]
    
    #Simplify names of columns
    simplified_columns_in = [simplify_name(col) for col in columns_in]
    simplified_columns_out = [simplify_name(col) for col in columns_out]
    simplified_columns_names = [simplify_name(col) for col in column_results]
    
    #Replace column names on  the column results dictionary
    simplified_column_results = {}

    # Iterate over both the original column names and their corresponding simplified names
    for old_key, new_key in zip(column_results.keys(), simplified_columns_names):
        simplified_column_results[new_key] = column_results[old_key]  # Assign the same list of results

    # Determine the maximum number of rows needed
    n_rows = max(len(columns_in), len(columns_out))

    # Create the figure and axes with 2 columns and n_rows
    fig, axes = plt.subplots(n_rows, 2, figsize=(15, n_rows * 5))

    # Ensure axes are treated as a 2D array for easier indexing
    axes = axes if isinstance(axes, np.ndarray) and len(axes.shape) == 2 else np.array([axes]).reshape(n_rows, 2)

    # Plot "_IN" columns in the first column
    for i, col in enumerate(simplified_columns_in):
        ax = axes[i, 0]  # Access the subplot for this position
        ax.plot(pumping_rates, simplified_column_results[col], marker='o', label=col)
        ax.set_xlabel('Pumping Rate (m³/day)')
        ax.set_ylabel(f'{simplify_name(col)} (m³/day)')
        ax.grid(True)
        ax.legend()

    # Plot "_OUT" columns in the second column
    for i, col in enumerate(simplified_columns_out):
        ax = axes[i, 1]  # Access the subplot for this position
        ax.plot(pumping_rates, simplified_column_results[col], marker='o', label=col)
        ax.set_xlabel('Pumping Rate (|m³/day|)')
        ax.set_ylabel(f'{simplify_name(col)} (m³/day)')
        ax.grid(True)
        ax.legend()

    # Hide any unused subplots
    for i in range(n_rows):
        if i >= len(columns_in):
            axes[i, 0].axis('off')  # Hide unused plots in the first column
        if i >= len(columns_out):
            axes[i, 1].axis('off')  # Hide unused plots in the second column

    plt.tight_layout()
    
    if save_budget:
        image_path = os.path.join(output_dir, f"modpump6_water budget.png")
        fig.savefig(image_path, dpi=300)
        plt.close(fig)         

    # --------------------------------------------------------------------- #
    # -------------------------- PLOTS WATER TO WELLS --------------------- #
    # --------------------------------------------------------------------- #

    # Plot induced recharge, natural discharge, and captured discharge vs pumping rate
    fig2, axs2 = plt.subplots(3, 1, figsize=(12, 8))

    # Induced Recharge vs Pumping Rate
    axs2[0].plot(pumping_rates, induced_recharge_results, marker='o', label='Induced Inflows')
    axs2[0].set_xlabel('Pumping Rate (m³/day)')
    axs2[0].set_ylabel('Induced Inflows (m³/day)')
    axs2[0].grid(True)
    axs2[0].legend()

    # Natural Discharge vs Pumping Rate
    axs2[1].plot(pumping_rates, natural_discharge_results, marker='o', label='Natural Outflows', color='green')
    axs2[1].set_xlabel('Pumping Rate (m³/day)')
    axs2[1].set_ylabel('Natural Outflows (m³/day)')
    axs2[1].grid(True)
    axs2[1].legend()

    # Captured Discharge vs Pumping Rate
    axs2[2].plot(pumping_rates, captured_discharge_results, marker='o', label='Captured Outflows', color='orange')
    axs2[2].set_xlabel('Pumping Rate (m³/day)')
    axs2[2].set_ylabel('Captured Outflows (m³/day)')
    axs2[2].grid(True)
    axs2[2].legend()

    plt.tight_layout()
    
    if save_wells:
        image_path = os.path.join(output_dir, f"modpump6_water to wells.png")
        fig2.savefig(image_path, dpi=300)
        plt.close(fig2) 
