import time
start_time = time.time()
import shutil
import os
import subprocess
import pandas as pd
import sys
import matplotlib.pyplot as plt

# Import local modules
sys.path.append('..')
from mlibs import modpump6 # type: ignore

ITERATE = False  # Set to True to run the iteration process
PLOT = True  # Set to True to plot the results after all iterations

# --------------------------------------------------------------------------------------- #
# ------------------------------- USER INPUTS ------------------------------------------- #
# --------------------------------------------------------------------------------------- #

# Set paths to setup add model files (absolute paths)
setup_file = "C:/Users/cmarinriver/Projects/ConfinedLab/setup.xlsx" # Absolute paths
sust_yield_file = "C:/Users/cmarinriver/Projects/ConfinedLab/Sustainable_yield.py" # Absolute paths

# Set path to parent dir containing mlibs modules (absolute path)
mlibs_path = "C:/Users/cmarinriver/Projects/ConfinedLab" # Absolute paths

# Define a general output folder for all results of the parameter analysis (absolute path)
output_folder = "C:/Users/cmarinriver/Projects/ConfinedLab/par_results" # Absolute paths

# Define the parameter to analyze (must be in the parameter_analysis sheet of the setup file)
parameter = "kv_02"

# --------------------------------------------------------------------------------------- #
# ------------------------------- ITERATE PARAMETERS ------------------------------------ #
# --------------------------------------------------------------------------------------- #

if ITERATE:

    # --------------------------------------------------------------------------------------- #
    # ------------------------------- PREPARE ITERATION FILE -------------------------------- #
    # --------------------------------------------------------------------------------------- #

    os.makedirs(output_folder, exist_ok=True)

    folder, filename = os.path.split(sust_yield_file)
    name, ext = os.path.splitext(filename)
    new_filename = f"{name}_it{ext}"
    new_file_path = os.path.join(folder, new_filename)

    # Copy the original file
    shutil.copy(sust_yield_file, new_file_path)

    # --------------------------------------------------------------------------------------- #
    # ------------------------- UPDATE PARAMETERS AND SCRIPT FILE --------------------------- #
    # --------------------------------------------------------------------------------------- #

    # Open file contining the parameters used by modflow6
    par_df = pd.read_excel(setup_file, sheet_name="parameters", index_col=0)

    # Open file with the parameter values for for each iteration
    par_val_df = pd.read_excel(setup_file, sheet_name="parameter_analysis", index_col=0)

    # Identify iteration columns and number of iterations (all except par_name)
    iter_cols = [c for c in par_val_df.columns if c not in ["par_name", "comment"]]
    n_iterations = len(iter_cols)

    # Loop over each iteration (parameter group)
    for i, col in enumerate(iter_cols, start=1):
        print(f"\n--- Running simulation for parameter group {i}/{n_iterations} with {col} ---")

        for row in par_val_df.index:
        # Replace parameter value
            par_df.loc[row, "value"] = par_val_df.loc[row, col]

        # Write updated parameters sheet back to Excel (overwrite only that sheet)
        with pd.ExcelWriter(setup_file, mode="a", if_sheet_exists="replace") as writer:
            par_df.to_excel(writer, sheet_name="parameters", index=True)

        # Read and modify the iteration file with right paths and setup file
        with open(new_file_path, "r") as f:
            lines = f.readlines()

        setup_replaced = False
        output_folder_repaced = False
        for i, line in enumerate(lines):
            if "sys.path.append('..')" in line:
                lines[i] = f"sys.path.append(r'{mlibs_path}')\n"
            if not setup_replaced and line.strip().startswith("setup_file ="):
                lines[i] = f"setup_file = r'{setup_file}' # Excel file containing model setup parameters\n"
                setup_replaced = True
            if not output_folder_repaced and line.strip().startswith("output_folder ="):
                lines[i] = f"output_folder = r'{output_folder}/{col}' # Output folder for each parameter iteration results\n"
                output_folder_repaced = True

        # Save back the modified file
        with open(new_file_path, "w") as f:
            f.writelines(lines)
        print("Iteration script created at:", new_file_path)

        # --------------------------------------------------------------------------------------- #
        # ----------------------- ITERATE SUSTAINABLE YIELD ESTIMATION -------------------------- #
        # --------------------------------------------------------------------------------------- #
        
        # Create a unique model workspace directory name based on the parameter value
        model_ws = os.path.join(output_folder, f"{col}")
        os.makedirs(model_ws, exist_ok=True)
        
        # Copy the iteration script into the unique model workspace folder and get the path
        shutil.copy(new_file_path, model_ws)
        script_path = os.path.join(model_ws, new_filename)
        
        # Run the script inside the unique model workspace folder
        # You can use subprocess to execute the script in that directory
        subprocess.run(["python", script_path], cwd=model_ws)

        print(f"Model run completed for iteration {i} with q={col}, model_ws={model_ws}")

# --------------------------------------------------------------------------------------- #
# ------------------------------- PLOT RESULTS ------------------------------------------ #
# --------------------------------------------------------------------------------------- #


if PLOT:
    # --- Load parameter mapping once ---
    par_val_df = pd.read_excel(setup_file, sheet_name="parameter_analysis", index_col=0)

    results = []

    # Collect all results in a single pass
    for subfolder in sorted(os.listdir(output_folder)):
        subfolder_path = os.path.join(output_folder, subfolder)
        csv_file = os.path.join(subfolder_path, "sustainable_yield_summary.csv")

        if not (os.path.isdir(subfolder_path) and os.path.exists(csv_file)):
            continue

        # Read yield data
        df = pd.read_csv(csv_file)

        # Get parameter value corresponding to this subfolder
        col_name = par_val_df.loc[parameter, subfolder]

        # Store only required columns with renamed parameter value
        results.append(
            df[["Planning_Horizon", "Sustainable_Yield"]]
            .rename(columns={"Sustainable_Yield": col_name})
        )

    # --- Merge all results efficiently ---
    final_results = pd.concat(results, axis=1)
    final_results = final_results.loc[:, ~final_results.columns.duplicated()]  # remove duplicate Planning_Horizon

    # --- Plotting ---
    param_cols = sorted([c for c in final_results.columns if c != "Planning_Horizon"], key=float)

    plt.figure(figsize=(10, 6))

    # Drop rows with missing values in parameter columns
    complete_rows = final_results.dropna(subset=param_cols)

    for _, row in complete_rows.iterrows():
        plt.plot(
            [float(c) for c in param_cols],
            row[param_cols],
            label=f"{row['Planning_Horizon']} years"
        )

    plt.xscale("log")
    plt.xlabel("Parameter value")
    plt.ylabel("Sustainable Yield [m³/day]")
    plt.title("Yield vs Parameter values")
    plt.legend(title="Planning Horizon")
    plt.grid(True)
    plt.tight_layout()

    # --- Save outputs ---
    plot_path = os.path.join(output_folder, "yield_vs_parameter.png")
    csv_path = os.path.join(output_folder, "parameter_results.csv")

    plt.savefig(plot_path, dpi=300)
    final_results.to_csv(csv_path, index=False)

    print(f"Saved plot to {plot_path}")
    print(f"Saved dataframe to {csv_path}")

end_time = time.time()
print(f"Total execution time: {end_time - start_time:.2f} seconds")

