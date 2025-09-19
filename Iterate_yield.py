import time
start_time = time.time()
import shutil
import os
import subprocess
import pandas as pd
import sys

# Import local modules
sys.path.append('..')
from mlibs import modpump6 # type: ignore

# --------------------------------------------------------------------------------------- #
# ------------------------------- USER INPUTS ------------------------------------------- #
# --------------------------------------------------------------------------------------- #

setup_file = "C:/Users/cmarinriver/Projects/ConfinedLab/setup.xlsx" # Absolute paths
model_file = "C:/Users/cmarinriver/Projects/ConfinedLab/Model.py" # Absolute paths

mlibs_path = "C:/Users/cmarinriver/Projects/ConfinedLab" # Absolute paths

output_dir = "C:/Users/cmarinriver/Projects/ConfinedLab/sust_yield_results" # Absolute paths
plot_dir = os.path.join(output_dir, "plots")
os.makedirs(output_dir, exist_ok=True)
os.makedirs(plot_dir, exist_ok=True)

# Define model workspace name subscript (used to taylor output paths)
model_ws_name = "mf" 

# Set output file basenames as written by the model output (not full paths)
model_name = 'DEESACt'
budget_file_name = f"{model_name}_budget.csv"
zonebud_file_name = "zonebud.csv"
head_file_name = "head_obs_t.csv"

# --------------------------------------------------------------------------------------- #
# ------------------------------- PREPARE ITERATION FILE -------------------------------- #
# --------------------------------------------------------------------------------------- #

folder, filename = os.path.split(model_file)
name, ext = os.path.splitext(filename)
new_filename = f"{name}_it{ext}"
new_file_path = os.path.join(folder, new_filename)

# Copy the original file
shutil.copy(model_file, new_file_path)

# Read and modify the new file with placeholders
with open(new_file_path, "r") as f:
    lines = f.readlines()

setup_replaced = False
for i, line in enumerate(lines):
    if "sys.path.append('..')" in line:
        lines[i] = f"sys.path.append(r'{mlibs_path}')\n"
    if not setup_replaced and line.strip().startswith("setup_file ="):
        lines[i] = f"setup_file = r'{setup_file}' # Excel file containing model setup parameters\n"
        setup_replaced = True

# Save back the modified file
with open(new_file_path, "w") as f:
    f.writelines(lines)
print("Iteration script created at:", new_file_path)

# --------------------------------------------------------------------------------------- #
# ------------------------------- UPDATE AND RUN MODEL ---------------------------------- #
# --------------------------------------------------------------------------------------- #

# Load setup
# file contining the pumping rates used by modflow6
well_df = pd.read_excel(setup_file, sheet_name="wells")

# q-values for each iteration
q_df = pd.read_excel(setup_file, sheet_name="q_values_tr")       

# Identify iteration columns (all except well_id + time)
iter_cols = [c for c in q_df.columns if c not in ["well_id", "time", "comment"]]
n_iterations = len(iter_cols)

for i, col in enumerate(iter_cols, start=1):

    print(f"\n--- Running iteration {i}/{n_iterations} with {col} ---")

    # ------------------------------------------------------------------------------------- #
    # -------------------------------- PREPARE SETUP FILE --------------------------------- #
    # ------------------------------------------------------------------------------------- #

    # Merge well_df with the selected q column
    merged = well_df.drop(columns=["q"]).merge(
        q_df[["well_id", "time", col]],
        on=["well_id", "time"],
        how="left")
    
    # Rename current iteration q value column to "q"
    merged = merged.rename(columns={col: "q"})

    # Write updated wells sheet back to Excel (overwrite only that sheet)
    with pd.ExcelWriter(setup_file, mode="a", if_sheet_exists="replace") as writer:
        merged.to_excel(writer, sheet_name="wells", index=False)

    # --- Run your model here ---
    # run_model(setup_file)

    # --- Optionally, save results tagged by iteration ---
    # save_results(iteration=i)

    # --------------------------------------------------------------------------------------- #
    # ------------------------------- ITERATE MODEL ----------------------------------------- #
    # --------------------------------------------------------------------------------------- #

    # Create a unique model workspace directory name based on the parameter value
    model_ws = os.path.join(output_dir, f"{model_ws_name}_it_{col}")
    
    # Create the directory for model_ws if it doesn't exist
    os.makedirs(model_ws, exist_ok=True)
    
    # Copy the iteration script into the unique model workspace folder and get the path
    shutil.copy(new_file_path, model_ws)
    script_path = os.path.join(model_ws, new_filename)
    
    # Run the script inside the unique model workspace folder
    # You can use subprocess to execute the script in that directory
    subprocess.run(["python", script_path], cwd=model_ws)

    print(f"Model run completed for iteration {i} with q={col}, model_ws={model_ws}")


# --------------------------------------------------------------------------------------- #
# ------------------------------- MANAGE OUTPUT FILES ----------------------------------- #
# --------------------------------------------------------------------------------------- #

# Define a destination directory to summarize results
results_folder = os.path.join(output_dir, "Summary_iterations")
os.makedirs(results_folder, exist_ok=True)

# Loop through the sub-folders in the output directory to get relevant files
for folder_name in os.listdir(output_dir):
    # Check if the folder matches the pattern "model_ws_name_it_xxxx"
    if folder_name.startswith(f"{model_ws_name}_it_"):
        folder_path = os.path.join(output_dir, folder_name)
        mf_path = os.path.join(folder_path, model_ws_name, "output")

        # Only proceed if the unique model workspace subfolder exists
        if os.path.exists(mf_path):
            # Extract the "parameter_xxxx" part from the folder name
            code = folder_name.split(f"{model_ws_name}_")[1]

            # Define the source files
            budget_file = os.path.join(mf_path, budget_file_name)
            zonebud_file = os.path.join(mf_path, zonebud_file_name)
            head_obs_file = os.path.join(mf_path, head_file_name)

            # Define the destination files
            budget_dest = os.path.join(results_folder, f"{os.path.splitext(budget_file_name)[0]}_{code}.csv")
            zonebud_dest = os.path.join(results_folder, f"{os.path.splitext(zonebud_file_name)[0]}_{code}.csv")
            head_obs_dest = os.path.join(results_folder, f"{os.path.splitext(head_file_name)[0]}_{code}.csv")

            # Copy files if they exist
            if os.path.exists(budget_file):
                shutil.copy(budget_file, budget_dest)
                print(f"Copied {budget_file} to {budget_dest}")

            if os.path.exists(zonebud_file):
                shutil.copy(zonebud_file, zonebud_dest)
                print(f"Copied {zonebud_file} to {zonebud_dest}")

            if os.path.exists(head_obs_file):
                shutil.copy(head_obs_file, head_obs_dest)
                print(f"Copied {head_obs_file} to {head_obs_dest}")

end_time = time.time()
print(f"Total execution time: {end_time - start_time:.2f} seconds")

