import time
start_time = time.time()
import shutil
import os
import subprocess

# --------------------------------------------------------------------------------------- #
# ------------------------------- USER INPUTS ------------------------------------------- #
# --------------------------------------------------------------------------------------- #

# Input the original model script file absolute paths
model_file = "C:/Users/cmarinriver/Projects/ConfinedLab/Model.py"
mlibs_path = "C:/Users/cmarinriver/Projects/ConfinedLab"

output_dir = "C:/Users/cmarinriver/Projects/ConfinedLab/sust_yield_results"

# Define model workspace name subscript (used to taylor output paths)
model_ws_name = "mf" 

# Set output file basenames as written by the model output (not full paths)
model_name = 'DEESAC'
budget_file_name = f"{model_name}t_budget.csv"
zonebud_file_name = "zonebud.csv"
head_file_name = "head_obs_t.csv"

# Values of parameter to iterate over
q_values = [-0, -2, -4, -8, -12, -16, -20]
parameter = "q"


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

q_replaced = False
for i, line in enumerate(lines):
    if "sys.path.append('..')" in line:
        lines[i] = f"sys.path.append(r'{mlibs_path}')\n"
    if not q_replaced and line.strip().startswith("q ="):
        lines[i] = "q = {Q_VALUE} # Pumping rate in m3/d\n"
        q_replaced = True

# Save back the modified file
with open(new_file_path, "w") as f:
    f.writelines(lines)
print("Iteration script created at:", new_file_path)


# --------------------------------------------------------------------------------------- #
# ------------------------------- ITERATE MODEL ----------------------------------------- #
# --------------------------------------------------------------------------------------- #

# Iterate over the k_values
for q in q_values:
    # Create a unique model workspace directory name based on the parameter value
    model_ws = os.path.join(output_dir, f"{model_ws_name}_{parameter}_{q}")
    
    # Create the directory for model_ws if it doesn't exist
    os.makedirs(model_ws, exist_ok=True)
    
    # Copy the iteration script into the unique model workspace folder and get the path
    shutil.copy(new_file_path, model_ws)
    script_path = os.path.join(model_ws, new_filename)
    
    # Replace placeholder with the desired pumping rate
    with open(script_path, 'r') as file:
        script_content = file.read()
    script_content = script_content.replace("{Q_VALUE}", str(q))

    # Replace {MLIBS_PATH} with the absolute path to 'mlibs'
    script_content = script_content.replace("{MLIBS_PATH}", mlibs_path)
    
    with open(script_path, 'w') as file:
        file.write(script_content)
    
    # Run the script inside the unique model workspace folder
    # You can use subprocess to execute the script in that directory
    subprocess.run(["python", script_path], cwd=model_ws)

    print(f"Model run completed for {parameter}={q}, model_ws={model_ws}")


# --------------------------------------------------------------------------------------- #
# ------------------------------- MANAGE OUTPUT FILES ----------------------------------- #
# --------------------------------------------------------------------------------------- #

# Define a destination directory to summarize results
results_folder = os.path.join(output_dir, "Summary_Results")
os.makedirs(results_folder, exist_ok=True)

# Loop through the sub-folders in the output directory to get relevant files
for folder_name in os.listdir(output_dir):
    # Check if the folder matches the pattern "model_ws_name_parameter_xxxx"
    if folder_name.startswith(f"{model_ws_name}_{parameter}_"):
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