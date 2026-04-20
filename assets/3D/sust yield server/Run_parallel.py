import os
import shutil
import subprocess
import pandas as pd
from openpyxl import load_workbook
from concurrent.futures import ProcessPoolExecutor

# -------------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------------
BASE_DIR = os.path.abspath(os.path.dirname(__file__))   # ConfinedLab
SETUP_FILE = os.path.join(BASE_DIR, "setup.xlsx")

SCRIPTS_TO_COPY = [
    "Sustainable_yield.py",
    "Model.py",
    # "Parameter_analysis.py"
]

SCRIPT_TO_RUN = "Sustainable_yield.py"  # The script to run in each case

PREFIX = "kv_" # Prefix for parameter value columns in parameter_analysis sheet

# -------------------------------------------------------------
# LOAD PARAMETER SHEETS
# -------------------------------------------------------------
df_par_sets = pd.read_excel(SETUP_FILE, sheet_name="parameter_analysis")
df_parameters = pd.read_excel(SETUP_FILE, sheet_name="parameters")

par_names = df_par_sets["par_name"].tolist()
parv_columns = [col for col in df_par_sets.columns if col.startswith(PREFIX)]

print(f"Found parameter sets: {parv_columns}")

# -------------------------------------------------------------
# PREPARE CASE (RUNS SEQUENTIALLY)
# -------------------------------------------------------------
def prepare_case(parv_col):

    case_dir = os.path.join(BASE_DIR, parv_col)
    os.makedirs(case_dir, exist_ok=True)

    print(f"\n=== Preparing case: {parv_col} ===")

    # 1. Copy scripts
    for script in SCRIPTS_TO_COPY:
        shutil.copy(
            os.path.join(BASE_DIR, script),
            os.path.join(case_dir, script)
        )

    # 2. Copy setup.xlsx
    setup_out = os.path.join(case_dir, "setup.xlsx")
    shutil.copy(SETUP_FILE, setup_out)

    # ---------------------------------------------------------
    # Replace "parameters" sheet using Pandas
    # ---------------------------------------------------------

    # Match by par_name
    # For each parameter name in parameters sheet: replace its value
    for idx, row in df_parameters.iterrows():
        pname = row["par_name"]

        # Only replace if parameter exists in parameter_analysis
        if pname in df_par_sets["par_name"].values:
            new_value = df_par_sets.loc[df_par_sets["par_name"] == pname, parv_col].values[0]
            df_parameters.at[idx, "value"] = new_value

    # ---------------------------------------------------------
    # Update pumping rates for initial iteration
    # ---------------------------------------------------------
    #q_values_init contains the pumping rates for the initial model run that is performed in the 
    #sustainable_yield.py script. This run is meant to create the model files that are needed
    #for subsequent iterations when using the . These pumping rates are adapted for the different parameter sets. 

    df_qinit = pd.read_excel(SETUP_FILE, sheet_name="q_values_init")

    # Get q_init for this case (parv_col == folder name)
    q_init_row = df_qinit.loc[df_qinit["par_name"] == parv_col]

    if q_init_row.empty:
        raise ValueError(f"No q_init found for {parv_col}")

    q_init = q_init_row["q_init"].values[0]

    # ---------------------------------------------------------
    # 1. Update wells_st: all q = q_init
    # ---------------------------------------------------------
    df_wells_st = pd.read_excel(SETUP_FILE, sheet_name="wells_st")
    df_wells_st["q"] = q_init

    # ---------------------------------------------------------
    # 2. Update wells: q = q_init only where time != 0
    # ---------------------------------------------------------
    df_wells = pd.read_excel(SETUP_FILE, sheet_name="wells")

    mask = df_wells["time"] != 0
    df_wells.loc[mask, "q"] = q_init

    # ---------------------------------------------------------
    # Write the updated "parameters" sheet back to the file
    # ---------------------------------------------------------
    with pd.ExcelWriter(setup_out, mode="a", if_sheet_exists="replace") as writer:
        df_parameters.to_excel(writer, sheet_name="parameters", index=False)
        df_wells_st.to_excel(writer, sheet_name="wells_st", index=False)
        df_wells.to_excel(writer, sheet_name="wells", index=False)
    return case_dir


# -------------------------------------------------------------
# RUN CASE (PROCESSPOOL EXECUTOR)
# -------------------------------------------------------------
def run_case(parv_col):

    case_dir = os.path.join(BASE_DIR, parv_col)
    print(f"Running simulation for {parv_col}")

    # IMPORTANT: Use subprocess.run, NOT subprocess.Popen
    subprocess.run(
        ["python", SCRIPT_TO_RUN],
        cwd=case_dir)

    return f"{parv_col} completed."


# -------------------------------------------------------------
# PARALLEL RUN USING PROCESS POOL EXECUTOR
# -------------------------------------------------------------
if __name__ == "__main__":

    print("\n=== Preparing all cases ===")
    for parv in parv_columns:
        prepare_case(parv)

    print("\n=== Running all cases in parallel ===")
    num_workers = min(len(parv_columns), os.cpu_count())
    print(f"Using {num_workers} workers.\n")

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(run_case, parv_columns))

    print("\n=== All parallel runs completed ===")
    for r in results:
        print(r)
