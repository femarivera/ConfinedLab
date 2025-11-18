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
    "Parameter_analysis.py"
]

# -------------------------------------------------------------
# LOAD PARAMETER SHEETS ONCE
# -------------------------------------------------------------
df_par_sets = pd.read_excel(SETUP_FILE, sheet_name="parameter_analysis")
df_parameters = pd.read_excel(SETUP_FILE, sheet_name="parameters")

par_names = df_par_sets["par_name"].tolist()
parv_columns = [col for col in df_par_sets.columns if col.startswith("parv_")]

print(f"Found parameter sets: {parv_columns}")


# -------------------------------------------------------------
# FUNCTION 1 — PREPARE CASE (RUNS SEQUENTIALLY)
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
    # Write the updated "parameters" sheet back to the file
    # ---------------------------------------------------------
    with pd.ExcelWriter(setup_out, mode="a", if_sheet_exists="replace") as writer:
        df_parameters.to_excel(writer, sheet_name="parameters", index=False)

    return case_dir


# -------------------------------------------------------------
# FUNCTION 2 — RUN CASE (PROCESSPOOL EXECUTOR)
# -------------------------------------------------------------
def run_case(parv_col):

    case_dir = os.path.join(BASE_DIR, parv_col)
    print(f"Running sustainable_yield.py for {parv_col}")

    # IMPORTANT: Use subprocess.run, NOT subprocess.Popen
    subprocess.run(
        ["python", "Sustainable_yield.py"],
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
