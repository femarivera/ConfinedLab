import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
sys.path.append('C:/Users/cmarinriver/Models')
from mlibs import modplot6, modtransient6, modpump6, modgeom6, modbound6 # type: ignore

#Iterated parameter 
parameter = "q"
parameter_name = "Pumping rate [m$^3$/day]"
recharge = -40
leakage_threshold = recharge

# Define paths
input_folder = "C:/Users/cmarinriver/Projects/ConfinedLab/sust_yield_results/Summary_Results"
output_folder = "C:/Users/cmarinriver/Projects/ConfinedLab/sust_yield_results/Plots"
os.makedirs(output_folder, exist_ok=True)

model_name = "DEESACt"
plt.rcParams.update({'font.size': 16})  # Adjust the number to increase/decrease font size
plt.rcParams["figure.figsize"] = (14, 12)  # Set default figure size

#################################### SUSTAINABLE YIELD ################################################3
# Function to find where a curve crosses a given threshold
def find_threshold_crossing(x, y, threshold=0):
    for i in range(len(y) - 1):
        if (y[i] - threshold) * (y[i + 1] - threshold) < 0:  # Sign change detected
            x0, x1 = x[i], x[i + 1]
            y0, y1 = y[i] - threshold, y[i + 1] - threshold  # Shift y by threshold
            return x0 - y0 * (x1 - x0) / (y1 - y0)  # Linear interpolation
    return None  # No crossing found

########################## 10 YEARS
# Set the desired time value
time_target = 6003000
# Prepare the plot
fig, ax = plt.subplots(figsize=(14, 12))
# List to store extracted data for both curves
net_leakageCaq_data = []
net_leakageUnc_data = []
net_leakageCaq2_data = []
outflow_data = []
drn_out_data = []
# Loop through all zonebud_.csv files in the input folder
for file_name in os.listdir(input_folder):
    if file_name.startswith("zonebud_") and file_name.endswith(".csv"):
        # Extract the numeric code
        code = file_name.split(f"zonebud_{parameter}")[1].split(".csv")[0][1:]
        try:
            code = float(code)  # Convert to number for sorting
        except ValueError:
            continue  # Skip if conversion fails
        # Read the CSV file
        file_path = os.path.join(input_folder, file_name)
        data = pd.read_csv(file_path)
        # Process DRN-OUT for all zones combined
        if 'totim' in data.columns and "RIV-OUT" in data.columns and "RIV-IN" in data.columns:
            # Find the row(s) closest to the target time
            time_diff = abs(data['totim'] - time_target)
            closest_idx = time_diff.idxmin()
            closest_time = data.loc[closest_idx, 'totim']
            # Filter all rows at the closest time step
            time_filtered = data[data['totim'] == closest_time]
            total_drn_out = time_filtered["RIV-OUT"].sum() - time_filtered["RIV-IN"].sum()
            code_value = -code  # Keep it consistent with your x-axis
            drn_out_data.append((code_value, total_drn_out))
        # Extract data for zone 3
        zone_data = data[data['zone'] == 3]
        # Process Net leakage to confined aquifer from above
        if 'totim' in data.columns and "TO ZONE 2" in data.columns and "FROM ZONE 2" in data.columns and "TO ZONE 4" in data.columns and "FROM ZONE 4" in data.columns:
            zone_data["Net Flow Caq"] = zone_data["TO ZONE 2"] + zone_data["TO ZONE 4"] - zone_data["FROM ZONE 2"] - zone_data["FROM ZONE 4"]
            time_diff = abs(zone_data['totim'] - time_target)
            closest_idx = time_diff.idxmin()  # Get the index of the closest time
            # Extract the corresponding value at this time step for Net Leakage
            code_value = -code  # Multiply code by -1 to make it positive
            net_flow_value = zone_data.loc[closest_idx, "Net Flow Caq"]
            net_leakageCaq_data.append((code_value, net_flow_value))
        # Process Net leakage to confined aquifer
        if 'totim' in data.columns and "TO ZONE 4" in data.columns and "FROM ZONE 4" in data.columns:
            zone_data["Net Flow Caq2"] = zone_data["FROM ZONE 4"] - zone_data["TO ZONE 4"]
            time_diff = abs(zone_data['totim'] - time_target)
            closest_idx = time_diff.idxmin()  # Get the index of the closest time
            # Extract the corresponding value at this time step for Net Leakage
            code_value = -code  # Multiply code by -1 to make it positive
            net_flow_value = zone_data.loc[closest_idx, "Net Flow Caq2"]
            net_leakageCaq2_data.append((code_value, net_flow_value))
        # Process Outflow Confined (GHB-OUT - GHB-IN)
        column_name_out = "GHB-OUT"
        column_name_in = "GHB-IN"
        if 'totim' in data.columns and column_name_out in data.columns and column_name_in in data.columns:
            time_diff = abs(zone_data['totim'] - time_target)
            closest_idx = time_diff.idxmin()  # Get the index of the closest time
            # Extract the corresponding values at this time step for GHB-OUT and GHB-IN
            ghb_out_value = zone_data.loc[closest_idx, column_name_out]
            ghb_in_value = zone_data.loc[closest_idx, column_name_in]
            ghb_diff_value = ghb_out_value - ghb_in_value  # Calculate the difference
            outflow_data.append((code_value, ghb_diff_value))  # Store the difference
# Sort the plot data based on the modified code values (multiplied by -1)
drn_out_data.sort(key=lambda x: x[0])
net_leakageCaq_data.sort(key=lambda x: x[0])
outflow_data.sort(key=lambda x: x[0])
# Extract sorted code values and corresponding net flow and GHB-OUT values
drn_codes, drn_out_values = zip(*drn_out_data)
net_codesCaq, net_flowsCaq = zip(*net_leakageCaq_data)
outflow_codes, ghb_out_values = zip(*outflow_data)
# Plot both curves
ax.plot(drn_codes, drn_out_values, marker='s', linestyle='-', color='orange', label="Discharge to river network")
ax.plot(net_codesCaq, net_flowsCaq, marker='o', linestyle='-', color='b', label="Net leakage to confined aquifer")
ax.plot(outflow_codes, ghb_out_values, marker='x', linestyle='-', color='r', label="Lateral outflow from confined aquifer")

initial_drn_value = drn_out_values[0]
if initial_drn_value != 0:
    threshold_drn = initial_drn_value - 0.1 * abs(initial_drn_value)
else:
    threshold_drn = -0.1

# Add a horizontal line at thresholds
ax.axhline(0, color='r', linewidth=1, linestyle = '--')
ax.axhline(leakage_threshold, color='b', linewidth=1, linestyle = '--')
ax.axhline(threshold_drn, color='orange', linewidth=1, linestyle = '--')
# Find the threshold crossing (zero) for both curves
zero_cross_drn = find_threshold_crossing(drn_codes, drn_out_values, threshold= threshold_drn)
zero_cross_net_leakageCaq = find_threshold_crossing(net_codesCaq, net_flowsCaq, threshold=leakage_threshold)
zero_cross_outflow = find_threshold_crossing(outflow_codes, ghb_out_values, threshold=0)
# Select the smaller threshold crossing value
qs_candidates = list(filter(lambda x: x is not None, [
    zero_cross_drn,
    zero_cross_net_leakageCaq,
    zero_cross_outflow
]))

if qs_candidates:
    qs_value = min(qs_candidates)
    # Annotate Qs on the plot
    ax.axvline(qs_value, color='g', linestyle='--', linewidth=1)
    ax.text(qs_value, max(max(drn_out_values),
                          max(net_flowsCaq), 
                          max(ghb_out_values)),
            f"Qs < {qs_value:.2f} m³/day", color='g', fontsize=14, ha='right',
            bbox=dict(facecolor='white', alpha=0.7))
# Customize the plot
ax.set_title("Sustainable yield estimation - 10 years after pumping")
ax.set_xlabel("Pumping rate Rate [m³/day]")
ax.set_ylabel("Flow Rate [m³/day]")
ax.grid(True)
# Add a legend
ax.legend()
# Save the plot
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "Q vs flow tp010.png"), bbox_inches='tight')

########################## 25 YEARS
# Set the desired time value
time_target = 6007500
# Prepare the plot
fig, ax = plt.subplots(figsize=(14, 12))
# List to store extracted data for both curves
net_leakageCaq_data = []
net_leakageUnc_data = []
net_leakageCaq2_data = []
outflow_data = []
drn_out_data = []
# Loop through all zonebud_.csv files in the input folder
for file_name in os.listdir(input_folder):
    if file_name.startswith("zonebud_") and file_name.endswith(".csv"):
        # Extract the numeric code
        code = file_name.split(f"zonebud_{parameter}")[1].split(".csv")[0][1:]
        try:
            code = float(code)  # Convert to number for sorting
        except ValueError:
            continue  # Skip if conversion fails
        # Read the CSV file
        file_path = os.path.join(input_folder, file_name)
        data = pd.read_csv(file_path)
        # Process DRN-OUT for all zones combined
        if 'totim' in data.columns and "RIV-OUT" in data.columns and "RIV-IN" in data.columns:
            # Find the row(s) closest to the target time
            time_diff = abs(data['totim'] - time_target)
            closest_idx = time_diff.idxmin()
            closest_time = data.loc[closest_idx, 'totim']
            # Filter all rows at the closest time step
            time_filtered = data[data['totim'] == closest_time]
            total_drn_out = time_filtered["RIV-OUT"].sum() - time_filtered["RIV-IN"].sum()
            code_value = -code  # Keep it consistent with your x-axis
            drn_out_data.append((code_value, total_drn_out))
        # Extract data for zone 3
        zone_data = data[data['zone'] == 3]
        # Process Net leakage to confined aquifer from above
        if 'totim' in data.columns and "TO ZONE 2" in data.columns and "FROM ZONE 2" in data.columns and "TO ZONE 4" in data.columns and "FROM ZONE 4" in data.columns:
            zone_data["Net Flow Caq"] = zone_data["TO ZONE 2"] + zone_data["TO ZONE 4"] - zone_data["FROM ZONE 2"] - zone_data["FROM ZONE 4"]
            time_diff = abs(zone_data['totim'] - time_target)
            closest_idx = time_diff.idxmin()  # Get the index of the closest time
            # Extract the corresponding value at this time step for Net Leakage
            code_value = -code  # Multiply code by -1 to make it positive
            net_flow_value = zone_data.loc[closest_idx, "Net Flow Caq"]
            net_leakageCaq_data.append((code_value, net_flow_value))
        # Process Net leakage to confined aquifer
        if 'totim' in data.columns and "TO ZONE 4" in data.columns and "FROM ZONE 4" in data.columns:
            zone_data["Net Flow Caq2"] = zone_data["FROM ZONE 4"] - zone_data["TO ZONE 4"]
            time_diff = abs(zone_data['totim'] - time_target)
            closest_idx = time_diff.idxmin()  # Get the index of the closest time
            # Extract the corresponding value at this time step for Net Leakage
            code_value = -code  # Multiply code by -1 to make it positive
            net_flow_value = zone_data.loc[closest_idx, "Net Flow Caq2"]
            net_leakageCaq2_data.append((code_value, net_flow_value))
        # Process Outflow Confined (GHB-OUT - GHB-IN)
        column_name_out = "GHB-OUT"
        column_name_in = "GHB-IN"
        if 'totim' in data.columns and column_name_out in data.columns and column_name_in in data.columns:
            time_diff = abs(zone_data['totim'] - time_target)
            closest_idx = time_diff.idxmin()  # Get the index of the closest time
            # Extract the corresponding values at this time step for GHB-OUT and GHB-IN
            ghb_out_value = zone_data.loc[closest_idx, column_name_out]
            ghb_in_value = zone_data.loc[closest_idx, column_name_in]
            ghb_diff_value = ghb_out_value - ghb_in_value  # Calculate the difference
            outflow_data.append((code_value, ghb_diff_value))  # Store the difference
# Sort the plot data based on the modified code values (multiplied by -1)
drn_out_data.sort(key=lambda x: x[0])
net_leakageCaq_data.sort(key=lambda x: x[0])
outflow_data.sort(key=lambda x: x[0])
# Extract sorted code values and corresponding net flow and GHB-OUT values
drn_codes, drn_out_values = zip(*drn_out_data)
net_codesCaq, net_flowsCaq = zip(*net_leakageCaq_data)
outflow_codes, ghb_out_values = zip(*outflow_data)
# Plot both curves
ax.plot(drn_codes, drn_out_values, marker='s', linestyle='-', color='orange', label="Discharge to river network")
ax.plot(net_codesCaq, net_flowsCaq, marker='o', linestyle='-', color='b', label="Net leakage to confined aquifer")
ax.plot(outflow_codes, ghb_out_values, marker='x', linestyle='-', color='r', label="Lateral outflow from confined aquifer")

initial_drn_value = drn_out_values[0]
if initial_drn_value != 0:
    threshold_drn = initial_drn_value - 0.1 * abs(initial_drn_value)
else:
    threshold_drn = -0.1

# Add a horizontal line at thresholds
ax.axhline(0, color='r', linewidth=1, linestyle = '--')
ax.axhline(leakage_threshold, color='b', linewidth=1, linestyle = '--')
ax.axhline(threshold_drn, color='orange', linewidth=1, linestyle = '--')
# Find the threshold crossing (zero) for both curves
zero_cross_drn = find_threshold_crossing(drn_codes, drn_out_values, threshold= threshold_drn)
zero_cross_net_leakageCaq = find_threshold_crossing(net_codesCaq, net_flowsCaq, threshold=leakage_threshold)
zero_cross_outflow = find_threshold_crossing(outflow_codes, ghb_out_values, threshold=0)
# Select the smaller threshold crossing value
qs_candidates = list(filter(lambda x: x is not None, [
    zero_cross_drn,
    zero_cross_net_leakageCaq,
    zero_cross_outflow
]))

if qs_candidates:
    qs_value = min(qs_candidates)
    # Annotate Qs on the plot
    ax.axvline(qs_value, color='g', linestyle='--', linewidth=1)
    ax.text(qs_value, max(max(drn_out_values),
                          max(net_flowsCaq), 
                          max(ghb_out_values)),
            f"Qs < {qs_value:.2f} m³/day", color='g', fontsize=14, ha='right',
            bbox=dict(facecolor='white', alpha=0.7))
# Customize the plot
ax.set_title("Sustainable yield estimation - 25 years after pumping")
ax.set_xlabel("Pumping rate Rate [m³/day]")
ax.set_ylabel("Flow Rate [m³/day]")
ax.grid(True)
# Add a legend
ax.legend()
# Save the plot
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "Q vs flow tp025.png"), bbox_inches='tight')

########################## 50 YEARS
# Set the desired time value
time_target = 6015000
# Prepare the plot
fig, ax = plt.subplots(figsize=(14, 12))
# List to store extracted data for both curves
net_leakageCaq_data = []
net_leakageUnc_data = []
net_leakageCaq2_data = []
outflow_data = []
drn_out_data = []
# Loop through all zonebud_.csv files in the input folder
for file_name in os.listdir(input_folder):
    if file_name.startswith("zonebud_") and file_name.endswith(".csv"):
        # Extract the numeric code
        code = file_name.split(f"zonebud_{parameter}")[1].split(".csv")[0][1:]
        try:
            code = float(code)  # Convert to number for sorting
        except ValueError:
            continue  # Skip if conversion fails
        # Read the CSV file
        file_path = os.path.join(input_folder, file_name)
        data = pd.read_csv(file_path)
        # Process DRN-OUT for all zones combined
        if 'totim' in data.columns and "RIV-OUT" in data.columns and "RIV-IN" in data.columns:
            # Find the row(s) closest to the target time
            time_diff = abs(data['totim'] - time_target)
            closest_idx = time_diff.idxmin()
            closest_time = data.loc[closest_idx, 'totim']
            # Filter all rows at the closest time step
            time_filtered = data[data['totim'] == closest_time]
            total_drn_out = time_filtered["RIV-OUT"].sum() - time_filtered["RIV-IN"].sum()
            code_value = -code  # Keep it consistent with your x-axis
            drn_out_data.append((code_value, total_drn_out))
        # Extract data for zone 3
        zone_data = data[data['zone'] == 3]
        # Process Net leakage to confined aquifer from above
        if 'totim' in data.columns and "TO ZONE 2" in data.columns and "FROM ZONE 2" in data.columns and "TO ZONE 4" in data.columns and "FROM ZONE 4" in data.columns:
            zone_data["Net Flow Caq"] = zone_data["TO ZONE 2"] + zone_data["TO ZONE 4"] - zone_data["FROM ZONE 2"] - zone_data["FROM ZONE 4"]
            time_diff = abs(zone_data['totim'] - time_target)
            closest_idx = time_diff.idxmin()  # Get the index of the closest time
            # Extract the corresponding value at this time step for Net Leakage
            code_value = -code  # Multiply code by -1 to make it positive
            net_flow_value = zone_data.loc[closest_idx, "Net Flow Caq"]
            net_leakageCaq_data.append((code_value, net_flow_value))
        # Process Net leakage to confined aquifer
        if 'totim' in data.columns and "TO ZONE 4" in data.columns and "FROM ZONE 4" in data.columns:
            zone_data["Net Flow Caq2"] = zone_data["FROM ZONE 4"] - zone_data["TO ZONE 4"]
            time_diff = abs(zone_data['totim'] - time_target)
            closest_idx = time_diff.idxmin()  # Get the index of the closest time
            # Extract the corresponding value at this time step for Net Leakage
            code_value = -code  # Multiply code by -1 to make it positive
            net_flow_value = zone_data.loc[closest_idx, "Net Flow Caq2"]
            net_leakageCaq2_data.append((code_value, net_flow_value))
        # Process Outflow Confined (GHB-OUT - GHB-IN)
        column_name_out = "GHB-OUT"
        column_name_in = "GHB-IN"
        if 'totim' in data.columns and column_name_out in data.columns and column_name_in in data.columns:
            time_diff = abs(zone_data['totim'] - time_target)
            closest_idx = time_diff.idxmin()  # Get the index of the closest time
            # Extract the corresponding values at this time step for GHB-OUT and GHB-IN
            ghb_out_value = zone_data.loc[closest_idx, column_name_out]
            ghb_in_value = zone_data.loc[closest_idx, column_name_in]
            ghb_diff_value = ghb_out_value - ghb_in_value  # Calculate the difference
            outflow_data.append((code_value, ghb_diff_value))  # Store the difference
# Sort the plot data based on the modified code values (multiplied by -1)
drn_out_data.sort(key=lambda x: x[0])
net_leakageCaq_data.sort(key=lambda x: x[0])
outflow_data.sort(key=lambda x: x[0])
# Extract sorted code values and corresponding net flow and GHB-OUT values
drn_codes, drn_out_values = zip(*drn_out_data)
net_codesCaq, net_flowsCaq = zip(*net_leakageCaq_data)
outflow_codes, ghb_out_values = zip(*outflow_data)
# Plot both curves
ax.plot(drn_codes, drn_out_values, marker='s', linestyle='-', color='orange', label="Discharge to river network")
ax.plot(net_codesCaq, net_flowsCaq, marker='o', linestyle='-', color='b', label="Net leakage to confined aquifer")
ax.plot(outflow_codes, ghb_out_values, marker='x', linestyle='-', color='r', label="Lateral outflow from confined aquifer")

initial_drn_value = drn_out_values[0]
if initial_drn_value != 0:
    threshold_drn = initial_drn_value - 0.1 * abs(initial_drn_value)
else:
    threshold_drn = -0.1

# Add a horizontal line at thresholds
ax.axhline(0, color='r', linewidth=1, linestyle = '--')
ax.axhline(leakage_threshold, color='b', linewidth=1, linestyle = '--')
ax.axhline(threshold_drn, color='orange', linewidth=1, linestyle = '--')
# Find the threshold crossing (zero) for both curves
zero_cross_drn = find_threshold_crossing(drn_codes, drn_out_values, threshold= threshold_drn)
zero_cross_net_leakageCaq = find_threshold_crossing(net_codesCaq, net_flowsCaq, threshold=leakage_threshold)
zero_cross_outflow = find_threshold_crossing(outflow_codes, ghb_out_values, threshold=0)
# Select the smaller threshold crossing value
qs_candidates = list(filter(lambda x: x is not None, [
    zero_cross_drn,
    zero_cross_net_leakageCaq,
    zero_cross_outflow
]))

if qs_candidates:
    qs_value = min(qs_candidates)
    # Annotate Qs on the plot
    ax.axvline(qs_value, color='g', linestyle='--', linewidth=1)
    ax.text(qs_value, max(max(drn_out_values),
                          max(net_flowsCaq), 
                          max(ghb_out_values)),
            f"Qs < {qs_value:.2f} m³/day", color='g', fontsize=14, ha='right',
            bbox=dict(facecolor='white', alpha=0.7))
# Customize the plot
ax.set_title("Sustainable yield estimation - 50 years after pumping")
ax.set_xlabel("Pumping rate Rate [m³/day]")
ax.set_ylabel("Flow Rate [m³/day]")
ax.grid(True)
# Add a legend
ax.legend()
# Save the plot
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "Q vs flow tp050.png"), bbox_inches='tight')

########################## 100 YEARS
# Set the desired time value
time_target = 6030000
# Prepare the plot
fig, ax = plt.subplots(figsize=(14, 12))
# List to store extracted data for both curves
net_leakageCaq_data = []
net_leakageUnc_data = []
net_leakageCaq2_data = []
outflow_data = []
drn_out_data = []
# Loop through all zonebud_.csv files in the input folder
for file_name in os.listdir(input_folder):
    if file_name.startswith("zonebud_") and file_name.endswith(".csv"):
        # Extract the numeric code
        code = file_name.split(f"zonebud_{parameter}")[1].split(".csv")[0][1:]
        try:
            code = float(code)  # Convert to number for sorting
        except ValueError:
            continue  # Skip if conversion fails
        # Read the CSV file
        file_path = os.path.join(input_folder, file_name)
        data = pd.read_csv(file_path)
        # Process DRN-OUT for all zones combined
        if 'totim' in data.columns and "RIV-OUT" in data.columns and "RIV-IN" in data.columns:
            # Find the row(s) closest to the target time
            time_diff = abs(data['totim'] - time_target)
            closest_idx = time_diff.idxmin()
            closest_time = data.loc[closest_idx, 'totim']
            # Filter all rows at the closest time step
            time_filtered = data[data['totim'] == closest_time]
            total_drn_out = time_filtered["RIV-OUT"].sum() - time_filtered["RIV-IN"].sum()
            code_value = -code  # Keep it consistent with your x-axis
            drn_out_data.append((code_value, total_drn_out))
        # Extract data for zone 3
        zone_data = data[data['zone'] == 3]
        # Process Net leakage to confined aquifer from above
        if 'totim' in data.columns and "TO ZONE 2" in data.columns and "FROM ZONE 2" in data.columns and "TO ZONE 4" in data.columns and "FROM ZONE 4" in data.columns:
            zone_data["Net Flow Caq"] = zone_data["TO ZONE 2"] + zone_data["TO ZONE 4"] - zone_data["FROM ZONE 2"] - zone_data["FROM ZONE 4"]
            time_diff = abs(zone_data['totim'] - time_target)
            closest_idx = time_diff.idxmin()  # Get the index of the closest time
            # Extract the corresponding value at this time step for Net Leakage
            code_value = -code  # Multiply code by -1 to make it positive
            net_flow_value = zone_data.loc[closest_idx, "Net Flow Caq"]
            net_leakageCaq_data.append((code_value, net_flow_value))
        # Process Net leakage to confined aquifer
        if 'totim' in data.columns and "TO ZONE 4" in data.columns and "FROM ZONE 4" in data.columns:
            zone_data["Net Flow Caq2"] = zone_data["FROM ZONE 4"] - zone_data["TO ZONE 4"]
            time_diff = abs(zone_data['totim'] - time_target)
            closest_idx = time_diff.idxmin()  # Get the index of the closest time
            # Extract the corresponding value at this time step for Net Leakage
            code_value = -code  # Multiply code by -1 to make it positive
            net_flow_value = zone_data.loc[closest_idx, "Net Flow Caq2"]
            net_leakageCaq2_data.append((code_value, net_flow_value))
        # Process Outflow Confined (GHB-OUT - GHB-IN)
        column_name_out = "GHB-OUT"
        column_name_in = "GHB-IN"
        if 'totim' in data.columns and column_name_out in data.columns and column_name_in in data.columns:
            time_diff = abs(zone_data['totim'] - time_target)
            closest_idx = time_diff.idxmin()  # Get the index of the closest time
            # Extract the corresponding values at this time step for GHB-OUT and GHB-IN
            ghb_out_value = zone_data.loc[closest_idx, column_name_out]
            ghb_in_value = zone_data.loc[closest_idx, column_name_in]
            ghb_diff_value = ghb_out_value - ghb_in_value  # Calculate the difference
            outflow_data.append((code_value, ghb_diff_value))  # Store the difference
# Sort the plot data based on the modified code values (multiplied by -1)
drn_out_data.sort(key=lambda x: x[0])
net_leakageCaq_data.sort(key=lambda x: x[0])
outflow_data.sort(key=lambda x: x[0])
# Extract sorted code values and corresponding net flow and GHB-OUT values
drn_codes, drn_out_values = zip(*drn_out_data)
net_codesCaq, net_flowsCaq = zip(*net_leakageCaq_data)
outflow_codes, ghb_out_values = zip(*outflow_data)
# Plot both curves
ax.plot(drn_codes, drn_out_values, marker='s', linestyle='-', color='orange', label="Discharge to river network")
ax.plot(net_codesCaq, net_flowsCaq, marker='o', linestyle='-', color='b', label="Net leakage to confined aquifer")
ax.plot(outflow_codes, ghb_out_values, marker='x', linestyle='-', color='r', label="Lateral outflow from confined aquifer")

initial_drn_value = drn_out_values[0]
if initial_drn_value != 0:
    threshold_drn = initial_drn_value - 0.1 * abs(initial_drn_value)
else:
    threshold_drn = -0.1

# Add a horizontal line at thresholds
ax.axhline(0, color='r', linewidth=1, linestyle = '--')
ax.axhline(leakage_threshold, color='b', linewidth=1, linestyle = '--')
ax.axhline(threshold_drn, color='orange', linewidth=1, linestyle = '--')
# Find the threshold crossing (zero) for both curves
zero_cross_drn = find_threshold_crossing(drn_codes, drn_out_values, threshold= threshold_drn)
zero_cross_net_leakageCaq = find_threshold_crossing(net_codesCaq, net_flowsCaq, threshold=leakage_threshold)
zero_cross_outflow = find_threshold_crossing(outflow_codes, ghb_out_values, threshold=0)
# Select the smaller threshold crossing value
qs_candidates = list(filter(lambda x: x is not None, [
    zero_cross_drn,
    zero_cross_net_leakageCaq,
    zero_cross_outflow
]))

if qs_candidates:
    qs_value = min(qs_candidates)
    # Annotate Qs on the plot
    ax.axvline(qs_value, color='g', linestyle='--', linewidth=1)
    ax.text(qs_value, max(max(drn_out_values),
                          max(net_flowsCaq), 
                          max(ghb_out_values)),
            f"Qs < {qs_value:.2f} m³/day", color='g', fontsize=14, ha='right',
            bbox=dict(facecolor='white', alpha=0.7))
# Customize the plot
ax.set_title("Sustainable yield estimation - 100 years after pumping")
ax.set_xlabel("Pumping rate Rate [m³/day]")
ax.set_ylabel("Flow Rate [m³/day]")
ax.grid(True)
# Add a legend
ax.legend()
# Save the plot
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "Q vs flow tp100.png"), bbox_inches='tight')

########################## 150 YEARS
# Set the desired time value
time_target = 6045000
# Prepare the plot
fig, ax = plt.subplots(figsize=(14, 12))
# List to store extracted data for both curves
net_leakageCaq_data = []
net_leakageUnc_data = []
net_leakageCaq2_data = []
outflow_data = []
drn_out_data = []
# Loop through all zonebud_.csv files in the input folder
for file_name in os.listdir(input_folder):
    if file_name.startswith("zonebud_") and file_name.endswith(".csv"):
        # Extract the numeric code
        code = file_name.split(f"zonebud_{parameter}")[1].split(".csv")[0][1:]
        try:
            code = float(code)  # Convert to number for sorting
        except ValueError:
            continue  # Skip if conversion fails
        # Read the CSV file
        file_path = os.path.join(input_folder, file_name)
        data = pd.read_csv(file_path)
        # Process DRN-OUT for all zones combined
        if 'totim' in data.columns and "RIV-OUT" in data.columns and "RIV-IN" in data.columns:
            # Find the row(s) closest to the target time
            time_diff = abs(data['totim'] - time_target)
            closest_idx = time_diff.idxmin()
            closest_time = data.loc[closest_idx, 'totim']
            # Filter all rows at the closest time step
            time_filtered = data[data['totim'] == closest_time]
            total_drn_out = time_filtered["RIV-OUT"].sum() - time_filtered["RIV-IN"].sum()
            code_value = -code  # Keep it consistent with your x-axis
            drn_out_data.append((code_value, total_drn_out))
        # Extract data for zone 3
        zone_data = data[data['zone'] == 3]
        # Process Net leakage to confined aquifer from above
        if 'totim' in data.columns and "TO ZONE 2" in data.columns and "FROM ZONE 2" in data.columns and "TO ZONE 4" in data.columns and "FROM ZONE 4" in data.columns:
            zone_data["Net Flow Caq"] = zone_data["TO ZONE 2"] + zone_data["TO ZONE 4"] - zone_data["FROM ZONE 2"] - zone_data["FROM ZONE 4"]
            time_diff = abs(zone_data['totim'] - time_target)
            closest_idx = time_diff.idxmin()  # Get the index of the closest time
            # Extract the corresponding value at this time step for Net Leakage
            code_value = -code  # Multiply code by -1 to make it positive
            net_flow_value = zone_data.loc[closest_idx, "Net Flow Caq"]
            net_leakageCaq_data.append((code_value, net_flow_value))
        # Process Net leakage to confined aquifer
        if 'totim' in data.columns and "TO ZONE 4" in data.columns and "FROM ZONE 4" in data.columns:
            zone_data["Net Flow Caq2"] = zone_data["FROM ZONE 4"] - zone_data["TO ZONE 4"]
            time_diff = abs(zone_data['totim'] - time_target)
            closest_idx = time_diff.idxmin()  # Get the index of the closest time
            # Extract the corresponding value at this time step for Net Leakage
            code_value = -code  # Multiply code by -1 to make it positive
            net_flow_value = zone_data.loc[closest_idx, "Net Flow Caq2"]
            net_leakageCaq2_data.append((code_value, net_flow_value))
        # Process Outflow Confined (GHB-OUT - GHB-IN)
        column_name_out = "GHB-OUT"
        column_name_in = "GHB-IN"
        if 'totim' in data.columns and column_name_out in data.columns and column_name_in in data.columns:
            time_diff = abs(zone_data['totim'] - time_target)
            closest_idx = time_diff.idxmin()  # Get the index of the closest time
            # Extract the corresponding values at this time step for GHB-OUT and GHB-IN
            ghb_out_value = zone_data.loc[closest_idx, column_name_out]
            ghb_in_value = zone_data.loc[closest_idx, column_name_in]
            ghb_diff_value = ghb_out_value - ghb_in_value  # Calculate the difference
            outflow_data.append((code_value, ghb_diff_value))  # Store the difference
# Sort the plot data based on the modified code values (multiplied by -1)
drn_out_data.sort(key=lambda x: x[0])
net_leakageCaq_data.sort(key=lambda x: x[0])
outflow_data.sort(key=lambda x: x[0])
# Extract sorted code values and corresponding net flow and GHB-OUT values
drn_codes, drn_out_values = zip(*drn_out_data)
net_codesCaq, net_flowsCaq = zip(*net_leakageCaq_data)
outflow_codes, ghb_out_values = zip(*outflow_data)
# Plot both curves
ax.plot(drn_codes, drn_out_values, marker='s', linestyle='-', color='orange', label="Discharge to river network")
ax.plot(net_codesCaq, net_flowsCaq, marker='o', linestyle='-', color='b', label="Net leakage to confined aquifer")
ax.plot(outflow_codes, ghb_out_values, marker='x', linestyle='-', color='r', label="Lateral outflow from confined aquifer")

initial_drn_value = drn_out_values[0]
if initial_drn_value != 0:
    threshold_drn = initial_drn_value - 0.1 * abs(initial_drn_value)
else:
    threshold_drn = -0.1

# Add a horizontal line at thresholds
ax.axhline(0, color='r', linewidth=1, linestyle = '--')
ax.axhline(leakage_threshold, color='b', linewidth=1, linestyle = '--')
ax.axhline(threshold_drn, color='orange', linewidth=1, linestyle = '--')
# Find the threshold crossing (zero) for both curves
zero_cross_drn = find_threshold_crossing(drn_codes, drn_out_values, threshold= threshold_drn)
zero_cross_net_leakageCaq = find_threshold_crossing(net_codesCaq, net_flowsCaq, threshold=leakage_threshold)
zero_cross_outflow = find_threshold_crossing(outflow_codes, ghb_out_values, threshold=0)
# Select the smaller threshold crossing value
qs_candidates = list(filter(lambda x: x is not None, [
    zero_cross_drn,
    zero_cross_net_leakageCaq,
    zero_cross_outflow
]))

if qs_candidates:
    qs_value = min(qs_candidates)
    # Annotate Qs on the plot
    ax.axvline(qs_value, color='g', linestyle='--', linewidth=1)
    ax.text(qs_value, max(max(drn_out_values),
                          max(net_flowsCaq), 
                          max(ghb_out_values)),
            f"Qs < {qs_value:.2f} m³/day", color='g', fontsize=14, ha='right',
            bbox=dict(facecolor='white', alpha=0.7))
# Customize the plot
ax.set_title("Sustainable yield estimation - 150 years after pumping")
ax.set_xlabel("Pumping rate Rate [m³/day]")
ax.set_ylabel("Flow Rate [m³/day]")
ax.grid(True)
# Add a legend
ax.legend()
# Save the plot
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "Q vs flow tp150.png"), bbox_inches='tight')

########################## 200 YEARS
# Set the desired time value
time_target = 6060000
# Prepare the plot
fig, ax = plt.subplots(figsize=(14, 12))
# List to store extracted data for both curves
net_leakageCaq_data = []
net_leakageUnc_data = []
net_leakageCaq2_data = []
outflow_data = []
drn_out_data = []
# Loop through all zonebud_.csv files in the input folder
for file_name in os.listdir(input_folder):
    if file_name.startswith("zonebud_") and file_name.endswith(".csv"):
        # Extract the numeric code
        code = file_name.split(f"zonebud_{parameter}")[1].split(".csv")[0][1:]
        try:
            code = float(code)  # Convert to number for sorting
        except ValueError:
            continue  # Skip if conversion fails
        # Read the CSV file
        file_path = os.path.join(input_folder, file_name)
        data = pd.read_csv(file_path)
        # Process DRN-OUT for all zones combined
        if 'totim' in data.columns and "RIV-OUT" in data.columns and "RIV-IN" in data.columns:
            # Find the row(s) closest to the target time
            time_diff = abs(data['totim'] - time_target)
            closest_idx = time_diff.idxmin()
            closest_time = data.loc[closest_idx, 'totim']
            # Filter all rows at the closest time step
            time_filtered = data[data['totim'] == closest_time]
            total_drn_out = time_filtered["RIV-OUT"].sum() - time_filtered["RIV-IN"].sum()
            code_value = -code  # Keep it consistent with your x-axis
            drn_out_data.append((code_value, total_drn_out))
        # Extract data for zone 3
        zone_data = data[data['zone'] == 3]
        # Process Net leakage to confined aquifer from above
        if 'totim' in data.columns and "TO ZONE 2" in data.columns and "FROM ZONE 2" in data.columns and "TO ZONE 4" in data.columns and "FROM ZONE 4" in data.columns:
            zone_data["Net Flow Caq"] = zone_data["TO ZONE 2"] + zone_data["TO ZONE 4"] - zone_data["FROM ZONE 2"] - zone_data["FROM ZONE 4"]
            time_diff = abs(zone_data['totim'] - time_target)
            closest_idx = time_diff.idxmin()  # Get the index of the closest time
            # Extract the corresponding value at this time step for Net Leakage
            code_value = -code  # Multiply code by -1 to make it positive
            net_flow_value = zone_data.loc[closest_idx, "Net Flow Caq"]
            net_leakageCaq_data.append((code_value, net_flow_value))
        # Process Net leakage to confined aquifer
        if 'totim' in data.columns and "TO ZONE 4" in data.columns and "FROM ZONE 4" in data.columns:
            zone_data["Net Flow Caq2"] = zone_data["FROM ZONE 4"] - zone_data["TO ZONE 4"]
            time_diff = abs(zone_data['totim'] - time_target)
            closest_idx = time_diff.idxmin()  # Get the index of the closest time
            # Extract the corresponding value at this time step for Net Leakage
            code_value = -code  # Multiply code by -1 to make it positive
            net_flow_value = zone_data.loc[closest_idx, "Net Flow Caq2"]
            net_leakageCaq2_data.append((code_value, net_flow_value))
        # Process Outflow Confined (GHB-OUT - GHB-IN)
        column_name_out = "GHB-OUT"
        column_name_in = "GHB-IN"
        if 'totim' in data.columns and column_name_out in data.columns and column_name_in in data.columns:
            time_diff = abs(zone_data['totim'] - time_target)
            closest_idx = time_diff.idxmin()  # Get the index of the closest time
            # Extract the corresponding values at this time step for GHB-OUT and GHB-IN
            ghb_out_value = zone_data.loc[closest_idx, column_name_out]
            ghb_in_value = zone_data.loc[closest_idx, column_name_in]
            ghb_diff_value = ghb_out_value - ghb_in_value  # Calculate the difference
            outflow_data.append((code_value, ghb_diff_value))  # Store the difference
# Sort the plot data based on the modified code values (multiplied by -1)
drn_out_data.sort(key=lambda x: x[0])
net_leakageCaq_data.sort(key=lambda x: x[0])
outflow_data.sort(key=lambda x: x[0])
# Extract sorted code values and corresponding net flow and GHB-OUT values
drn_codes, drn_out_values = zip(*drn_out_data)
net_codesCaq, net_flowsCaq = zip(*net_leakageCaq_data)
outflow_codes, ghb_out_values = zip(*outflow_data)
# Plot both curves
ax.plot(drn_codes, drn_out_values, marker='s', linestyle='-', color='orange', label="Discharge to river network")
ax.plot(net_codesCaq, net_flowsCaq, marker='o', linestyle='-', color='b', label="Net leakage to confined aquifer")
ax.plot(outflow_codes, ghb_out_values, marker='x', linestyle='-', color='r', label="Lateral outflow from confined aquifer")

initial_drn_value = drn_out_values[0]
if initial_drn_value != 0:
    threshold_drn = initial_drn_value - 0.1 * abs(initial_drn_value)
else:
    threshold_drn = -0.1

# Add a horizontal line at thresholds
ax.axhline(0, color='r', linewidth=1, linestyle = '--')
ax.axhline(leakage_threshold, color='b', linewidth=1, linestyle = '--')
ax.axhline(threshold_drn, color='orange', linewidth=1, linestyle = '--')
# Find the threshold crossing (zero) for both curves
zero_cross_drn = find_threshold_crossing(drn_codes, drn_out_values, threshold= threshold_drn)
zero_cross_net_leakageCaq = find_threshold_crossing(net_codesCaq, net_flowsCaq, threshold=leakage_threshold)
zero_cross_outflow = find_threshold_crossing(outflow_codes, ghb_out_values, threshold=0)
# Select the smaller threshold crossing value
qs_candidates = list(filter(lambda x: x is not None, [
    zero_cross_drn,
    zero_cross_net_leakageCaq,
    zero_cross_outflow
]))

if qs_candidates:
    qs_value = min(qs_candidates)
    # Annotate Qs on the plot
    ax.axvline(qs_value, color='g', linestyle='--', linewidth=1)
    ax.text(qs_value, max(max(drn_out_values),
                          max(net_flowsCaq), 
                          max(ghb_out_values)),
            f"Qs < {qs_value:.2f} m³/day", color='g', fontsize=14, ha='right',
            bbox=dict(facecolor='white', alpha=0.7))
# Customize the plot
ax.set_title("Sustainable yield estimation - 200 years after pumping")
ax.set_xlabel("Pumping rate Rate [m³/day]")
ax.set_ylabel("Flow Rate [m³/day]")
ax.grid(True)
# Add a legend
ax.legend()
# Save the plot
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "Q vs flow tp200.png"), bbox_inches='tight')

########################## 500 YEARS
# Set the desired time value
time_target = 6150000
# Prepare the plot
fig, ax = plt.subplots(figsize=(14, 12))
# List to store extracted data for both curves
net_leakageCaq_data = []
net_leakageUnc_data = []
net_leakageCaq2_data = []
outflow_data = []
drn_out_data = []
# Loop through all zonebud_.csv files in the input folder
for file_name in os.listdir(input_folder):
    if file_name.startswith("zonebud_") and file_name.endswith(".csv"):
        # Extract the numeric code
        code = file_name.split(f"zonebud_{parameter}")[1].split(".csv")[0][1:]
        try:
            code = float(code)  # Convert to number for sorting
        except ValueError:
            continue  # Skip if conversion fails
        # Read the CSV file
        file_path = os.path.join(input_folder, file_name)
        data = pd.read_csv(file_path)
        # Process DRN-OUT for all zones combined
        if 'totim' in data.columns and "RIV-OUT" in data.columns and "RIV-IN" in data.columns:
            # Find the row(s) closest to the target time
            time_diff = abs(data['totim'] - time_target)
            closest_idx = time_diff.idxmin()
            closest_time = data.loc[closest_idx, 'totim']
            # Filter all rows at the closest time step
            time_filtered = data[data['totim'] == closest_time]
            total_drn_out = time_filtered["RIV-OUT"].sum() - time_filtered["RIV-IN"].sum()
            code_value = -code  # Keep it consistent with your x-axis
            drn_out_data.append((code_value, total_drn_out))
        # Extract data for zone 3
        zone_data = data[data['zone'] == 3]
        # Process Net leakage to confined aquifer from above
        if 'totim' in data.columns and "TO ZONE 2" in data.columns and "FROM ZONE 2" in data.columns and "TO ZONE 4" in data.columns and "FROM ZONE 4" in data.columns:
            zone_data["Net Flow Caq"] = zone_data["TO ZONE 2"] + zone_data["TO ZONE 4"] - zone_data["FROM ZONE 2"] - zone_data["FROM ZONE 4"]
            time_diff = abs(zone_data['totim'] - time_target)
            closest_idx = time_diff.idxmin()  # Get the index of the closest time
            # Extract the corresponding value at this time step for Net Leakage
            code_value = -code  # Multiply code by -1 to make it positive
            net_flow_value = zone_data.loc[closest_idx, "Net Flow Caq"]
            net_leakageCaq_data.append((code_value, net_flow_value))
        # Process Net leakage to confined aquifer
        if 'totim' in data.columns and "TO ZONE 4" in data.columns and "FROM ZONE 4" in data.columns:
            zone_data["Net Flow Caq2"] = zone_data["FROM ZONE 4"] - zone_data["TO ZONE 4"]
            time_diff = abs(zone_data['totim'] - time_target)
            closest_idx = time_diff.idxmin()  # Get the index of the closest time
            # Extract the corresponding value at this time step for Net Leakage
            code_value = -code  # Multiply code by -1 to make it positive
            net_flow_value = zone_data.loc[closest_idx, "Net Flow Caq2"]
            net_leakageCaq2_data.append((code_value, net_flow_value))
        # Process Outflow Confined (GHB-OUT - GHB-IN)
        column_name_out = "GHB-OUT"
        column_name_in = "GHB-IN"
        if 'totim' in data.columns and column_name_out in data.columns and column_name_in in data.columns:
            time_diff = abs(zone_data['totim'] - time_target)
            closest_idx = time_diff.idxmin()  # Get the index of the closest time
            # Extract the corresponding values at this time step for GHB-OUT and GHB-IN
            ghb_out_value = zone_data.loc[closest_idx, column_name_out]
            ghb_in_value = zone_data.loc[closest_idx, column_name_in]
            ghb_diff_value = ghb_out_value - ghb_in_value  # Calculate the difference
            outflow_data.append((code_value, ghb_diff_value))  # Store the difference
# Sort the plot data based on the modified code values (multiplied by -1)
drn_out_data.sort(key=lambda x: x[0])
net_leakageCaq_data.sort(key=lambda x: x[0])
outflow_data.sort(key=lambda x: x[0])
# Extract sorted code values and corresponding net flow and GHB-OUT values
drn_codes, drn_out_values = zip(*drn_out_data)
net_codesCaq, net_flowsCaq = zip(*net_leakageCaq_data)
outflow_codes, ghb_out_values = zip(*outflow_data)
# Plot both curves
ax.plot(drn_codes, drn_out_values, marker='s', linestyle='-', color='orange', label="Discharge to river network")
ax.plot(net_codesCaq, net_flowsCaq, marker='o', linestyle='-', color='b', label="Net leakage to confined aquifer")
ax.plot(outflow_codes, ghb_out_values, marker='x', linestyle='-', color='r', label="Lateral outflow from confined aquifer")

initial_drn_value = drn_out_values[0]
if initial_drn_value != 0:
    threshold_drn = initial_drn_value - 0.1 * abs(initial_drn_value)
else:
    threshold_drn = -0.1

# Add a horizontal line at thresholds
ax.axhline(0, color='r', linewidth=1, linestyle = '--')
ax.axhline(leakage_threshold, color='b', linewidth=1, linestyle = '--')
ax.axhline(threshold_drn, color='orange', linewidth=1, linestyle = '--')
# Find the threshold crossing (zero) for both curves
zero_cross_drn = find_threshold_crossing(drn_codes, drn_out_values, threshold= threshold_drn)
zero_cross_net_leakageCaq = find_threshold_crossing(net_codesCaq, net_flowsCaq, threshold=leakage_threshold)
zero_cross_outflow = find_threshold_crossing(outflow_codes, ghb_out_values, threshold=0)
# Select the smaller threshold crossing value
qs_candidates = list(filter(lambda x: x is not None, [
    zero_cross_drn,
    zero_cross_net_leakageCaq,
    zero_cross_outflow
]))

if qs_candidates:
    qs_value = min(qs_candidates)
    # Annotate Qs on the plot
    ax.axvline(qs_value, color='g', linestyle='--', linewidth=1)
    ax.text(qs_value, max(max(drn_out_values),
                          max(net_flowsCaq), 
                          max(ghb_out_values)),
            f"Qs < {qs_value:.2f} m³/day", color='g', fontsize=14, ha='right',
            bbox=dict(facecolor='white', alpha=0.7))
# Customize the plot
ax.set_title("Sustainable yield estimation - 500 years after pumping")
ax.set_xlabel("Pumping rate Rate [m³/day]")
ax.set_ylabel("Flow Rate [m³/day]")
ax.grid(True)
# Add a legend
ax.legend()
# Save the plot
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "Q vs flow tp500.png"), bbox_inches='tight')

########################## 1000 YEARS
# Set the desired time value
time_target = 6300000
# Prepare the plot
fig, ax = plt.subplots(figsize=(14, 12))
# List to store extracted data for both curves
net_leakageCaq_data = []
net_leakageUnc_data = []
net_leakageCaq2_data = []
outflow_data = []
drn_out_data = []
# Loop through all zonebud_.csv files in the input folder
for file_name in os.listdir(input_folder):
    if file_name.startswith("zonebud_") and file_name.endswith(".csv"):
        # Extract the numeric code
        code = file_name.split(f"zonebud_{parameter}")[1].split(".csv")[0][1:]
        try:
            code = float(code)  # Convert to number for sorting
        except ValueError:
            continue  # Skip if conversion fails
        # Read the CSV file
        file_path = os.path.join(input_folder, file_name)
        data = pd.read_csv(file_path)
        # Process DRN-OUT for all zones combined
        if 'totim' in data.columns and "RIV-OUT" in data.columns and "RIV-IN" in data.columns:
            # Find the row(s) closest to the target time
            time_diff = abs(data['totim'] - time_target)
            closest_idx = time_diff.idxmin()
            closest_time = data.loc[closest_idx, 'totim']
            # Filter all rows at the closest time step
            time_filtered = data[data['totim'] == closest_time]
            total_drn_out = time_filtered["RIV-OUT"].sum() - time_filtered["RIV-IN"].sum()
            code_value = -code  # Keep it consistent with your x-axis
            drn_out_data.append((code_value, total_drn_out))
        # Extract data for zone 3
        zone_data = data[data['zone'] == 3]
        # Process Net leakage to confined aquifer from above
        if 'totim' in data.columns and "TO ZONE 2" in data.columns and "FROM ZONE 2" in data.columns and "TO ZONE 4" in data.columns and "FROM ZONE 4" in data.columns:
            zone_data["Net Flow Caq"] = zone_data["TO ZONE 2"] + zone_data["TO ZONE 4"] - zone_data["FROM ZONE 2"] - zone_data["FROM ZONE 4"]
            time_diff = abs(zone_data['totim'] - time_target)
            closest_idx = time_diff.idxmin()  # Get the index of the closest time
            # Extract the corresponding value at this time step for Net Leakage
            code_value = -code  # Multiply code by -1 to make it positive
            net_flow_value = zone_data.loc[closest_idx, "Net Flow Caq"]
            net_leakageCaq_data.append((code_value, net_flow_value))
        # Process Net leakage to confined aquifer
        if 'totim' in data.columns and "TO ZONE 4" in data.columns and "FROM ZONE 4" in data.columns:
            zone_data["Net Flow Caq2"] = zone_data["FROM ZONE 4"] - zone_data["TO ZONE 4"]
            time_diff = abs(zone_data['totim'] - time_target)
            closest_idx = time_diff.idxmin()  # Get the index of the closest time
            # Extract the corresponding value at this time step for Net Leakage
            code_value = -code  # Multiply code by -1 to make it positive
            net_flow_value = zone_data.loc[closest_idx, "Net Flow Caq2"]
            net_leakageCaq2_data.append((code_value, net_flow_value))
        # Process Outflow Confined (GHB-OUT - GHB-IN)
        column_name_out = "GHB-OUT"
        column_name_in = "GHB-IN"
        if 'totim' in data.columns and column_name_out in data.columns and column_name_in in data.columns:
            time_diff = abs(zone_data['totim'] - time_target)
            closest_idx = time_diff.idxmin()  # Get the index of the closest time
            # Extract the corresponding values at this time step for GHB-OUT and GHB-IN
            ghb_out_value = zone_data.loc[closest_idx, column_name_out]
            ghb_in_value = zone_data.loc[closest_idx, column_name_in]
            ghb_diff_value = ghb_out_value - ghb_in_value  # Calculate the difference
            outflow_data.append((code_value, ghb_diff_value))  # Store the difference
# Sort the plot data based on the modified code values (multiplied by -1)
drn_out_data.sort(key=lambda x: x[0])
net_leakageCaq_data.sort(key=lambda x: x[0])
outflow_data.sort(key=lambda x: x[0])
# Extract sorted code values and corresponding net flow and GHB-OUT values
drn_codes, drn_out_values = zip(*drn_out_data)
net_codesCaq, net_flowsCaq = zip(*net_leakageCaq_data)
outflow_codes, ghb_out_values = zip(*outflow_data)
# Plot both curves
ax.plot(drn_codes, drn_out_values, marker='s', linestyle='-', color='orange', label="Discharge to river network")
ax.plot(net_codesCaq, net_flowsCaq, marker='o', linestyle='-', color='b', label="Net leakage to confined aquifer")
ax.plot(outflow_codes, ghb_out_values, marker='x', linestyle='-', color='r', label="Lateral outflow from confined aquifer")

initial_drn_value = drn_out_values[0]
if initial_drn_value != 0:
    threshold_drn = initial_drn_value - 0.1 * abs(initial_drn_value)
else:
    threshold_drn = -0.1

# Add a horizontal line at thresholds
ax.axhline(0, color='r', linewidth=1, linestyle = '--')
ax.axhline(leakage_threshold, color='b', linewidth=1, linestyle = '--')
ax.axhline(threshold_drn, color='orange', linewidth=1, linestyle = '--')
# Find the threshold crossing (zero) for both curves
zero_cross_drn = find_threshold_crossing(drn_codes, drn_out_values, threshold= threshold_drn)
zero_cross_net_leakageCaq = find_threshold_crossing(net_codesCaq, net_flowsCaq, threshold=leakage_threshold)
zero_cross_outflow = find_threshold_crossing(outflow_codes, ghb_out_values, threshold=0)
# Select the smaller threshold crossing value
qs_candidates = list(filter(lambda x: x is not None, [
    zero_cross_drn,
    zero_cross_net_leakageCaq,
    zero_cross_outflow
]))

if qs_candidates:
    qs_value = min(qs_candidates)
    # Annotate Qs on the plot
    ax.axvline(qs_value, color='g', linestyle='--', linewidth=1)
    ax.text(qs_value, max(max(drn_out_values),
                          max(net_flowsCaq), 
                          max(ghb_out_values)),
            f"Qs < {qs_value:.2f} m³/day", color='g', fontsize=14, ha='right',
            bbox=dict(facecolor='white', alpha=0.7))
# Customize the plot
ax.set_title("Sustainable yield estimation - 1000 years after pumping")
ax.set_xlabel("Pumping rate Rate [m³/day]")
ax.set_ylabel("Flow Rate [m³/day]")
ax.grid(True)
# Add a legend
ax.legend()
# Save the plot
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "Q vs flow tp1000.png"), bbox_inches='tight')


########################## 10000 YEARS
# Set the desired time value
time_target = 9000000
# Prepare the plot
fig, ax = plt.subplots(figsize=(14, 12))
# List to store extracted data for both curves
net_leakageCaq_data = []
net_leakageUnc_data = []
net_leakageCaq2_data = []
outflow_data = []
drn_out_data = []
# Loop through all zonebud_.csv files in the input folder
for file_name in os.listdir(input_folder):
    if file_name.startswith("zonebud_") and file_name.endswith(".csv"):
        # Extract the numeric code
        code = file_name.split(f"zonebud_{parameter}")[1].split(".csv")[0][1:]
        try:
            code = float(code)  # Convert to number for sorting
        except ValueError:
            continue  # Skip if conversion fails
        # Read the CSV file
        file_path = os.path.join(input_folder, file_name)
        data = pd.read_csv(file_path)
        # Process DRN-OUT for all zones combined
        if 'totim' in data.columns and "RIV-OUT" in data.columns and "RIV-IN" in data.columns:
            # Find the row(s) closest to the target time
            time_diff = abs(data['totim'] - time_target)
            closest_idx = time_diff.idxmin()
            closest_time = data.loc[closest_idx, 'totim']
            # Filter all rows at the closest time step
            time_filtered = data[data['totim'] == closest_time]
            total_drn_out = time_filtered["RIV-OUT"].sum() - time_filtered["RIV-IN"].sum()
            code_value = -code  # Keep it consistent with your x-axis
            drn_out_data.append((code_value, total_drn_out))
        # Extract data for zone 3
        zone_data = data[data['zone'] == 3]
        # Process Net leakage to confined aquifer from above
        if 'totim' in data.columns and "TO ZONE 2" in data.columns and "FROM ZONE 2" in data.columns and "TO ZONE 4" in data.columns and "FROM ZONE 4" in data.columns:
            zone_data["Net Flow Caq"] = zone_data["TO ZONE 2"] + zone_data["TO ZONE 4"] - zone_data["FROM ZONE 2"] - zone_data["FROM ZONE 4"]
            time_diff = abs(zone_data['totim'] - time_target)
            closest_idx = time_diff.idxmin()  # Get the index of the closest time
            # Extract the corresponding value at this time step for Net Leakage
            code_value = -code  # Multiply code by -1 to make it positive
            net_flow_value = zone_data.loc[closest_idx, "Net Flow Caq"]
            net_leakageCaq_data.append((code_value, net_flow_value))
        # Process Net leakage to confined aquifer
        if 'totim' in data.columns and "TO ZONE 4" in data.columns and "FROM ZONE 4" in data.columns:
            zone_data["Net Flow Caq2"] = zone_data["FROM ZONE 4"] - zone_data["TO ZONE 4"]
            time_diff = abs(zone_data['totim'] - time_target)
            closest_idx = time_diff.idxmin()  # Get the index of the closest time
            # Extract the corresponding value at this time step for Net Leakage
            code_value = -code  # Multiply code by -1 to make it positive
            net_flow_value = zone_data.loc[closest_idx, "Net Flow Caq2"]
            net_leakageCaq2_data.append((code_value, net_flow_value))
        # Process Outflow Confined (GHB-OUT - GHB-IN)
        column_name_out = "GHB-OUT"
        column_name_in = "GHB-IN"
        if 'totim' in data.columns and column_name_out in data.columns and column_name_in in data.columns:
            time_diff = abs(zone_data['totim'] - time_target)
            closest_idx = time_diff.idxmin()  # Get the index of the closest time
            # Extract the corresponding values at this time step for GHB-OUT and GHB-IN
            ghb_out_value = zone_data.loc[closest_idx, column_name_out]
            ghb_in_value = zone_data.loc[closest_idx, column_name_in]
            ghb_diff_value = ghb_out_value - ghb_in_value  # Calculate the difference
            outflow_data.append((code_value, ghb_diff_value))  # Store the difference
# Sort the plot data based on the modified code values (multiplied by -1)
drn_out_data.sort(key=lambda x: x[0])
net_leakageCaq_data.sort(key=lambda x: x[0])
outflow_data.sort(key=lambda x: x[0])
# Extract sorted code values and corresponding net flow and GHB-OUT values
drn_codes, drn_out_values = zip(*drn_out_data)
net_codesCaq, net_flowsCaq = zip(*net_leakageCaq_data)
outflow_codes, ghb_out_values = zip(*outflow_data)
# Plot both curves
ax.plot(drn_codes, drn_out_values, marker='s', linestyle='-', color='orange', label="Discharge to river network")
ax.plot(net_codesCaq, net_flowsCaq, marker='o', linestyle='-', color='b', label="Net leakage to confined aquifer")
ax.plot(outflow_codes, ghb_out_values, marker='x', linestyle='-', color='r', label="Lateral outflow from confined aquifer")

initial_drn_value = drn_out_values[0]
if initial_drn_value != 0:
    threshold_drn = initial_drn_value - 0.1 * abs(initial_drn_value)
else:
    threshold_drn = -0.1

# Add a horizontal line at thresholds
ax.axhline(0, color='r', linewidth=1, linestyle = '--')
ax.axhline(leakage_threshold, color='b', linewidth=1, linestyle = '--')
ax.axhline(threshold_drn, color='orange', linewidth=1, linestyle = '--')
# Find the threshold crossing (zero) for both curves
zero_cross_drn = find_threshold_crossing(drn_codes, drn_out_values, threshold= threshold_drn)
zero_cross_net_leakageCaq = find_threshold_crossing(net_codesCaq, net_flowsCaq, threshold=leakage_threshold)
zero_cross_outflow = find_threshold_crossing(outflow_codes, ghb_out_values, threshold=0)
# Select the smaller threshold crossing value
qs_candidates = list(filter(lambda x: x is not None, [
    zero_cross_drn,
    zero_cross_net_leakageCaq,
    zero_cross_outflow
]))

if qs_candidates:
    qs_value = min(qs_candidates)
    # Annotate Qs on the plot
    ax.axvline(qs_value, color='g', linestyle='--', linewidth=1)
    ax.text(qs_value, max(max(drn_out_values),
                          max(net_flowsCaq), 
                          max(ghb_out_values)),
            f"Qs < {qs_value:.2f} m³/day", color='g', fontsize=14, ha='right',
            bbox=dict(facecolor='white', alpha=0.7))
# Customize the plot
ax.set_title("Sustainable yield estimation - 10000 years after pumping")
ax.set_xlabel("Pumping rate Rate [m³/day]")
ax.set_ylabel("Flow Rate [m³/day]")
ax.grid(True)
# Add a legend
ax.legend()
# Save the plot
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "Q vs flow tp10000.png"), bbox_inches='tight')

########################## 100 YEARS
# Set the desired time value
time_target = 12000000
# Prepare the plot
fig, ax = plt.subplots(figsize=(14, 12))
# List to store extracted data for both curves
net_leakageCaq_data = []
net_leakageUnc_data = []
net_leakageCaq2_data = []
outflow_data = []
drn_out_data = []
# Loop through all zonebud_.csv files in the input folder
for file_name in os.listdir(input_folder):
    if file_name.startswith("zonebud_") and file_name.endswith(".csv"):
        # Extract the numeric code
        code = file_name.split(f"zonebud_{parameter}")[1].split(".csv")[0][1:]
        try:
            code = float(code)  # Convert to number for sorting
        except ValueError:
            continue  # Skip if conversion fails
        # Read the CSV file
        file_path = os.path.join(input_folder, file_name)
        data = pd.read_csv(file_path)
        # Process DRN-OUT for all zones combined
        if 'totim' in data.columns and "RIV-OUT" in data.columns and "RIV-IN" in data.columns:
            # Find the row(s) closest to the target time
            time_diff = abs(data['totim'] - time_target)
            closest_idx = time_diff.idxmin()
            closest_time = data.loc[closest_idx, 'totim']
            # Filter all rows at the closest time step
            time_filtered = data[data['totim'] == closest_time]
            total_drn_out = time_filtered["RIV-OUT"].sum() - time_filtered["RIV-IN"].sum()
            code_value = -code  # Keep it consistent with your x-axis
            drn_out_data.append((code_value, total_drn_out))
        # Extract data for zone 3
        zone_data = data[data['zone'] == 3]
        # Process Net leakage to confined aquifer from above
        if 'totim' in data.columns and "TO ZONE 2" in data.columns and "FROM ZONE 2" in data.columns and "TO ZONE 4" in data.columns and "FROM ZONE 4" in data.columns:
            zone_data["Net Flow Caq"] = zone_data["TO ZONE 2"] + zone_data["TO ZONE 4"] - zone_data["FROM ZONE 2"] - zone_data["FROM ZONE 4"]
            time_diff = abs(zone_data['totim'] - time_target)
            closest_idx = time_diff.idxmin()  # Get the index of the closest time
            # Extract the corresponding value at this time step for Net Leakage
            code_value = -code  # Multiply code by -1 to make it positive
            net_flow_value = zone_data.loc[closest_idx, "Net Flow Caq"]
            net_leakageCaq_data.append((code_value, net_flow_value))
        # Process Net leakage to confined aquifer
        if 'totim' in data.columns and "TO ZONE 4" in data.columns and "FROM ZONE 4" in data.columns:
            zone_data["Net Flow Caq2"] = zone_data["FROM ZONE 4"] - zone_data["TO ZONE 4"]
            time_diff = abs(zone_data['totim'] - time_target)
            closest_idx = time_diff.idxmin()  # Get the index of the closest time
            # Extract the corresponding value at this time step for Net Leakage
            code_value = -code  # Multiply code by -1 to make it positive
            net_flow_value = zone_data.loc[closest_idx, "Net Flow Caq2"]
            net_leakageCaq2_data.append((code_value, net_flow_value))
        # Process Outflow Confined (GHB-OUT - GHB-IN)
        column_name_out = "GHB-OUT"
        column_name_in = "GHB-IN"
        if 'totim' in data.columns and column_name_out in data.columns and column_name_in in data.columns:
            time_diff = abs(zone_data['totim'] - time_target)
            closest_idx = time_diff.idxmin()  # Get the index of the closest time
            # Extract the corresponding values at this time step for GHB-OUT and GHB-IN
            ghb_out_value = zone_data.loc[closest_idx, column_name_out]
            ghb_in_value = zone_data.loc[closest_idx, column_name_in]
            ghb_diff_value = ghb_out_value - ghb_in_value  # Calculate the difference
            outflow_data.append((code_value, ghb_diff_value))  # Store the difference
# Sort the plot data based on the modified code values (multiplied by -1)
drn_out_data.sort(key=lambda x: x[0])
net_leakageCaq_data.sort(key=lambda x: x[0])
outflow_data.sort(key=lambda x: x[0])
# Extract sorted code values and corresponding net flow and GHB-OUT values
drn_codes, drn_out_values = zip(*drn_out_data)
net_codesCaq, net_flowsCaq = zip(*net_leakageCaq_data)
outflow_codes, ghb_out_values = zip(*outflow_data)
# Plot both curves
ax.plot(drn_codes, drn_out_values, marker='s', linestyle='-', color='orange', label="Discharge to river network")
ax.plot(net_codesCaq, net_flowsCaq, marker='o', linestyle='-', color='b', label="Net leakage to confined aquifer")
ax.plot(outflow_codes, ghb_out_values, marker='x', linestyle='-', color='r', label="Lateral outflow from confined aquifer")

initial_drn_value = drn_out_values[0]
if initial_drn_value != 0:
    threshold_drn = initial_drn_value - 0.1 * abs(initial_drn_value)
else:
    threshold_drn = -0.1

# Add a horizontal line at thresholds
ax.axhline(0, color='r', linewidth=1, linestyle = '--')
ax.axhline(leakage_threshold, color='b', linewidth=1, linestyle = '--')
ax.axhline(threshold_drn, color='orange', linewidth=1, linestyle = '--')
# Find the threshold crossing (zero) for both curves
zero_cross_drn = find_threshold_crossing(drn_codes, drn_out_values, threshold= threshold_drn)
zero_cross_net_leakageCaq = find_threshold_crossing(net_codesCaq, net_flowsCaq, threshold=leakage_threshold)
zero_cross_outflow = find_threshold_crossing(outflow_codes, ghb_out_values, threshold=0)
# Select the smaller threshold crossing value
qs_candidates = list(filter(lambda x: x is not None, [
    zero_cross_drn,
    zero_cross_net_leakageCaq,
    zero_cross_outflow
]))

if qs_candidates:
    qs_value = min(qs_candidates)
    # Annotate Qs on the plot
    ax.axvline(qs_value, color='g', linestyle='--', linewidth=1)
    ax.text(qs_value, max(max(drn_out_values),
                          max(net_flowsCaq), 
                          max(ghb_out_values)),
            f"Qs < {qs_value:.2f} m³/day", color='g', fontsize=14, ha='right',
            bbox=dict(facecolor='white', alpha=0.7))
# Customize the plot
ax.set_title("Sustainable yield estimation - 20000 years after pumping")
ax.set_xlabel("Pumping rate Rate [m³/day]")
ax.set_ylabel("Flow Rate [m³/day]")
ax.grid(True)
# Add a legend
ax.legend()
# Save the plot
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "Q vs flow tp20000.png"), bbox_inches='tight')

###################### HEAD DIFFERENCE 100 YEARS
# Define target time
time_target = 6030000  
# Initialize storage
head_diff_data = []
# Loop through head observation files
for file_name in os.listdir(input_folder):
    if file_name.startswith("head_obs_") and file_name.endswith("_t.csv"):
        try:
            code = float(file_name.split(f"head_obs_{parameter}")[1].split("_t.csv")[0][1:])
        except ValueError:
            continue
        # Read CSV
        file_path = os.path.join(input_folder, file_name)
        data = pd.read_csv(file_path)
        if 'time' in data.columns and data.shape[1] > 1:
            closest_idx = (data['time'] - time_target).abs().idxmin()
            head_diff = data.iloc[closest_idx, 3] - data.iloc[closest_idx, 1]  # Confined - Unconfined
            head_diff_data.append((-code, head_diff))
# Sort data
head_diff_data.sort()
pumping_rates, head_diffs = zip(*head_diff_data)
# Plot
fig, ax = plt.subplots()
ax.plot(pumping_rates, head_diffs, marker='o', linestyle='-', color='b', label="Drawdown Difference (Confined - Unconfined)")
# Customize plot
ax.set_title("Sustainable yield estimation - 100 years after pumping")
ax.set_xlabel("Pumping Rate [m³/day]")
ax.set_ylabel("Head Difference (Confined - Unconfined) [m]")
ax.axhline(0, color='black', linewidth=1, linestyle='--')
# Find the threshold crossing (zero) for both curves
zero_cross = find_threshold_crossing(pumping_rates, head_diffs, threshold=-100)
qs_value = zero_cross
# Annotate Qs on the plot
if qs_value is not None:
    ax.axvline(qs_value, color='g', linestyle='--', linewidth=1)
    ax.text(qs_value, max(head_diffs),
            f"Qs < {qs_value:.2f} m³/day", color='g', fontsize=14, ha='right', bbox=dict(facecolor='white', alpha=0.7))
ax.grid(True)
ax.legend()
# Save plot
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "Q vs head difference tp.png"), bbox_inches='tight')

############################# HEADS 100 YEARS
# Define the target time step
time_target = 6030000 
# Initialize storage for data
confined_data = []
unconfined_data = []
# Loop through all head observation files
for file_name in os.listdir(input_folder):
    if file_name.startswith("head_obs_") and file_name.endswith("_t.csv"):
        # Extract pumping rate from filename
        code = file_name.split(f"head_obs_{parameter}")[1].split("_t.csv")[0][1:]
        try:
            code = float(code)  # Convert to number for sorting
        except ValueError:
            continue  # Skip if conversion fails
        file_path = os.path.join(input_folder, file_name)
        data = pd.read_csv(file_path)
        if 'time' in data.columns and data.shape[1] > 1:
            time_values = data['time']  # Convert time units
            # Find closest time step
            time_diff = abs(time_values - time_target)
            closest_idx = time_diff.idxmin()
            # Extract head values for confined and unconfined aquifers
            confined_head = data.iloc[closest_idx, 3]  # Confined aquifer (column 3)
            unconfined_head = data.iloc[closest_idx, 1]  # Unconfined aquifer (column 1)
            confined_data.append((-code, confined_head))
            unconfined_data.append((-code, unconfined_head))
# Sort data by pumping rate
confined_data.sort(key=lambda x: x[0])
unconfined_data.sort(key=lambda x: x[0])
# Extract sorted values for plotting
confined_codes, confined_heads = zip(*confined_data)
unconfined_codes, unconfined_heads = zip(*unconfined_data)
# Plot
fig, ax = plt.subplots()
ax.plot(confined_codes, confined_heads, marker='o', linestyle='-', color='b', label="Confined Aquifer")
ax.plot(unconfined_codes, unconfined_heads, marker='x', linestyle='-', color='r', label="Unconfined Aquifer")
# Customize the plot
ax.set_title(f"Sustainable yield estimation - 100 years after pumping")
ax.set_xlabel("Pumping Rate [m³/day]")
ax.set_ylabel("Head at pumping well [meters]")
ax.axhline(0, color='black', linewidth=1, linestyle='--')
# Find the threshold crossing (zero) for both curves
zero_cross_confined = find_threshold_crossing(confined_codes, confined_heads, threshold=0)
zero_cross_unconfined = find_threshold_crossing(unconfined_codes, unconfined_heads, threshold=0)
# Select the smaller threshold crossing value
qs_value = min(filter(None, [zero_cross_confined, zero_cross_unconfined]))
# Annotate Qs on the plot
if qs_value is not None:
    ax.axvline(qs_value, color='g', linestyle='--', linewidth=1)
    ax.text(qs_value, max(max(confined_heads), max(unconfined_heads)),
            f"Qs < {qs_value:.2f} m³/day", color='g', fontsize=14, ha='right', bbox=dict(facecolor='white', alpha=0.7))
ax.grid(True)
ax.legend()
# Save the plot
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "Q vs head tp.png"), bbox_inches='tight')

