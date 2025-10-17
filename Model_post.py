# ==========================================================================================
#  Model.py - MODFLOW 6 Model post-processing
# ==========================================================================================
#
#  Author: MARIN RIVERA Carlos Felipe
#  Organization: Bordeaux INP, Lab EPOC, Université de Bordeaux
#  Project: Funded by the OneWater PEPR DEESAC Project
#
#  DESCRIPTION:
#  ------------
#  This script postporcesses and analyzes MODFLOW 6 groundwater flow model of a synthetic
#  multilayer aquifer system for the ConfinedLab project. It integrates model construction,  
#  steady state and transient simulation, and post-processing utilities for flow and budget 
#  analysis.
#
#  USAGE:
#  ------
#  Configure model parameters, execute the simulation, and generate outputs
#  for further analysis and visualization.
#
# ==========================================================================================

# ==========================================================================================
#  Model.py - MODFLOW 6 Model post-processing
# ==========================================================================================
#
#  Author: MARIN RIVERA Carlos Felipe
#  Organization: Bordeaux INP, Lab EPOC, Université de Bordeaux
#  Project: Funded by the OneWater PEPR DEESAC Project
#
#  DESCRIPTION:
#  ------------
#  This script postporcesses and analyzes MODFLOW 6 groundwater flow model of a synthetic
#  multilayer aquifer system for the ConfinedLab project. It integrates model construction,  
#  steady state and transient simulation, and post-processing utilities for flow and budget 
#  analysis.
#
#  USAGE:
#  ------
#  Configure model parameters, execute the simulation, and generate outputs
#  for further analysis and visualization.
#
# ==========================================================================================

# ---------------------------------------------------------------------------------------- #
# ------------------------------------- IMPORT MODULES ----------------------------------- #
# ---------------------------------------------------------------------------------------- #
import time
start_time = time.time()
import os
import sys
import numpy as np
import pandas as pd
import flopy

# Plot settings
#matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rc('font', family='serif', size=9)
sgcol_width = 9/2.54
mdcol_width = 14/2.54
dbcol_width = 19/2.54

# Import local modules
sys.path.append('..')
from mlibs import modpar6, modplot6, modtransient6, modpump6, modgeom6, modbound6 # type: ignore

# Check the current and parent directories
current_dir = os.getcwd()
print("Current Directory:", current_dir)
parent_dir = os.path.dirname(current_dir)
print("Parent Directory:", parent_dir)

# Dispose of warnings
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# ------------------------------------------------------------------------------- #
# --------------------------- MODEL RUN CONTROL --------------------------------- #
# ------------------------------------------------------------------------------- #
rivers = False # If True, includes river package instead of drain package
heterogeneity = False # If True, generates random hydraulic conductivity fields

if rivers:
    boundary_keywords = ["GHB", "WEL", "RIV"]
else:
    boundary_keywords = ["GHB", "WEL", "DRN"]

# ------------------------------------------------------------------------------- #
# --------------------------------- MODEL SETUP --------------------------------- #
# ------------------------------------------------------------------------------- #

# Set model directory and model names
model_ws = 'mf'
model_name = 'DEESAC'
model_name_tr = 'DEESACt'
output_folder = f"{model_ws}/output"
figure_folder = f"{model_ws}/fig"
gis_folder = "C:/Users/cmarinriver/Projects/ConfinedLab/gis" 
setup_file = "setup.xlsx" # Excel file containing model setup parameters

os.makedirs(output_folder, exist_ok=True)
os.makedirs(figure_folder, exist_ok=True)
os.makedirs(gis_folder, exist_ok=True)

# Set model hydraulic parameters from setup file
par_df = pd.read_excel(setup_file, sheet_name="parameters", index_col=0)
def par_df_to_1Darray(df, prefix):
    subset = df[df.index.str.startswith(prefix)]
    subset = subset.sort_index()  # Ensure correct order
    return subset["value"].to_numpy()
kh = par_df_to_1Darray(par_df, "kh") # Horizontal hydraulic conductivity in m/d
kv = par_df_to_1Darray(par_df, "kv") # Vertical hydraulic conductivity in m/d
sy = par_df_to_1Darray(par_df, "sy") # Specific yield (adimensional)
ss = par_df_to_1Darray(par_df, "ss") # Specific storage (m-1)
drn_cond = par_df_to_1Darray(par_df, "drn_cond") # Drain conductance (m2/d)
R = par_df_to_1Darray(par_df, "rech") # Recharge (m/d)

# Set model grid parameters
grid_df = pd.read_excel(setup_file, sheet_name="grid")
nlay = int(grid_df["nlay"][0]) # Number of layers
ncol = int(grid_df["ncol"][0]) # Number of columns
nrow = int(grid_df["nrow"][0]) # Number of rows (single row for 2D cross section)
length = float(grid_df["lcol"][0]) # Total lenght of model in meters
width = float(grid_df["lrow"][0]) # Total width of model in meters
dcol = int(grid_df["dcol"][0]) # Column size in meters
drow = int(grid_df["drow"][0]) # Row size in meters

# ------------------------------------------------------------------------------- #
# --------------------------- GEOMETRY GENERATION ------------------------------- #
# ------------------------------------------------------------------------------- #

# Set synthetic geometry generation parameters
geom_df = pd.read_excel(setup_file, sheet_name="geometry")
outcrop_z = geom_df["outcrop_z"].to_numpy() # Elevation (Just used when SLOPE is set to False)
outcrop_zmax = geom_df["outcrop_zmax"].to_numpy() # Elevation (Just used when SLOPE are set to True)
outcrop_zmin = geom_df["outcrop_zmin"].to_numpy() # Elevation (Just used when SLOPE are set to True)
base_thicknesses = geom_df["base_thicknesses"].to_numpy() # Layer thickness in meters
outcrop_cells = geom_df["outcrop_cells"].to_numpy() # Cell ID where the unit starts outcropping (measured from left to right)
zones = geom_df["zone"].to_numpy() # Zone ID for each layer

# Create idomain, irch and recharge arrays
epsilon = float(geom_df["epsilon"].iloc[0]) # Minimum allowed cell thickness in meters
transition = int(geom_df["transition_cells"].iloc[0]) # Transitions cells
idomain = modgeom6.compute_idomain(nlay, nrow, ncol, outcrop_cells)
ztop = modgeom6.compute_top(idomain, outcrop_z, transition=True, slope=True,
                            transition_cells=transition, transition_type="contain", 
                            outcrop_zmin=outcrop_zmin, outcrop_zmax=outcrop_zmax)
thickness_array = modgeom6.compute_thickness(idomain, base_thicknesses, 
                                             transition=True, transition_type="extend", 
                                             transition_cells=transition)
zbot = modgeom6.compute_bottom(ztop, thickness_array)
idomain = modgeom6.idomain_from_thickness(thickness_array, epsilon)
ztop_array = modgeom6.compute_ztop_array(ztop, zbot)
irch = modgeom6.compute_irch(idomain)
R_array = modgeom6.compute_recharge(irch, R)
zone_array = modgeom6.compute_3Darray(zones, idomain, dtype = int)
kh_array = modgeom6.compute_3Darray(kh, idomain)
kv_array = modgeom6.compute_3Darray(kv, idomain)

# ------------------------------------------------------------------------------- #
# ----------------------- RANDOM PARAMETER FIELDS ------------------------------- #
# ------------------------------------------------------------------------------- #

if heterogeneity:
    # Generate random hydraulic conductivity fields for each layer
    kh0 = modpar6.generate_random_field((nrow, ncol), "exponential",
                                        geom_mean=kh[0], sill=0.3, nugget=0.0,
                                        range_param=15000, drow=drow, dcol=dcol,
                                        param_type="K", seed=0)

    kh1 = modpar6.generate_random_field((nrow, ncol), "exponential",
                                        geom_mean=kh[1], sill=0.3, nugget=0.0,
                                        range_param=15000, drow=drow, dcol=dcol,
                                        param_type="K", seed=1)

    kh2 = modpar6.generate_random_field((nrow, ncol), "exponential",
                                        geom_mean=kh[2], sill=0.3, nugget=0.0,
                                        range_param=15000, drow=drow, dcol=dcol,
                                        param_type="K", seed=2)

    kh3 = modpar6.generate_random_field((nrow, ncol), "exponential",
                                        geom_mean=kh[3], sill=0.3, nugget=0.0,
                                        range_param=15000, drow=drow, dcol=dcol,
                                        param_type="K", seed=3)

    kh4 = modpar6.generate_random_field((nrow, ncol), "exponential",
                                        geom_mean=kh[4], sill=0.3, nugget=0.0,
                                        range_param=15000, drow=drow, dcol=dcol,
                                        param_type="K", seed=4)

    kh_array = modpar6.stack_fields_to_3D([kh0, kh1, kh2, kh3, kh4], nlay, nrow, ncol)
    kv_array = kh_array / 10  # Assume kv is one-tenth of kh

# ------------------------------------------------------------------------------- #
# ----------------------- LAYER SUBDIVISION -------------------------------------- #
# ------------------------------------------------------------------------------- #
nsub = geom_df["nsub"].to_list() # Number of subdivisions per layer

#Subdivide layers
nlay, idomain, ztop_array, zbot = modgeom6.subdivide_layers(idomain, ztop_array, zbot, nsub)

# Subdivide other arrays (size nlay)
kh = modgeom6.subdivide_array(kh, nsub)
sy = modgeom6.subdivide_array(sy, nsub)
ss = modgeom6.subdivide_array(ss, nsub)
drn_cond = modgeom6.subdivide_array(drn_cond, nsub)
base_thicknesses = modgeom6.subdivide_array(base_thicknesses, nsub)
R = modgeom6.subdivide_array(R, nsub)
zone_array = modgeom6.subdivide_array(zone_array, nsub)
kh_array = modgeom6.subdivide_array(kh_array, nsub)
kv_array = modgeom6.subdivide_array(kv_array, nsub)

#Update thickness, irch, R_array, zone_array and kh_array
thickness_array = ztop_array - zbot
irch = modgeom6.compute_irch(idomain)
R_array = modgeom6.compute_recharge(irch, R)

# ------------------------------------------------------------------------------- #
# ----------------------- LOAD EXISTING MODEL ----------------------------------- #        
# ------------------------------------------------------------------------------- #

sim = flopy.mf6.MFSimulation.load(sim_name = model_name, 
                             sim_ws = model_ws, 
                             exe_name = "mf6")

gwf = sim.gwf[0]

# ------------------------------------------------------------------------------- #
# ----------------------- PLOT AND POSTPROCESS ---------------------------------- #
# ------------------------------------------------------------------------------- #

tdis_df = pd.read_excel(setup_file, sheet_name="tdis")
nper = len(tdis_df.index)
perlen = tdis_df["perlen"].tolist()
nstp = tdis_df["nstp"].tolist()
tsmult = tdis_df["tsmult"].tolist()
perioddata = list(zip(perlen, nstp, tsmult))

sp_num = 0
ts_num = 0
layer = 0
elapsed_time = modtransient6.elapsed_time(perioddata, sp_num, ts_num)

head = gwf.output.head().get_data(kstpkper=(ts_num, sp_num)) #kstpkper = time step and stress period.
bud = gwf.output.budget()
spdis = bud.get_data(text='DATA-SPDIS', kstpkper=(ts_num, sp_num))[layer] 
qx, qy, qz = flopy.utils.postprocessing.get_specific_discharge(spdis, gwf)

hobj = flopy.utils.HeadFile(f"{output_folder}/{model_name_tr}.hds")
transient_heads = hobj.get_alldata()
times_list = hobj.get_times()

hobj_ss = flopy.utils.HeadFile(f"{output_folder}/{model_name}.hds")
steady_state_heads = hobj_ss.get_data()

# cb = gwf.oc.output.budget()
# cb.get_data(idx=0, full3D=True) #Get cell budget file

budget_file_t = f"{output_folder}/{model_name_tr}_budget.csv"

zonebud_file_t = f"{output_folder}/zonebud.csv"

head_file_t = f"{output_folder}/head_obs_t.csv"

plot_heads = False
plot_budget = False
animate = False
response_times = False
diffusion_animation = True

#--------------------------------------- HEADS ---------------------------------------------#
if plot_heads:

    modplot6.plot_map_view(gwf, head, 
            f"{figure_folder}/map_heads_L1.png", 
            boundary_keywords=["WEL", "GHB"], 
            layer=0, flow_dir=False, contours=True,show=False, save=True,
            grid=True, figsize=(10, 10), fontsize=14,title="Model map view Layer 1_t")

    modplot6.plot_map_view(gwf, head,
            f"{figure_folder}/map_heads_L2.png", 
            boundary_keywords=["WEL", "GHB"], 
            layer=7, flow_dir=False, contours=True,show=False, save=True,
            grid=True, figsize=(10, 10), fontsize=14,title="Model map view Layer 2_t")
    
    modplot6.plot_map_view(gwf, head, 
            f"{figure_folder}/map_heads_L3.png", 
            boundary_keywords=["WEL", "GHB"], 
            layer=12, flow_dir=False, contours=True,show=False, save=True,
            grid=True, figsize=(10, 10), fontsize=14,title="Model map view Layer 3_t")
    
    modplot6.plot_map_view(gwf, head,
            f"{figure_folder}/map_heads_L4.png", 
            boundary_keywords=["WEL", "GHB"], 
            layer=17, flow_dir=False, contours=True,show=False, save=True,
            grid=True, figsize=(10, 10), fontsize=14,title="Model map view Layer 4_t")
    
    modplot6.plot_map_view(gwf, head,
            f"{figure_folder}/map_heads_L5.png", 
            boundary_keywords=["WEL", "GHB"], 
            layer=22, flow_dir=False, contours=True,show=False, save=True,
            grid=True, figsize=(10, 10), fontsize=14,title="Model map view Layer 5_t")

    modplot6.plot_cross_section_row(gwf, head, nrow//2, 
                                    f"{figure_folder}/cross_section_heads_t.png",
                                    boundary_keywords = ["WEL"], ve=100,
                                    flow_dir = False, surface = False, layers=False,
                                    show=False, save=True, figsize = (19, 4),
                                    title=f"Cross section at time {elapsed_time} days")
    modplot6.plot_cross_section_row(gwf, head, nrow//2, 
                                    f"{figure_folder}/cross_section_heads_t_qdir.png",
                                    boundary_keywords = ["WEL"], ve=100,
                                    flow_dir = True, qx=qx, qy=qy, qz=qz, 
                                    surface = True, layers=False,
                                    show=False, save=True, figsize = (19, 4),
                                    title=f"Cross section at time {elapsed_time} days")

    modtransient6.plot_head_time_series(head_file_t, 
                                        gwf, 
                                        f"{figure_folder}/head_ts.png",
                                        show = False, 
                                        save = True, 
                                        tau = None, 
                                        time_units="years")

#--------------------------------------- FLOW BUDGET ---------------------------------------------#
if plot_budget: 
    modtransient6.process_csv_budget(budget_file_t)  

    modtransient6.plot_bud_sum_transient(budget_file_t, elapsed_time, 
                                        f"{figure_folder}/bud_sum_t.png", 
                                        show = False, save=True)

    modtransient6.plot_bud_time_series(budget_file_t,  
                                    f"{figure_folder}/budget_ts.png", 
                                    show=False, save=True, 
                                    time_units="years")

    modtransient6.plot_water_to_wells(budget_file_t, 
                                    f"{figure_folder}/water_to_wells.png", 
                                    show=False, save=True, 
                                    time_units="years")

    modtransient6.plot_net_flow_time_series(budget_file_t,
                                            f"{figure_folder}/net_flow_ts.png",
                                            show=False, save=True, tau = None, 
                                            time_units="years")

    modtransient6.plot_storage_change_rate(budget_file_t, 
                        f"{figure_folder}/storage_change_rate.png", 
                        show=False, 
                        save=True, 
                        figsize=(14, 12), 
                        fontsize=14,
                        xlim=None,  # Tuple for x-axis limits
                        ylim=None,
                        time_units="years")

    #--------------------------------------- ZONE BUDGET ---------------------------------------------#

    modtransient6.process_csv_zonebudget(zonebud_file_t)
    modtransient6.plot_zone_budget(zonebud_file_t,
                                    output_folder, 
                                    figure_folder, 
                                    show=False, save=True, 
                                    zone_descriptions = {
                                    1: "Unconfined Aquifer",
                                    2: "Aquitard",
                                    3: "Confined Aquifer",
                                    4: "Aquitard",
                                    5: "Confined Aquifer"}, 
                                    time_units="years")

    modtransient6.plot_water_to_wells_zonebud(zonebud_file_t, figure_folder, 
                                                show=False, save=True, 
                                                time_units="years")

# ------------------------------------- RESPONSE TIMES ------------------------------------------ #
stability_threshold = 0.0005
threshold_absolute = 0.05
threshold_percent = 1
histogram_bins = 50

if response_times:

    start = 3600000 #start of the step change in model units
    start_time_years = start / 360  # Convert to years assuming 360 days/year
    start_step = 1 # Corresponding time step index

    modtransient6.tr_storage_change_rate_zones(zonebud_file_t, output_folder, figure_folder, 
                show=False, save_csv=True, save_fig=True, 
                figsize=(14, 12), fontsize=14,
                xlim=None, ylim=None, threshold=threshold_percent, threshold_type="relative", 
                start_time=start_time_years+1, fig_name="tr_storage_change_rate_zones_relative.png")

    modtransient6.tr_storage_change_rate(zonebud_file_t, output_folder, figure_folder, 
                show=False, save_csv=True, save_fig=True, 
                figsize=(14, 12), fontsize=14,
                xlim=None, ylim=None, threshold=threshold_percent, threshold_type="relative", 
                start_time=start_time_years+1, fig_name="tr_storage_change_rate_relative.png")
    
    modtransient6.tr_storage_change_rate_zones(zonebud_file_t, output_folder, figure_folder, 
                show=False, save_csv=False, save_fig=True, 
                figsize=(14, 12), fontsize=14,
                xlim=None, ylim=None, threshold=threshold_absolute, threshold_type="absolute", 
                start_time=start_time_years+1, fig_name="tr_storage_change_rate_zones_absolute.png")

    modtransient6.tr_storage_change_rate(zonebud_file_t, output_folder, figure_folder, 
                show=False, save_csv=False, save_fig=True, 
                figsize=(14, 12), fontsize=14,
                xlim=None, ylim=None, threshold=threshold_absolute, threshold_type="absolute", 
                start_time=start_time_years+1, fig_name="tr_storage_change_rate_absolute.png")
    
    modtransient6.response_time_array_relative( gwf,
                                                steady_state_heads,
                                                transient_heads,
                                                times_list,
                                                threshold_percent=threshold_percent,
                                                stability_threshold=stability_threshold,
                                                array_output_folder=output_folder,
                                                fig_output_folder=figure_folder,
                                                save_array=True,
                                                save_plot=True,
                                                show_plot=False,
                                                start_step=start_step,
                                                boundary_keywords=["WEL"],
                                                fill="start",
                                                bounds="95p",
                                                ve=100,
                                                fig_name="Response_time_relative_01.png")
    
    modtransient6.response_time_array_relative(   gwf,
                                                    steady_state_heads,
                                                    transient_heads,
                                                    times_list,
                                                    threshold_percent=threshold_percent,
                                                    stability_threshold=stability_threshold,
                                                    array_output_folder=output_folder,
                                                    fig_output_folder=figure_folder,
                                                    save_array=True,
                                                    save_plot=True,
                                                    show_plot=False,
                                                    start_step=start_step,
                                                    boundary_keywords=["WEL"],
                                                    max_initial_diff=True,
                                                    fill="start",
                                                    fig_name="Response_time_relative_max_initial_diff_01.png",
                                                    ve=100)
    
    modtransient6.response_time_array_absolute(gwf,
                                                steady_state_heads,
                                                transient_heads,
                                                times_list,
                                                threshold=threshold_percent,
                                                threshold_type="relative",
                                                stability_threshold=stability_threshold,
                                                array_output_folder=output_folder,
                                                fig_output_folder=figure_folder,
                                                save_array=True,
                                                save_plot=True,
                                                show_plot=False,
                                                start_step=start_step,
                                                boundary_keywords=["WEL"],
                                                fill="start",
                                                ve=100,
                                                fig_name="Response_time_absolute_01.png")

    modtransient6.response_time_array_relative( gwf,
                                                steady_state_heads,
                                                transient_heads,
                                                times_list,
                                                threshold_percent=threshold_percent,
                                                stability_threshold=stability_threshold,
                                                array_output_folder=output_folder,
                                                fig_output_folder=figure_folder,
                                                save_array=True,
                                                save_plot=True,
                                                show_plot=False,
                                                start_step=start_step,
                                                boundary_keywords=["WEL"],
                                                fill="nan",
                                                bounds="95p",
                                                ve=100,
                                                fig_name="Response_time_relative_02.png",
                                                histogram=True,
                                                histogram_bins=histogram_bins,
                                                histogram_name="Response_time_relative_02_histogram.png")
    
    modtransient6.response_time_array_relative(   gwf,
                                                    steady_state_heads,
                                                    transient_heads,
                                                    times_list,
                                                    threshold_percent=threshold_percent,
                                                    stability_threshold=stability_threshold,
                                                    array_output_folder=output_folder,
                                                    fig_output_folder=figure_folder,
                                                    save_array=True,
                                                    save_plot=True,
                                                    show_plot=False,
                                                    start_step=start_step,
                                                    boundary_keywords=["WEL"],
                                                    max_initial_diff=True,
                                                    fill="nan",
                                                    fig_name="Response_time_relative_max_initial_diff_02.png",
                                                    ve=100, histogram=True, histogram_bins=histogram_bins,
                                                    histogram_name="Response_time_relative_max_initial_diff_02_histogram.png")
    
    modtransient6.response_time_array_absolute(gwf,
                                                steady_state_heads,
                                                transient_heads,
                                                times_list,
                                                threshold=threshold_percent,
                                                threshold_type="relative",
                                                stability_threshold=stability_threshold,
                                                array_output_folder=output_folder,
                                                fig_output_folder=figure_folder,
                                                save_array=True,
                                                save_plot=True,
                                                show_plot=False,
                                                start_step=start_step,
                                                boundary_keywords=["WEL"],
                                                fill="nan",
                                                ve=100,
                                                fig_name="Response_time_absolute_02.png",
                                                histogram=True,
                                                histogram_bins=histogram_bins,
                                                histogram_name="Response_time_absolute_02_histogram.png")

    modtransient6.response_time_array_relative( gwf,
                                                steady_state_heads,
                                                transient_heads,
                                                times_list,
                                                threshold_percent=threshold_percent,
                                                stability_threshold=stability_threshold,
                                                array_output_folder=output_folder,
                                                fig_output_folder=figure_folder,
                                                save_array=True,
                                                save_plot=True,
                                                show_plot=False,
                                                start_step=start_step,
                                                boundary_keywords=["WEL"],
                                                fill="max",
                                                bounds="95p",
                                                ve=100,
                                                fig_name="Response_time_relative_03.png")
    
    modtransient6.response_time_array_relative(   gwf,
                                                    steady_state_heads,
                                                    transient_heads,
                                                    times_list,
                                                    threshold_percent=threshold_percent,
                                                    stability_threshold=stability_threshold,
                                                    array_output_folder=output_folder,
                                                    fig_output_folder=figure_folder,
                                                    save_array=True,
                                                    save_plot=True,
                                                    show_plot=False,
                                                    start_step=start_step,
                                                    boundary_keywords=["WEL"],
                                                    max_initial_diff=True,
                                                    fill="max",
                                                    fig_name="Response_time_relative_max_initial_diff_03.png",
                                                    ve=100)
    
    modtransient6.response_time_array_absolute(gwf,
                                                steady_state_heads,
                                                transient_heads,
                                                times_list,
                                                threshold=threshold_percent,
                                                threshold_type="relative",
                                                stability_threshold=stability_threshold,
                                                array_output_folder=output_folder,
                                                fig_output_folder=figure_folder,
                                                save_array=True,
                                                save_plot=True,
                                                show_plot=False,
                                                start_step=start_step,
                                                boundary_keywords=["WEL"],
                                                fill="max",
                                                ve=100,
                                                fig_name="Response_time_absolute_03.png")
           
    modtransient6.absolute_head_diffusion_zones(transient_heads, steady_state_heads, 
                                                times_list, zone_array, 
                                                start_step=start_step,
                                                threshold=threshold_percent, threshold_type="relative",
                                                stability_threshold=stability_threshold,
                                                array_output_folder=output_folder, fig_output_folder=figure_folder,
                                                save_fig=True, show_fig=False, 
                                                zone_descriptions = {
                                                1: "Unconfined Aquifer",
                                                2: "Aquitard",
                                                3: "Confined Aquifer",
                                                4: "Aquitard",
                                                5: "Confined Aquifer"}, 
                                                bounds="95p")
    
    modtransient6.absolute_head_diffusion(transient_heads, steady_state_heads, times_list,
                                            start_step=start_step, 
                                            threshold=threshold_percent, threshold_type="relative",
                                            stability_threshold=stability_threshold,
                                            fig_output_folder=figure_folder, 
                                            save_fig=True, show_fig=False, bounds = "95p")
    
    modtransient6.relative_head_diffusion_zones(transient_heads, steady_state_heads, 
                                                times_list, zone_array,
                                                start_step=start_step, 
                                                threshold_percent=threshold_percent, 
                                                stability_threshold=stability_threshold,
                                                array_output_folder=output_folder, fig_output_folder=figure_folder,
                                                save_fig=True, show_fig=False,
                                                zone_descriptions = {
                                                1: "Unconfined Aquifer",
                                                2: "Aquitard",
                                                3: "Confined Aquifer",
                                                4: "Aquitard",
                                                5: "Confined Aquifer"}, 
                                                bounds="stdev")
    
    modtransient6.relative_head_diffusion(transient_heads, steady_state_heads, times_list,
                                            start_step=start_step, 
                                            threshold_percent=threshold_percent, stability_threshold=stability_threshold,
                                            fig_output_folder=figure_folder, 
                                            save_fig=True, show_fig=False, bounds="stdev")
    
    #--------------------------------------- RESIDUAL DIFFUSION ANIMATION ---------------------------------------------#

if diffusion_animation:

    start = 3600000 #start of the step change in model units
    step = 3600000 #Size of the steps in model units
    n = 50
    end = start + (step*n)
    vmax = None

    for t in range(start, end, step):
        res_diff_array = modtransient6.plot_residual_diffusion(  gwf=gwf,
                                                start_time=start,
                                                time=t,
                                                perioddata=perioddata,
                                                nrow=nrow//2,
                                                transient_heads=transient_heads,
                                                steady_state_heads=steady_state_heads,
                                                title=f"Absolute residual difussion in hydraulic heads after {int((t - start)/360)} years",
                                                label="Head difference (m)",
                                                vmin=0,
                                                vmax=vmax,
                                                save=True,
                                                ve=100,
                                                output_folder=f"{figure_folder}/Residual_diffusion", 
                                                plot_name = f"Residual_diffusion_{t}.png",
                                                boundary_keywords=["WEL"])
        
        if vmax is None:
            vmax = np.nanmax(res_diff_array)
    
    print("Creating diffusion animation...")
    modplot6.animate(f"{figure_folder}/Residual_diffusion", f"{figure_folder}/Residual_diffusion.gif", duration=250)

#--------------------------------------- TRANSIENT ANIMATION ---------------------------------------------#

if animate:
    modplot6.plot_animation(gwf, transient_heads, nrow//2, 
                                f"{figure_folder}/cross_sections_tr",
                                f"{figure_folder}/cross_section_animation_tr.gif",
                                boundary_keywords = ["WEL"], ve = 100,
                                flow_dir = False, qx=qx, qy=qy, qz=qz, 
                                surface = True, layers=False,
                                show=False, save=True, figsize = (19, 4), 
                                gif_start=0, gif_step=20, duration=250)

end_time = time.time()
print(f"Total execution time: {end_time - start_time:.2f} seconds")