# ==========================================================================================
#  Model.py - MODFLOW 6 Model Setup and Execution Script
# ==========================================================================================
#
#  Author: MARIN RIVERA Carlos Felipe
#  Organization: Bordeaux INP, Lab EPOC, Université de Bordeaux
#  Project: Funded by the OneWater PEPR DEESAC Project
#
#  DESCRIPTION:
#  ------------
#  This script sets up, runs, and analyzes MODFLOW 6 groundwater flow model of a synthetic
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
# ----------------------------------------- STEADY STATE --------------------------------- #
# ---------------------------------------------------------------------------------------- #

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
from flopy.utils.geometry import Point, LineString, MultiPoint
from flopy.discretization import StructuredGrid
from flopy.utils.gridintersect import GridIntersect
import flopy.utils.binaryfile as bf
import pyemu

# Plot settings
import matplotlib
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

boundary_keywords = ["GHB", "WEL", "DRN"] #List of boundaries used in the model for plotting
heterogeneity = False # If True, generates random hydraulic conductivity fields

STEADY = True # Runs the steady state model
plot_steady = True # Plots steady state outputs
iterate = False # Iterates pumping rates over steady state model. Uses q_values defined above

TRANSIENT = True # Runs the transient model
plot_transient = True # Plots transient outputs
animate = False # Animates transient cross sections

# ------------------------------------------------------------------------------- #
# --------------------------------- MODEL SETUP --------------------------------- #
# ------------------------------------------------------------------------------- #

# Set model directory and model names
model_ws = 'mf'
model_name = 'DEESAC'
model_name_tr = 'DEESACt'
output_folder = f"{model_ws}/output"
figure_folder = f"{model_ws}/fig"
os.makedirs(output_folder, exist_ok=True)
os.makedirs(figure_folder, exist_ok=True)

setup_file = "setup.xlsx" # Excel file containing model setup parameters

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

# Create idomain, irch and recharge arrays
epsilon = 0 # Minimum allowed cell thickness in meters
transition = 70 # Transitions cells (Just used when SMOOTH_TOPO is set to True)
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
zone_array = np.zeros((nlay, nrow, ncol), dtype=int) # Create zones array for zone budget
for i in range(nlay):
    zone_array[i, :, :] = i + 1
kh_array = modgeom6.compute_3Darray(kh, idomain)

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

# ------------------------------------------------------------------------------- #
# ----------------------- LAYER SUBDIVISION -------------------------------------- #
# ------------------------------------------------------------------------------- #

nsub = [5,5,5,5,5]
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

#Update thickness, irch, R_array, zone_array and kh_array
thickness_array = ztop_array - zbot
irch = modgeom6.compute_irch(idomain)
R_array = modgeom6.compute_recharge(irch, R)

# ------------------------------------------------------------------------------- #
# --------------------------- BUILDING SIMULATION ------------------------------- #
# ------------------------------------------------------------------------------- #

# Create the flopy simulation object
sim = flopy.mf6.MFSimulation(sim_name = model_name, 
                             sim_ws = model_ws, 
                             exe_name = "mf6")

#Create the temporal discretiztion
# List contaning tupples: (Stress periods lenght, time steps, multiplier)
# Lenght of list = number of stress periods
tdis = flopy.mf6.ModflowTdis(sim, 
                             pname = "tdis", 
                             time_units = "DAYS", 
                             nper = 1, 
                             perioddata = [(1, 1, 1)]) 

# Create the groundwater flow model object
gwf = flopy.mf6.ModflowGwf(sim, 
                           modelname = model_name, 
                           save_flows = True, 
                           newtonoptions = "NEWTON UNDER_RELAXATION",
                           model_nam_file = f"{model_name}.nam")

#Create the Iterative Model Solution
ims = flopy.mf6.ModflowIms(sim, pname="ims",
                           print_option="SUMMARY",
                           complexity="COMPLEX",
                           outer_dvclose=0.001,
                           outer_maximum=1000,
                           under_relaxation="NONE",
                           inner_maximum=1000,
                           inner_dvclose=0.001,
                           rcloserecord=0.0001,
                           linear_acceleration="BICGSTAB",
                           scaling_method="NONE",
                           reordering_method="NONE",
                           relaxation_factor=0.9,
                           filename=f"{model_name}.ims")
sim.register_ims_package(ims, [gwf.name])

# Set the spatial discretization package
dis = flopy.mf6.ModflowGwfdis(gwf, 
                              nlay=nlay, nrow=nrow, ncol=ncol, 
                              delr=dcol, delc=drow, 
                              top=ztop, botm=zbot, idomain=idomain,
                              filename=f"{model_name}.dis")

# Set the initial conditions
ic = flopy.mf6.ModflowGwfic(gwf, 
                            pname = "ic", 
                            strt = ztop_array,
                            filename=f"{model_name}.ic")

# Set the Node Property Flow package
npf = flopy.mf6.ModflowGwfnpf(gwf,
                              pname = "npf",
                              save_specific_discharge = True,
                              icelltype= modgeom6.subdivide_array(np.array([1, 1, 1, 1, 1]), nsub), 
                              k=kh_array,
                              k33=kh_array/10,
                              filename=f"{model_name}.npf")

# Output control
oc = flopy.mf6.ModflowGwfoc(
    gwf,
    pname = "oc",
    head_filerecord = f"output/{model_name}.hds",
    budget_filerecord = f"output/{model_name}.cbb",
    budgetcsv_filerecord = f"output/{model_name}_budget.csv",
    saverecord = [("HEAD", "ALL"), ("BUDGET", "ALL")],
    printrecord = [("HEAD", "ALL"),("BUDGET", "ALL")], 
    filename = f"{model_name}.oc")

# --------------------------- BOUNDARY CONDITIONS ------------------------------- #

#River package
# riv_cells = modbound6.extract_active_cells_zone(irch, idomain, zone_array, nrow//2, nrow//2, 0, ncol-25, zones = [1,2,3,4,5])
# #riv_cells = riv_cells[:-1] # leave the last cell
# riv_spd1 = modbound6.create_riv_spd(
#     riv_cells,
#     ztop_array,
#     thickness_array,
#     drn_cond,
#     drow,
#     river_width=1,
#     riverbed_thickness=1,
#     stage_type="absolute",
#     a=0,
#     b=1,
#     conc=None)
# riv1 = flopy.mf6.ModflowGwfriv(gwf, 
#                               pname = "riv",
#                               save_flows = True,
#                               stress_period_data = riv_spd1,
#                               filename = f"{model_name}.riv")

# Drain package
drn_cells1 = modbound6.extract_active_cells_zone(irch, idomain, zone_array, nrow//2, nrow//2, 0, ncol-2, zones = [1,2,3,4,5])
drn_cells = drn_cells1 
drn_spd = modbound6.create_drn_spd(
    drn_cells,
    ztop_array,
    thickness_array,
    drn_cond,
    drow,
    drain_width=1,
    drainbed_thickness=1,
    elev_type="absolute",
    a=0,
    conc=None)
drn = flopy.mf6.ModflowGwfdrn(gwf, 
                              pname = "drn",
                              save_flows = True,
                              stress_period_data = drn_spd,
                              filename = f"{model_name}.drn")

# Recharge package
rch = flopy.mf6.ModflowGwfrcha(gwf, 
                               pname = "rch",
                               save_flows = True,
                               fixed_cell= False,
                               irch=irch,
                               recharge = R_array,
                               filename = f"{model_name}.rcha")

# Well package
well_df = pd.read_excel(setup_file, sheet_name="wells")
wel_spd = {}
wel_spd[0] = []
for well_id, group in well_df.groupby("well_id"):
    # Extract unique lay, row, col for this well
    lay = group["lay"].iloc[0]
    row = group["row"].iloc[0]
    col = group["col"].iloc[0]
    
    # Find the pumping rate at the first time step (or modify as needed)
    #q0 = group.loc[group["time"] == 0, "q"].iloc[0]  # assumes time=0 exists

    q0 = -2
    
    # Append tuple to list
    wel_spd[0].append((lay, row, col, q0, well_id))

wel = flopy.mf6.ModflowGwfwel(gwf, 
                              pname = "wel",
                              save_flows = True,
                              boundnames=True,
                              stress_period_data = wel_spd, 
                              filename = f"{model_name}.wel")

# General head boundary package
# GHB in the lateral outflow
ghb_1 = ztop_array[0,0,ncol-1] # Head in the GHB
ghb_spd1 = {}
ghb_spd1[0] = [
    ((ilay, 0, ncol-1), ghb_1, 100 * kh[ilay] * base_thicknesses[ilay] * width, f"Layer{ilay}")
    for ilay in range(nlay)]

# GHB in the top of first layer
# ghb_cells2 = modbound6.extract_active_cells_range(irch, idomain, nrow//2, nrow//2,col_start=ncol-25, col_end=ncol-2)
# ghb_spd2 = {}
# ghb_spd2[0] = [((k, i, j), ztop_array[k,i,j], 100 * kh[k]*dcol*width, "top_ghb") for (k, i, j) in ghb_cells2]
# ghb_spd1[0].extend(ghb_spd2[0])

ghb = flopy.mf6.ModflowGwfghb(gwf,
                                pname="ghb",
                                print_input=True,
                                print_flows=True,
                                save_flows=True,
                                boundnames=True,
                                filename = f"{model_name}.ghb",
                                stress_period_data=ghb_spd1)

# --------------------------------------------------------------------------- #    
# ---------------------------- RUN SIMULATION ------------------------------- #
# --------------------------------------------------------------------------- #  

if STEADY:
    sim.write_simulation()
    sim.run_simulation()

    # -------------------------- ZONE BUDGET -------------------------- #

    zonebud = gwf.output.zonebudget(zone_array)
    zonebud.change_model_ws(output_folder)
    zonebud.write_input()
    zonebud.run_model()

    # -------------------------- OUTPUTS -------------------------- #

    head = gwf.output.head().get_data()
    steady_state_heads = head
    bud = gwf.output.budget()
    spdis = bud.get_data(text='DATA-SPDIS')[0]
    qx, qy, qz = flopy.utils.postprocessing.get_specific_discharge(spdis, gwf)
    budget_file = f"{output_folder}/{model_name}_budget.csv"

    # -------------------------- PLOTTING -------------------------- #

    if plot_steady:
        modplot6.plot_cross_section_row(gwf, head, qx, qy, qz, nrow//2, 
                                f"{figure_folder}/cross_section_heads.png",
                                boundary_keywords = ["WEL"],
                                flow_dir = False, surface = False, 
                                show=False, save=True, figsize=(19, 4), layers = False, 
                                title="Cross section - Steady state simulation")
        modplot6.plot_cross_section_row(gwf, head, qx, qy, qz, nrow//2, 
                                f"{figure_folder}/cross_section_heads_qdir.png",
                                boundary_keywords = ["WEL"],
                                flow_dir = True, surface = True, 
                                show=False, save=True, figsize=(19, 4), layers = False, 
                                title="Cross section - Steady state simulation")

        modplot6.plot_bud_sum_steady(budget_file, 
                                 f"{figure_folder}/bud_sum_ss.png", 
                                show=False, save=True, figsize=(14, 5), fontsize=14)

        modplot6.plot_cross_section_array(gwf, 
                             nrow//2, 
                             f"{figure_folder}/cross_section_kh.png", 
                             boundary_keywords=None, 
                             show = False, 
                             save = True, 
                             figsize=(19, 5),
                             fontsize=14,
                             log=True,
                             array=kh_array,
                             label="Hydraulic Conductivity (m/d)", 
                             title="Model layers")

        modplot6.plot_cross_section_array(  gwf,
                                            nrow//2,
                                            f"{figure_folder}/cross_section_layers.png",
                                            boundary_keywords= boundary_keywords,
                                            show=False,
                                            save=True,
                                            ax=None,
                                            figsize=(19, 6),
                                            fontsize=14,
                                            array=None,
                                            title="Boundary conditions",
                                            colorbar=False,
                                            log=False)      
        
        # Plot heads with im.show
        #masked_head = np.where(idomain == 0, np.nan, head)
        #plt.imshow(masked_head[:,0,:], aspect=300, interpolation=None)
        #plt.colorbar()
        #plt.show()
    # ----------------------------------------------------------------------------- #
    # -------------------------- ITERATION PUMPING RATES -------------------------- #
    # ----------------------------------------------------------------------------- #

    if iterate:
        # Set pumping wells
        q_df = pd.read_excel(setup_file, sheet_name="q_values_st", index_col=0)
        q_values = [tuple(row) for row in q_df.iloc[:, :].values]
        q_ref = tuple(q_df.loc[well_id][0] for well_id in q_df.index) # Reference pumping rates for each well (first value, generally zero)

        # Run the iterate_pumping_rate function 
        modpump6.iterate_pumping_rate_steady(model_ws, sim, gwf, wel_spd, wel, q_values, q_ref, budget_file, nrow//2,
                                        f"{figure_folder}",
                                        f"{output_folder}/{model_name}_modpump6_ss.csv",
                                        boundary_keywords = ["WEL"],
                                        animate = True, animation_name = "cross_section_animation_ss.gif",
                                        duration = 250, #In seconds, duration of each frame
                                        save_budget = True, save_wells = True, save_csv = True)

if TRANSIENT:
    
    # -------------------------------------------------------------------------------- #
    # -------------------------------------- TRANSIENT ------------------------------- #
    # -------------------------------------------------------------------------------- #

    # -------------------------------------------------------------------------------- #
    # ---------------------------------- IMPORT MODULES ------------------------------ #
    # -------------------------------------------------------------------------------- #

    # Check the current and parent directories
    current_dir = os.getcwd()
    print("Current Directory:", current_dir)
    parent_dir = os.path.dirname(current_dir)
    print("Parent Directory:", parent_dir)

    # Dispose of warnings
    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)
      
    # ------------------------------------------------------------------------------- #
    # --------------------------------- MODEL SETUP --------------------------------- #
    # ------------------------------------------------------------------------------- #
    
    # Recall groundater flow model object from simulation
    gwf = sim.gwf[0]
    gwf.model_nam_file = f"{model_name_tr}.nam"
    ncol = gwf.dis.ncol.get_data()
    nrow = gwf.dis.nrow.get_data()
    nlay = gwf.dis.nlay.get_data()

    # ----------------------------- UPDATE PACKAGES ----------------------------------- #

    # Update time discretization
    tdis_df = pd.read_excel(setup_file, sheet_name="tdis")
    nper = len(tdis_df.index)
    perlen = tdis_df["perlen"].tolist()
    nstp = tdis_df["nstp"].tolist()
    tsmult = tdis_df["tsmult"].tolist()
    perioddata = list(zip(perlen, nstp, tsmult))
    tdis = sim.tdis
    tdis.nper = nper
    tdis.perioddata = perioddata

    # Update the initial conditions (this block is ignored since steady_state={0: True} in the storage package)
    ic = gwf.ic
    if STEADY:
        ic.strt = steady_state_heads
    else:
        ic.strt = ztop_array
    ic.filename = f"{model_name_tr}.ic"

    # Create storage package for transient simulation
    sto = flopy.mf6.ModflowGwfsto(
        gwf,
        pname="sto",
        iconvert = 1, #Unconfined/confined mixed storage is used
        sy=sy, #Specific yield
        ss=ss, #If not specified, flopy uses default value of 1e-5 m-1
        ss_confined_only=True,
        steady_state={0: True}, # First stress period is steady state
        transient={1: True}, 
        filename=f"{model_name_tr}.sto")

    # Update output control
    oc = gwf.oc
    oc.head_filerecord = f"output/{model_name_tr}.hds"
    oc.budget_filerecord = f"output/{model_name_tr}.cbb"
    oc.budgetcsv_filerecord = f"output/{model_name_tr}_budget.csv"
    oc.filename = f"{model_name_tr}.oc"

    # ---------------------------- UPDATE TRANSIENT BOUNDARY CONDITIONS -------------------------- #
    # ---------------------------- Update transient recharge package
    rch = flopy.mf6.ModflowGwfrcha(gwf, 
                                pname = "rch",
                                save_flows = True,
                                fixed_cell= True,
                                irch=irch,
                                recharge = "TIMEARRAYSERIES recharge", 
                                filename = f"{model_name_tr}.rcha")
    #Manual step changes
    #tas_data = {0 : R_array,
    #        3000000 : R_array,
    #        20000000 : R_array} 

    # Load your recharge series CSV
    df = pd.read_excel(setup_file, sheet_name="transient_recharge")

    # For recharge specified per layer
    # Extract time steps and R time series per layer
    time_steps = df.iloc[:, 0].values  # shape (n_times_steps,)
    R_vectors = df.iloc[:, 1:].values  # shape (n_time_steps, n_layers)
    # Compute recharge arrays per time step
    tas_data = {}
    for i, t in enumerate(time_steps):
        R = modgeom6.subdivide_array(R_vectors[i], nsub)
        recharge_array = modgeom6.compute_recharge(irch, R)  # shape (nrow, ncol)
        tas_data[t] = recharge_array

    # For single recharge series
    # tas_data = dict(zip(df.iloc[:, 0], df.iloc[:, 1]))

    rch.tas.initialize(
        filename="recharge_rates.ts",
        tas_array=tas_data,
        time_series_namerecord="recharge",
        interpolation_methodrecord="stepwise",)

    # ---------------------------- Update transient well package
    wel_spd = {}
    wel_spd[0] = []
    for well_id, group in well_df.groupby("well_id"):
        # Extract unique lay, row, col for this well
        lay = group["lay"].iloc[0]
        row = group["row"].iloc[0]
        col = group["col"].iloc[0]
        
        # Append tuple to list (uses key "wells" for time series initialization later)
        wel_spd[0].append((lay, row, col, well_id, well_id))

    wel = flopy.mf6.ModflowGwfwel(gwf, 
                              pname = "wel",
                              save_flows = True,
                              boundnames=True,
                              stress_period_data = wel_spd, 
                              filename = f"{model_name_tr}.wel")
    
    # Get unique wells ids and timesteps from well_df
    well_ids = sorted(well_df["well_id"].unique())
    all_times = sorted(well_df["time"].unique())

    ts_data = []
    last_q = {well: None for well in well_ids} # Prepare a dict to store the last known q for each well
    
    for t in all_times:
        row = [t]  # start an list with the timestep
        for well in well_ids:
            # Get the row in well_df with this well and time
            matching = well_df[(well_df["well_id"] == well) & (well_df["time"] == t)]
            if not matching.empty:
                q = matching["q"].iloc[0]
                last_q[well] = q  # update last known pumping rate
            else:
                q = last_q[well]  # use previous pumping rate
            row.append(q) # Append the pumping rate the row
        ts_data.append(tuple(row)) # Append the tuple to the data list

    wel.ts.initialize(
        filename="well_rates.ts",
        timeseries=ts_data,
        time_series_namerecord=well_ids,
        interpolation_methodrecord=["stepwise"]*len(well_ids))

    # ------------------------- OBSERVATIONS ------------------------------ #

    # Head observations
    obs_df = pd.read_excel(setup_file, sheet_name="observations")
    obs_recarray = {f"output/head_obs_t.csv": [
        (row["obs_label"], row["obs_type"], (row["lay"], row["row"], row["col"]))
        for _, row in obs_df.iterrows()]}

    obs_package = flopy.mf6.ModflowUtlobs(
        gwf,
        pname="head_obs_t",
        print_input=True,
        continuous=obs_recarray, 
        filename=f"{model_name_tr}.obs")
    
    # ---------------------------- RUN SIMULATION ------------------------------- #

    sim.write_simulation()
    sim.run_simulation()

    # ------------------------------ ZONE BUDGET -------------------------------- #

    zonebud= gwf.output.zonebudget(zone_array)
    zonebud.change_model_ws(output_folder)
    zonebud.write_input()
    zonebud.run_model()

    # --------------------------------------------------------------------------- #
    # ------------------------------- PLOTTING ---------------------------------- #
    # --------------------------------------------------------------------------- #
    if plot_transient:
        
        # Select time step, period, and layer to plot
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

        cb = gwf.oc.output.budget()
        cb.get_data(idx=0, full3D=True) #Get cell budget file

        budget_file_t = f"{output_folder}/{model_name_tr}_budget.csv"

        zonebud_file_t = f"{output_folder}/zonebud.csv"

        head_file_t = f"{output_folder}/head_obs_t.csv"

        #--------------------------------------- HEADS ---------------------------------------------#

        modplot6.plot_cross_section_row(gwf, head, qx, qy, qz, nrow//2, 
                                        f"{figure_folder}/cross_section_heads_t.png",
                                        boundary_keywords = ["WEL"],
                                        flow_dir = False, surface = False, layers=False,
                                        show=False, save=True, figsize = (19, 4),
                                        title=f"Cross section at time {elapsed_time} days")
        modplot6.plot_cross_section_row(gwf, head, qx, qy, qz, nrow//2, 
                                        f"{figure_folder}/cross_section_heads_t_qdir.png",
                                        boundary_keywords = ["WEL"],
                                        flow_dir = True, surface = True, layers=False,
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
        modtransient6.plot_zone_budget(zonebud_file_t, figure_folder, show=False, save=True, 
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


        # ------------------------------------- RELAXATION TIMES ------------------------------------------ #
        
        # start = 3600000 #start of the step change in model units
        # step = 3600 #Size of the steps in model units
        # n = 150
        # end = start + (step*n)

        # for t in range(start, end, step):
        #     modtransient6.plot_residual_diffusion(  gwf=gwf,
        #                                             start_time=start,
        #                                             time=t,
        #                                             perioddata=perioddata,
        #                                             nrow=nrow//2,
        #                                             transient_heads=transient_heads,
        #                                             steady_state_heads=steady_state_heads,
        #                                             title=f"Absolute residual difussion in hydraulic heads after {int((t - start)/360)} years",
        #                                             label="Head difference (m)",
        #                                             vmin=0,
        #                                             vmax=250,
        #                                             save=True,
        #                                             output_folder=f"{figure_folder}/Residual_diffusion", 
        #                                             plot_name = f"Residual_diffusion_{t}.png" )

        # modplot6.animate(f"{figure_folder}/Residual_diffusion", f"{figure_folder}/Residual_diffusion.gif", duration=250)

        # modtransient6.animate_sto_cb_cross_section( gwf,
        #                                             cb, # CellBudgetFile object
        #                                             nrow//2,  # Row index for cross-section
        #                                             f"{figure_folder}/cross_sections_sto",   # Folder to save individual plots
        #                                             f"{figure_folder}/animation_sto.gif",    # Path to save GIF
        #                                             boundary_keywords=None,
        #                                             show=False, save=True,
        #                                             figsize=(19, 6), fontsize=14,
        #                                             gif_start=30, gif_step=50, duration=250,
        #                                             vmin=-10e-12, vmax=0)
        #--------------------------------------- TRANSIENT ANIMATION ---------------------------------------------#

        if animate:
            modplot6.plot_animation(gwf, transient_heads, qx, qy, qz, nrow//2, 
                                        f"{figure_folder}/cross_sections_tr",
                                        f"{figure_folder}/cross_section_animation_tr.gif",
                                        boundary_keywords = ["WEL"],
                                        flow_dir = False, surface = True, layers=False,
                                        show=False, save=True, figsize = (19, 4), 
                                        gif_start=0, gif_step=20, duration=250)

end_time = time.time()
print(f"Total execution time: {end_time - start_time:.2f} seconds")