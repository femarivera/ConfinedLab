# ==========================================================================================
#  Model.py - MODFLOW 6 Flow Model Setup, Execution, Post-Processing and Analysis Script
# ==========================================================================================
#
#  Author: MARIN RIVERA Carlos Felipe
#  Organization: Bordeaux INP, Lab EPOC, Université de Bordeaux
#  Project: OneWater PEPR DEESAC Project
#
#  DESCRIPTION:
#  ------------
#  This script sets up, runs, and analyzes a MODFLOW6 groundwater flow model of a synthetic
#  multilayer aquifer system for the ConfinedLab project. It integrates model construction,  
#  steady state and transient simulation, and post-processing utilities for flow and budget 
#  analysis using FloPY utilities and custom modules.
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
from pprint import pformat

# Plot settings
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

# --------------------------- LOAD OR RUN OPTIONS ------------------------------- #
LOAD_MODEL = False # Loads an existing model to postprocess.

STEADY = True # Builds and runs a steady state simulation
TRANSIENT = True # Builds and runs a transient simulation

# ------------------------------- POSTPROCESS OPTIONS --------------------------- #
post_steady = True # Postprocess steady state outputs

post_transient = True # Postprocess transient outputs
response_times = True # Estimates response times from model transient files
animate = False # Animates transient cross sections
plot_maps = True # If True, plots map views of heads and flows

# --------------------- MODEL STRUCTURE OPTIONS --------------------------------- #
subdivide_layers = True

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

# ------------------------------------------------------------------------------- #
# --------------------------- STRUCTURED GRID GENERATION ------------------------ #
# ------------------------------------------------------------------------------- #

# Set model grid parameters from set up file
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

# Build synthetic geometry from setup file parameters
geom_df = pd.read_excel(setup_file, sheet_name="geometry")
outcrop_z = geom_df["outcrop_z"].to_numpy() # Elevation (Just used when SLOPE is set to False)
outcrop_zmax = geom_df["outcrop_zmax"].to_numpy() # Elevation (Just used when SLOPE are set to True)
outcrop_zmin = geom_df["outcrop_zmin"].to_numpy() # Elevation (Just used when SLOPE are set to True)
base_thicknesses = geom_df["base_thicknesses"].to_numpy() # Maximum layer thickness in meters
outcrop_cells = geom_df["outcrop_cells"].to_numpy() # Cell ID where the unit starts outcropping (measured from left to right)
zones = geom_df["zone"].to_numpy() # Zone ID for each layer
epsilon = float(geom_df["epsilon"].iloc[0]) # Minimum allowed cell thickness in meters
transition = int(geom_df["transition_cells"].iloc[0]) # Transition cells for layer thickness and top elevation changes

# Create initial idomain based on layer outcrop and confined areas
idomain = modgeom6.compute_idomain(nlay, nrow, ncol, outcrop_cells)

# Create topography 2D array, thickness 3D array, zbot 3D array
topo = modgeom6.compute_top(idomain, outcrop_z, transition=True, slope=True,
                            transition_cells=transition, transition_type="contain", 
                            outcrop_zmin=outcrop_zmin, outcrop_zmax=outcrop_zmax)
thickness_array = modgeom6.compute_thickness(idomain, base_thicknesses, 
                                             transition=True, transition_type="extend", 
                                             transition_cells=transition)
zbot = modgeom6.compute_bottom(topo, thickness_array)

# Create ztop 3D array (top elevation for each cell)
ztop = modgeom6.compute_ztop_array(topo, zbot)

# Create geological interfaces array for plotting
interfaces = np.concatenate(([topo], zbot), axis=0)

# Update idomain as needed (deactivate thin cells or specific layers,zones for example)
idomain = modgeom6.idomain_from_thickness(thickness_array, epsilon)

# Create zones 3D array for zonal budget
zone_array = modgeom6.compute_3Darray(zones, nlay, nrow, ncol, dtype = int)

# ------------------------------------------------------------------------------- #
# ------------------------------ MODEL PARAMETERS ------------------------------- #
# ------------------------------------------------------------------------------- #

# Set model hydraulic parameters per layer from setup file
par_df = pd.read_excel(setup_file, sheet_name="parameters", index_col=0)

# Get parameters per layer as 1D arrays (size nlay)
kh = modpar6.par_df_to_1Darray(par_df, "kh") # Horizontal hydraulic conductivity in m/d
kv = modpar6.par_df_to_1Darray(par_df, "kv") # Vertical hydraulic conductivity in m/d
sy = modpar6.par_df_to_1Darray(par_df, "sy") # Specific yield (adimensional)
ss = modpar6.par_df_to_1Darray(par_df, "ss") # Specific storage (m-1)
drn_cond = modpar6.par_df_to_1Darray(par_df, "drn_cond") # Hydraulic conductivity of drain bed (m/d)
recharge = modpar6.par_df_to_1Darray(par_df, "rech") # Recharge (m/d)

# Create 3D hydraulic parameter arrays
kh_array = modgeom6.compute_3Darray(kh, nlay, nrow, ncol)
kv_array = modgeom6.compute_3Darray(kv, nlay, nrow, ncol)
ss_array = modgeom6.compute_3Darray(ss, nlay, nrow, ncol)
sy_array = modgeom6.compute_3Darray(sy, nlay, nrow, ncol)

# Create irch and recharge 2D arrays 
irch = modgeom6.compute_irch(idomain)
R_array = modgeom6.compute_recharge(irch, recharge)

# Set well locations and pumping rates for transient and steady state simulations
well_st_df = pd.read_excel(setup_file, sheet_name="wells_st") # Well locations and steady state pumping rates
q0 = well_st_df["q"].sum() # Get the total steady state pumping rate for reference
well_df = pd.read_excel(setup_file, sheet_name="wells") # Well locations and transient pumping rates

# ------------------------------------------------------------------------------- #
# ----------------------- LAYER SUBDIVISION ------------------------------------- #
# ------------------------------------------------------------------------------- #
if subdivide_layers:

    nsub = geom_df["nsub"].to_list() # Number of subdivisions per layer

    # Subdivide layers: get new nlay, and idomain, ztop, zbot, and thickness 3D arrays
    nlay, idomain, ztop, zbot, thickness_array = modgeom6.subdivide_layers(idomain, ztop, zbot, nsub)

    # Update idomain as needed (deactivate thin cells or specific layers for example)
    idomain = modgeom6.idomain_from_thickness(thickness_array, epsilon)

    # Subdivide 1D arrays (new size nlay)
    kh = modgeom6.subdivide_array(kh, nsub)
    kv = modgeom6.subdivide_array(kv, nsub)
    sy = modgeom6.subdivide_array(sy, nsub)
    ss = modgeom6.subdivide_array(ss, nsub)
    drn_cond = modgeom6.subdivide_array(drn_cond, nsub)
    recharge = modgeom6.subdivide_array(recharge, nsub)

    # Subdivide 3D arrays (new size nlay, nrow, ncol)
    zone_array = modgeom6.subdivide_array(zone_array, nsub)
    kh_array = modgeom6.subdivide_array(kh_array, nsub)
    kv_array = modgeom6.subdivide_array(kv_array, nsub)
    ss_array = modgeom6.subdivide_array(ss_array, nsub)
    sy_array = modgeom6.subdivide_array(sy_array, nsub)

    # Recompute irch and recharge 2D arrays
    irch = modgeom6.compute_irch(idomain)
    R_array = modgeom6.compute_recharge(irch, recharge)

    # Compute storage coefficient, sto_cell_type, transmissivity, and diffusivity 3D arrays
    sy_cells = modbound6.extract_active_cells_range(irch, idomain, 0, nrow-1, 0, ncol-1) # List of top most active cells
    storage_coeff = modgeom6.storage_coefficient(sy_cells, idomain, ss_array, sy_array, thickness_array)
    sto_cell_type = modgeom6.storage_cell_type(sy_cells, idomain)
    transmissivity = kh_array * thickness_array
    diffusivity = np.where(idomain==1, transmissivity / storage_coeff, np.nan)

# ------------------------------------------------------------------------------- #
# -------------------------- RESPONSE TIME PARAMETERS --------------------------- #
# ------------------------------------------------------------------------------- #

# Read response time parameters from setup file
response_df = pd.read_excel(setup_file, sheet_name="response_times")

# Convert the sheet into a dictionary: {parameter_name: value}
params = response_df.set_index('Parameter')['value'].to_dict()

# Convert values to appropriate types (Defaults provided)
stability_threshold = float(params.get('stability_threshold', 0.0005))
start = int(params.get('start', 0))
start_step = int(params.get('start_step', 1))
threshold_absolute = float(params.get('threshold_absolute', 0.01))
threshold_percent = float(params.get('threshold_percent', 5))
step = int(params.get('step', 1))
n = int(params.get('n', 100))

# For histogram_bins, handle empty cells as None
histogram_bins_raw = params.get('histogram_bins')
histogram_bins = int(histogram_bins_raw) if pd.notna(histogram_bins_raw) else None

# ------------------------------------------------------------------------------- #
# ----------------------------- LOAD EXISTING SIMULATION ------------------------ #
# ------------------------------------------------------------------------------- #
if LOAD_MODEL:

    print("Loading existing model files...")

    # --- Load the existing simulation ---
    sim = flopy.mf6.MFSimulation.load(
    sim_name=model_name,
    sim_ws=model_ws,
    exe_name="mf6")

    # --- Get the GWF model (first one if only one exists) ---
    gwf = sim.gwf[0]

    # --- Retrieve the WEL package ---
    # Option 1: direct attribute (if it exists)
    try:
        wel = gwf.wel
    except AttributeError:
        # Option 2: safe method (always works)
        wel = gwf.get_package("WEL")

    # --- Retrieve WEL stress period data ---
    wel_spd = {
    kper: [(cid[0], cid[1], cid[2], q, name) for cid, q, name in zip(d['cellid'], d['q'], d['boundname'])]
    for kper, d in wel.stress_period_data.get_data().items()}

    tdis = sim.tdis
    perioddata = tdis.perioddata.get_data()

# --------------------------------------------------------------------------------- #
# ----------------------------------- BUILD STEADY STATE -------------------------- #
# --------------------------------------------------------------------------------- #
if STEADY:

    print("Building and running steady state simulation...")

    # Save zone_array
    np.save(f"{model_ws}/zone_array.npy", zone_array)

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
                            outer_dvclose=0.000001,
                            outer_maximum=2000,
                            under_relaxation="SIMPLE",
                            under_relaxation_gamma=0.1,
                            inner_maximum=2000,
                            inner_dvclose=0.000001,
                            rcloserecord=0.00001,
                            linear_acceleration="BICGSTAB",
                            scaling_method="NONE",
                            reordering_method="NONE",
                            relaxation_factor=0.97,
                            filename=f"{model_name}.ims")
    sim.register_ims_package(ims, [gwf.name])

    # Set the spatial discretization package
    dis = flopy.mf6.ModflowGwfdis(gwf, 
                                nlay=nlay, nrow=nrow, ncol=ncol, 
                                delr=dcol, delc=drow, 
                                top=topo, botm=zbot, idomain=idomain,
                                filename=f"{model_name}.dis")

    # Set the initial conditions 
    ic = flopy.mf6.ModflowGwfic(gwf, 
                                pname = "ic", 
                                strt = np.repeat(topo[np.newaxis, :, :], nlay, axis=0),
                                filename=f"{model_name}.ic")

    # Set the Node Property Flow package
    npf = flopy.mf6.ModflowGwfnpf(gwf,
                                pname = "npf",
                                save_specific_discharge = True,
                                save_flows= True,
                                save_saturation= True,
                                icelltype=1, 
                                k=kh_array,
                                k33=kv_array,
                                filename=f"{model_name}.npf")  

    # Output control
    oc = flopy.mf6.ModflowGwfoc(
        gwf,
        pname = "oc",
        head_filerecord = f"output/{model_name}.hds",
        budget_filerecord = f"output/{model_name}.cbb",
        budgetcsv_filerecord = f"output/{model_name}_budget.csv",
        saverecord = [("HEAD", "ALL"), ("BUDGET", "ALL")],
        printrecord = [("HEAD", "LAST"),("BUDGET", "LAST")], 
        filename = f"{model_name}.oc")

    # --------------------------- BOUNDARY CONDITIONS ------------------------------- #
    # Drain package
    drn_cells = modbound6.extract_active_cells_range(irch, idomain, 0, nrow-1, 0, ncol-2)
    drn_spd = modbound6.create_drn_spd(
        drn_cells,
        ztop,
        thickness_array,
        drn_cond,
        drain_length=drow,
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
                                irch=irch,
                                recharge = R_array,
                                filename = f"{model_name}.rcha")

    # Well package
    wel_spd = {}
    wel_spd[0] = []
    for well_id, group in well_st_df.groupby("well_id"):
        # Extract unique lay, row, col, q for this well
        lay = int(group["lay"].iloc[0])
        row = int(group["row"].iloc[0])
        col = int(group["col"].iloc[0])
        q = float(group["q"].iloc[0]) 
        # Append tuple to stress period data
        wel_spd[0].append((lay, row, col, q, well_id))
    wel = flopy.mf6.ModflowGwfwel(gwf, 
                                pname = "wel",
                                save_flows = True,
                                boundnames=True,
                                stress_period_data = wel_spd, 
                                filename = f"{model_name}.wel")

    # General Head Boundary package
    ghb_1 = ztop[0,0,ncol-1] # Head in the GHB
    ghb_spd1 = {}
    ghb_spd1[0] = [
        ((ilay, irow, ncol-1), ghb_1, kh[ilay] * thickness_array[ilay, irow, ncol-1] * drow, f"Layer{ilay}")
        for ilay in range(nlay)
        for irow in range(nrow)] #Conductance set to transmissivity of the cell
    ghb = flopy.mf6.ModflowGwfghb(gwf,
                                    pname="ghb",
                                    print_input=True,
                                    print_flows=True,
                                    save_flows=True,
                                    boundnames=True,
                                    filename = f"{model_name}.ghb",
                                    stress_period_data=ghb_spd1)
    
    # ---------------------------------- OBSERVATIONS --------------------------------------- #

    # Head observations
    obs_df = pd.read_excel(setup_file, sheet_name="observations")
    obs_recarray = {f"output/head_obs.csv": [
        (row["obs_label"], row["obs_type"], (row["lay"], row["row"], row["col"]))
        for _, row in obs_df.iterrows()]}

    obs_package = flopy.mf6.ModflowUtlobs(
        gwf,
        pname="head_obs",
        print_input=True,
        continuous=obs_recarray, 
        filename=f"{model_name}.obs")
    
    # --------------------------------------------------------------------------------- #    
    # ------------------------------ RUN STEADY STATE --------------------------------- #
    # --------------------------------------------------------------------------------- #  

    sim.write_simulation()
    sim.run_simulation()

    # -------------------------- ZONE BUDGET -------------------------- #

    zonebud = gwf.output.zonebudget(zone_array)
    zonebud.change_model_ws(output_folder)
    zonebud.write_input()
    zonebud.run_model()

# --------------------------------------------------------------------------------- #
# --------------------------- POSTPROCESS STEADY STATE ---------------------------- #
# --------------------------------------------------------------------------------- #

# -------------------------- OUTPUTS -------------------------- #

print("Reading steady state outputs...")

head_file_path = f"{output_folder}/{model_name}.hds"
hobj_ss = flopy.utils.HeadFile(head_file_path)
steady_state_heads = hobj_ss.get_data() # Or steady_state_heads = gwf.output.head().get_data()

budget_file_path = f"{output_folder}/{model_name}.cbb"
bud = flopy.utils.CellBudgetFile(budget_file_path, precision="double") # or bud = gwf.output.budget()
spdis = bud.get_data(text='DATA-SPDIS')[0]
qx, qy, qz = flopy.utils.postprocessing.get_specific_discharge(spdis, gwf)
budget_file = f"{output_folder}/{model_name}_budget.csv"

ve= length/(6*(ztop.max()-zbot.min())) # Vertical exaggeration for plots

if post_steady:

    print("Postprocessing steady state simulation...")

    if plot_maps:
        if steady_state_heads.shape[-2]>=2 and steady_state_heads.shape[-1]>=2:

            modplot6.plot_map_view(gwf, head_path=head_file_path, 
                                    output_path=f"{figure_folder}/map_heads_L1.png", 
                                    boundary_keywords=["WEL", "GHB"], 
                                    layer=0, flow_dir=False, contours=True,show=False, save=True,
                                    grid=True, figsize=(10, 10), fontsize=14,title="Model map view Layer 1")

            modplot6.plot_map_view(gwf, head_path=head_file_path, 
                                    output_path=f"{figure_folder}/map_heads_L2.png", 
                                    boundary_keywords=["WEL", "GHB"], 
                                    layer=7, flow_dir=False, contours=True,show=False, save=True,
                                    grid=True, figsize=(10, 10), fontsize=14,title="Model map view Layer 2")

            modplot6.plot_map_view(gwf, head_path=head_file_path, 
                                    output_path=f"{figure_folder}/map_heads_L3.png", 
                                    boundary_keywords=["WEL", "GHB"], 
                                    layer=12, flow_dir=False, contours=True,show=False, save=True,
                                    grid=True, figsize=(10, 10), fontsize=14,title="Model map view Layer 3")

            modplot6.plot_map_view(gwf, head_path=head_file_path, 
                                    output_path=f"{figure_folder}/map_heads_L4.png", 
                                    boundary_keywords=["WEL", "GHB"], 
                                    layer=17, flow_dir=False, contours=True,show=False, save=True,
                                    grid=True, figsize=(10, 10), fontsize=14,title="Model map view Layer 4")

            modplot6.plot_map_view(gwf, head_path=head_file_path, 
                                    output_path=f"{figure_folder}/map_heads_L5.png", 
                                    boundary_keywords=["WEL", "GHB"], 
                                    layer=22, flow_dir=False, contours=True,show=False, save=True,
                                    grid=True, figsize=(10, 10), fontsize=14,title="Model map view Layer 5")
    
    modplot6.plot_cross_section_row(gwf, 
                                    head_path=head_file_path, 
                                    row=nrow//2, 
                                    output_path=f"{figure_folder}/cross_section_heads.png",
                                    boundary_keywords = ["WEL"],
                                    flow_dir = False, surface = False, ve=ve,
                                    show=False, save=True, figsize=(19, 4), layers = False, 
                                    title="Cross section - Steady state simulation", interfaces=interfaces)
    modplot6.plot_cross_section_row(gwf, 
                                    head_path=head_file_path, 
                                    row = nrow//2, 
                                    output_path=f"{figure_folder}/cross_section_heads_qdir.png",
                                    boundary_keywords = ["WEL"], ve=ve,
                                    flow_dir = True, cbb_path=budget_file_path, 
                                    surface = True, 
                                    show=False, save=True, figsize=(19, 4), layers = False, 
                                    title="Cross section - Steady state simulation", interfaces=interfaces)

    modplot6.plot_bud_sum_steady(budget_file,
                                    f"{figure_folder}/bud_sum_ss.png",
                                    show=False, save=True, figsize=(14, 5), fontsize=14)

    modplot6.plot_cross_section_array(gwf, 
                                        row = nrow//2, 
                                        output_path=f"{figure_folder}/cross_section_kh.png", 
                                        boundary_keywords=None, 
                                        show = False, 
                                        save = True, 
                                        figsize=(19, 5),
                                        fontsize=14,
                                        ve=ve,
                                        log=True,
                                        array=kh_array,
                                        label="Hydraulic Conductivity (m/d)", 
                                        title="Model layers", 
                                        interfaces=interfaces)
    modplot6.plot_cross_section_array(gwf, 
                                        row = nrow//2, 
                                        output_path=f"{figure_folder}/cross_section_diffusivity.png", 
                                        boundary_keywords=None, 
                                        show = False, 
                                        save = True, 
                                        figsize=(19, 5),
                                        fontsize=14,
                                        ve=ve,
                                        log=True,
                                        array=diffusivity,
                                        label="Hydraulic diffusivity (m/d)", 
                                        title="Model layers", 
                                        interfaces=interfaces)

    modplot6.plot_cross_section_array(gwf,
                                      row= nrow//2,
                                      output_path=f"{figure_folder}/cross_section_layers.png",
                                      boundary_keywords= boundary_keywords,
                                      show=False,
                                      save=True,
                                      ax=None,
                                      figsize=(19, 6),
                                      fontsize=14,
                                      ve=ve,
                                      array=None,
                                      title="Boundary conditions",
                                      colorbar=False,
                                      interfaces=interfaces)   

# ---------------------------------------------------------------------------------------- #
# ------------------------------------ TRANSIENT SIMULAITON ------------------------------ #
# ---------------------------------------------------------------------------------------- #

if TRANSIENT: 

    print ("Building and running transient simulation...")

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
        ic.strt = np.repeat(topo[np.newaxis, :, :], nlay, axis=0)
    ic.filename = f"{model_name_tr}.ic"

    # Storage formulation
    sto = flopy.mf6.ModflowGwfsto(
        gwf,
        pname="sto",
        iconvert = sto_cell_type, #Confined only in outcrop cells (just uses ss), convertible elswhere
        storagecoefficient = True, # Uses ss as storage coefficient instead of specific storage
        sy=sy, #It is used just for convertible cells
        ss=storage_coeff, #Using an array that has sy for outcrop cells, and ss*thickness elsewhere
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

    # ---------------------------- Update transient recharge package ---------------------------- #
    rch = flopy.mf6.ModflowGwfrcha(gwf, 
                                pname = "rch",
                                save_flows = True,
                                irch=irch,
                                recharge = "TIMEARRAYSERIES recharge", 
                                filename = f"{model_name_tr}.rcha")

    # Load your recharge series CSV
    df = pd.read_excel(setup_file, sheet_name="transient_recharge")

    # ---- For recharge specified per layer
    # Extract time steps and R time series per layer
    time_steps = df.iloc[:, 0].values  # shape (n_times_steps,)
    R_vectors = df.iloc[:, 1:].values  # shape (n_time_steps, n_layers)
    # Compute recharge arrays per time step
    tas_data = {}
    for i, t in enumerate(time_steps):
        recharge = modgeom6.subdivide_array(R_vectors[i], nsub)
        recharge_array = modgeom6.compute_recharge(irch, recharge)  # shape (nrow, ncol)
        tas_data[t] = recharge_array

    # ---- For single recharge series
    # tas_data = dict(zip(df.iloc[:, 0], df.iloc[:, 1]))

    rch.tas.initialize(
        filename="recharge_rates.ts",
        tas_array=tas_data,
        time_series_namerecord="recharge",
        interpolation_methodrecord="stepwise",)

    # ---------------------------- Update transient well package ---------------------------- #
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
        row = [t]  # start a list with the timestep
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

    # ---------------------------------- OBSERVATIONS --------------------------------------- #

    # Head observations
    obs_df = pd.read_excel(setup_file, sheet_name="observations")
    obs_recarray = {f"output/head_obs_t.csv": [
        (row["obs_label"], row["obs_type"], (row["lay"], row["row"], row["col"]))
        for _, row in obs_df.iterrows()]}

    obs_package = flopy.mf6.ModflowUtlobs(
        gwf,
        pname="head_obs",
        print_input=True,
        continuous=obs_recarray, 
        filename=f"{model_name_tr}.obs")

    # --------------------------------------------------------------------------- #    
    # ----------------------------- RUN TRANSIENT ------------------------------- #
    # --------------------------------------------------------------------------- # 

    sim.write_simulation()
    sim.run_simulation()

    # ------------------------------ ZONE BUDGET -------------------------------- #

    zonebud= gwf.output.zonebudget(zone_array)
    zonebud.change_model_ws(output_folder)
    zonebud.write_input()
    zonebud.run_model()

# --------------------------------------------------------------------------- #
# ------------------------------ POSTPROCESS TRANSIENT ---------------------- #
# --------------------------------------------------------------------------- #

# -------------------------- OUTPUTS -------------------------- #

print("Reading transient simulation outputs...")

time_step_plot = 0  # Time step to plot (0 = first time step)

budget_file_path = f"{output_folder}/{model_name_tr}.cbb"
cb =  flopy.utils.CellBudgetFile(budget_file_path, precision = "double")  # gwf.output.budget()
steps = cb.get_kstpkper()
kstpkper = steps[0]
spdis = cb.get_data(text='DATA-SPDIS', kstpkper=kstpkper)[0]
qx, qy, qz = flopy.utils.postprocessing.get_specific_discharge(spdis, gwf)

transient_head_file_path = f"{output_folder}/{model_name_tr}.hds"
hobj_tr = flopy.utils.HeadFile(transient_head_file_path)
transient_heads = hobj_tr.get_alldata()
times_list = hobj_tr.get_times()

budget_file_t = f"{output_folder}/{model_name_tr}_budget.csv"
zonebud_file_t = f"{output_folder}/zonebud.csv"
head_file_t = f"{output_folder}/head_obs_t.csv"

modtransient6.process_csv_budget(budget_file_t)
modtransient6.process_csv_zonebudget(zonebud_file_t)  

if post_transient:
  
    print("Postprocessing transient simulation...")

    #--------------------------------------- HEADS ---------------------------------------------#
    if plot_maps:
        if transient_heads[time_step_plot].shape[-2]>=2 and transient_heads[time_step_plot].shape[-1]>=2:

            modplot6.plot_map_view(gwf, transient_head_file_path, 
                    f"{figure_folder}/map_heads_L1_t.png", 
                    boundary_keywords=["WEL", "GHB"], 
                    layer=0, flow_dir=False, contours=True,show=False, save=True,
                    grid=True, figsize=(10, 10), fontsize=14,title="Model map view Layer 1_t", 
                    transient=True, time_step=time_step_plot)

            modplot6.plot_map_view(gwf, transient_head_file_path,
                    f"{figure_folder}/map_heads_L2_t.png", 
                    boundary_keywords=["WEL", "GHB"], 
                    layer=7, flow_dir=False, contours=True,show=False, save=True,
                    grid=True, figsize=(10, 10), fontsize=14,title="Model map view Layer 2_t", 
                    transient=True, time_step=time_step_plot)

            modplot6.plot_map_view(gwf, transient_head_file_path, 
                    f"{figure_folder}/map_heads_L3_t.png", 
                    boundary_keywords=["WEL", "GHB"], 
                    layer=12, flow_dir=False, contours=True,show=False, save=True,
                    grid=True, figsize=(10, 10), fontsize=14,title="Model map view Layer 3_t", 
                    transient=True, time_step=time_step_plot)
            
            modplot6.plot_map_view(gwf, transient_head_file_path,
                    f"{figure_folder}/map_heads_L4_t.png", 
                    boundary_keywords=["WEL", "GHB"], 
                    layer=17, flow_dir=False, contours=True,show=False, save=True,
                    grid=True, figsize=(10, 10), fontsize=14,title="Model map view Layer 4_t",
                    transient=True, time_step=time_step_plot)
            
            modplot6.plot_map_view(gwf, transient_head_file_path,
                    f"{figure_folder}/map_heads_L5_t.png", 
                    boundary_keywords=["WEL", "GHB"], 
                    layer=22, flow_dir=False, contours=True,show=False, save=True,
                    grid=True, figsize=(10, 10), fontsize=14,title="Model map view Layer 5_t",
                    transient=True, time_step=time_step_plot)

    modplot6.plot_cross_section_row(gwf, transient_head_file_path, nrow//2, 
                                    f"{figure_folder}/cross_section_heads_t.png",
                                    boundary_keywords = ["WEL"], ve=ve,
                                    flow_dir = False, surface = False, layers=False,
                                    show=False, save=True, figsize = (19, 4),
                                    title=f"Cross section at time {times_list[time_step_plot]} days", 
                                    transient=True, time_step=time_step_plot, interfaces=interfaces)
    modplot6.plot_cross_section_row(gwf, transient_head_file_path, nrow//2, 
                                    f"{figure_folder}/cross_section_heads_t_qdir.png",
                                    boundary_keywords = ["WEL"], ve=ve,
                                    flow_dir = True, cbb_path=budget_file_path, 
                                    surface = True, layers=False,
                                    show=False, save=True, figsize = (19, 4),
                                    title=f"Cross section at time {times_list[time_step_plot]} days",
                                    transient=True, time_step=time_step_plot, interfaces=interfaces)
    modtransient6.plot_head_time_series(head_file_t, 
                                        gwf, 
                                        f"{figure_folder}/head_ts.png",
                                        show = False, 
                                        save = True, 
                                        tau = None, 
                                        time_units="years")

    #--------------------------------------- FLOW BUDGET ---------------------------------------------#

    modtransient6.plot_bud_sum_transient(budget_file_t, times_list[time_step_plot], 
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

if response_times:
    print("Estimating response times...")

    # ----------------------------- Response time: Absolute residual diffusion threshold
    tr_abs_mean, tr_abs_median, tr_abs_95p, tr_abs_max = modtransient6.response_time_array_absolute(
                                        gwf,
                                        steady_state_heads,
                                        transient_heads,
                                        times_list,
                                        threshold=threshold_absolute,
                                        threshold_type="absolute",
                                        stability_threshold=stability_threshold,
                                        start_step=start_step,
                                        save_array=True, #just save the response time array once
                                        save_plot=True,
                                        show_plot=False,
                                        boundary_keywords=["WEL"],
                                        fill="nan",
                                        ve=ve,
                                        array_output_folder=output_folder,
                                        array_name="response_time_absolute.npy",
                                        fig_output_folder=figure_folder,
                                        fig_name="Response_time_absolute_02.png",
                                        histogram=True, #just plot histogram once
                                        histogram_bins=histogram_bins,
                                        histogram_name="Response_time_absolute_histogram.png",
                                        interfaces=interfaces) 
    
    modtransient6.absolute_head_diffusion_zones(transient_heads, 
                                                steady_state_heads, 
                                                times_list, 
                                                zone_array, 
                                                start_step=start_step,
                                                threshold=threshold_absolute, 
                                                threshold_type="absolute",
                                                stability_threshold=stability_threshold,
                                                csv_output_folder=output_folder, 
                                                summary_csv_name="tr_zones_absolute_diffusion.csv",
                                                save_fig=True, show_fig=False,
                                                fig_output_folder=figure_folder, 
                                                fig_name = "diff_absolute_zones.png",                                                
                                                zone_descriptions = {
                                                1: "Unconfined Aquifer",
                                                2: "Aquitard",
                                                3: "Confined Aquifer",
                                                4: "Aquitard",
                                                5: "Confined Aquifer"},
                                                center="median",
                                                bounds="95p")
    
    modtransient6.absolute_head_diffusion(transient_heads, 
                                          steady_state_heads, 
                                          times_list,
                                          start_step=start_step, 
                                          threshold=threshold_absolute, 
                                          threshold_type="absolute",
                                          stability_threshold=stability_threshold,
                                          save_array=False,
                                          save_fig=True, show_fig=False,
                                          array_output_folder=output_folder,
                                          array_name = "diff_array_absolute.npy",
                                          fig_output_folder=figure_folder,
                                          fig_name = "diff_absolute_total.png",
                                          center="median", 
                                          bounds = "95p")

    # ----------------------------- Response time: Relative global threshold
    tr_rel_global_mean, tr_rel_global_median, tr_rel_global_95p, tr_rel_global_max = modtransient6.response_time_array_relative(gwf,
                                               steady_state_heads,
                                               transient_heads,
                                               times_list,
                                               threshold_percent=threshold_percent,
                                               stability_threshold=stability_threshold,
                                               start_step=start_step,
                                               save_array=True,
                                               save_plot=True,
                                               show_plot=False,
                                               boundary_keywords=["WEL"],
                                               max_initial_diff=True,
                                               fill="nan",
                                               ve=ve,
                                               array_output_folder=output_folder,
                                               array_name="response_time_relative_global.npy",
                                               fig_output_folder=figure_folder,
                                               fig_name="Response_time_rel_global_02.png",
                                               histogram=True, 
                                               histogram_bins=histogram_bins,
                                               histogram_name="Response_time_rel_global_histogram.png",
                                               interfaces=interfaces)

    modtransient6.relative_head_diffusion_zones(transient_heads, 
                                                steady_state_heads, 
                                                times_list, 
                                                zone_array,
                                                start_step=start_step, 
                                                threshold_percent=threshold_percent, 
                                                stability_threshold=stability_threshold,
                                                csv_output_folder=output_folder,
                                                summary_csv_name="tr_zones_relative_global.csv",
                                                save_fig=True, show_fig=False,
                                                fig_output_folder=figure_folder,
                                                fig_name = "diff_rel_global.png",
                                                zone_descriptions = {
                                                1: "Unconfined Aquifer",
                                                2: "Aquitard",
                                                3: "Confined Aquifer",
                                                4: "Aquitard",
                                                5: "Confined Aquifer"},
                                                center="median", 
                                                bounds="95p", 
                                                max_initial_diff=True)

    modtransient6.relative_head_diffusion(transient_heads, 
                                          steady_state_heads, 
                                          times_list,
                                          start_step=start_step,
                                          threshold_percent=threshold_percent, 
                                          stability_threshold=stability_threshold,
                                          save_array=False,
                                          save_fig=True, 
                                          show_fig=False,
                                          array_output_folder=output_folder,
                                          array_name="diff_array_rel_global.npy",
                                          fig_output_folder=figure_folder, 
                                          fig_name="diff_rel_global_total.png",
                                          center="median",
                                          bounds="95p",
                                          max_initial_diff=True)

    # ----------------------------- Response time: Relative local threshold
    tr_rel_local_mean, tr_rel_local_median, tr_rel_local_95p, tr_rel_local_max = modtransient6.response_time_array_relative(gwf,
                                               steady_state_heads,
                                               transient_heads,
                                               times_list,
                                               threshold_percent=threshold_percent,
                                               stability_threshold=stability_threshold,
                                               start_step=start_step,
                                               save_array=True,
                                               save_plot=True,
                                               show_plot=False,
                                               boundary_keywords=["WEL"],
                                               max_initial_diff=False,
                                               fill="nan",
                                               ve=ve,
                                               array_output_folder=output_folder,
                                               array_name="response_time_relative_local.npy",
                                               fig_output_folder=figure_folder,
                                               fig_name="Response_time_rel_local_02.png",
                                               histogram=True,
                                               histogram_bins=histogram_bins,
                                               histogram_name="Response_time_rel_local_histogram.png",
                                               interfaces=interfaces)

    modtransient6.relative_head_diffusion_zones(transient_heads, 
                                                steady_state_heads, 
                                                times_list, 
                                                zone_array,
                                                start_step=start_step, 
                                                threshold_percent=threshold_percent, 
                                                stability_threshold=stability_threshold,
                                                csv_output_folder=output_folder,
                                                summary_csv_name="tr_zones_relative_local.csv",
                                                save_fig=True, show_fig=False,
                                                fig_output_folder=figure_folder,
                                                fig_name = "diff_rel_local.png",
                                                zone_descriptions = {
                                                1: "Unconfined Aquifer",
                                                2: "Aquitard",
                                                3: "Confined Aquifer",
                                                4: "Aquitard",
                                                5: "Confined Aquifer"},
                                                center="median", 
                                                bounds="95p", 
                                                max_initial_diff=False)

    modtransient6.relative_head_diffusion(transient_heads, 
                                          steady_state_heads, 
                                          times_list,
                                          start_step=start_step,
                                          threshold_percent=threshold_percent, 
                                          stability_threshold=stability_threshold,
                                          save_array=False,
                                          save_fig=True, 
                                          show_fig=False,
                                          array_output_folder=output_folder,
                                          array_name="diff_array_rel_local.npy",
                                          fig_output_folder=figure_folder, 
                                          fig_name="diff_rel_local_total.png",
                                          center="median",
                                          bounds="95p",
                                          max_initial_diff=False)
    
    #------------------------------- Maximum drawdown or head difference
    res_diff_array = modtransient6.plot_residual_diffusion(  gwf=gwf,
                                            time=start,
                                            perioddata=perioddata,
                                            nrow=nrow//2,
                                            transient_heads=transient_heads,
                                            steady_state_heads=steady_state_heads,
                                            title=f"Maximum absolute residual difussion in hydraulic heads",
                                            label="Head difference (m)",
                                            vmin=0,
                                            vmax=None,
                                            save=True,
                                            ve=ve,
                                            output_folder=f"{figure_folder}/Residual_diffusion",
                                            plot_name=f"Residual_diffusion_{start}.png",
                                            boundary_keywords=["WEL"], 
                                            interfaces=interfaces)
    vmax = np.nanmax(res_diff_array)

    #--------------------------------------- SUMMARY CSV ---------------------------------------------#
    print(  f"Total pumping rate [m3/day]: {q0}",
            f"Maximum drawdown [m]: {vmax}",
            f"tr_abs_mean= {tr_abs_mean}",
            f"tr_abs_median= {tr_abs_median}",
            f"tr_abs_95p= {tr_abs_95p}",
            f"tr_abs_max= {tr_abs_max}",
            f"tr_rel_global_mean= {tr_rel_global_mean}",
            f"tr_rel_global_median= {tr_rel_global_median}",
            f"tr_rel_global_95p= {tr_rel_global_95p}",
            f"tr_rel_global_max= {tr_rel_global_max}",
            f"tr_rel_local_mean= {tr_rel_local_mean}",
            f"tr_rel_local_median= {tr_rel_local_median}",
            f"tr_rel_local_95p= {tr_rel_local_95p}",
            f"tr_rel_local_max= {tr_rel_local_max}",
            sep="\n"
            )
    
    summary_data = {
        "Total pumping rate [m3/day]": [q0],
        "Maximum drawdown [m]": [vmax],
        "tr_abs_mean": [tr_abs_mean],
        "tr_abs_median": [tr_abs_median],
        "tr_abs_95p": [tr_abs_95p],
        "tr_abs_max": [tr_abs_max],
        "tr_rel_global_mean": [tr_rel_global_mean],
        "tr_rel_global_median": [tr_rel_global_median],
        "tr_rel_global_95p": [tr_rel_global_95p],
        "tr_rel_global_max": [tr_rel_global_max],
        "tr_rel_local_mean": [tr_rel_local_mean],
        "tr_rel_local_median": [tr_rel_local_median],
        "tr_rel_local_95p": [tr_rel_local_95p],
        "tr_rel_local_max": [tr_rel_local_max]}

    csv_path = os.path.join(output_folder, "tr_total_summary.csv")
    pd.DataFrame(summary_data).to_csv(csv_path, index=False)

cb.close()
del cb
del zonebud
modpump6.analyze_cbb_boundaries(
    cbb_path   = f"{output_folder}/{model_name_tr}.cbb",
    csv_out    = f"{output_folder}/cbb_summary.csv",
    delete_cbb = True,
    run_label  = None,)

#--------------------------------------- TRANSIENT ANIMATION ---------------------------------------------#

if animate:
    
    # print("Creating residual diffusion animation...")
    # end = start + (step*n)
    # vmax = None
    # for t in range(start, end, step):
    #     res_diff_array = modtransient6.plot_residual_diffusion(  gwf=gwf,
    #                                             time=t,
    #                                             perioddata=perioddata,
    #                                             nrow=nrow//2,
    #                                             transient_heads=transient_heads,
    #                                             steady_state_heads=steady_state_heads,
    #                                             title=f"Absolute residual difussion in hydraulic heads after {int((t - start)/360)} years",
    #                                             label="Head difference (m)",
    #                                             vmin=0,
    #                                             vmax=vmax,
    #                                             save=True,
    #                                             ve=ve,
    #                                             output_folder=f"{figure_folder}/Residual_diffusion", 
    #                                             plot_name = f"Residual_diffusion_{t}.png",
    #                                             boundary_keywords=["WEL"],
    #                                             interfaces=interfaces)
        
    #     if vmax is None:
    #         vmax = np.nanmax(res_diff_array)
    
    # modplot6.animate(f"{figure_folder}/Residual_diffusion", f"{figure_folder}/Residual_diffusion.gif", duration=250)

    # print("Creating transient head animation...")
    # end = start + (step*n)
    # vmax = None
    # for t in range(start, end, step):
    #     tr_heads_array = modtransient6.plot_transient_heads(  gwf=gwf,
    #                                             idomain=idomain,
    #                                             time=t,
    #                                             perioddata=perioddata,
    #                                             nrow=nrow//2,
    #                                             transient_heads=transient_heads,
    #                                             title=f"Hydraulic heads after {int((t - start)/360)} years",
    #                                             label="Head (m)",
    #                                             vmin=0,
    #                                             vmax=vmax,
    #                                             save=True,
    #                                             ve=ve,
    #                                             output_folder=f"{figure_folder}/Transient_heads", 
    #                                             plot_name = f"Transient_heads_{t}.png",
    #                                             boundary_keywords=["WEL"],
    #                                             interfaces=interfaces)
        
    #     if vmax is None:
    #         vmax = np.nanmax(tr_heads_array)
    
    # modplot6.animate(f"{figure_folder}/Transient_heads", f"{figure_folder}/Transient_heads.gif", duration=250)

    # print("Creating transient head animation with head time series...")
    # end = start + (step*n)
    # vmax = None
    # for t in range(start, end, step):
    #     tr_heads_array = modtransient6.plot_transient_heads_tr(gwf=gwf,
    #                                         idomain=idomain,
    #                                         time=t,
    #                                         perioddata=perioddata,
    #                                         nrow=nrow//2,
    #                                         transient_heads=transient_heads,
    #                                         times_list=times_list,
    #                                         cell=(12,0,400),
    #                                         start_time=start,
    #                                         end_time=end,
    #                                         title=f"Hydraulic heads {int((t - start)/360)} years after the start of pumping",
    #                                         label="Head [m]",
    #                                         vmin=-150,
    #                                         vmax=vmax,
    #                                         save=True,
    #                                         ve=ve,
    #                                         output_folder=f"{figure_folder}/Transient_heads_ts",
    #                                         plot_name=f"Transient_heads_{t}.png",
    #                                         boundary_keywords=["WEL"],
    #                                         interfaces=interfaces)
        
    #     if vmax is None:
    #         vmax = np.nanmax(tr_heads_array)

    # modplot6.animate(f"{figure_folder}/Transient_heads_ts", f"{figure_folder}/Transient_heads_ts.gif", duration=250)

    print("Creating transient Residual diffusion animation with residual diffusion time series...")
    end = start + (step*n)
    vmax = None
    for t in range(start, end, step):
        res_diff_array = modtransient6.plot_residual_diffusion_tr(gwf=gwf,
                                            time=t,
                                            perioddata=perioddata,
                                            nrow=nrow//2,
                                            transient_heads=transient_heads,
                                            steady_state_heads=steady_state_heads,
                                            times_list=times_list,
                                            cell=(12,0,400),
                                            start_time=start,
                                            end_time=end,
                                            title=f"Hydraulic heads {int((t - start)/360)} years after the start of pumping",
                                            label="Absolute residual diffusion [m]",
                                            vmin=0,
                                            vmax=vmax,
                                            save=True,
                                            ve=ve,
                                            output_folder=f"{figure_folder}/Res_diff_ts",
                                            plot_name=f"Res_diff_ts_{t}.png",
                                            boundary_keywords=["WEL"],
                                            interfaces=interfaces)
        
        if vmax is None:
            vmax = np.nanmax(res_diff_array)

    modplot6.animate(f"{figure_folder}/Res_diff_ts", f"{figure_folder}/Res_diff_ts.gif", duration=250)

    # print("Creating transient head animation with capture time series...")
    # end = start + (step*n)
    # vmax = None
    # for t in range(start, end, step):
    #     tr_heads_array = modtransient6.plot_transient_heads_capture(gwf=gwf,
    #                                         idomain=idomain,
    #                                         time=t,
    #                                         perioddata=perioddata,
    #                                         nrow=nrow//2,
    #                                         transient_heads=transient_heads,
    #                                         csv_path=budget_file_t,
    #                                         start_time=start,
    #                                         end_time=end,
    #                                         title=f"Hydraulic heads {int((t - start)/360)} years after the start of pumping",
    #                                         label="Head [m]",
    #                                         vmin=-150,
    #                                         vmax=vmax,
    #                                         save=True,
    #                                         ve=ve,
    #                                         output_folder=f"{figure_folder}/Transient_heads_sto",
    #                                         plot_name=f"Transient_heads_{t}.png",
    #                                         boundary_keywords=["WEL"],
    #                                         interfaces=interfaces)
        
    #     if vmax is None:
    #         vmax = np.nanmax(tr_heads_array)

    # modplot6.animate(f"{figure_folder}/Transient_heads_sto", f"{figure_folder}/Transient_heads_sto.gif", duration=250)

end_time = time.time()
print(f"Total execution time: {(end_time - start_time)/60:.2f} minutes")