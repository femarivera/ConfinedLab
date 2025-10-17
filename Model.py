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
river_shapefile = False # If True, extracts river cells from shapefile

heterogeneity = False # If True, generates random hydraulic conductivity fields

STEADY = True # Runs the steady state model
plot_steady = True # Plots steady state outputs
iterate = False # Iterates pumping rates over steady state model

TRANSIENT = True # Runs the transient model
plot_transient = True # Plots transient outputs
animate = False # Animates transient cross sections

plot_maps = False # If True, plots map views of heads and flows (just works when model is 3D or 2D Horizontal)

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
recharge = par_df_to_1Darray(par_df, "rech") # Recharge (m/d)

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
R_array = modgeom6.compute_recharge(irch, recharge)
zone_array = modgeom6.compute_3Darray(zones, idomain, dtype = int)
kh_array = modgeom6.compute_3Darray(kh, idomain)
kv_array = modgeom6.compute_3Darray(kv, idomain)
unconfined_areas = modgeom6.compute_unconfined_areas(irch, idomain)
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
recharge = modgeom6.subdivide_array(recharge, nsub)
zone_array = modgeom6.subdivide_array(zone_array, nsub)
kh_array = modgeom6.subdivide_array(kh_array, nsub)
kv_array = modgeom6.subdivide_array(kv_array, nsub)
unconfined_areas = modgeom6.subdivide_array(unconfined_areas, nsub)

#Update thickness, irch, R_array, zone_array and kh_array
thickness_array = ztop_array - zbot
irch = modgeom6.compute_irch(idomain)
R_array = modgeom6.compute_recharge(irch, recharge)

# Compute storage coefficient array, transmissivity, and diffusivity
storage_coeff = modgeom6.compute_storage_coefficient(unconfined_areas, sy, ss, thickness_array)
transmissivity = kh_array * thickness_array
diffusivity = transmissivity / storage_coeff

# ------------------------------------------------------------------------------- #
# --------------------------- BUILDING SIMULATION ------------------------------- #
# ------------------------------------------------------------------------------- #

#Save zone_array
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
                           outer_dvclose=0.0001,
                           outer_maximum=5000,
                           under_relaxation="NONE",
                           inner_maximum=5000,
                           inner_dvclose=0.0001,
                           rcloserecord=0.001,
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
                              top=ztop, botm=zbot, idomain=idomain,
                              filename=f"{model_name}.dis")

# Set the initial conditions
strt = np.repeat(ztop[np.newaxis, :, :], nlay, axis=0)
ic = flopy.mf6.ModflowGwfic(gwf, 
                            pname = "ic", 
                            strt = strt,
                            filename=f"{model_name}.ic")

# Set the Node Property Flow package
npf = flopy.mf6.ModflowGwfnpf(gwf,
                              pname = "npf",
                              save_specific_discharge = True,
                              icelltype= modgeom6.subdivide_array(np.array([1, 1, 1, 1, 1]), nsub), 
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
    printrecord = [("HEAD", "ALL"),("BUDGET", "LAST")], 
    filename = f"{model_name}.oc")

# --------------------------- BOUNDARY CONDITIONS ------------------------------- #

if rivers:
    #River package
    if river_shapefile: 
        modbound6.export_grid_topview(nrow, ncol, drow, dcol, irch, out_shp=f"{gis_folder}/grid_topview.shp", crs="EPSG:4326")
        riv_cells = modbound6.active_cells_from_line(f"{gis_folder}/grid_topview.shp", f"{gis_folder}/river.shp")
    else:
        riv_cells = modbound6.extract_active_cells_range(irch, idomain, nrow//2, nrow//2, 0, ncol-2)
    riv_spd = modbound6.create_riv_spd(
        riv_cells,
        ztop_array,
        thickness_array,
        drn_cond,
        drow,
        river_width=1,
        riverbed_thickness=1,
        stage_type="absolute",
        a=0,
        b=1,
        conc=None)
    riv = flopy.mf6.ModflowGwfriv(gwf, 
                                pname = "riv",
                                save_flows = True,
                                stress_period_data = riv_spd,
                                filename = f"{model_name}.riv")
else: 
    # Drain package
    drn_cells = modbound6.extract_active_cells_zone(irch, idomain, zone_array, 0, nrow-1, 0, ncol-2, zones = [1, 2, 3, 4, 5])
    # drn_cells = [t for t in drn_cells if t not in riv_cells] 
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
    
    # Find the STEADY STATE pumping rate from parameters
    q0 = par_df[par_df.index=="q_0"].iloc[0,0]
    
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
    ((ilay, irow, ncol-1), ghb_1, 1000 * base_thicknesses[ilay] * width, f"Layer{ilay}")
    for ilay in range(nlay)
    for irow in range(nrow)]

# GHB in the top of first layer
# ghb_cells2 = modbound6.extract_active_cells_range(irch, idomain, 0, nrow-1,col_start=ncol-25, col_end=ncol-2)
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

        if plot_maps:

            modplot6.plot_map_view(gwf, head, 
                    f"{figure_folder}/map_heads_L1.png", 
                    boundary_keywords=["WEL", "GHB"], 
                    layer=0, flow_dir=False, contours=True,show=False, save=True,
                    grid=True, figsize=(10, 10), fontsize=14,title="Model map view Layer 1")

            modplot6.plot_map_view(gwf, head,  
                    f"{figure_folder}/map_heads_L2.png", 
                    boundary_keywords=["WEL", "GHB"], 
                    layer=7, flow_dir=False, contours=True,show=False, save=True,
                    grid=True, figsize=(10, 10), fontsize=14,title="Model map view Layer 2")
            
            modplot6.plot_map_view(gwf, head,
                    f"{figure_folder}/map_heads_L3.png", 
                    boundary_keywords=["WEL", "GHB"], 
                    layer=12, flow_dir=False, contours=True,show=False, save=True,
                    grid=True, figsize=(10, 10), fontsize=14,title="Model map view Layer 3")
            
            modplot6.plot_map_view(gwf, head,
                    f"{figure_folder}/map_heads_L4.png", 
                    boundary_keywords=["WEL", "GHB"], 
                    layer=17, flow_dir=False, contours=True,show=False, save=True,
                    grid=True, figsize=(10, 10), fontsize=14,title="Model map view Layer 4")
            
            modplot6.plot_map_view(gwf, head,
                    f"{figure_folder}/map_heads_L5.png", 
                    boundary_keywords=["WEL", "GHB"], 
                    layer=22, flow_dir=False, contours=True,show=False, save=True,
                    grid=True, figsize=(10, 10), fontsize=14,title="Model map view Layer 5")
        
        modplot6.plot_cross_section_row(gwf, head, nrow//2, 
                                f"{figure_folder}/cross_section_heads.png",
                                boundary_keywords = ["WEL"],
                                flow_dir = False, surface = False, ve=100,
                                show=False, save=True, figsize=(19, 4), layers = False, 
                                title="Cross section - Steady state simulation")
        modplot6.plot_cross_section_row(gwf, head, nrow//2, 
                                f"{figure_folder}/cross_section_heads_qdir.png",
                                boundary_keywords = ["WEL"], ve=100,
                                flow_dir = True, qx=qx, qy=qy, qz=qz, surface = True, 
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
                             ve=100,
                             log=True,
                             array=kh_array,
                             label="Hydraulic Conductivity (m/d)", 
                             title="Model layers")
        
        modplot6.plot_cross_section_array(gwf, 
                             nrow//2, 
                             f"{figure_folder}/cross_section_diffusivity.png", 
                             boundary_keywords=None, 
                             show = False, 
                             save = True, 
                             figsize=(19, 5),
                             fontsize=14,
                             ve=100,
                             log=True,
                             array=diffusivity,
                             label="Hydraulic diffusivity (m²/day)", 
                             title="Model layers")
        
        modplot6.plot_cross_section_array(gwf, 
                             nrow//2, 
                             f"{figure_folder}/cross_section_outcrops.png", 
                             boundary_keywords=None, 
                             show = False, 
                             save = True, 
                             figsize=(19, 5),
                             fontsize=14,
                             ve=100,
                             log=False,
                             array=unconfined_areas,
                             label="Unconfined areas (1=Unconfined, 0=Confined)", 
                             title="Unconfined areas")

        modplot6.plot_cross_section_array(  gwf,
                                            nrow//2,
                                            f"{figure_folder}/cross_section_layers.png",
                                            boundary_keywords= boundary_keywords,
                                            show=False,
                                            save=True,
                                            ax=None,
                                            figsize=(19, 6),
                                            fontsize=14,
                                            ve=100,
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
                                        boundary_keywords = ["WEL"], ve=100,
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
        ic.strt = strt
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
    
    # # Create storage package for transient simulation
    # sto = flopy.mf6.ModflowGwfsto(
    #     gwf,
    #     pname="sto",
    #     iconvert = 0, #Confined storage is used
    #     storagecoefficient = True,
    #     ss=storage_coeff, #Using an array that has sy for unconfined areas, and ss*thickness for confined areas
    #     steady_state={0: True}, # First stress period is steady state
    #     transient={1: True}, 
    #     filename=f"{model_name_tr}.sto")

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
        times_list = hobj.get_times()

        # cb = gwf.oc.output.budget()
        # cb.get_data(idx=0, full3D=True) #Get cell budget file

        budget_file_t = f"{output_folder}/{model_name_tr}_budget.csv"

        zonebud_file_t = f"{output_folder}/zonebud.csv"

        head_file_t = f"{output_folder}/head_obs_t.csv"

        #--------------------------------------- HEADS ---------------------------------------------#
        if plot_maps:

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
        if STEADY:

            stability_threshold = 0.0005
            threshold_absolute = 0.05
            threshold_percent = 1

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
                                                        histogram_bins=100,
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
                                                            ve=100, histogram=True, histogram_bins=100,
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
                                                        histogram_bins=100,
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

            # start = 3600000 #start of the step change in model units
            # step = 36000 #Size of the steps in model units
            # n = 60
            # end = start + (step*n)
            # vmax = None

            # for t in range(start, end, step):
            #     res_diff_array = modtransient6.plot_residual_diffusion(  gwf=gwf,
            #                                             start_time=start,
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
            #                                             ve=100,
            #                                             output_folder=f"{figure_folder}/Residual_diffusion", 
            #                                             plot_name = f"Residual_diffusion_{t}.png",
            #                                             boundary_keywords=["WEL"])
                
            #     if vmax is None:
            #         vmax = np.nanmax(res_diff_array)

            # modplot6.animate(f"{figure_folder}/Residual_diffusion", f"{figure_folder}/Residual_diffusion.gif", duration=250)

            

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