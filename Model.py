# ---------------------------------------------------------------------------------------- #
# ----------------------------------------- STEADY STATE --------------------------------- #
# ---------------------------------------------------------------------------------------- #

# ------------------------------------------------------------------------ #
# ----------------------------- IMPORT MODULES --------------------------- #
# ------------------------------------------------------------------------ #
import time
start_time = time.time()
import os
import sys
import numpy as np
import pandas as pd
import geopandas as gpd

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
import matplotlib.gridspec as gspec
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

# Set model parameters
k = np.array([250, 1e-5, 25, 1e-5, 25]) #Horizontal hydraulic conductivity in m/d
R = np.array([8e-4, 0, 8e-4, 0, 8e-4]) #Arid/Semi-arid conditions rates in m/d
sy = np.array([0.25, 0.25, 0.25, 0.25, 0.25]) # Specific yield for Unconfined cells (adimentional)
ss = np.array([1e-5, 1e-5, 1e-5, 1e-5, 1e-5]) # type: ignore # Specific storage for Confined cells (m-1)
q = -15 # Pumping rate in m3/d
well_loc = (2, 0, 400) # Well location (layer, row, column)

# Set model grid parameters
nlay = 5 # Number of layers
ncol = 600 # Number of columns
nrow = 1 # Number of rows (single row for 2D cross section)
length = 600000 # Total lenght of model in meters
width = 1 # Total width of model in meters
dcol = length/ncol # Column size in meters
drow = width/nrow # Row size in meters

# Define synthetic geometry generation parameters
epsilon = 0 # Minimum allowed cell thickness in meters
outcrop_z = np.array([100, 150, 200, 250, 350]) # Elevation (Just used when SLOPE is set to False)
outcrop_zmax = np.array([200, 300, 400, 500, 500]) # Elevation (Just used when SLOPE are set to True)
outcrop_zmin = np.array([0, 200, 300, 400, 500]) # Elevation (Just used when SLOPE are set to True)
base_thicknesses = np.array([300, 150, 200, 150, 200]) # Layer thickness in meters
outcrop_cells = np.array([200, 150, 100, 50, 0]) 
transition = 50 # Transitions cells (Just used when SMOOTH_TOPO is set to True)

# ------------------------------------------------------------------------------- #
# --------------------------- MODEL RUN CONTROL --------------------------------- #
# ------------------------------------------------------------------------------- #

boundary_keywords = ["GHB", "WEL", "RIV"] #List of boundaries used in the model for plotting
heterogeneity = True # If True, generates random hydraulic conductivity fields

STEADY = True # Runs the steady state model
plot_steady = True # Plots steady state outputs
iterate = False # Iterates pumping rates over steady state model

TRANSIENT = True
plot_transient = True
animate = True

# ------------------------------------------------------------------------------- #
# --------------------------- GEOMETRY GENERATION ------------------------------- #
# ------------------------------------------------------------------------------- #

# Create idomain, irch and recharge arrays
idomain = modgeom6.compute_idomain(nlay, nrow, ncol, outcrop_cells)
ztop = modgeom6.compute_top(idomain, outcrop_z, transition=True, slope=True,
                            transition_cells=transition, transition_type="contain", 
                            outcrop_zmin=outcrop_zmin, outcrop_zmax=outcrop_zmax)
thickness_array = modgeom6.compute_thickness(idomain, base_thicknesses, 
                                             transition=True, transition_type="contain", 
                                             transition_cells=transition)
zbot = modgeom6.compute_bottom(ztop, thickness_array)
idomain = modgeom6.idomain_from_thickness(thickness_array, epsilon)
ztop_array = modgeom6.compute_ztop_array(ztop, zbot)
irch = modgeom6.compute_irch(idomain)
R_array = modgeom6.compute_recharge(irch, R)
zone_array = np.zeros((nlay, nrow, ncol), dtype=int) # Create zones array for zone budget
for i in range(nlay):
    zone_array[i, :, :] = i + 1
k_array = modgeom6.compute_3Darray(k, idomain)
# Plot the arrays to check consistency
#plt.imshow(idomain[0,:,:], cmap='viridis', interpolation='nearest', aspect=300)
#plt.colorbar()  # Add color bar to show scale
#plt.title('iDomain')
#plt.show()
#plt.imshow(R_array, cmap='viridis', interpolation='nearest', aspect=300)
#plt.colorbar()  # Add color bar to show scale
#plt.title('Recharge')
#plt.show()


# ------------------------------------------------------------------------------- #
# ----------------------- RANDOM PARAMETER FIELDS ------------------------------- #
# ------------------------------------------------------------------------------- #

if heterogeneity:
    # Generate random hydraulic conductivity fields for each layer
    k0 = modpar6.generate_random_field((nrow, ncol), "exponential",
                                        geom_mean=k[0], sill=0.3, nugget=0.0,
                                        range_param=15000, drow=drow, dcol=dcol,
                                        param_type="K", seed=0)

    k1 = modpar6.generate_random_field((nrow, ncol), "exponential",
                                        geom_mean=k[1], sill=0.3, nugget=0.0,
                                        range_param=15000, drow=drow, dcol=dcol,
                                        param_type="K", seed=1)

    k2 = modpar6.generate_random_field((nrow, ncol), "exponential",
                                        geom_mean=k[2], sill=0.3, nugget=0.0,
                                        range_param=15000, drow=drow, dcol=dcol,
                                        param_type="K", seed=2)

    k3 = modpar6.generate_random_field((nrow, ncol), "exponential",
                                        geom_mean=k[3], sill=0.3, nugget=0.0,
                                        range_param=15000, drow=drow, dcol=dcol,
                                        param_type="K", seed=3)

    k4 = modpar6.generate_random_field((nrow, ncol), "exponential",
                                        geom_mean=k[4], sill=0.3, nugget=0.0,
                                        range_param=15000, drow=drow, dcol=dcol,
                                        param_type="K", seed=4)

    k_array = modpar6.stack_fields_to_3D([k0, k1, k2, k3, k4], nlay, nrow, ncol)

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
                           outer_dvclose=0.01,
                           outer_maximum=10000,
                           under_relaxation="NONE",
                           inner_maximum=10000,
                           inner_dvclose=0.01,
                           rcloserecord=0.01,
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
                              icelltype=[1, 1, 1, 1, 1], 
                              k=k_array,
                              k33=k_array/10,
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

# River package
riv_cells1 = modbound6.extract_active_cells_n_range(irch, idomain, n=20, col_start=0, col_end=200)
riv_cells2 = modbound6.extract_active_cells_n_range(irch, idomain, n=75, col_start=200, col_end=ncol-1)
riv_cells = riv_cells1 + riv_cells2
riv_spd1 = modbound6.create_riv_spd(
    riv_cells,
    ztop_array,
    thickness_array,
    np.array([k[0]/100,k[1]*10,k[2]/10,k[3]*10,k[4]/10]),
    drow,
    river_width=1,
    riverbed_thickness=1,
    stage_type="proportion",
    a=0.20,
    b=1,
    conc=None)
riv1 = flopy.mf6.ModflowGwfriv(gwf, 
                              pname = "riv",
                              save_flows = True,
                              stress_period_data = riv_spd1,
                              filename = f"{model_name}.riv")

# Drain package
drn_cells1 = modbound6.extract_active_cells_range(irch, idomain, nrow//2, nrow//2, 0, 202)
#drn_cells2 = modbound6.extract_active_cells_range(irch, idomain, nrow//2, nrow//2, 150, 202)
drn_cells = drn_cells1 #+ drn_cells2
drn_spd = modbound6.create_drn_spd(
    drn_cells,
    ztop_array,
    thickness_array,
    100000*k,
    drow,
    drain_width=1,
    drainbed_thickness=1,
    elev_type="absolute",
    a=1,
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
wel_spd = {}
wel_spd[0] = [(well_loc[0], well_loc[1], well_loc[2], 0, "WELL1")]
wel = flopy.mf6.ModflowGwfwel(gwf, 
                              pname = "wel",
                              save_flows = True,
                              boundnames=True,
                              stress_period_data = wel_spd, 
                              filename = f"{model_name}.wel")

# General head boundary package
ghb_1 = ztop_array[0,0,ncol-1]-(0.15*base_thicknesses[0])
ghb_spd1 = {}
ghb_spd1[0] = [((0, 0, ncol-1), ghb_1, k[0]*base_thicknesses[0]*width, "Unconfined"),
                ((1, 0, ncol-1), ghb_1, k[1]*base_thicknesses[1]*width, "Aqt1"),
                ((2, 0, ncol-1), ghb_1, k[2]*base_thicknesses[2]*width, "Caq1"),
                ((3, 0, ncol-1), ghb_1, k[3]*base_thicknesses[3]*width, "Aqt2"),
                ((4, 0, ncol-1), ghb_1, k[4]*base_thicknesses[4]*width, "Caq2")]
ghb1 = flopy.mf6.ModflowGwfghb(gwf,
                                pname="ghb1",
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
                                boundary_keywords = boundary_keywords,
                                flow_dir = False, surface = True, 
                                show=False, save=True, figsize=(19, 4), layers = True, 
                                title="Cross section - Steady state simulation")

        modplot6.plot_bud_sum_steady(budget_file, 
                                 f"{figure_folder}/bud_sum_ss.png", 
                                show=False, save=True, figsize=(14, 5), fontsize=14)

        modplot6.plot_cross_section_array(gwf, 
                             k_array, 
                             nrow//2, 
                             f"{figure_folder}/cross_section_layers.png", 
                             boundary_keywords=boundary_keywords, 
                             show = False, 
                             save = True, 
                             figsize=(19, 5),
                             fontsize=14,
                             log=True,
                             label="Hydraulic Conductivity (m/d)", 
                             title="Model layers")        
        
        # Plot heads with im.show
        #masked_head = np.where(idomain == 0, np.nan, head)
        #plt.imshow(masked_head[:,0,:], aspect=300, interpolation=None)
        #plt.colorbar()
        #plt.show()

    # ----------------------------------------------------------------------------- #
    # -------------------------- ITERATION PUMPING RATES -------------------------- #
    # ----------------------------------------------------------------------------- #

    if iterate:
        # Define the pumping rates to iterate over in steady state conditions
        q_values = [-0, -0.4, -2, -4, -8, -12, -16, -20, -24, -28, -32, -36, -40, -60]
        
        # Run the iterate_pumping_rate function 
        modpump6.iterate_pumping_rate_steady(sim, gwf, wel_spd, wel, q_values, model_ws, 
                                        f"{figure_folder}/cross_sections_ss",
                                        f"{figure_folder}",
                                        budget_file,
                                        nrow//2,
                                        boundary_keywords = boundary_keywords,
                                        animate = True,
                                        save_budget = True,
                                        save_wells = True)

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
    nper = 4
    perlen = [0]*1 + [7560000]*1 + [19800]*1 + [360000]*1
    nstp = [1]*1 + [210]*1 +[55]*1 + [100]*1
    tsmult = [1]*1 + [1]*1 + [1]*1 + [1]*1
    perioddata = list(zip(perlen, nstp, tsmult))
    tdis = sim.tdis
    tdis.nper = nper
    tdis.perioddata = perioddata

    # Update the initial conditions
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
        iconvert = 1, #Unonfined/confined mixed storage is used
        sy=sy, #Specific yield
        ss=ss, #If not specified, flopy uses default value of 1e-5 m-1
        steady_state={0: True},
        transient={1: True}, 
        filename=f"{model_name_tr}.sto")

    # Update output control
    oc = gwf.oc
    oc.head_filerecord = f"output/{model_name_tr}.hds"
    oc.budget_filerecord = f"output/{model_name_tr}.cbb"
    oc.budgetcsv_filerecord = f"output/{model_name_tr}_budget.csv"
    oc.filename = f"{model_name_tr}.oc"
    # ---------------------------- UPDATE TRANSIENT BOUNDARY CONDITIONS -------------------------- #
    # Update transient recharge package
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
    df = pd.read_csv(f"{model_ws}/transient_recharge.csv", delimiter=';')

    # For recharge specified per layer
    # Extract time steps and R time series per layer
    time_steps = df.iloc[:, 0].values  # shape (n_times_steps,)
    R_vectors = df.iloc[:, 1:].values  # shape (n_time_steps, n_layers)
    # Compute recharge arrays per time step
    tas_data = {}
    for i, t in enumerate(time_steps):
        R = R_vectors[i]
        recharge_array = modgeom6.compute_recharge(irch, R)  # shape (nrow, ncol)
        tas_data[t] = recharge_array

    # For single recharge series
    # tas_data = dict(zip(df.iloc[:, 0], df.iloc[:, 1]))

    rch.tas.initialize(
        filename="recharge_rates.ts",
        tas_array=tas_data,
        time_series_namerecord="recharge",
        interpolation_methodrecord="stepwise",)

    # Update transient well package
    wel_spd = {}
    wel_spd[0] = [(well_loc[0], well_loc[1], well_loc[2],"wells1")]
    wel = flopy.mf6.ModflowGwfwel(gwf, 
                              pname = "wel",
                              save_flows = True,
                              boundnames=True,
                              stress_period_data = wel_spd, 
                              filename = f"{model_name_tr}.wel")
    ts_data = [(0, 0),
               (7560000 , q),
               (20000000 , q)]
    wel.ts.initialize(
        filename="well_rates.ts",
        timeseries=ts_data,
        time_series_namerecord=["wells1"],
        interpolation_methodrecord=["stepwise"])

    # ------------------------- OBSERVATIONS ------------------------------ #

    # Head obervations
    obs_recarray = {
        f"output/head_obs_t.csv": [ ("Head at pumping well - Unconfined Aquifer", "HEAD", (0, 0, well_loc[2])),
                            ("Head at pumping well - Aquitard", "HEAD", (1, 0, well_loc[2])),
                            ("Head at pumping well - Confined Aquifer", "HEAD", (2, 0, well_loc[2])),
                            ("Head at pumping well - Aquitard", "HEAD", (3, 0, well_loc[2])),
                            ("Head at pumping well - Confined Aquifer", "HEAD", (4, 0, well_loc[2]))]}

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
        ts_num = 0
        sp_num = 0 
        layer = 0
        elapsed_time = modtransient6.elapsed_time(perioddata, sp_num, ts_num)

        head = gwf.output.head().get_data(kstpkper=(ts_num, sp_num)) #kstpkper = time step and stress period.
        bud = gwf.output.budget()
        spdis = bud.get_data(text='DATA-SPDIS', kstpkper=(ts_num, sp_num))[layer] 
        qx, qy, qz = flopy.utils.postprocessing.get_specific_discharge(spdis, gwf)

        hobj = flopy.utils.HeadFile(f"{output_folder}/{model_name_tr}.hds")
        transient_heads = hobj.get_alldata()

        budget_file_t = f"{output_folder}/{model_name_tr}_budget.csv"

        zonebud_file_t = f"{output_folder}/zonebud.csv"

        head_file_t = f"{output_folder}/head_obs_t.csv"

        #--------------------------------------- HEADS ---------------------------------------------#

        modplot6.plot_cross_section_row(gwf, head, qx, qy, qz, nrow//2, 
                                        f"{figure_folder}/cross_section_heads_t.png",
                                        boundary_keywords = boundary_keywords,
                                        flow_dir = False, surface = True, layers=True,
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

        #--------------------------------------- ZONE BUDGET ---------------------------------------------#
        
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

        #--------------------------------------- TRANSIENT ANIMATION ---------------------------------------------#

        if animate:
            modplot6.plot_animation(gwf, transient_heads, qx, qy, qz, nrow//2, 
                                        f"{figure_folder}/cross_sections_tr",
                                        f"{figure_folder}/heads_transient.gif",
                                        boundary_keywords = boundary_keywords,
                                        flow_dir = False, surface = True, layers=True,
                                        show=False, save=True, figsize = (19, 4), 
                                        gif_start=0, gif_step=20, duration=2)

end_time = time.time()
print(f"Total execution time: {end_time - start_time:.2f} seconds")