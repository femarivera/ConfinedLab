# ---------------------------------------------------------------------------------------- #
# ----------------------------------------- STEADY STATE --------------------------------- #
# ---------------------------------------------------------------------------------------- #

# ------------------------------------------------------------------------ #
# ----------------------------- IMPORT MODULES --------------------------- #
# ------------------------------------------------------------------------ #

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
from mlibs import modplot6, modtransient6, modpump6, modgeom6, modbound6 # type: ignore

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
transient_name = 'DEESACt'

# Set model parameters
k = np.array([250, 1e-3, 25, 1e-3, 25]) #Horizontal hydraulic conductivity in m/d
R = np.array([8e-4, 0, 8e-4, 0, 8e-4]) #Arid/Semi-arid conditions rates in m/d
sy = np.array([0.25, 0.25, 0.25, 0.25, 0.25]) # Specific yield for Unconfined cells (adimentional)
ss = np.array([1e-5, 1e-5, 1e-5, 1e-5, 1e-5]) # type: ignore # Specific storage for Confined cells (m-1)
q = -45 # Pumping rate in m3/d
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

STEADY = True # Runs the steady state model
plot_steady = True # Plots steady state outputs
iterate = False # Iterates pumping rates over steady state model

TRANSIENT = True
plot_transient = True
animate = False

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
                           outer_dvclose=0.1,
                           outer_maximum=10000,
                           under_relaxation="NONE",
                           inner_maximum=10000,
                           inner_dvclose=0.1,
                           rcloserecord=0.1,
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
                              k=k,
                              k33=k/10,
                              filename=f"{model_name}.npf")

# Output control
oc = flopy.mf6.ModflowGwfoc(
    gwf,
    pname = "oc",
    head_filerecord = f"{model_name}.hds",
    budget_filerecord = f"{model_name}.cbb",
    budgetcsv_filerecord = f"{model_name}_budget.csv",
    saverecord = [("HEAD", "ALL"), ("BUDGET", "ALL")],
    printrecord = [("HEAD", "ALL"),("BUDGET", "ALL")])

# --------------------------- BOUNDARY CONDITIONS ------------------------------- #

# River package
riv_cells1 = modbound6.extract_active_cells_n(irch, idomain, n=20)
riv_spd1 = modbound6.create_riv_spd(
    riv_cells1,
    ztop_array,
    thickness_array,
    np.array([k[0]/100,k[1]*10,k[2]/10,k[3]*10,k[4]/10]),
    drow,
    river_width=1,
    riverbed_thickness=1,
    stage_type="proportion",
    a=0.1,
    b=1,
    conc=None)
riv1 = flopy.mf6.ModflowGwfriv(gwf, 
                              pname = "riv",
                              save_flows = True,
                              stress_period_data = riv_spd1,
                              filename = f"{model_name}.riv")

# Drain package
drn_cells1 = modbound6.extract_active_cells_range(irch, idomain, nrow//2, nrow//2, 50, 110)
drn_cells2 = modbound6.extract_active_cells_range(irch, idomain, 0, nrow-1, 150, 210)
#drn_cells3 = [(3,0,j) for j in range(100, 110)] + [(1,0,j) for j in range(200, 210)]
drn_cells = drn_cells1 + drn_cells2 #+ drn_cells3
drn_spd = modbound6.create_drn_spd(
    drn_cells,
    ztop_array,
    thickness_array,
    1000000*k,
    drow,
    drain_width=1,
    drainbed_thickness=1,
    elev_type="absolute",
    a=10,
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
                                filename = f"{model_name}_1.ghb",
                                stress_period_data=ghb_spd1)

# --------------------------------------------------------------------------- #    
# ---------------------------- RUN SIMULATION ------------------------------- #
# --------------------------------------------------------------------------- #  

if STEADY:
    sim.write_simulation()
    sim.run_simulation()

    # -------------------------- ZONE BUDGET -------------------------- #

    zonbud = gwf.output.zonebudget(zone_array)
    zonbud.change_model_ws(model_ws)
    zonbud.write_input()
    zonbud.run_model()

    # -------------------------- OUTPUTS -------------------------- #

    head = gwf.output.head().get_data()
    steady_state_heads = gwf.output.head().get_data()
    bud = gwf.output.budget()
    spdis = bud.get_data(text='DATA-SPDIS')[0]
    qx, qy, qz = flopy.utils.postprocessing.get_specific_discharge(spdis, gwf)

    # -------------------------- PLOTTING -------------------------- #

    if plot_steady:
        modplot6.plot_cross_section_row(gwf, head, qx, qy, qz, nrow//2, 
                                f"{model_ws}/fig/cross_section_heads.png",
                                boundary_keywords = boundary_keywords,
                                flow_dir = True, surface = True, 
                                show=False, save=True, figsize=(19, 4), layers = True, 
                                title="Cross section - Steady state simulation")

        modplot6.plot_bud_sum_steady(f"{model_ws}/{model_name}_budget.csv", 
                                 f"{model_ws}/fig/bud_sum_ss.png", 
                                show=False, save=True, figsize=(14, 5), fontsize=14)

        modplot6.plot_cross_section_array(gwf, 
                             k_array, 
                             nrow//2, 
                             f"{model_ws}/fig/cross_section_layers.png", 
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
        csv_file_name = f"{model_name}_budget.csv"

        # Run the iterate_pumping_rate function 
        modpump6.iterate_pumping_rate_steady(sim, gwf, wel_spd, wel, q_values, model_ws, 
                                        f"{model_ws}/cross sections",
                                        f"{model_ws}/fig",
                                        csv_file_name,
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
    ncol = gwf.dis.ncol.get_data()
    nrow = gwf.dis.nrow.get_data()
    nlay = gwf.dis.nlay.get_data()

    # ----------------------------- UPDATE PACKAGES ----------------------------------- #

    # Update time discretization
    nper = 4
    perlen = [0]*1 + [6000000]*1 + [60000]*1 + [6000000]*1
    nstp = [1]*1 + [60]*1 +[300]*1 + [60]*1
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

    # Create storage package for transient simulation
    sto = flopy.mf6.ModflowGwfsto(
        gwf,
        pname="sto",
        iconvert = 1, #Unonfined/confined mixed storage is used
        sy=sy, #Specific yield
        ss=ss, #If not specified, flopy uses default value of 1e-5 m-1
        steady_state={0: True},
        transient={1: True})

    # Update output control
    oc = gwf.oc
    oc.head_filerecord = f"{transient_name}.hds"
    oc.budget_filerecord = f"{transient_name}.cbb"
    oc.budgetcsv_filerecord = f"{transient_name}_budget.csv"

    # ---------------------------- UPDATE TRANSIENT BOUNDARY CONDITIONS -------------------------- #
    # Update transient recharge package
    rch = flopy.mf6.ModflowGwfrcha(gwf, 
                                pname = "rch",
                                save_flows = True,
                                fixed_cell= True,
                                irch=irch,
                                recharge = "TIMEARRAYSERIES recharge", 
                                filename = f"{transient_name}.rcha")
    tas_data = {0 : R_array,
            3000000 : R_array,
            20000000 : R_array}
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
                              filename = f"{transient_name}.wel")
    ts_data = [(0, 0),
               (6000000 , q),
               (20000000 , q)]
    wel.ts.initialize(
        filename="well_rates.ts",
        timeseries=ts_data,
        time_series_namerecord=["wells1"],
        interpolation_methodrecord=["stepwise"])

    # ------------------------- OBSERVATIONS ------------------------------ #

    # Head obervations
    obs_recarray = {
        "head_obs_t.csv": [("Head at pumping well - Unconfined Aquifer", "HEAD", (0, 0, well_loc[2])),
                        ("Head at pumping well - Aquitard", "HEAD", (1, 0, well_loc[2])),
                        ("Head at pumping well - Confined Aquifer", "HEAD", (2, 0, well_loc[2])),
                        ("Head at pumping well - Aquitard", "HEAD", (3, 0, well_loc[2])),
                        ("Head at pumping well - Confined Aquifer", "HEAD", (4, 0, well_loc[2]))]}

    obs_package = flopy.mf6.ModflowUtlobs(
        gwf,
        pname="head_obs_t",
        print_input=True,
        continuous=obs_recarray)
    
    # ---------------------------- RUN SIMULATION ------------------------------- #

    sim.write_simulation()
    sim.run_simulation()

    # ------------------------------ ZONE BUDGET -------------------------------- #

    zonbud = gwf.output.zonebudget(zone_array)
    zonbud.change_model_ws(model_ws)
    zonbud.write_input()
    zonbud.run_model()

    # --------------------------------------------------------------------------- #
    # ------------------------------- PLOTTING ---------------------------------- #
    # --------------------------------------------------------------------------- #
    if plot_transient:
        # Select time step, period, and layer to plot
        ts_num = 0
        sp_num = nper - 1 
        layer = 0
        time = modtransient6.elapsed_time(perioddata, sp_num, ts_num)

        head = gwf.output.head().get_data(kstpkper=(ts_num, sp_num)) #kstpkper = time step and stress period.
        bud = gwf.output.budget()
        spdis = bud.get_data(text='DATA-SPDIS', kstpkper=(ts_num, sp_num))[layer] 
        qx, qy, qz = flopy.utils.postprocessing.get_specific_discharge(spdis, gwf)

        hobj = flopy.utils.HeadFile(f"{model_ws}/{transient_name}.hds")
        heads = hobj.get_alldata()

        fig_dir = os.path.join(model_ws, "fig")
        os.makedirs(fig_dir, exist_ok=True) 

        #--------------------------------------- HEADS ---------------------------------------------#

        modplot6.plot_cross_section_row(gwf, head, qx, qy, qz, nrow//2, 
                                        f"{model_ws}/fig/cross_section_heads_t.png",
                                        boundary_keywords = boundary_keywords,
                                        flow_dir = True, surface = True, layers=True,
                                        show=False, save=True, figsize = (19, 4),
                                        title=f"Cross section at time {time} days")

        modtransient6.plot_head_time_series("head_obs_t.csv", 
                                            gwf, 
                                            f"{model_ws}/fig/head_ts.png",
                                            show = False, 
                                            save = True, 
                                            tau = None)

        #--------------------------------------- FLOW BUDGET ---------------------------------------------#

        file_path = f"{model_ws}/{transient_name}_budget.csv"
        modtransient6.process_csv_budget(file_path)  

        modtransient6.plot_bud_sum_transient(file_path, time, 
                                            f"{model_ws}/fig/bud_sum_t.png", 
                                            show = False, save=True)

        modtransient6.plot_bud_time_series(file_path,  
                                        f"{model_ws}/fig/budget_ts.png", 
                                        show=False, save=True)

        modtransient6.plot_water_to_wells(file_path, 
                                        f"{model_ws}/fig/water_to_wells.png", 
                                        show=False, 
                                        save=True)

        modtransient6.plot_net_flow_time_series(file_path,
                                                f"{model_ws}/fig/net_flow_ts.png",
                                                show=False, save=True, tau = None)

        #--------------------------------------- ZONE BUDGET ---------------------------------------------#
        zone_bud_path = f"{model_ws}/zonebud.csv"
        modtransient6.plot_zone_budget(zone_bud_path, fig_dir, show=False, save=True, zone_descriptions = {
                1: "Unconfined Aquifer",
                2: "Aquitard",
                3: "Confined Aquifer",
                4: "Aquitard",
                5: "Confined Aquifer"})
        modtransient6.plot_water_to_wells_zonebud(zone_bud_path, fig_dir, show=False, save=True)

        #--------------------------------------- TRANSIENT ANIMATION ---------------------------------------------#

        if animate:
            modplot6.plot_animation(gwf, heads, qx, qy, qz, nrow//2, 
                                        f"{model_ws}/sections",
                                        f"{model_ws}/fig/heads_transient.gif",
                                        boundary_keywords = boundary_keywords,
                                        flow_dir = False, surface = True, layers=True,
                                        show=False, save=True, figsize = (19, 4), gif_start=0, gif_step=10)