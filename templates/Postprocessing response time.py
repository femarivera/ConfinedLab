# ==========================================================================================
#  Postprocessing a set of flow simulations and response time estimations
# ==========================================================================================
#
#  Author: MARIN RIVERA Carlos Felipe
#  Organization: Bordeaux INP, Lab EPOC, Université de Bordeaux
#  Project: OneWater PEPR DEESAC Project
#
#  DESCRIPTION:
#  ------------
#  This script collects results and postprocesses outputs from a set of flow simulations
#  and response time estimations. 
#
#  USAGE:
#  ------
#  Set folder path containing simulation results and parameters of the functions
#
# ==========================================================================================

import warnings
import sys
# Import local modules
sys.path.append('..')
from mlibs import modpar6, modplot6, modtransient6, modpump6, modgeom6, modbound6 # type: ignore
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Set folder path containing simulation results
path = r"D:\DEESAC\Response time\01 Recharge decrease 50percent - Base case"

# Set parameters for analysis
anis = 1
B=900 #Maximum thickness of the system 
L=600000 #Maximum length of the system
B1=150 #Aquitard mean thickness
bf = 1 #Thickness factor with respect to base case
lf = 1 #Length factor with respect to base case
thickness_dict= {1:200*bf,
                2: 150*bf,
                3: 200*bf,
                4: 150*bf,
                5: 200*bf,}
length_dict = {1: 350000*lf,
                2: 400000*lf,
                3: 500000*lf,
                4: 550000*lf,
                5: 600000*lf,}
unc_length_dict = {1: 350000*lf,
                2: 50000*lf,
                3: 100000*lf,
                4: 50000*lf,
                5: 50000*lf,}

# Collect results and postprocess outputs
df_analysis = modtransient6.analyze_results(path,
                           thickness_dict=thickness_dict,
                            length_dict = length_dict,
                            unc_length_dict = unc_length_dict,
                            B=B, L=L,
                            subfolder_keyword="kv_")
df_analysis.to_csv(path + "/tr_analysis.csv", index=False)

# Scatter plots
modtransient6.loglog_scatter_df(
    df=df_analysis,
    x_column="tr_zone",
    y_column="tr_95p_vol_zone",
    zone_value=5,
    color_zone_value=2,
    color_column="Dv",
    color_bar_label="Aquitard vertical diffusivity [m²/s]",
    xlabel="Analytical response time [years]",
    ylabel="Simulated response time [years]",
    marker_size=70,
    min_val=1, max_val=1e8, 
    cmap = "plasma_r", 
    SAVE=True,
    output_path=path + "/tr_scatter_zone5.png")

modtransient6.loglog_scatter_df(
    df=df_analysis,
    x_column="tr_mixed_zone",
    y_column="tr_95p_vol_zone",
    zone_value=5,
    color_zone_value=2,
    color_column="Dv",
    color_bar_label="Aquitard vertical diffusivity [m²/s]",
    xlabel="Analytical response time [years]",
    ylabel="Simulated response time [years]",
    marker_size=70,
    min_val=1, max_val=1e8, 
    cmap = "plasma_r",
    SAVE=True,
    output_path=path + "/tr_scatter_mixed_zone5.png")

modtransient6.loglog_scatter_df(
    df=df_analysis,
    x_column="tr_basin",
    y_column= "tr_95p_vol",
    zone_value=5, # This zone is not relevant, since the values are basin scale
    color_zone_value=5, # Each zone of the same simulation has the same basin scale values
    color_column="Dv_eq",
    color_bar_label="Equivalent vertical diffusivity [m²/s]",
    xlabel="Analytical response time [years]",
    ylabel="Simulated response time [years]",
    marker_size=70,
    cmap = "plasma_r",
    SAVE=True,
    output_path=path + "/tr_scatter_basin.png")

# Plot contour maps of response time as a function of equivalent diffusivities
modtransient6.loglog_contours_df(
    df_analysis,
    x_col=df_analysis["Dh_eq"],
    y_col=df_analysis["Dv_eq"],
    z_col=df_analysis["tr_95p_vol"],
    B=2*B,
    L=L,
    B1=2*B1,
    anis=None,
    x_label="Equivalent horizontal diffusivity [m²/s]",
    y_label="Equivalent vertical diffusivity [m²/s]",
    z_label="Response time [years]",
    y_max_log=-1.5,
    y_min_log=-7.5,
    x_min_log=-2.3, x_max_log=1.6,
    grid_n=100,
    SAVE=True,
    output_path_interpolation=path + "/tr_contours_basin.png"
    )

modtransient6.loglog_contours_df(
    df_analysis,
    x_col=df_analysis[df_analysis["zone"] == 5]["Dh"],
    y_col=df_analysis[df_analysis["zone"] == 4]["Dv"],
    z_col=df_analysis[df_analysis["zone"] == 5]["tr_95p_vol_zone"],
    # B=B,
    # L=L,
    # B1=B1,
    anis=None,
    x_label="Aquifer horizontal diffusivity [m²/s]",
    y_label="Aquitard vertical diffusivity [m²/s]",
    z_label="Response time [years]",
    grid_n=100,
    SAVE=True,
    output_path_interpolation=path + "/tr_contours_zone5.png"
    )