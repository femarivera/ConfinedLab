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
path = r"."
subfolder_keyword = "kv_"
# Set parameters for analysis
anis = 1
bf = 0.5 #Thickness factor with respect to base case
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
B=sum(thickness_dict.values()) #Maximum thickness of the system 
L=max(length_dict.values()) #Maximum length of the system
B_threshold=sum(v for k, v in thickness_dict.items() if k % 2 == 0) / sum(k % 2 == 0 for k in thickness_dict) #Aquitard mean thickness

# Collect results and postprocess outputs
df_analysis = modtransient6.analyze_results(path,
                            thickness_dict=thickness_dict,
                            length_dict = length_dict,
                            unc_length_dict = unc_length_dict,
                            subfolder_keyword=subfolder_keyword,
                            volume_weighted=True,
                            B=B, 
                            L=L,)
df_analysis.to_csv(path + "/tr_analysis.csv", index=False)

# Scatter plots existing analytical formulations vs simulated response time per aquifer zone
modtransient6.loglog_scatter_df(
    df=df_analysis,
    x_column="tr_zone",
    y_column="tr_95p_vol_zone",
    zone_value=5,
    color_zone_value=5,
    color_column="Dv_eq",
    color_bar_label="Equivalent vertical diffusivity [m²/s]",
    xlabel="Analytical response time [years]",
    ylabel="Simulated response time [years]",
    marker_size=70,
    min_val=1, max_val=1e6, 
    cmap = "plasma_r", 
    SAVE=True,
    output_path=path + "/tr_scatter_zone5.png")

modtransient6.loglog_scatter_df(
    df=df_analysis,
    x_column="tr_mixed_zone",
    y_column="tr_95p_vol_zone",
    zone_value=5,
    color_zone_value=5,
    color_column="Dv_eq",
    color_bar_label="Equivalent vertical diffusivity [m²/s]",
    xlabel="Analytical response time [years]",
    ylabel="Simulated response time [years]",
    marker_size=70,
    min_val=1, max_val=1e6, 
    cmap = "plasma_r",
    SAVE=True,
    output_path=path + "/tr_scatter_mixed_zone5.png")

modtransient6.loglog_scatter_df(
    df=df_analysis,
    x_column="tr_zone",
    y_column="tr_95p_vol_zone",
    zone_value=3,
    color_zone_value=3,
    color_column="Dv_eq",
    color_bar_label="Equivalent vertical diffusivity [m²/s]",
    xlabel="Analytical response time [years]",
    ylabel="Simulated response time [years]",
    marker_size=70,
    min_val=1, max_val=1e6, 
    cmap = "plasma_r", 
    SAVE=True,
    output_path=path + "/tr_scatter_zone3.png")

modtransient6.loglog_scatter_df(
    df=df_analysis,
    x_column="tr_mixed_zone",
    y_column="tr_95p_vol_zone",
    zone_value=3,
    color_zone_value=3,
    color_column="Dv_eq",
    color_bar_label="Equivalent vertical diffusivity [m²/s]",
    xlabel="Analytical response time [years]",
    ylabel="Simulated response time [years]",
    marker_size=70,
    min_val=1, max_val=1e6, 
    cmap = "plasma_r",
    SAVE=True,
    output_path=path + "/tr_scatter_mixed_zone3.png")

# Scatter plots revised analytical formulation vs simulated response time per aquifer zone
modtransient6.loglog_scatter_df(
    df=df_analysis,
    x_column="tr_aquifer",
    y_column= "tr_95p_vol_zone",
    zone_value=5, 
    color_zone_value=5,
    color_column="Dv_eq",
    color_bar_label="Equivalent vertical diffusivity [m²/s]",
    xlabel="Analytical response time [years]",
    ylabel="Simulated response time [years]",
    marker_size=70,
    cmap = "plasma_r",
    min_val=1, max_val=1e6,
    SAVE=True,
    output_path=path + "/tr_scatter_revised_zone5.png")

modtransient6.loglog_scatter_df(
    df=df_analysis,
    x_column="tr_aquifer",
    y_column= "tr_95p_vol_zone",
    zone_value=3, 
    color_zone_value=3,
    color_column="Dv_eq",
    color_bar_label="Equivalent vertical diffusivity [m²/s]",
    xlabel="Analytical response time [years]",
    ylabel="Simulated response time [years]",
    marker_size=70,
    cmap = "plasma_r",
    min_val=1, max_val=1e6,
    SAVE=True,
    output_path=path + "/tr_scatter_revised_zone3.png")

# Scatter plots revised analytical formulation vs simulated response time basin-scale
modtransient6.loglog_scatter_df(
    df=df_analysis,
    x_column="tr_basin",
    y_column= "tr_95p_vol_seq",
    zone_value=5,
    color_zone_value=5,
    color_column="Dv_eq",
    color_bar_label="Equivalent vertical diffusivity [m²/s]",
    xlabel="Analytical response time [years]",
    ylabel="Simulated response time [years]",
    marker_size=70,
    cmap = "plasma_r",
    min_val=1, max_val=1e6,
    SAVE=True,
    output_path=path + "/tr_scatter_basin_zone5.png")

modtransient6.loglog_scatter_df(
    df=df_analysis,
    x_column="tr_basin",
    y_column= "tr_95p_vol_seq",
    zone_value=3,
    color_zone_value=3,
    color_column="Dv_eq",
    color_bar_label="Equivalent vertical diffusivity [m²/s]",
    xlabel="Analytical response time [years]",
    ylabel="Simulated response time [years]",
    marker_size=70,
    cmap = "plasma_r",
    min_val=1, max_val=1e6,
    SAVE=True,
    output_path=path + "/tr_scatter_basin_zone3.png")

# Plot contour maps of basin-scale response time as a function of equivalent diffusivities
modtransient6.loglog_contours_df(
    df_analysis,
    zone=5,
    x_col="Dh_eq",
    y_col="Dv_eq",
    z_col="tr_95p_vol_seq",
    B=B,
    L=L,
    plot_threshold=True,
    plot_B_threshold=True,
    x_label="Equivalent horizontal diffusivity [m²/s]",
    y_label="Equivalent vertical diffusivity [m²/s]",
    z_label="Response time [years]",
    y_max_log=-1.5, y_min_log=-7.5,
    x_min_log=-2.33, x_max_log=1.65,
    grid_n=100,
    SAVE=True,
    output_path_interpolation=path + "/tr_contours_basin_zone5.png")

modtransient6.loglog_contours_df(
    df_analysis,
    zone=3,
    x_col="Dh_eq",
    y_col="Dv_eq",
    z_col="tr_95p_vol_seq",
    B=B,
    L=L,
    plot_threshold=True,
    plot_B_threshold=True,
    x_label="Equivalent horizontal diffusivity [m²/s]",
    y_label="Equivalent vertical diffusivity [m²/s]",
    z_label="Response time [years]",
    y_max_log=-1.5, y_min_log=-7.5,
    x_min_log=-2.33, x_max_log=1.65,
    grid_n=100,
    SAVE=True,
    output_path_interpolation=path + "/tr_contours_basin_zone3.png")

# Plot contour maps of aquifer-scale response time as a function of equivalent diffusivities
modtransient6.loglog_contours_df(
    df_analysis,
    zone=3,
    x_col="Dh_eq",
    y_col="Dv_eq",
    z_col="tr_95p_vol_zone",
    B=B,
    L=L,
    plot_threshold=True,
    plot_B_threshold=True,
    x_label="Equivalent horizontal diffusivity [m²/s]",
    y_label="Equivalent vertical diffusivity [m²/s]",
    z_label="Response time [years]",
    y_max_log=-1.5, y_min_log=-7.5,
    x_min_log=-2.33, x_max_log=1.65,
    grid_n=100,
    SAVE=True,
    output_path_interpolation=path + "/tr_contours_zone3.png")

modtransient6.loglog_contours_df(
    df_analysis,
    zone=5,
    x_col="Dh_eq",
    y_col="Dv_eq",
    z_col="tr_95p_vol_zone",
    B=B,
    L=L,
    plot_threshold=True,
    plot_B_threshold=True,
    x_label="Equivalent horizontal diffusivity [m²/s]",
    y_label="Equivalent vertical diffusivity [m²/s]",
    z_label="Response time [years]",
    y_max_log=-1.5, y_min_log=-7.5,
    x_min_log=-2.33, x_max_log=1.65,
    grid_n=100,
    SAVE=True,
    output_path_interpolation=path + "/tr_contours_zone5.png")
