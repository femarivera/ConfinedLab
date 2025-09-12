# ==========================================================================================
#  modplot6.py - Modular Plotting Utilities for MODFLOW 6 Model Visualization
# ==========================================================================================
#
#  Author: MARIN RIVERA Carlos Felipe
#  Organization: Bordeaux INP, Lab EPOC, Université de Bordeaux
#  Project: Funded by the OneWater PEPR DEESAC Project 
#
#  DESCRIPTION:
#  ------------
#  As part of the ConfinedLab project, this module provides flexible utilities 
#  for visualizing MODFLOW 6 groundwater model results (including flow, transport, and particle tracking). 
# 
#  MAIN FEATURES:
#  --------------
#  - Plot 2D and 3D model grids, heads, and boundary conditions.
#  - Visualize cross-sections and budget summaries.
#  - Transient simulations and animations.

def plot_map_view(gwf, 
                  head, 
                  qx, 
                  qy, 
                  output_path, 
                  boundary_keywords=None, 
                  layer=0, 
                  flow_dir=False, 
                  contours=False,
                  show=False, 
                  save=False,
                  grid=True,
                  figsize=(10, 10),
                  fontsize=14,
                  title="Model map view"):
    """
    Plots a map view for a MODFLOW 6 groundwater flow model.

    Args:
        gwf (flopy.mf6.ModflowGwf): Groundwater flow model object.
        head (numpy.ndarray): Head array for the model.
        qx, qy (numpy.ndarray): Flow vectors in x and y directions.
        output_path (str): File path to save the plot.
        boundary_keywords (list of str): Keywords for boundary condition columns to include.
        layer (int): Model layer to plot.
        flow_dir (bool): Whether to plot flow vectors.
        contours (bool): Whether to plot contours.
        show (bool): Whether to display the plot.
        save (bool): Whether to save the plot to file.
        figsize (tuple): Size of the figure.
        fontsize (int): Font size for plot labels.

    Outputs:
        Displays the map view plot and/or saves it to a file.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import flopy
    import os

    # Input checks
    if gwf is None:
        raise ValueError("gwf (MODFLOW 6 model object) must be provided.")
    if head is None:
        raise ValueError("head array must be provided.")
    if qx is None or qy is None:
        raise ValueError("qx and qy (flow vectors) must be provided.")
    if not isinstance(output_path, str) or not output_path:
        raise ValueError("output_path must be a non-empty string.")
    if boundary_keywords is not None and not isinstance(boundary_keywords, list):
        raise ValueError("boundary_keywords must be a list of strings or None.")
    if not isinstance(layer, int) or layer < 0:
        raise ValueError("layer must be a non-negative integer.")
    if not isinstance(flow_dir, bool):
        raise ValueError("flow_dir must be a boolean.")
    if not isinstance(contours, bool):
        raise ValueError("contours must be a boolean.")
    if not isinstance(show, bool):
        raise ValueError("show must be a boolean.")
    if not isinstance(save, bool):
        raise ValueError("save must be a boolean.")
    if not (isinstance(figsize, tuple) and len(figsize) == 2):
        raise ValueError("figsize must be a tuple of length 2.")
    if not isinstance(fontsize, (int, float)):
        raise ValueError("fontsize must be a number.")
    if title is not None and not isinstance(title, str):
        raise ValueError("title must be a string or None.")    
    
    # Default color mapping based on boundary condition type
    color_map = {
        "RIV": "blue",
        "WEL": "red",
        "GHB": "black",
        "DRN": "gray",
        "CHD": "purple"
    }

    # Mask inactive cells
    idomain = gwf.modelgrid.idomain
    masked_head = np.where(idomain == 0, np.nan, head)

    # Initialize the figure and axes
    fig, ax = plt.subplots(1, 1, figsize=figsize, constrained_layout=True)
    ax.set_title(title, fontsize=fontsize)

    # Compute minimum and maximum head values for color scaling
    vmin, vmax = np.nanmin(masked_head), np.nanmax(masked_head)

    # Create the map view object
    modelmap = flopy.plot.PlotMapView(model=gwf, ax=ax, layer=layer)
    # Plot the heads
    pa = modelmap.plot_array(masked_head, vmin=vmin, vmax=vmax)

    # Add contours
    if contours:
        contour_intervals = np.arange(vmin, vmax + 1, (vmax-vmin)/10)
        contours = modelmap.contour_array(masked_head, levels=contour_intervals, colors="black")
        ax.clabel(contours, fmt="%2.1f")

    # Plot the grid
    if grid:
        modelmap.plot_grid(lw=0.1, color="0.5")

    # Plot flow vectors
    if flow_dir:
        modelmap.plot_vector(qx, qy, normalize=True, color="white", headwidth=2, headlength=1, headaxislength=1, scale=60)

    # Dynamically plot boundary conditions based on keywords
    if boundary_keywords:
        for bc in boundary_keywords:
            # Determine color based on the keyword
            bc_color = None
            for key in color_map:
                if key in bc:  # Check if the keyword contains the key
                    bc_color = color_map[key]
                    break
            # Plot the boundary condition with the appropriate color
            if bc_color:
                modelmap.plot_bc(bc, color=bc_color)

    # Add colorbar
    cb = plt.colorbar(pa, shrink=0.5, ax=ax)
    cb.set_label("Head [m]", fontsize=fontsize)

    # Adjust layout and show plot
    plt.ioff()
    if show:
        plt.show()

    # Save plot
    if save:
        # Create directory if it does not exist
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        fig.savefig(output_path, dpi=300)
        plt.close(fig)

def plot_cross_section_row(gwf, 
                           head,
                           qx,
                           qy,
                           qz,
                           row, 
                           output_path, 
                           boundary_keywords=None, 
                           flow_dir=False, 
                           surface=False,
                           layers=False, 
                           show=False, 
                           save=False, 
                           ax=None,
                           figsize=(19, 6),
                           fontsize=14, 
                           title="Cross section"):
    """
    Plots a cross-section for a MODFLOW 6 groundwater flow model along a specified row.

    Args:
        gwf (flopy.mf6.ModflowGwf): Groundwater flow model object.
        head (np.ndarray): Head array for the model (or any other array to plot).
        qx, qy, qz (np.ndarray): Flow vectors in x, y, and z directions.
        row (int): Row number for the cross-section.
        output_path (str): File path to save the plot.
        boundary_keywords (list of str, optional): Keywords for boundary condition columns to include.
        flow_dir (bool, optional): Whether to include flow direction vectors.
        surface (bool, optional): Whether to include the surface head plot.
        layers (bool, optional): Whether to include layer legend.
        show (bool, optional): Whether to display the plot.
        save (bool, optional): Whether to save the plot to file.
        ax (matplotlib.axes.Axes, optional): Matplotlib axis to plot on. If None, a new figure is created.
        figsize (tuple, optional): Size of the figure.
        fontsize (int, optional): Font size for plot labels.
        title (str, optional): Title for the plot.

    Outputs:
        Displays the cross-section plot and/or saves it to a file.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import flopy
    from matplotlib.cm import get_cmap
    from matplotlib.lines import Line2D
    import os

    # Input checks
    if gwf is None:
        raise ValueError("gwf (MODFLOW 6 model object) must be provided.")
    if head is None:
        raise ValueError("head array must be provided.")
    if qx is None or qy is None or qz is None:
        raise ValueError("qx, qy, and qz (flow vectors) must be provided.")
    if not isinstance(row, int) or row < 0:
        raise ValueError("row must be a non-negative integer.")
    if not isinstance(output_path, str) or not output_path:
        raise ValueError("output_path must be a non-empty string.")
    if boundary_keywords is not None and not isinstance(boundary_keywords, list):
        raise ValueError("boundary_keywords must be a list of strings or None.")
    if not isinstance(flow_dir, bool):
        raise ValueError("flow_dir must be a boolean.")
    if not isinstance(surface, bool):
        raise ValueError("surface must be a boolean.")
    if not isinstance(layers, bool):
        raise ValueError("layers must be a boolean.")
    if not isinstance(show, bool):
        raise ValueError("show must be a boolean.")
    if not isinstance(save, bool):
        raise ValueError("save must be a boolean.")
    if not (isinstance(figsize, tuple) and len(figsize) == 2):
        raise ValueError("figsize must be a tuple of length 2.")
    if not isinstance(fontsize, (int, float)):
        raise ValueError("fontsize must be a number.")
    if title is not None and not isinstance(title, str):
        raise ValueError("title must be a string or None.")
    
    # Default color mapping based on boundary condition type
    color_map = {
        "RIV": "blue",
        "WEL": "red",
        "GHB": "black",
        "DRN": "gray",
        "CHD": "purple"
    }

    # Validate row index
    nrow, ncol = gwf.modelgrid.nrow, gwf.modelgrid.ncol
    assert 0 <= row < nrow, f"Row index {row} is out of bounds for grid with {nrow} rows."

    # If no axis is provided, create a new figure and axis
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.set_title(title, fontsize=fontsize)

    # Mask inactive cells
    idomain = gwf.modelgrid.idomain
    masked_head = np.where(idomain == 0, np.nan, head)

    # Compute minimum and maximum head values for color scaling
    vmin, vmax = np.nanmin(masked_head), np.nanmax(masked_head)

    # Create the cross-section object
    section = flopy.plot.PlotCrossSection(
        model=gwf,
        ax=ax,
        line={"row": row}
    )

    # Plot the array
    pa = section.plot_array(masked_head, head=masked_head, vmin=vmin, vmax=vmax)

    # Plot surface for each layer with a gradient of blues
    if surface:
        cmap = get_cmap("Blues")
        num_layers = masked_head.shape[0]
        layer_colors = []  # Store colors for legend
        for layer in range(num_layers):
            # Assign a color based on the layer index
            color = cmap((layer + 1) / num_layers)  # Normalize the layer index
            section.plot_surface(masked_head[layer, :, :], color=color, lw=1)
            layer_colors.append((color, f"Layer {layer + 1}"))

    # Plot the grid lines
    section.plot_grid(lw=0.05, color="0")
    
    # Plot flow vectors
    if flow_dir:
        section.plot_vector(qx, qy, qz, normalize=True, color="white", head=masked_head,
                            hstep=int(ncol//50), headwidth=2, headlength=1, headaxislength=1, scale=60)

    # Dynamically plot boundary conditions based on keywords
    if boundary_keywords:
        for bc in boundary_keywords:
            # Determine color based on the keyword
            bc_color = None
            for key in color_map:
                if key in bc:  # Check if the keyword contains the key
                    bc_color = color_map[key]
                    break
            # Plot the boundary condition with the appropriate color
            if bc_color:
                section.plot_bc(bc, color=bc_color)

    # Add colorbar
    cb = plt.colorbar(pa, ax=ax)
    cb.set_label("Head [m]", fontsize=fontsize)

    # Add legend for layers with unique entries
    if surface:
        legend_handles = [Line2D([0], [0], color=color, lw=2, label=label) for color, label in layer_colors]
        if layers:
            ax.legend(handles=legend_handles, loc="lower left", title="Layers", fontsize=fontsize/1.5)

    # Show and save the plot
    plt.ioff()
    if show:
        plt.show()

    # Save plot
    if save:
        # Create directory if it does not exist
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        fig = ax.get_figure()
        fig.savefig(output_path, dpi=300)
        plt.close(fig)  

def plot_cross_section_col(gwf, 
                           head, 
                           qx, 
                           qy, 
                           qz, 
                           col, 
                           output_path, 
                           boundary_keywords=None, 
                           flow_dir=False, 
                           surface=False, 
                           layers=False,
                           show=False, 
                           save=False, 
                           ax=None,
                           figsize=(19, 6),
                           fontsize=14, 
                           title = "Cross section"):
    """
    Plots a cross-section for a MODFLOW 6 groundwater flow model along a specified column.

    Args:
        gwf (flopy.mf6.ModflowGwf): Groundwater flow model object.
        head (numpy.ndarray): Head array for the model (or any other array to plot).
        qx, qy, qz (numpy.ndarray): Flow vectors in x, y, and z directions.
        col (int): Column number for the cross-section.
        output_path (str): File path to save the plot.
        boundary_keywords (list of str, optional): Keywords for boundary condition columns to include.
        flow_dir (bool, optional): Whether to include flow direction vectors.
        surface (bool, optional): Whether to include the surface head plot.
        layers (bool, optional): Whether to include layer legend.
        show (bool, optional): Whether to display the plot.
        save (bool, optional): Whether to save the plot to file.
        ax (matplotlib.axes.Axes, optional): Matplotlib axis to plot on. If None, a new figure is created.
        figsize (tuple, optional): Size of the figure.
        fontsize (int, optional): Font size for plot labels.
        title (str, optional): Title for the plot.

    Outputs:
        Displays the cross-section plot and/or saves it to a file.
    """

    import matplotlib.pyplot as plt
    import numpy as np
    import flopy
    from matplotlib.cm import get_cmap
    from matplotlib.lines import Line2D
    import os

    # Input checks
    if gwf is None:
        raise ValueError("gwf (MODFLOW 6 model object) must be provided.")
    if head is None:
        raise ValueError("head array must be provided.")
    if qx is None or qy is None or qz is None:
        raise ValueError("qx, qy, and qz (flow vectors) must be provided.")
    if not isinstance(col, int) or col < 0:
        raise ValueError("col must be a non-negative integer.")
    if not isinstance(output_path, str) or not output_path:
        raise ValueError("output_path must be a non-empty string.")
    if boundary_keywords is not None and not isinstance(boundary_keywords, list):
        raise ValueError("boundary_keywords must be a list of strings or None.")
    if not isinstance(flow_dir, bool):
        raise ValueError("flow_dir must be a boolean.")
    if not isinstance(surface, bool):
        raise ValueError("surface must be a boolean.")
    if not isinstance(layers, bool):
        raise ValueError("layers must be a boolean.")
    if not isinstance(show, bool):
        raise ValueError("show must be a boolean.")
    if not isinstance(save, bool):
        raise ValueError("save must be a boolean.")
    if not (isinstance(figsize, tuple) and len(figsize) == 2):
        raise ValueError("figsize must be a tuple of length 2.")
    if not isinstance(fontsize, (int, float)):
        raise ValueError("fontsize must be a number.")
    if title is not None and not isinstance(title, str):
        raise ValueError("title must be a string or None.")

    # Default color mapping based on boundary condition type
    color_map = {
        "RIV": "blue",
        "WEL": "red",
        "GHB": "black",
        "DRN": "gray",
        "CHD": "purple"
    }

    # Validate row index
    nrow, ncol = gwf.modelgrid.nrow, gwf.modelgrid.ncol
    assert 0 <= col < ncol, f"RColumn index {col} is out of bounds for grid with {ncol} columns."

    # If no axis is provided, create a new figure and axis
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.set_title(title, fontsize=fontsize)

    # Mask inactive cells
    idomain = gwf.modelgrid.idomain
    masked_head = np.where(idomain == 0, np.nan, head)

    # Compute minimum and maximum head values for color scaling
    vmin, vmax = np.nanmin(masked_head), np.nanmax(masked_head)

    # Create the cross-section object
    section = flopy.plot.PlotCrossSection(
        model=gwf,
        ax=ax,
        line={"column": col}
    )

    # Plot the array
    pa = section.plot_array(masked_head, head=masked_head, vmin=vmin, vmax=vmax)

    # Plot surface for each layer with a gradient of blues
    if surface:
        cmap = get_cmap("Blues")
        num_layers = masked_head.shape[0]
        layer_colors = []  # Store colors for legend
        for layer in range(num_layers):
            # Assign a color based on the layer index
            color = cmap((layer + 1) / num_layers)  # Normalize the layer index
            section.plot_surface(masked_head[layer, :, :], color=color, lw=2)
            layer_colors.append((color, f"Layer {layer + 1}"))

    # Plot the grid lines
    section.plot_grid(lw=0.1, color="0.5")

    # Plot flow vectors
    if flow_dir:
        section.plot_vector(qx, qy, qz, normalize=True, color="white", head=masked_head, 
                            hstep=int(nrow//50), headwidth=2, headlength=1, headaxislength=1, scale=60)
        
    # Dynamically plot boundary conditions based on keywords
    if boundary_keywords:
        for bc in boundary_keywords:
            # Determine color based on the keyword
            bc_color = None
            for key in color_map:
                if key in bc:  # Check if the keyword contains the key
                    bc_color = color_map[key]
                    break
            # Plot the boundary condition with the appropriate color
            if bc_color:
                section.plot_bc(bc, color=bc_color)

    # Add colorbar
    cb = plt.colorbar(pa, ax=ax)
    cb.set_label("Head [m]", fontsize=fontsize)

    # Add legend for layers with unique entries
    if surface:
        legend_handles = [Line2D([0], [0], color=color, lw=2, label=label) for color, label in layer_colors]
        if layers:
            ax.legend(handles=legend_handles, loc="lower left", title="Layers", fontsize=fontsize/1.5)

    # Show and save the plot
    plt.ioff()
    if show:
        plt.show()

    # Save plot
    if save:
        # Create directory if it does not exist
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        fig = ax.get_figure()
        fig.savefig(output_path, dpi=300)
        plt.close(fig)  

def plot_bud_sum_steady(file_path, 
                        output_path, 
                        show=False, 
                        save=False, 
                        figsize=(19, 6), 
                        fontsize=14):
    """
    Creates bar plots for inflow, outflow, and total flows based on a budget summary CSV file
    output of a MODFLOW 6 steady-state simulation.

    Args:
        file_path (str): Path to the budget CSV file. The file should have one row,
                        with columns ending in _IN, _OUT, and containing TOTAL_IN and TOTAL_OUT.
        output_path (str): Path to save the output figure.
        show (bool): Whether to display the plot.
        save (bool): Whether to save the plot to a file.
        figsize (tuple): Size of the figure.
        fontsize (int): Font size for plot labels.

    Outputs:
        Displays and/or saves a single figure with three subplots showing inflow, outflow, and total flows.
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    import os

    # Input checks
    if not isinstance(file_path, str) or not file_path:
        raise ValueError("file_path must be a non-empty string.")
    if not isinstance(output_path, str) or not output_path:
        raise ValueError("output_path must be a non-empty string.")
    if not isinstance(show, bool):
        raise ValueError("show must be a boolean.")
    if not isinstance(save, bool):
        raise ValueError("save must be a boolean.")
    if not (isinstance(figsize, tuple) and len(figsize) == 2):
        raise ValueError("figsize must be a tuple of length 2.")
    if not isinstance(fontsize, (int, float)):
        raise ValueError("fontsize must be a number.")

    # Load the CSV file
    data = pd.read_csv(file_path)

    # Simplify column names
    def simplify_name(name):
        """
        Simplify the name by extracting content inside parentheses if present.
        If no parentheses, replace underscores with spaces. If no underscores, fallback to original name.
        """
        if '(' in name and ')' in name:
            simplified = name.split('(')[1].split(')')[0].strip()  # Extract text inside parentheses
        elif '_' in name:
            simplified = name.replace('_', ' ')  # Replace underscores with spaces
        else:
         simplified = name.strip()  # Fallback to original name
        return simplified

    # Identify columns
    columns_in = [col for col in data.columns if col.endswith("_IN") and col != "TOTAL_IN"]
    columns_out = [col for col in data.columns if col.endswith("_OUT") and col != "TOTAL_OUT"]
    columns_total = ["TOTAL_IN", "TOTAL_OUT"]

    # Prepare data for plots
    data_in = data[columns_in].iloc[0]
    data_out = data[columns_out].iloc[0]
    data_total = data[columns_total].iloc[0]

    # Simplify column names for plotting
    columns_in_simplified = [simplify_name(col) for col in columns_in]
    columns_out_simplified = [simplify_name(col) for col in columns_out]
    columns_total_simplified = [simplify_name(col) for col in columns_total]

    # Create a figure with subplots
    fig, axs = plt.subplots(1, 3, figsize=figsize)

    # Determine the common y-axis range based on the "Total Inflow and Outflow" plot
    common_ylim = (0, max(max(data_in.values), max(data_out.values), max(data_total.values)) * 1.1)  # Add 10% padding

    # Plot inflow components
    axs[0].bar(columns_in_simplified, data_in.values, color="blue")
    axs[0].set_title("Inflow Components", fontsize=fontsize)
    axs[0].set_xlabel("Component")
    axs[0].set_ylabel("m³/day")
    axs[0].set_ylim(common_ylim) 
    for i, val in enumerate(data_in.values):
        axs[0].text(i, val, f"{val:.2f}", ha="center", va="bottom")

    # Plot outflow components
    axs[1].bar(columns_out_simplified, data_out.values, color="red")
    axs[1].set_title("Outflow Components", fontsize=fontsize)
    axs[1].set_xlabel("Component")
    #axs[1].set_ylabel("m³/day")
    axs[1].set_ylim(common_ylim) 
    for i, val in enumerate(data_out.values):
        axs[1].text(i, val, f"{val:.2f}", ha="center", va="bottom")

    # Plot total inflow and outflow
    axs[2].bar(columns_total_simplified, data_total.values, color="green")
    axs[2].set_title("Total Inflow and Outflow", fontsize=fontsize)
    axs[2].set_xlabel("Component")
    #axs[2].set_ylabel("m³/day")
    axs[2].set_ylim(common_ylim) 
    for i, val in enumerate(data_total.values):
        axs[2].text(i, val, f"{val:.2f}", ha="center", va="bottom")

    # Adjust layout and show plot
    plt.ioff()
    if show:
        plt.show()

    # Save plot
    if save:
        # Create directory if it does not exist
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        fig.savefig(output_path, dpi=300)
        plt.close(fig)

def plot_cross_section_array(gwf, 
                             array, 
                             row, 
                             output_path, 
                             boundary_keywords=None, 
                             show = False, 
                             save = False, 
                             ax=None,
                             figsize=(19, 6),
                             fontsize=14, 
                             title="Cross section",
                             colorbar = True,
                             log=False,
                             label="Legend"):
    """
    Plots a cross-section for a MODFLOW 6 groundwater flow model along a specified row.

    Args:
        gwf (flopy.mf6.ModflowGwf): Groundwater flow model object.
        array (numpy.ndarray): Any array to plot matching the shape and dimensions of the model grid.
        row (int): Row number for the cross-section.
        output_path (str): File path to save the plot.
        boundary_keywords (list of str, optional): Keywords for boundary condition columns to include.
        show (bool, optional): Whether to display the plot.
        save (bool, optional): Whether to save the plot to file.
        ax (matplotlib.axes.Axes, optional): Matplotlib axis to plot on. If None, a new figure is created.
        figsize (tuple, optional): Size of the figure.
        fontsize (int, optional): Font size for plot labels.
        title (str, optional): Title for the plot.
        colorbar (bool, optional): Whether to include a colorbar.
        label (str, optional): Label for the colorbar.

    Outputs:
        Displays the cross-section plot and/or saves it to a file.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import flopy
    from matplotlib.cm import get_cmap
    from matplotlib.colors import LogNorm
    import os

    # Input checks

    if gwf is None:
        raise ValueError("gwf (MODFLOW 6 model object) must be provided.")
    if array is None:
        raise ValueError("array to plot must be provided.")
    if not isinstance(row, int) or row < 0:
        raise ValueError("row must be a non-negative integer.")
    if not isinstance(output_path, str) or not output_path:
        raise ValueError("output_path must be a non-empty string.")
    if boundary_keywords is not None and not isinstance(boundary_keywords, list):
        raise ValueError("boundary_keywords must be a list of strings or None.")
    if not isinstance(show, bool):
        raise ValueError("show must be a boolean.")
    if not isinstance(save, bool):
        raise ValueError("save must be a boolean.")
    if not (isinstance(figsize, tuple) and len(figsize) == 2):
        raise ValueError("figsize must be a tuple of length 2.")
    if not isinstance(fontsize, (int, float)):
        raise ValueError("fontsize must be a number.")
    if title is not None and not isinstance(title, str):
        raise ValueError("title must be a string or None.")
    if not isinstance(colorbar, bool):
        raise ValueError("colorbar must be a boolean.")
    if label is not None and not isinstance(label, str):
        raise ValueError("label must be a string or None.")

    # Default color mapping based on boundary condition type
    color_map = {
        "RIV": "blue",
        "WEL": "red",
        "GHB": "black",
        "DRN": "gray",
        "CHD": "purple"
    }

    # Validate row index
    nrow, ncol = gwf.modelgrid.nrow, gwf.modelgrid.ncol
    assert 0 <= row < nrow, f"Row index {row} is out of bounds for grid with {nrow} rows."

    # If no axis is provided, create a new figure and axis
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.set_title(title, fontsize=fontsize)
  
    # Compute minimum and maximum head values for color scaling
    vmin, vmax = np.nanmin(array), np.nanmax(array)
    
    # Create the cross-section object
    section = flopy.plot.PlotCrossSection(
        model=gwf,
        ax=ax,
        line={"row": row}
    )
    
    # Plot the array
    if log:
        norm = LogNorm(vmin=vmin, vmax=vmax)
        pa = section.plot_array(array, vmin=vmin, vmax=vmax, cmap=get_cmap("cividis_r"), norm=norm)
    else:    
        pa = section.plot_array(array, vmin=vmin, vmax=vmax, cmap=get_cmap("cividis_r"))

    # Plot the grid lines
    section.plot_grid(lw=0.1, color="0.5")
    
    # Dynamically plot boundary conditions based on keywords
    if boundary_keywords:
        for bc in boundary_keywords:
            # Determine color based on the keyword
            bc_color = None
            for key in color_map:
                if key in bc:  # Check if the keyword contains the key
                    bc_color = color_map[key]
                    break
            # Plot the boundary condition with the appropriate color
            if bc_color:
                section.plot_bc(bc, color=bc_color)

    # Add colorbar
    if colorbar:
        cb = plt.colorbar(pa, ax=ax)
        cb.set_label(label, fontsize=fontsize)

    # Show and save the plot
    plt.ioff()
    if show:
        plt.show()
    
    #Save plot
    if save:
        # Create directory if it does not exist
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        fig.savefig(output_path, dpi=300)
        plt.close(fig)  

def plot_animation(gwf, heads, 
                    qx, qy, qz, nrow, 
                    cs_output_folder,
                    gif_output_path,
                    boundary_keywords=None,
                    show=False, save=False, 
                    flow_dir=True, surface=True, layers=True,
                    figsize=(19, 4), fontsize=14, 
                    gif_start=0, gif_step=1):
    """
    Plot cross-sections for all time steps in the heads array, save images, and create an animation.

    Args:
        gwf (flopy.mf6.ModflowGwf): Groundwater flow model object.
        heads (np.ndarray): 4D numpy array of heads (time, layer, row, column).
        qx, qy, qz (np.ndarray): Flow components.
        nrow (int): Row index for cross-section.
        cs_output_folder (str): Directory to save the cross-section images.
        gif_output_path (str): Path to save the generated animation GIF.
        boundary_keywords (list of str, optional): List of boundary conditions keywords.
        show (bool, optional): Whether to display the plots.
        save (bool, optional): Whether to save the plots to files.
        flow_dir (bool, optional): Whether to plot flow directions.
        surface (bool, optional): Whether to plot the surface.
        layers (bool, optional): Whether to include layer legend.
        figsize (tuple, optional): Figure size for plots.
        fontsize (int, optional): Font size for plot labels.
        gif_start (int, optional): First time step to include in the animation.
        gif_step (int, optional): Step size for time steps in the animation.

    Outputs:
        Saves cross-section images for each time step and creates an animated GIF.
    """
    import os
    import matplotlib.pyplot as plt
    from matplotlib.animation import PillowWriter, FuncAnimation
    import imageio

    # Input checks
    if heads.ndim != 4:
        raise ValueError("heads must be a 4D array (time, layer, row, column).")
    if not isinstance(nrow, int) or nrow < 0:
        raise ValueError("nrow must be a non-negative integer.")
    if not isinstance(cs_output_folder, str) or not cs_output_folder:
        raise ValueError("cs_output_folder must be a non-empty string.")
    if not isinstance(gif_output_path, str) or not gif_output_path:
        raise ValueError("gif_output_path must be a non-empty string.")
    if boundary_keywords is not None and not isinstance(boundary_keywords, list):
        raise ValueError("boundary_keywords must be a list of strings or None.")
    if not isinstance(show, bool):
        raise ValueError("show must be a boolean.")
    if not isinstance(save, bool):
        raise ValueError("save must be a boolean.")
    if not isinstance(flow_dir, bool):
        raise ValueError("flow_dir must be a boolean.")
    if not isinstance(surface, bool):
        raise ValueError("surface must be a boolean.")
    if not isinstance(layers, bool):
        raise ValueError("layers must be a boolean.")
    if not (isinstance(figsize, tuple) and len(figsize) == 2):
        raise ValueError("figsize must be a tuple of length 2.")
    if not isinstance(fontsize, (int, float)):
        raise ValueError("fontsize must be a number.")
    if not isinstance(gif_start, int) or gif_start < 0:
        raise ValueError("gif_start must be a non-negative integer.")
    if not isinstance(gif_step, int) or gif_step < 1:
        raise ValueError("gif_step must be a positive integer.")

    # Ensure the output folder exists
    os.makedirs(cs_output_folder, exist_ok=True)

    num_timesteps = heads.shape[0]  # Number of time steps
    image_paths = []

    for tstep in range(gif_start, num_timesteps, gif_step):
        output_path = os.path.join(cs_output_folder, f"cross_section_heads_{tstep}.png")
        image_paths.append(output_path)
        
        plot_cross_section_row(
            gwf, heads[tstep, :, :, :], qx, qy, qz, nrow,
            output_path,
            boundary_keywords=boundary_keywords,
            flow_dir=flow_dir, surface=surface, layers=layers,
            show=show, save=save, figsize=figsize, fontsize=fontsize,
            title=f"Cross section - time step : {tstep}"
        )
        
        print(f"Saved cross-section plot for time step {tstep} at {output_path}")

    # Create the GIF animation
    with imageio.get_writer(gif_output_path, mode='I', duration=0.5) as writer:
        for image_path in image_paths:
            image = imageio.imread(image_path)
            writer.append_data(image)

    print(f"Animation saved at {gif_output_path}")
    print("All cross-section plots and animation generated and saved.")

def fix_mppth_file(fpth):
    """
    Fixes malformed scientific notation in a file, where 'E' is missing before the exponent.
    For example, changes '0.99292660-100' to '0.99292660E-100'.

    Args:
        fpth (str): Path to the file to be corrected.

    Outputs:
        Overwrites the file with corrected scientific notation.
        Prints a message indicating the file has been corrected.
    """
    import re

    # Input checks
    if not isinstance(fpth, str) or not fpth:
        raise ValueError("fpth must be a non-empty string (file path).")

    # Read the file
    with open(fpth, 'r') as file:
        lines = file.readlines()
    
    # Regex pattern to find numbers with a missing 'E' before + or -
    pattern = re.compile(r'(?<=[0-9])(?=[+-][0-9]{2,})')

    # Fix lines
    fixed_lines = [pattern.sub('E', line) for line in lines]

    # Write corrected lines back to file (overwrite)
    with open(fpth, 'w') as file:
        file.writelines(fixed_lines)

    print(f"File '{fpth}' has been corrected for scientific notation issues.")