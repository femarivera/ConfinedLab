# Functions for plotting Steady State outcomes of a MODFLOW 6 GW Flow Simulation
# Import local modules

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
        modelmap.plot_vector(qx, qy, normalize=True, color="white", headwidth=2, headlength=1, headaxislength=1)

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
        head (numpy.ndarray): Head array for the model (or any other array to plot).
        qx, qy, qz (numpy.ndarray): Flow vectors in x, y, and z directions.
        row (int): Row number for the cross-section.
        output_path (str): File path to save the plot.
        boundary_keywords (list of str): Keywords for boundary condition columns to include.
        flow_dir (bool): Whether to include flow direction vectors.
        surface (bool): Whether to include the surface head plot.
        show (bool): Whether to display the plot.
        save (bool): Whether to save the plot to file.
        ax (matplotlib.axes.Axes): Matplotlib axis to plot on. If None, a new figure is created.
        figsize (tuple): Size of the figure.
        fontsize (int): Font size for plot labels.

    Outputs:
        Displays the cross-section plot and/or saves it to a file.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import flopy
    from matplotlib.cm import get_cmap
    from matplotlib.lines import Line2D
    import os

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
        section.plot_vector(qx, qy, qz, normalize=True, color="white", head=masked_head, hstep=5, 
                            headwidth=2, headlength=1, headaxislength=1, scale=50)

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
    Plots a cross-section for a MODFLOW 6 groundwater flow model along a specified row.

    Args:
        gwf (flopy.mf6.ModflowGwf): Groundwater flow model object.
        head (numpy.ndarray): Head array for the model (or any other array to plot).
        qx, qy, qz (numpy.ndarray): Flow vectors in x, y, and z directions.
        row (int): Row number for the cross-section.
        output_path (str): File path to save the plot.
        boundary_keywords (list of str): Keywords for boundary condition columns to include.
        flow_dir (bool): Whether to include flow direction vectors.
        surface (bool): Whether to include the surface head plot.
        show (bool): Whether to display the plot.
        save (bool): Whether to save the plot to file.
        ax (matplotlib.axes.Axes): Matplotlib axis to plot on. If None, a new figure is created.
        figsize (tuple): Size of the figure.
        fontsize (int): Font size for plot labels.

    Outputs:
        Displays the cross-section plot and/or saves it to a file.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import flopy
    from matplotlib.cm import get_cmap
    from matplotlib.lines import Line2D
    import os

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
        section.plot_vector(qx, qy, qz, normalize=True, color="white", head=masked_head, hstep=5, 
                            headwidth=2, headlength=1, headaxislength=1, scale=50)

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
                        figsize=(19, 5), 
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
                             figsize=(19, 5),
                             fontsize=14, 
                             title="Cross section",
                             colorbar = False,
                             label="Legend"):
    """
    Plots a cross-section for a MODFLOW 6 groundwater flow model along a specified row.

    Args:
        gwf (flopy.mf6.ModflowGwf): Groundwater flow model object.
        array (numpy.ndarray): Any array to plot.
        qx, qy, qz (numpy.ndarray): Flow vectors in x, y, and z directions.
        row (int): Row number for the cross-section.
        boundary_keywords (list of str): Keywords for boundary condition columns to include.
     
    Outputs:
        Displays the cross-section plot.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import flopy
    from matplotlib.cm import get_cmap
    from matplotlib.lines import Line2D
    import os

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

def plot_cross_sections_animation_transient(gwf, heads, qx, qy, qz, nrow, cs_output_folder,
                                      gif_output_path,
                                      boundary_keywords=None,
                                      show=False, save=False, 
                                      flow_dir=True, surface=True, layers=True,
                                      figsize=(19, 4), fontsize=14, gif_start=0, gif_step=1):
    """
    Plot cross-sections for all time steps in the heads array, save images, and create an animation.
    
    Parameters:
    - gwf: Groundwater model object.
    - heads: 4D numpy array of heads (time, layer, row, column).
    - qx, qy, qz: Flow components.
    - nrow: Row index for cross-section.
    - output_folder: Directory to save the cross-section images.
    - gif_output_path: Path to save the generated animation GIF.
    - boundary_keywords: List of boundary conditions keywords.
    - flow_dir: Whether to plot flow directions.
    - surface: Whether to plot the surface.
    - figsize: Figure size for plots.
    """
    import os
    import matplotlib.pyplot as plt
    from matplotlib.animation import PillowWriter, FuncAnimation
    import imageio

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
    
    Parameters:
        fpth (str): Path to the file to be corrected.
    """
    import re
    import os

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