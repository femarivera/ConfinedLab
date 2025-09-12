# ==========================================================================================
#  modgeom6.py - Modular Utilities for Synthetic Geometry Generation of Multilayer Systems
# ==========================================================================================
#
#  Author: MARIN RIVERA Carlos Felipe
#  Organization: Bordeaux INP, Lab EPOC, Université de Bordeaux
#  Project: Funded by the OneWater PEPR DEESAC Project 
#
#  DESCRIPTION:
#  ------------
#  As part of the ConfinedLab project, this module provides robust, flexible, and well-documented utilities 
#  for generating and manipulating synthetic multilayer groundwater model geometries using structured grids.
#  The approach generates a 3D rectangular grid with defined number of layers, rows, and columns. The geometry
#  to be generated will always correspond to a multiayer system with a dip direction along the column axis. 
#
#  MAIN FEATURES:
#  --------------
#  - Create idomain arrays for left- or right-dipping systems with customizable outcrop and cofined areas.
#  - Compute top elevation arrays with options for smooth linear transitions and sloping topography.
#  - Compute thickness arrays with or without transition zones.
#  - Calculate bottom elevations, irch arrays, recharge arrays, and more.
#  - All functions include input validation for clarity and robustness.
#  
#  DEPENDENCIES:
#  -------------
#  - numpy
# ==========================================================================================
#  MAIN FUNCTIONS - RECOMMENDED WORKFLOW FOR SYNTHETIC MULTILAYER GEOMETRY GENERATION
# ==========================================================================================
#  Default parameters on the main functions are set to generate geometries for systems with defined outcropping 
#  and confined areas. This usually generates a simple yet realistic geometry of a multilayer system representing
#  the typical stratigraphic configuration of a sedimentary basin without faulting or folding. 


def compute_idomain(nlay, nrow, ncol, outcrop_cells, direction = "right"):
    """
    Create an idomain array for a synthetic multilayer system dipping to the left or right.

    Parameters:
        nlay (int): Number of layers.
        nrow (int): Number of rows.
        ncol (int): Number of columns.
        outcrop_cells (1D array): 1D array of length (nlay), with the column indices (int) representing
            the threshold for each layer. For 'left', each layer is active for columns <= outcrop_cells[i].
            For 'right', each layer is active for columns >= outcrop_cells[i].
        direction (str): "right" (default) for right-dipping (confined to the right side), 
                         "left" for left-dipping (confined to the left side).

    Returns:
        idomain (numpy.ndarray): 3D array (nlay, nrow, ncol) with 1 for active and 0 for inactive cells.
    """

    import numpy as np

    #input checks
    outcrop_cells = np.asarray(outcrop_cells)
    if len(outcrop_cells) != nlay:
        raise ValueError("outcrop_cells must have length equal to nlay.")
    if direction not in ("left", "right"):
        raise ValueError("direction must be 'left' or 'right'.")
    if np.any(outcrop_cells < 0) or np.any(outcrop_cells > ncol):
        raise ValueError("All outcrop_cells values must be in the range [0, ncol].")

    if direction == "left":
        if not np.all(np.diff(outcrop_cells) > 0):
            raise ValueError("For 'left', outcrop_cells must be strictly ascending.")
    else:  # direction == "right"
        if not np.all(np.diff(outcrop_cells) < 0):
            raise ValueError("For 'right', outcrop_cells must be strictly descending.")

    # Initialize idomain array with ones (active cells)
    idomain = np.ones((nlay, nrow, ncol), dtype=int)
    # Apply the condition for each layer using the corresponding outcrop length
    for layer in range(nlay - 1):
        L = int(outcrop_cells[layer])
        if direction == "left":
            idomain[layer, :, L:] = 0  
        else:  # direction == "right"
            idomain[layer, :, :L] = 0 

    # Last layer remains fully active
    return idomain

def compute_top(
    idomain,
    outcrop_z,
    transition=True,
    slope=True,
    direction="right",
    transition_cells=None,
    transition_type="contain",
    outcrop_zmin=None,
    outcrop_zmax=None
):
    """
    Generalized function to compute top elevations for multilayer systems with options for:
    - left/right dipping systems,
    - smooth transitions between outcrops,
    - contained or extended transitions,
    - sloping or flat outcrop elevations.

    Parameters:
        idomain (numpy.ndarray): 3D array (nlay, nrow, ncol) indicating active (1) and inactive (0) cells.
        outcrop_z (1D array-like): Array of top elevations for each layer (length nlay).
        transition (bool): If True, add transition zones between outcrops. If False, use simple top assignment.
        slope (bool): If True (default), use sloping topography (requires outcrop_zmin and outcrop_zmax). If False, use flat outcrop_z.
        direction (str): "left" or "right" (default "right"). Direction of system dip/outcrop.
        transition_cells (int or None): Number of columns for the transition zone between layers. Required if transition=True.
        transition_type (str): "contain" (default) or "extend". "contain" keeps transitions within idomain, "extend" allows transitions to extend beyond.
        outcrop_zmin (1D array-like or None): Minimum elevation for each layer (required if slope=True).
        outcrop_zmax (1D array-like or None): Maximum elevation for each layer (required if slope=True).

    Returns:
        top (numpy.ndarray): 2D array (nrow, ncol) of top elevations.
    """
    import numpy as np

    # Input checks
    if idomain.ndim != 3:
        raise ValueError("idomain must be a 3D array (nlay, nrow, ncol).")
    nlay, nrow, ncol = idomain.shape
    outcrop_z = np.asarray(outcrop_z)
    if outcrop_z.shape[0] != nlay:
        raise ValueError("outcrop_z must have length equal to nlay.")
    if direction not in ("left", "right"):
        raise ValueError("direction must be 'left' or 'right'.")

    if not transition:
        # No transition zone: ignore slope and related parameters
        top = np.zeros((nrow, ncol), dtype=float)
        active_layers = np.sum(idomain, axis=0)
        irch = nlay - active_layers
        for layer_id in range(nlay):
            top[irch == layer_id] = outcrop_z[layer_id]
        return top

    # If transition is True, check transition parameters
    if transition_cells is None or not isinstance(transition_cells, int) or transition_cells < 1:
        raise ValueError("transition_cells must be a positive integer when transition=True.")
    if transition_type not in ("contain", "extend"):
        raise ValueError("transition_type must be 'contain' or 'extend'.")

    # If slope is True, check slope parameters
    if slope:
        if outcrop_zmin is None or outcrop_zmax is None:
            raise ValueError("outcrop_zmin and outcrop_zmax must be provided when slope=True.")
        outcrop_zmin = np.asarray(outcrop_zmin)
        outcrop_zmax = np.asarray(outcrop_zmax)
        if outcrop_zmin.shape[0] != nlay or outcrop_zmax.shape[0] != nlay:
            raise ValueError("outcrop_zmin and outcrop_zmax must have length equal to nlay.")

    # Compute topmost active layer index (irch)
    active_layers = np.sum(idomain, axis=0)  # (nrow, ncol)
    irch = nlay - active_layers  # Topmost active layer index per cell
    top = np.zeros((nrow, ncol), dtype=float)

    # Step 1: Assign base top elevations
    if slope:
        for layer_id in range(nlay):
            for row in range(nrow):
                mask = (irch[row, :] == layer_id)
                n_cells = np.sum(mask)
                if n_cells > 0:
                    if direction == "right":
                        slope_vals = np.linspace(outcrop_zmax[layer_id], outcrop_zmin[layer_id], n_cells)
                    else:
                        slope_vals = np.linspace(outcrop_zmin[layer_id], outcrop_zmax[layer_id], n_cells)
                    top[row, mask] = slope_vals
    else:
        for layer_id in range(nlay):
            top[irch == layer_id] = outcrop_z[layer_id]

    # Step 2: Add transitions
    for layer_id in range(nlay):
        if direction == "right":
            # For right-dipping, transitions are to the left (lower column indices)
            transition_mask = (irch == layer_id) & (np.roll(irch, 1, axis=-1) == layer_id + 1)
        else:
            # For left-dipping, transitions are to the right (higher column indices)
            transition_mask = (irch == layer_id) & (np.roll(irch, -1, axis=-1) == layer_id + 1)

        for row in range(nrow):
            transition_indices = np.where(transition_mask[row, :])[0]
            for idx in transition_indices:
                if direction == "right":
                    if transition_type == "extend":
                        start = max(0, idx - transition_cells + 1)
                        end = idx + 1
                    else:  # "contain"
                        start = idx
                        end = min(ncol-1, idx + transition_cells)
                    n = end - start
                    if n > 1:
                        if slope:
                            top[row, start:end] = np.linspace(
                                outcrop_zmin[layer_id + 1], top[row, end], n
                            )
                        else:
                            top[row, start:end] = np.linspace(
                                outcrop_z[layer_id + 1], outcrop_z[layer_id], n
                            )
                else:
                    # direction == "left"
                    if transition_type == "extend":
                        start = idx
                        end = min(ncol-1, idx + transition_cells)
                    else:  # "contain"
                        start = max(0, idx - transition_cells + 1)
                        end = idx + 1
                    n = end - start
                    if n > 1:
                        if slope:
                            top[row, start:end] = np.linspace(
                                outcrop_zmax[layer_id], top[row, end], n
                            )
                        else:
                            top[row, start:end] = np.linspace(
                                outcrop_z[layer_id], outcrop_z[layer_id + 1], n
                            )

    return top

def compute_thickness(
    idomain,
    base_thicknesses,
    transition=True,
    transition_cells=None,
    transition_type="contain"
):
    """
    Generalized function to compute thickness arrays for multilayer systems with options for:
    - simple thickness assignment (no transition),
    - smooth transitions between active/inactive zones,
    - contained or extended transitions.

    Parameters:
        idomain (numpy.ndarray): 3D array (nlay, nrow, ncol) indicating active (1) and inactive (0) cells.
        base_thicknesses (1D array-like): Array of length nlay, with the base thickness for each model layer.
        transition (bool): If True, add transition zones between active/inactive areas. If False, use simple assignment.
        transition_cells (int or None): Number of columns for the transition zone. Required if transition=True.
        transition_type (str): "contain" (default) or "extend". "contain" keeps transitions within idomain, "extend" allows transitions to extend beyond.

    Returns:
        thickness_array (numpy.ndarray): 3D array (nlay, nrow, ncol) with thicknesses.
    """
    import numpy as np

    # Input checks
    if idomain.ndim != 3:
        raise ValueError("idomain must be a 3D array (nlay, nrow, ncol).")
    nlay, nrow, ncol = idomain.shape
    base_thicknesses = np.asarray(base_thicknesses)
    if base_thicknesses.shape[0] != nlay:
        raise ValueError("base_thicknesses must have length equal to nlay.")

    if not transition:
        # Simple assignment
        thickness_array = np.zeros_like(idomain, dtype=float)
        for layer in range(nlay):
            thickness_array[layer, :, :] = np.where(idomain[layer, :, :] == 1, base_thicknesses[layer], 0)

        return thickness_array

    # If transition is True, check transition parameters
    if transition_cells is None or not isinstance(transition_cells, int) or transition_cells < 1:
        raise ValueError("transition_cells must be a positive integer when transition=True.")
    if transition_type not in ("contain", "extend"):
        raise ValueError("transition_type must be 'contain' or 'extend'.")

    if transition_type == "contain":
        # Contained transition (within idomain)
        # Initialize the thickness array
        nlay, nrow, ncol = idomain.shape
        thickness_array = np.zeros_like(idomain, dtype=float)

        # Loop through layers and apply base thickness and smooth transition
        for layer in range(nlay):
            # Get base thickness for the current layer
            base_thickness = base_thicknesses[layer]
            
            # Set thickness for active cells (idomain == 1) to the base thickness
            thickness_array[layer, :, :] = np.where(idomain[layer, :, :] == 1, base_thickness, 0)
            
            # Now we smooth the transition from base thickness to zero within the same layer
            for row in range(nrow):
                for col in range(ncol):
                    if idomain[layer, row, col] == 0:  # If the cell is inactive
                        # Check if there are adjacent active cells to create a smooth transition
                        if col > 0 and idomain[layer, row, col - 1] == 1:  # Transition from left
                            transition_range = np.linspace(0, base_thickness, transition_cells)
                            # Apply transition over the next `transition_cells` columns
                            for t in range(min(transition_cells, ncol - col)):
                                thickness_array[layer, row, col - t] = transition_range[t]
                        elif col < ncol - 1 and idomain[layer, row, col + 1] == 1:  # Transition from right
                            transition_range = np.linspace(0, base_thickness, transition_cells)
                            # Apply transition over the previous `transition_cells` columns
                            for t in range(min(transition_cells, col + 1)):
                                thickness_array[layer, row, col + t] = transition_range[t]
        
        return thickness_array
    else:
        # Extended transition (beyond idomain)
        # Initialize the thickness array
        nlay, nrow, ncol = idomain.shape
        thickness_array = np.zeros_like(idomain, dtype=float)

        # Loop through layers and apply base thickness and smooth transition
        for layer in range(nlay):
            # Get base thickness for the current layer
            base_thickness = base_thicknesses[layer]
            
            # Set thickness for active cells (idomain == 1) to the base thickness
            thickness_array[layer, :, :] = np.where(idomain[layer, :, :] == 1, base_thickness, 0)
            
            # Now we smooth the transition from base thickness to zero within the same layer
            for row in range(nrow):
                for col in range(ncol):
                    if idomain[layer, row, col] == 0:  # If the cell is inactive
                        # Check if there are adjacent active cells to create a smooth transition
                        if col > 0 and idomain[layer, row, col - 1] == 1:  # Transition from left
                            transition_range = np.linspace(base_thickness, 0, transition_cells)
                            # Apply transition over the next `transition_cells` columns
                            for t in range(min(transition_cells, ncol - col)):
                                thickness_array[layer, row, col + t] = transition_range[t]
                        elif col < ncol - 1 and idomain[layer, row, col + 1] == 1:  # Transition from right
                            transition_range = np.linspace(base_thickness, 0, transition_cells)
                            # Apply transition over the previous `transition_cells` columns
                            for t in range(min(transition_cells, col + 1)):
                                thickness_array[layer, row, col - t] = transition_range[t]
        
        return thickness_array

def compute_bottom(ztop, thickness_array):
    """
    Compute the bottom elevations for each layer based on ztop and thickness_array.

    Parameters:
        ztop (ndarray): 2D array of shape (nrow, ncol), representing the top elevation of the model.
        thickness_array (ndarray): 3D array of shape (nlay, nrow, ncol), with the thickness for each layer.

    Returns:
        bottom (ndarray): 3D array of shape (nlay, nrow, ncol) representing the bottom elevations for each layer.
    """
    import numpy as np

    # Input checks
    if thickness_array.ndim != 3:
        raise ValueError("thickness_array must be a 3D array (nlay, nrow, ncol).")
    if ztop.ndim != 2:
        raise ValueError("ztop must be a 2D array (nrow, ncol).")
    nlay, nrow, ncol = thickness_array.shape
    if ztop.shape != (nrow, ncol):
        raise ValueError("ztop shape must match (nrow, ncol) of thickness_array.")

    # Initialize the bottom array
    bottom = np.zeros_like(thickness_array, dtype=float)

    # Compute the bottom elevations for each layer
    for layer in range(nlay):
        if layer == 0:
            # For the first layer, bottom elevation is simply ztop - thickness
            bottom[layer, :, :] = ztop - thickness_array[layer, :, :]
        else:
            # For subsequent layers, subtract the thickness of the current layer from the previous layer's bottom
            bottom[layer, :, :] = bottom[layer - 1, :, :] - thickness_array[layer, :, :]
    
    return bottom

def idomain_from_thickness(thickness_array, epsilon):
    """
    Create an idomain array from the thickness array.

    Parameters:
        thickness_array (numpy.ndarray): 3D array (nlay, nrow, ncol) containing thickness values for each layer.
        epsilon (float): Thickness threshold under which cells are deactivated.

    Returns:
        idomain (numpy.ndarray): 3D array (nlay, nrow, ncol) with 1 (active) where thickness > epsilon and 0 (inactive) otherwise.
    """
    import numpy as np

    # Input checks
    if thickness_array.ndim != 3:
        raise ValueError("thickness_array must be a 3D array (nlay, nrow, ncol).")
    if not isinstance(epsilon, (float, int)) or epsilon < 0:
        raise ValueError("epsilon must be a non-negative number.")

    # Set idomain to 1 (active) where thickness > epsilon, else 0 (inactive)
    idomain = np.where(thickness_array > epsilon, 1, 0)
    
    return idomain

def compute_irch(idomain):
    """
    Calculate the topmost active layer index (irch) for each cell.

    Parameters:
        idomain (numpy.ndarray): 3D array (nlay, nrow, ncol) with 1 for active and 0 for inactive cells.

    Returns:
        irch (numpy.ndarray): 2D array (nrow, ncol) where each value is the index of the topmost active layer.
    """

    import numpy as np

    # Input checks
    if idomain.ndim != 3:
        raise ValueError("idomain must be a 3D array (nlay, nrow, ncol).")
    
    # Calculate the number of layers
    nlay = idomain.shape[0]
    
    # Sum idomain across the layers
    active_layers = np.sum(idomain, axis=0)
    
    # Calculate irch
    irch = nlay - active_layers
    
    return irch

def compute_recharge(irch, R):
    """
    Compute the recharge (surface recharge) array based on irch and layer-specific recharge rates.

    Parameters:
        irch (numpy.ndarray): 2D array of shape (nrow, ncol), containing layer indices (0 to nlay-1).
        R (numpy.ndarray or list): 1D array of shape (nlay), containing recharge rates for each layer.

    Returns:
        numpy.ndarray: 2D array of shape (nrow, ncol) with recharge values assigned based on irch.
    """
    import numpy as np  

    # Input checks
    if irch.ndim != 2:
        raise ValueError("irch must be a 2D array (nrow, ncol).")
    R = np.asarray(R)
    nlay = np.max(irch) + 1
    if R.shape[0] != nlay:
        raise ValueError("R must have length equal to the number of layers in irch (max(irch)+1).")
    if np.any((irch < 0) | (irch >= nlay)):
        raise ValueError("All values in irch must be valid layer indices (0 to nlay-1).")

    # Initialize a recharge array with the same shape as irch
    rch = np.zeros_like(irch, dtype=float)

    # Assign recharge rates based on the layer index in irch
    for layer_idx, recharge_rate in enumerate(R):
        rch[irch == layer_idx] = recharge_rate

    return rch

def compute_ztop_array(ztop, zbot):
    """
    Create a 3D array of top elevations for each layer, where the first layer uses ztop and
    subsequent layers use the bottom elevation of the layer above. Usually used for starting conditions
    in certain groundwater models.

    Parameters:
        ztop (numpy.ndarray): 2D array (nrow, ncol), top elevation of the model.
        zbot (numpy.ndarray): 3D array (nlay, nrow, ncol), bottom elevations for each layer.

    Returns:
        ztop_array (numpy.ndarray): 3D array (nlay, nrow, ncol) of top elevations for each layer.
    """
    import numpy as np

    # Input checks
    if ztop.ndim != 2:
        raise ValueError("ztop must be a 2D array (nrow, ncol).")
    if zbot.ndim != 3:
        raise ValueError("zbot must be a 3D array (nlay, nrow, ncol).")
    nlay, nrow, ncol = zbot.shape
    if ztop.shape != (nrow, ncol):
        raise ValueError("ztop shape must match (nrow, ncol) of zbot.")

    # Initialize the ztop_array
    ztop_array = np.zeros((nlay, nrow, ncol))  # Initialize the start array
    
    # Assign ztop to the first layer
    ztop_array[0, :, :] = ztop
    
    # Assign each subsequent layer from zbot
    for i in range(1, nlay):
        ztop_array[i, :, :] = zbot[i - 1, :, :]
    
    return ztop_array

def compute_3Darray(values_1d, idomain):
    """
    Expands a 1D array of layer values to a 3D array, assigning each value to active cells in the corresponding layer.

    Args:
        values_1d (np.ndarray): 1D array of length nlay, with values for each layer.
        idomain (np.ndarray): 3D array of shape (nlay, nrow, ncol), with 1 for active and 0 for inactive cells.

    Returns:
        np.ndarray: 3D array of shape (nlay, nrow, ncol), with each active cell in layer i assigned values_1d[i], and np.nan elsewhere.

    Raises:
        ValueError: If input shapes are inconsistent or invalid.
    """
    import numpy as np

    # Input checks
    if not isinstance(values_1d, np.ndarray):
        raise ValueError("values_1d must be a numpy array.")
    if not isinstance(idomain, np.ndarray):
        raise ValueError("idomain must be a numpy array.")
    if idomain.ndim != 3:
        raise ValueError("idomain must be a 3D array (nlay, nrow, ncol).")
    if values_1d.ndim != 1:
        raise ValueError("values_1d must be a 1D array.")
    if values_1d.shape[0] != idomain.shape[0]:
        raise ValueError("Length of values_1d must match number of layers in idomain.")

    nlay, nrow, ncol = idomain.shape
    arr3d = np.full((nlay, nrow, ncol), np.nan, dtype=float)
    for ilay in range(nlay):
        arr3d[ilay][idomain[ilay] == 1] = values_1d[ilay]
    return arr3d

# ==========================================================================================
# ==========================================================================================
#  VARIANT FUNCTIONS - ADVANCED & EXPERIMENTAL SYNTHETIC GEOMETRY GENERATION
# ==========================================================================================
# The following functions provide additional flexibility. They allow you to
# experiment with left- or right-dipping systems, different transitions, and custom
# slope/top/thickness logic. Use these to explore a wider range of conceptual models.
# ==========================================================================================

#  NAMING CONVENTION:
#  ------------------
#  - Functions with 'left' or 'right' refer to the dip direction of the system.
#  - Functions with 'extend' allow transitions refer to different approaches to generate transitions between units.
#  - 'slope' functions use sloping tops; others use flat tops.
#
#  Combinig contained transitions for top generation with extended transitions for thickness generation, using transition cells that 
#  are larger than the outcropping area would lead to a more complex geometry.

# ==========================================================================================
#  IDOMAIN VARIANT FUNCTIONS - ADVANCED & EXPERIMENTAL SYNTHETIC GEOMETRY GENERATION
# ==========================================================================================

def compute_idomain_left(nlay, nrow, ncol, outcrop_cells):
    """
    Create an idomain array for a synthetic multilayer system dipping to the left (confined to the left).
    
    Parameters:
        nlay (int): Number of layers.
        nrow (int): Number of rows.
        ncol (int): Number of columns.
        outcrop_cells (1D array): 1D array of length (nlay), with the column indices (int) representing
            the threshold from where the layer becomes inactive. Each layer (i) is active for columns
            less or equal than outcrop_cell[i], and inactive after that.
            The last layer remains fully active.
            Outcrop_cells should be sorted in ascending order: Outcrop_cells[i] < outcrop_cells[i+1].
            Outcropping area in model grid for layer i corresponds to [outcrop_cells[i-1], outcrop_cells[i]] 
            and [0, outcrop_cells[0]] for the first layer.

    Returns:
        Idomain (numpy.ndarray): 3D array (nlay, nrow, ncol) with 1 for active and 0 for inactive cells.
    """
    import numpy as np

    # Input checks
    outcrop_cells = np.asarray(outcrop_cells)
    if len(outcrop_cells) != nlay:
        raise ValueError("outcrop_cells must have length equal to nlay.")
    if not np.all(np.diff(outcrop_cells) > 0):
        raise ValueError("outcrop_cells must be strictly ascending (each value less than the next).")
    if np.any(outcrop_cells < 0) or np.any(outcrop_cells > ncol):
        raise ValueError("All outcrop_cells values must be in the range [0, ncol].")

     # Initialize idomain array with ones (active cells)
    idomain = np.ones((nlay, nrow, ncol), dtype=int)
    # Apply the condition for each layer using the corresponding outcrop length
    for layer in range(nlay - 1):
        L = int(outcrop_cells[layer])
        idomain[layer, :, L:] = 0  # Set cells beyond the threshold for the current layer to 0

    # Last layer remains fully active (already set to 1)
    return idomain

def compute_idomain_right(nlay, nrow, ncol, outcrop_cells):
    """
    Create an idomain array for a synthetic multilayer system dipping to the right (outcropping to the left).
    
    Parameters:
        nlay (int): Number of layers.
        nrow (int): Number of rows.
        ncol (int): Number of columns.
        outcrop_cells (1D array): 1D array of length (nlay), with the column indices (int) representing
                the threshold from where the layer becomes active. Each layer is active (1) for columns 
                greater than or equal to the outcrop cell, and inactive before that.
                The last layer remains fully active.
                Outcrop_cells should be sorted in descending order: outcrop_cells[i] > outcrop_cells[i+1].
                Outcropping areas in model grid for layer i correspond to [outcrop_cells[i], outcrop_cells[i-1]] 
                and [outcrop_cells[0], ncol-1] for the first layer.
                

    Returns:
        Idomain (numpy.ndarray): 3D array (nlay, nrow, ncol) with 1 for active and 0 for inactive cells.
    """

    import numpy as np

    # Input checks
    outcrop_cells = np.asarray(outcrop_cells)
    if len(outcrop_cells) != nlay:
        raise ValueError("outcrop_cells must have length equal to nlay.")
    if not np.all(np.diff(outcrop_cells) < 0):
        raise ValueError("outcrop_cells must be strictly descending (each value greater than the next).")
    if np.any(outcrop_cells < 0) or np.any(outcrop_cells > ncol):
        raise ValueError("All outcrop_cells values must be in the range [0, ncol].")

    # Initialize idomain array with ones (active cells)
    idomain = np.ones((nlay, nrow, ncol), dtype=int)
    # Apply the condition for each layer using the corresponding outcrop length
    for layer in range(nlay - 1):  # Loop through all layers except the last one
        L = int(outcrop_cells[layer])
        idomain[layer, :, :L] = 0  # Set cells untill the threshold for the current layer to 0 (inactive)
    
    # Last layer remains fully active (already set to 1)
    return idomain

# ==========================================================================================
# TOP VARIANT FUNCTIONS - ADVANCED & EXPERIMENTAL SYNTHETIC GEOMETRY GENERATION
# ==========================================================================================
def compute_top_simple(idomain, outcrop_z):
    """
    Computes a 2D array of top elevations based on the uppermost active layer at each cell and the outcrop elevations
    defined per layer.

    Parameters:
        idomain (numpy.ndarray): 3D array (nlay, nrow, ncol) indicating active (1) and inactive (0) cells.
        outcrop_z (1D array-like): Array of length nlay, with outcropping top elevations for each layer.

    Returns:
        top (numpy.ndarray): 2D array (nrow, ncol) of top elevations.
    """
    import numpy as np

    # Input checks
    if idomain.ndim != 3:
        raise ValueError("idomain must be a 3D array (nlay, nrow, ncol).")
    nlay, nrow, ncol = idomain.shape    
    outcrop_z = np.asarray(outcrop_z)
    if outcrop_z.shape[0] != nlay:
        raise ValueError("outcrop_z must have length equal to nlay.")
    
    # Initialize top array
    top = np.zeros((nrow, ncol), dtype=float)
    
    # Calculate the number of active layers at each (row, col)
    active_layers = np.sum(idomain, axis=0)  # Shape will be (nrow, ncol)
    
    # The topmost active layer index at each cell
    irch = nlay - active_layers
    
    # Assign the correct top elevation for each layer
    for layer_id in range(nlay):
        top[irch == layer_id] = outcrop_z[layer_id]

    return top

def compute_slope_left_extend(idomain, outcrop_z_min, outcrop_z_max, transition_cells):
    """
    Compute top elevations with smooth transitions between outcropping zones adding linear slopes for each layer,
    for left-dipping systems. Transitions are extended beyond the active zones defined in `idomain`
    by the extent specified by the `transition_cells` parameter (number of columns).
    
    Note:
    This version extends the top elevation transition *beyond* the active zones defined in the input idomain
    by the extent specified by the `transition_cells` parameter. This can be useful for certain visualization
    or conceptual model setups, but it does not confine transitions strictly within each layer's original idomain.

    Parameters:
        idomain (numpy.ndarray): 3D array (nlay, nrow, ncol) indicating active (1) and inactive (0) cells.
        outcrop_z_min (1D array-like): Minimum elevation for each layer (length nlay).
        outcrop_z_max (1D array-like): Maximum elevation for each layer (length nlay).
        transition_cells (int): Number of columns for the transition zone between layers.

    Returns:
        top (numpy.ndarray): 2D array (nrow, ncol) of top elevations with extended transitions and slopes.
    """
    import numpy as np

    # Input checks
    if idomain.ndim != 3:
        raise ValueError("idomain must be a 3D array (nlay, nrow, ncol).")
    nlay, nrow, ncol = idomain.shape
    outcrop_z_min = np.asarray(outcrop_z_min)
    outcrop_z_max = np.asarray(outcrop_z_max)
    if outcrop_z_min.shape[0] != nlay or outcrop_z_max.shape[0] != nlay:
        raise ValueError("outcrop_z_min and outcrop_z_max must have length equal to nlay.")
    if not isinstance(transition_cells, int) or transition_cells < 1:
        raise ValueError("transition_cells must be a positive integer.")
    
    # Initialize top array
    top = np.zeros((nrow, ncol), dtype=float)

    # Compute the topmost active layer index at each cell
    active_layers = np.sum(idomain, axis=0)  # Shape will be (nrow, ncol)
    irch = nlay - active_layers  # Values will range from 0 to nlay-1

    # Step 1: Assign a sloping top elevation for each active layer
    for layer_id in range(nlay):
        for row in range(nrow):
            slope = np.linspace(outcrop_z_min[layer_id], 
                                outcrop_z_max[layer_id], 
                                np.sum(irch[row, :] == layer_id))
            top[row, irch[row, :] == layer_id] = slope

    # Step 2: Add smooth transition between zones that extebds beyond layer_id's domain
    for layer_id in range(nlay - 1):
        # Find cells at the boundary between adjacent layers
        transition_mask = (irch == layer_id) & (np.roll(irch, -1, axis=-1) == layer_id + 1)
        
        for row in range(nrow):
            transition_indices = np.where(transition_mask[row, :])[0]
            for idx in transition_indices:
                # Apply transition starting at boundary and extending forward
                start = idx
                end = min(ncol-1, idx + transition_cells)
                n = end - start
                if n > 1:
                    transition_range = np.linspace(outcrop_z_max[layer_id], top[row, end], n)
                    top[row, start:end] = transition_range
  
    return top

def compute_slope_left_contain(idomain, outcrop_z_min, outcrop_z_max, transition_cells):
    """
    Compute top elevations with smooth transitions between outcropping zones, adding linear slopes for each layer,
    for left-dipping systems. The transition zone is added strictly within the active area defined by `idomain`
    for each layer (it does not extend or modify the idomain).
    Note:
    This version ensures that the transition happens *within* the zone defined as active in `idomain`
    for each layer. The transition is performed from the end of the current layers domain up to the
    start of the next, without spilling into the next layer's active area.

    Parameters:
        idomain (numpy.ndarray): 3D array (nlay, nrow, ncol) indicating active (1) and inactive (0) cells.
        outcrop_z_min (1D array-like): Minimum elevation for each layer (length nlay).
        outcrop_z_max (1D array-like): Maximum elevation for each layer (length nlay).
        transition_cells (int): Number of columns for the transition zone between layers.

    Returns:
        top (numpy.ndarray): 2D array (nrow, ncol) of top elevations with contained transitions and slopes.
    """

    import numpy as np

    # Input checks
    if idomain.ndim != 3:
        raise ValueError("idomain must be a 3D array (nlay, nrow, ncol).")
    nlay, nrow, ncol = idomain.shape
    outcrop_z_min = np.asarray(outcrop_z_min)
    outcrop_z_max = np.asarray(outcrop_z_max)
    if outcrop_z_min.shape[0] != nlay or outcrop_z_max.shape[0] != nlay:
        raise ValueError("outcrop_z_min and outcrop_z_max must have length equal to nlay.")
    if not isinstance(transition_cells, int) or transition_cells < 1:
        raise ValueError("transition_cells must be a positive integer.")
    
    # Initialize top array
    top = np.zeros((nrow, ncol), dtype=float)

    # Compute topmost active layer index at each surface cell
    active_layers = np.sum(idomain, axis=0)  # Shape will be (nrow, ncol)
    irch = nlay - active_layers  # Values will range from 0 to nlay-1

    # Step 1: Assign a sloping top elevation for each active layer
    for layer_id in range(nlay):
        for row in range(nrow):
            slope = np.linspace(outcrop_z_min[layer_id], 
                                outcrop_z_max[layer_id], 
                                np.sum(irch[row, :] == layer_id))
            top[row, irch[row, :] == layer_id] = slope


    # Step 2: Add smooth transitions, but staying within the active zone of each layer
    for layer_id in range(nlay - 1):
        # Find cells at the boundary between adjacent layers
        transition_mask = (irch == layer_id) & (np.roll(irch, -1, axis=-1) == layer_id + 1)
        
        for row in range(nrow):
            transition_indices = np.where(transition_mask[row, :])[0]
            for idx in transition_indices:
                start = max(0, idx - transition_cells + 1)
                end = idx + 1
                n = end - start
                if n > 1:
                    transition_range = np.linspace(
                        outcrop_z_max[layer_id], top[row, end], n)
                    top[row, start:end] = transition_range
  
    return top

def compute_slope_right_extend(idomain, outcrop_z_min, outcrop_z_max, transition_cells):
    """
    Compute top elevations with smooth transitions between recharge zones, adding linear slopes for each layer,
    for right-dipping systems. The transition zone is extended beyond the active zones defined in `idomain`
    by the extent specified by the `transition_cells` parameter (number of columns).
    Note:
    This version extends the top elevation transition *beyond* the active zones defined in `idomain`
    by the extent specified by the `transition` parameter. This can be useful for certain visualization
    or conceptual model setups, but it does not confine transitions strictly within each layer's input idomain.

    Parameters:
        idomain (numpy.ndarray): 3D array (nlay, nrow, ncol) indicating active (1) and inactive (0) cells.
        outcrop_z_min (1D array-like): Minimum elevation for each layer (length nlay).
        outcrop_z_max (1D array-like): Maximum elevation for each layer (length nlay).
        transition_cells (int): Number of columns for the transition zone between layers.

    Returns:
        top (numpy.ndarray): 2D array (nrow, ncol) of top elevations with extended transitions and slopes.
    """

    import numpy as np

    # Input checks
    if idomain.ndim != 3:
        raise ValueError("idomain must be a 3D array (nlay, nrow, ncol).")
    nlay, nrow, ncol = idomain.shape
    outcrop_z_min = np.asarray(outcrop_z_min)
    outcrop_z_max = np.asarray(outcrop_z_max)
    if outcrop_z_min.shape[0] != nlay or outcrop_z_max.shape[0] != nlay:
        raise ValueError("outcrop_z_min and outcrop_z_max must have length equal to nlay.")
    if not isinstance(transition_cells, int) or transition_cells < 1:
        raise ValueError("transition_cells must be a positive integer.")

    # Initialize top array 
    top = np.zeros((nrow, ncol), dtype=float)

    # Compute topmost active layer index at each surface cell
    active_layers = np.sum(idomain, axis=0)  # Shape will be (nrow, ncol)
    irch = nlay - active_layers  # Values will range from 0 to nlay-1

    # Step 1: Assign a sloping top elevation for each active layer
    for layer_id in range(nlay):
        for row in range(nrow):
            slope = np.linspace(outcrop_z_max[layer_id], 
                                outcrop_z_min[layer_id], 
                                np.sum(irch[row, :] == layer_id))
            top[row, irch[row, :] == layer_id] = slope

    # Step 2: Add smooth transitions that extend beyond the active zone (to the left)
    for layer_id in range(nlay - 1):
        # Find cells at the boundary between adjacent layers
        transition_mask = (irch == layer_id) & (np.roll(irch, 1, axis=-1) == layer_id + 1)
        
        for row in range(nrow):
            transition_indices = np.where(transition_mask[row, :])[0]
            for idx in transition_indices:
                # Apply the transition over the specified number of cells
                start = max(0, idx - transition_cells + 1)
                end = idx + 1
                n = end - start
                if n > 1:
                    transition_range = np.linspace(
                        outcrop_z_min[layer_id + 1], top[row, end], n)
                    top[row, start:end] = transition_range
  
    return top

def compute_slope_right_contain(idomain, outcrop_z_min, outcrop_z_max, transition_cells):
    """
    Compute top elevations with smooth transitions between recharge zones, adding linear slopes for each layer,
    for right-dipping systems. The transition zone is added strictly within the active area defined by `idomain`
    for each layer (it does not extend or modify the idomain).
    Note:
    This version ensures that the transition happens *within* the zone defined as active in `idomain`
    for each layer. The transition is performed from the end of the current layers domain up to the
    start of the next, without spilling into the next layer's active area.

    Parameters:
        idomain (numpy.ndarray): 3D array (nlay, nrow, ncol) indicating active (1) and inactive (0) cells.
        outcrop_z_min (1D array-like): Minimum elevation for each layer (length nlay).
        outcrop_z_max (1D array-like): Maximum elevation for each layer (length nlay).
        transition_cells (int): Number of columns for the transition zone between layers.

    Returns:
        top (numpy.ndarray): 2D array (nrow, ncol) of top elevations with contained transitions and slopes.
    """
    import numpy as np

    # Input checks
    if idomain.ndim != 3:
        raise ValueError("idomain must be a 3D array (nlay, nrow, ncol).")
    nlay, nrow, ncol = idomain.shape
    outcrop_z_min = np.asarray(outcrop_z_min)
    outcrop_z_max = np.asarray(outcrop_z_max)
    if outcrop_z_min.shape[0] != nlay or outcrop_z_max.shape[0] != nlay:
        raise ValueError("outcrop_z_min and outcrop_z_max must have length equal to nlay.")
    if not isinstance(transition_cells, int) or transition_cells < 1:
        raise ValueError("transition_cells must be a positive integer.")
    
    # Initialize top array
    top = np.zeros((nrow, ncol), dtype=float)

    # Compute topmost active layer index at each surface cell
    active_layers = np.sum(idomain, axis=0)  # Shape will be (nrow, ncol)
    irch = nlay - active_layers  # Values will range from 0 to nlay-1

    # Step 1: Assign a sloping top elevation for each active layer
    for layer_id in range(nlay):
        for row in range(nrow):
            slope = np.linspace(outcrop_z_max[layer_id], 
                                outcrop_z_min[layer_id], 
                                np.sum(irch[row, :] == layer_id))
            top[row, irch[row, :] == layer_id] = slope

    # Step 2: Add smooth transitions, but staying within the active zone of each layer
    for layer_id in range(nlay - 1):
        # Find cells at the boundary between adjacent layers
        transition_mask = (irch == layer_id) & (np.roll(irch, 1, axis=-1) == layer_id + 1)
        
        for row in range(nrow):
            transition_indices = np.where(transition_mask[row, :])[0]
            for idx in transition_indices:
                start = idx
                end = min(ncol-1, idx + transition_cells)
                n = end - start
                if n > 1:
                    transition_range = np.linspace(outcrop_z_min[layer_id + 1], top[row, end], n)
                    top[row, start:end] = transition_range
 
    return top

def compute_top_left_extend(idomain, outcrop_z, transition_cells):
    """
    Compute top elevations with smooth transitions between outcroping zones using outcrop_z values,
    for left-dipping systems. The active area of zach layer is extended beyond the active zones defined in `idomain`
    by the extent specified by the `transition_cells` parameter (number of columns).

    Parameters:
        idomain (numpy.ndarray): 3D array (nlay, nrow, ncol) indicating active (1) and inactive (0) cells.
        outcrop_z (1D array-like): Array of top elevations for each layer (length nlay).
        transition_cells (int): Number of columns for the transition zone between layers.

    Returns:
        top (numpy.ndarray): 2D array (nrow, ncol) of top elevations with extended transitions.

    Note:
    This version extends the top elevation transition *beyond* the active zones defined in the input idomain
    by the extent specified by the `transition_cells` parameter. This can be useful for certain visualization
    or conceptual model setups, but it does not confine transitions strictly within each layer's original idomain.
    """
    import numpy as np

    # Input checks
    if idomain.ndim != 3:
        raise ValueError("idomain must be a 3D array (nlay, nrow, ncol).")
    nlay, nrow, ncol = idomain.shape
    outcrop_z = np.asarray(outcrop_z)
    if outcrop_z.shape[0] != nlay:
        raise ValueError("outcrop_z must have length equal to nlay.")
    if not isinstance(transition_cells, int) or transition_cells < 1:
        raise ValueError("transition_cells must be a positive integer.")
        
    #Initialize top array
    top = np.zeros((nrow, ncol), dtype=float)

    # Compute the topmost active layer index at each cell
    active_layers = np.sum(idomain, axis=0)  # (nrow, ncol)
    irch = nlay - active_layers  # Topmost active layer index per cell

    # Step 1: Assign base top elevations from outcrop_z
    for layer_id in range(nlay):
        top[irch == layer_id] = outcrop_z[layer_id]

    # Step 2: Add transitions that extend beyond layer_id's domain
    for layer_id in range(nlay - 1):
        # Find transition boundaries: from layer_id to layer_id+1
        transition_mask = (irch == layer_id) & (np.roll(irch, -1, axis=-1) == layer_id + 1)

        for row in range(nrow):
            transition_indices = np.where(transition_mask[row, :])[0]
            for idx in transition_indices:
                # Apply transition starting at boundary and extending forward
                start = idx
                end = min(ncol-1, idx + transition_cells)
                n = end - start
                if n > 1:
                    top[row, start:end] = np.linspace(
                        outcrop_z[layer_id], outcrop_z[layer_id + 1], n)

    return top

def compute_top_left_contain(idomain, outcrop_z, transition_cells):
    """
    Compute top elevations with smooth transitions between outcropping zones using outcrop_z values,
    for left-dipping systems. The transition zone is added strictly within the active area defined by `idomain`
    for each layer (it does not extend or modify the idomain).
    Note:
    This version ensures that the transition happens *within* the zone defined as active in `idomain`
    for each layer. The transition is performed from the end of the current layers domain up to the
    start of the next, without spilling into the next layer's active area.

    Parameters:
        idomain (numpy.ndarray): 3D array (nlay, nrow, ncol) indicating active (1) and inactive (0) cells.
        outcrop_z (1D array-like): Array of top elevations for each layer (length nlay).
        transition_cells (int): Number of columns for the transition zone between layers.

    Returns:
        top (numpy.ndarray): 2D array (nrow, ncol) of top elevations with contained transitions.
    """

    import numpy as np

    # Input checks
    if idomain.ndim != 3:
        raise ValueError("idomain must be a 3D array (nlay, nrow, ncol).")
    nlay, nrow, ncol = idomain.shape
    outcrop_z = np.asarray(outcrop_z)
    if outcrop_z.shape[0] != nlay:
        raise ValueError("outcrop_z must have length equal to nlay.")
    if not isinstance(transition_cells, int) or transition_cells < 1:
        raise ValueError("transition_cells must be a positive integer.")
    
    # Initialize top array
    top = np.zeros((nrow, ncol), dtype=float)

    # Compute topmost active layer index at each surface cell
    active_layers = np.sum(idomain, axis=0)  # (nrow, ncol)
    irch = nlay - active_layers  # topmost active layer

    # Step 1: Assign base top elevations from outcrop_z
    for layer_id in range(nlay):
        top[irch == layer_id] = outcrop_z[layer_id]

    # Step 2: Add smooth transitions, but staying within the active zone of each layer
    for layer_id in range(nlay - 1):
        # Identify boundary columns between layer_id and layer_id + 1
        transition_mask = (irch == layer_id) & (np.roll(irch, -1, axis=-1) == layer_id + 1)

        for row in range(nrow):
            transition_indices = np.where(transition_mask[row, :])[0]
            for idx in transition_indices:
                start = max(0, idx - transition_cells + 1)
                end = idx + 1  # include the boundary cell
                n = end - start
                if n > 1:
                    top[row, start:end] = np.linspace(
                        outcrop_z[layer_id], outcrop_z[layer_id + 1], n)

    return top

def compute_top_right_extend(idomain, outcrop_z, transition_cells):
    """
    Compute top elevations with smooth transitions between recharge zones using outcrop_z values,
    for right-dipping systems. The transition zone is extended beyond the active zones defined in `idomain`
    by the extent specified by the `transition_cells` parameter (number of columns).
    Note:
    This version extends the top elevation transition *beyond* the active zones defined in `idomain`
    by the extent specified by the `transition` parameter. This can be useful for certain visualization
    or conceptual model setups, but it does not confine transitions strictly within each layer's input idomain.

    Parameters:
        idomain (numpy.ndarray): 3D array (nlay, nrow, ncol) indicating active (1) and inactive (0) cells.
        outcrop_z (1D array-like): Array of top elevations for each layer (length nlay).
        transition_cells (int): Number of columns for the transition zone between layers.

    Returns:
        top (numpy.ndarray): 2D array (nrow, ncol) of top elevations with extended transitions.
    """
    import numpy as np

    # Input checks
    if idomain.ndim != 3:
        raise ValueError("idomain must be a 3D array (nlay, nrow, ncol).")
    nlay, nrow, ncol = idomain.shape
    outcrop_z = np.asarray(outcrop_z)
    if outcrop_z.shape[0] != nlay:
        raise ValueError("outcrop_z must have length equal to nlay.")
    if not isinstance(transition_cells, int) or transition_cells < 1:
        raise ValueError("transition_cells must be a positive integer.")
    
    # Initialize top array
    top = np.zeros((nrow, ncol), dtype=float)

    # Compute topmost active layer index at each surface cell
    active_layers = np.sum(idomain, axis=0)  # (nrow, ncol)
    irch = nlay - active_layers  # Topmost active layer index per cell

    # Step 1: Assign base top elevations from outcrop_z
    for layer_id in range(nlay):
        top[irch == layer_id] = outcrop_z[layer_id]

    # Step 2: Add transitions that extend beyond layer_id's domain
    for layer_id in range(nlay - 1):
        # Find transition boundaries: from layer_id to layer_id+1
        transition_mask = (irch == layer_id) & (np.roll(irch, 1, axis=-1) == layer_id + 1)

        for row in range(nrow):
            transition_indices = np.where(transition_mask[row, :])[0]
            for idx in transition_indices:
                # Apply transition starting at boundary and extending forward
                start = max(0, idx - transition_cells + 1)
                end = idx+1
                n = end - start
                if n > 1:
                    top[row, start:end] = np.linspace(
                        outcrop_z[layer_id+1], outcrop_z[layer_id], n)

    return top

def compute_top_right_contain(idomain, outcrop_z, transition_cells):
    """
    Compute top elevations with smooth transitions between outcropping zones using outcrop_z values,
    for right-dipping systems. The transition zone is added strictly within the active area defined by `idomain`
    for each layer (it does not extend or modify the idomain).
    Note:
    This version ensures that the transition happens *within* the zone defined as active in `idomain`
    for each layer. The transition is performed from the end of the current layers domain up to the
    start of the next, without spilling into the next layer's active area.

    Parameters:
        idomain (numpy.ndarray): 3D array (nlay, nrow, ncol) indicating active (1) and inactive (0) cells.
        outcrop_z (1D array-like): Array of top elevations for each layer (length nlay).
        transition_cells (int): Number of columns for the transition zone between layers.

    Returns:
        top (numpy.ndarray): 2D array (nrow, ncol) of top elevations with contained transitions.
    """
    import numpy as np

    # Input checks
    if idomain.ndim != 3:
        raise ValueError("idomain must be a 3D array (nlay, nrow, ncol).")
    nlay, nrow, ncol = idomain.shape
    outcrop_z = np.asarray(outcrop_z)
    if outcrop_z.shape[0] != nlay:
        raise ValueError("outcrop_z must have length equal to nlay.")
    if not isinstance(transition_cells, int) or transition_cells < 1:
        raise ValueError("transition_cells must be a positive integer.")

    # Initialize top array
    top = np.zeros((nrow, ncol), dtype=float)

    # Compute topmost active layer index at each surface cell
    active_layers = np.sum(idomain, axis=0)  # (nrow, ncol)
    irch = nlay - active_layers  # topmost active layer index at each surface cell

    # Step 1: Assign base top elevations from outcrop_z
    for layer_id in range(nlay):
        top[irch == layer_id] = outcrop_z[layer_id]

    # Step 2: Smooth transitions, but staying within the active zone of each layer
    for layer_id in range(nlay - 1):
        # Identify boundary columns between layer_id and layer_id + 1
        transition_mask = (irch == layer_id) & (np.roll(irch, 1, axis=-1) == layer_id + 1)

        for row in range(nrow):
            transition_indices = np.where(transition_mask[row, :])[0]
            for idx in transition_indices:
                start = idx 
                end = min(ncol-1, idx + transition_cells)
                n = end - start
                if n > 1:
                    top[row, start:end] = np.linspace(
                        outcrop_z[layer_id + 1], outcrop_z[layer_id], n)
    return top

# ==========================================================================================
# THICKNESS VARIANT FUNCTIONS - ADVANCED & EXPERIMENTAL SYNTHETIC GEOMETRY GENERATION
# ==========================================================================================

def compute_thickness_simple(idomain, base_thicknesses):
    """
    Compute the thickness array for each cell based on idomain and base_thicknesses.

    Parameters:
        idomain (numpy.ndarray): 3D array (nlay, nrow, ncol) indicating active (1) and inactive (0) cells.
        base_thicknesses (1D array-like): Array of length nlay, with the base thickness for each model layer.

    Returns:
        thickness_array (numpy.ndarray): 3D array (nlay, nrow, ncol) with thicknesses.
            Thickness is set to base_thickness for active cells (idomain == 1),
            and 0 for inactive cells (idomain == 0).
    """
    import numpy as np

    # Input checks
    if idomain.ndim != 3:
        raise ValueError("idomain must be a 3D array (nlay, nrow, ncol).")
    nlay, nrow, ncol = idomain.shape
    base_thicknesses = np.asarray(base_thicknesses)
    if base_thicknesses.shape[0] != nlay:
        raise ValueError("base_thicknesses must have length equal to nlay.")
    
    # Initialize the thickness array
    thickness_array = np.zeros_like(idomain, dtype=float)

    # Loop through each layer and assign thickness based on idomain
    for layer in range(nlay):
        thickness_array[layer, :, :] = np.where(idomain[layer, :, :] == 1, base_thicknesses[layer], 0)

    return thickness_array

def compute_thickness_extend(idomain, base_thicknesses, transition_cells):
    """
    Compute the thickness array with smooth transitions from base thickness to zero,
    extending the transition zone beyond the active area defined by idomain.
    Note:
    This version extends the top elevation transition *beyond* the active zones defined in `idomain`
    by the extent specified by the `transition` parameter. This can be useful for certain visualization
    or conceptual model setups, but it does not confine transitions strictly within each layer's domain.

    Parameters:
        idomain (numpy.ndarray): 3D array (nlay, nrow, ncol) indicating active (1) and inactive (0) cells.
        base_thicknesses (1D array-like): Array of length nlay, with the base thickness for each model layer.
        transition_cells (int): Number of columns for the transition zone beyond the active area.

    Returns:
        thickness_array (numpy.ndarray): 3D array (nlay, nrow, ncol) with thicknesses.
            Thickness is set to base_thickness for active cells (idomain == 1),
            smoothly transitions to 0 over transition_cells beyond the active area,
            and 0 for inactive cells outside the transition zone.
    """
    import numpy as np

    # Input checks
    if idomain.ndim != 3:
        raise ValueError("idomain must be a 3D array (nlay, nrow, ncol).")
    nlay, nrow, ncol = idomain.shape
    base_thicknesses = np.asarray(base_thicknesses)
    if base_thicknesses.shape[0] != nlay:
        raise ValueError("base_thicknesses must have length equal to nlay.")
    if not isinstance(transition_cells, int) or transition_cells < 1:
        raise ValueError("transition_cells must be a positive integer.")
    
    # Initialize the thickness array
    thickness_array = np.zeros_like(idomain, dtype=float)

    # Loop through layers and apply base thickness and smooth transition
    for layer in range(nlay):
        # Get base thickness for the current layer
        base_thickness = base_thicknesses[layer]
        
        # Set thickness for active cells (idomain == 1) to the base thickness
        thickness_array[layer, :, :] = np.where(idomain[layer, :, :] == 1, base_thickness, 0)
        
        # Now we smooth the transition from base thickness to zero within the same layer
        for row in range(nrow):
            for col in range(ncol):
                if idomain[layer, row, col] == 0:  # If the cell is inactive
                    # Check if there are adjacent active cells to create a smooth transition
                    if col > 0 and idomain[layer, row, col - 1] == 1:  # Transition from left
                        transition_range = np.linspace(base_thickness, 0, transition_cells)
                        # Apply transition over the next `transition_cells` columns
                        for t in range(min(transition_cells, ncol - col)):
                            thickness_array[layer, row, col + t] = transition_range[t]
                    elif col < ncol - 1 and idomain[layer, row, col + 1] == 1:  # Transition from right
                        transition_range = np.linspace(base_thickness, 0, transition_cells)
                        # Apply transition over the previous `transition_cells` columns
                        for t in range(min(transition_cells, col + 1)):
                            thickness_array[layer, row, col - t] = transition_range[t]
    
    return thickness_array

def compute_thickness_contain(idomain, base_thicknesses, transition_cells):
    """
    Compute the thickness array with smooth transitions from base thickness to zero,
    but keep the transition strictly within the active area defined by idomain.
    Note:
    This version ensures that the transition happens *within* the zone defined as active in `idomain`
    for each layer. The transition is performed from the end of the current layers domain up to the
    start of the next, without spilling into the next layer's active area.

    Parameters:
        idomain (numpy.ndarray): 3D array (nlay, nrow, ncol) indicating active (1) and inactive (0) cells.
        base_thicknesses (1D array-like): Array of length nlay, with the base thickness for each model layer.
        transition_cells (int): Number of columns for the transition zone within the active area.

    Returns:
        thickness_array (numpy.ndarray): 3D array (nlay, nrow, ncol) with thicknesses.
            Thickness is set to base_thickness for active cells (idomain == 1),
            smoothly transitions to 0 within the last transition_cells of the active area,
            and 0 for inactive cells.
    """
    import numpy as np

    # Input checks
    if idomain.ndim != 3:
        raise ValueError("idomain must be a 3D array (nlay, nrow, ncol).")
    nlay, nrow, ncol = idomain.shape
    base_thicknesses = np.asarray(base_thicknesses)
    if base_thicknesses.shape[0] != nlay:
        raise ValueError("base_thicknesses must have length equal to nlay.")
    if not isinstance(transition_cells, int) or transition_cells < 1:
        raise ValueError("transition_cells must be a positive integer.")

    # Initialize the thickness array
    thickness_array = np.zeros_like(idomain, dtype=float)

    # Loop through layers and apply base thickness and smooth transition
    for layer in range(nlay):
        # Get base thickness for the current layer
        base_thickness = base_thicknesses[layer]
        
        # Set thickness for active cells (idomain == 1) to the base thickness
        thickness_array[layer, :, :] = np.where(idomain[layer, :, :] == 1, base_thickness, 0)
        
        # Now we smooth the transition from base thickness to zero within the same layer
        for row in range(nrow):
            for col in range(ncol):
                if idomain[layer, row, col] == 0:  # If the cell is inactive
                    # Check if there are adjacent active cells to create a smooth transition
                    if col > 0 and idomain[layer, row, col - 1] == 1:  # Transition from left
                        transition_range = np.linspace(0, base_thickness, transition_cells)
                        # Apply transition over the next `transition_cells` columns
                        for t in range(min(transition_cells, ncol - col)):
                            thickness_array[layer, row, col - t] = transition_range[t]
                    elif col < ncol - 1 and idomain[layer, row, col + 1] == 1:  # Transition from right
                        transition_range = np.linspace(0, base_thickness, transition_cells)
                        # Apply transition over the previous `transition_cells` columns
                        for t in range(min(transition_cells, col + 1)):
                            thickness_array[layer, row, col + t] = transition_range[t]
    
    return thickness_array







