import os
import sys
import numpy as np
import pandas as pd

# Subscript 1 is used for left side dipping multi layer systems
# Subscript 2 is used for right side dipping mulyi layer systems

#Subscript a is used for extending layers beyond idomain
#Subqcrit b is used for constraining layers within idomain (recomended)

#Recomended use: idomain2, and subscripts 2b/b

#When combaining a top_b type function with a thickness_a funciton, a more complew structure and geometry is obtained

def create_idomain1(nlay, nrow, ncol, length, outcrop_length):
    """
    Create an idomain array based on outcrop lengths for each layer.
    This version is useful for multilayer systems where layers are dipping towards the left side
    and outcropping to the right side.
    
    Parameters:
    - nlay: Number of layers.
    - nrow: Number of rows.
    - ncol: Number of columns.
    - Length: Total length of the model cross section
    - outcrop_length: 1D array with shape (nlay - 1,) defining the outcrop lengths (thresholds) for each layer.
    
    Returns:
    - idomain: A 3D array of shape (nlay, nrow, ncol) indicating the active (1) or inactive (0) cells for each layer.
    """
    # Initialize idomain array with ones (active cells)
    idomain = np.ones((nlay, nrow, ncol), dtype=int)
    dcol = length/ncol
    # Apply the condition for each layer using the corresponding outcrop length
    for layer in range(nlay - 1):  # Loop through all layers except the last one
        L = int((outcrop_length[layer] / dcol))
        idomain[layer, :, L:] = 0  # Set cells beyond the threshold for the current layer to 0
    
    # For the last layer, no need to change idomain because it already defaults to 1 for all cells
    return idomain

def create_idomain2(nlay, nrow, ncol, length, outcrop_length):
    """
    Create an idomain array based on outcrop lengths for each layer.
    This version is useful for multilayer systems where layers are dipping towards the right side
    and outcropping to the left side.
    
    Parameters:
    - nlay: Number of layers.
    - nrow: Number of rows.
    - ncol: Number of columns.
    - Length: Total length of the model cross section
    - outcrop_length: 1D array with shape (nlay - 1,) defining the outcrop lengths (thresholds) for each layer.
    
    Returns:
    - idomain: A 3D array of shape (nlay, nrow, ncol) indicating the active (1) or inactive (0) cells for each layer.
    """
    # Initialize idomain array with ones (active cells)
    idomain = np.ones((nlay, nrow, ncol), dtype=int)
    dcol = length/ncol
    # Apply the condition for each layer using the corresponding outcrop length
    for layer in range(nlay - 1):  # Loop through all layers except the last one
        L = int((outcrop_length[layer] / dcol))
        idomain[layer, :, :L] = 0  # Set cells untill the threshold for the current layer to 0 (inactive)
    
    # For the last layer, no need to change idomain because it already defaults to 1 for all cells
    return idomain

def compute_top(idomain, outcrop_z):
    """
    Compute top elevations based on the active layers and outcrop_z values.
    Parameters:
    - idomain: array indicating the active layers (nlay x nrow x ncol).
    - outcrop_z: 1D array of outcropping top elevations for each layer (shape = nlay).
    Returns:
    - top: array of top elevations (shape = nrow x ncol).
    """
    nlay, nrow, ncol = idomain.shape
    top = np.zeros((nrow, ncol), dtype=float)
    
    # Calculate irch (active layers)
    active_layers = np.sum(idomain, axis=0)  # Shape will be (nrow, ncol)
    
    # Subtract from the number of layers to calculate irch (range from 0 to nlay-1)
    irch = nlay - active_layers
    
    # Assign the correct top elevation for each layer
    for layer_id in range(nlay):
        top[irch == layer_id] = outcrop_z[layer_id]

    return top

def compute_top_with_transition1a(idomain, length, outcrop_z, transition):
    """
    Compute top elevations with smooth transitions between recharge zones using outcrop_z values.
    
    Note:
    This version extends the top elevation transition *beyond* the active zones defined in `idomain`
    by the extent specified by the `transition` parameter. This can be useful for certain visualization
    or conceptual model setups, but it does not confine transitions strictly within each layer's domain.

    Parameters:
    - idomain: 3D array indicating active model cells (shape = nlay x nrow x ncol).
    - length: total length of the model cross section (in meters).
    - outcrop_z: 1D array of top elevations for each layer (shape = nlay).
    - transition: length of the transition zone between layers (in meters).

    Returns:
    - top: 2D array (shape = nrow x ncol) of top elevations with extended transitions.
    """
    nlay, nrow, ncol = idomain.shape
    top = np.zeros((nrow, ncol), dtype=float)
    dcol = length / ncol
    transition_cells = int(transition / dcol)

    # Compute which layer outcrops at each surface cell
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
                    top[row, start:end] = np.linspace(outcrop_z[layer_id], outcrop_z[layer_id + 1], n)

    return top

def compute_top_slope1a(idomain, length, outcrop_z_min, outcrop_z_max, transition):
    """
    Compute top elevations with smooth transitions between recharge zones using linear slopes for each layer.
    
    Note:
    This version extends the top elevation transition *beyond* the active zones defined in `idomain`
    by the extent specified by the `transition` parameter. This can be useful for certain visualization
    or conceptual model setups, but it does not confine transitions strictly within each layer's domain.

    Parameters:
    - idomain: 3D array indicating the active layers (shape = nlay x nrow x ncol).
    - length: Total length of the model cross-section (in horizontal units).
    - outcrop_z_min: 1D array of minimum elevations for each layer (shape = nlay).
    - outcrop_z_max: 1D array of maximum elevations for each layer (shape = nlay).
    - transition: Length of the transition zone between adjacent recharge zones.
    
    Returns:
    - top: 2D array of top elevations (shape = nrow x ncol) with transitions.
    """
    import numpy as np

    nlay, nrow, ncol = idomain.shape
    top = np.zeros((nrow, ncol), dtype=float)
    dcol = length / ncol
    transition_cells = int(transition / dcol)

    # Calculate irch (active layers)
    active_layers = np.sum(idomain, axis=0)  # Shape will be (nrow, ncol)
    irch = nlay - active_layers  # Values will range from 0 to nlay-1

    # Assign a sloping top elevation for each active layer
    for layer_id in range(nlay):
        for row in range(nrow):
            slope = np.linspace(outcrop_z_min[layer_id], 
                                outcrop_z_max[layer_id], 
                                np.sum(irch[row, :] == layer_id))
            top[row, irch[row, :] == layer_id] = slope

    # Smooth transition between zones
        # Smooth transition between zones
    for layer_id in range(nlay - 1):
        # Find cells at the boundary between adjacent layers
        transition_mask = (irch == layer_id) & (np.roll(irch, -1, axis=-1) == layer_id + 1)
        
        for row in range(nrow):
            transition_indices = np.where(transition_mask[row, :])[0]
            for idx in transition_indices:
                # Apply the transition over the specified number of cells
                start = idx
                end = min(ncol-1, idx + transition_cells)
                n = end - start
                if n > 1:
                    transition_range = np.linspace(outcrop_z_max[layer_id], top[row, end], n)
                    top[row, start:end] = transition_range
  
    return top

def compute_top_with_transition1b(idomain, length, outcrop_z, transition):
    """
    Compute top elevations with smooth transitions between recharge zones using outcrop_z values.
    Note:
    This version ensures that the transition happens *within* the zone defined as active in `idomain`
    for each layer. The transition is performed from the end of the current layers domain up to the
    start of the next, without spilling into the next layer's active area.

    Parameters:
    - idomain: array indicating the active layers (nlay x nrow x ncol).
    - length: total length of the model cross section (meters).
    - outcrop_z: 1D array of top elevations for each layer (shape = nlay).
    - transition: length of the transition zone (meters).

    Returns:
    - top: 2D array (nrow x ncol) of top elevations with smoothed transitions.
    """
    nlay, nrow, ncol = idomain.shape
    top = np.zeros((nrow, ncol), dtype=float)
    dcol = length / ncol
    transition_cells = int(transition / dcol)

    # Compute which layer outcrops at each surface column
    active_layers = np.sum(idomain, axis=0)  # (nrow, ncol)
    irch = nlay - active_layers  # topmost active layer index at each surface cell

    # Step 1: Assign base top elevations from outcrop_z
    for layer_id in range(nlay):
        top[irch == layer_id] = outcrop_z[layer_id]

    # Step 2: Smooth transitions, but staying within the active zone of each layer
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
                    top[row, start:end] = np.linspace(outcrop_z[layer_id], outcrop_z[layer_id + 1], n)

    return top

def compute_top_slope1b(idomain, length, outcrop_z_min, outcrop_z_max, transition):
    """
    Compute top elevations with smooth transitions between recharge zones using linear slopes for each layer.
    
    This version ensures that the transition happens *within* the zone defined as active in `idomain`
    for each layer. The transition is performed from the end of the current layers domain up to the
    start of the next, without spilling into the next layer's active area.

    Parameters:
    - idomain: 3D array indicating the active layers (shape = nlay x nrow x ncol).
    - length: Total length of the model cross-section (in horizontal units).
    - outcrop_z_min: 1D array of minimum elevations for each layer (shape = nlay).
    - outcrop_z_max: 1D array of maximum elevations for each layer (shape = nlay).
    - transition: Length of the transition zone between adjacent recharge zones.
    
    Returns:
    - top: 2D array of top elevations (shape = nrow x ncol) with transitions.
    """
    import numpy as np

    nlay, nrow, ncol = idomain.shape
    top = np.zeros((nrow, ncol), dtype=float)
    dcol = length / ncol
    transition_cells = int(transition / dcol)

    # Calculate irch (active layers)
    active_layers = np.sum(idomain, axis=0)  # Shape will be (nrow, ncol)
    irch = nlay - active_layers  # Values will range from 0 to nlay-1

    # Assign a sloping top elevation for each active layer
    for layer_id in range(nlay):
        for row in range(nrow):
            slope = np.linspace(outcrop_z_min[layer_id], 
                                outcrop_z_max[layer_id], 
                                np.sum(irch[row, :] == layer_id))
            top[row, irch[row, :] == layer_id] = slope

    # Smooth transition between zones
        # Smooth transition between zones
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
                    transition_range = np.linspace(outcrop_z_max[layer_id], top[row, end], n)
                    top[row, start:end] = transition_range
  
    return top

def compute_top_with_transition2a(idomain, length, outcrop_z, transition):
    """
    Compute top elevations with smooth transitions between recharge zones using outcrop_z values.
    
    Note:
    This version extends the top elevation transition *beyond* the active zones defined in `idomain`
    by the extent specified by the `transition` parameter. This can be useful for certain visualization
    or conceptual model setups, but it does not confine transitions strictly within each layer's domain.
    Used with idomain2 for right side dipping multilayer systems

    Parameters:
    - idomain: 3D array indicating active model cells (shape = nlay x nrow x ncol).
    - length: total length of the model cross section (in meters).
    - outcrop_z: 1D array of top elevations for each layer (shape = nlay).
    - transition: length of the transition zone between layers (in meters).

    Returns:
    - top: 2D array (shape = nrow x ncol) of top elevations with extended transitions.
    """
    nlay, nrow, ncol = idomain.shape
    top = np.zeros((nrow, ncol), dtype=float)
    dcol = length / ncol
    transition_cells = int(transition / dcol)

    # Compute which layer outcrops at each surface cell
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
                    top[row, start:end] = np.linspace(outcrop_z[layer_id+1], outcrop_z[layer_id], n)

    return top

def compute_top_slope2a(idomain, length, outcrop_z_min, outcrop_z_max, transition):
    """
    Compute top elevations with smooth transitions between recharge zones using linear slopes for each layer.
    
    Note:
    This version extends the top elevation transition *beyond* the active zones defined in `idomain`
    by the extent specified by the `transition` parameter. This can be useful for certain visualization
    or conceptual model setups, but it does not confine transitions strictly within each layer's domain.

    Parameters:
    - idomain: 3D array indicating the active layers (shape = nlay x nrow x ncol).
    - length: Total length of the model cross-section (in horizontal units).
    - outcrop_z_min: 1D array of minimum elevations for each layer (shape = nlay).
    - outcrop_z_max: 1D array of maximum elevations for each layer (shape = nlay).
    - transition: Length of the transition zone between adjacent recharge zones.
    
    Returns:
    - top: 2D array of top elevations (shape = nrow x ncol) with transitions.
    """
    import numpy as np

    nlay, nrow, ncol = idomain.shape
    top = np.zeros((nrow, ncol), dtype=float)
    dcol = length / ncol
    transition_cells = int(transition / dcol)

    # Calculate irch (active layers)
    active_layers = np.sum(idomain, axis=0)  # Shape will be (nrow, ncol)
    irch = nlay - active_layers  # Values will range from 0 to nlay-1

    # Assign a sloping top elevation for each active layer
    for layer_id in range(nlay):
        for row in range(nrow):
            slope = np.linspace(outcrop_z_max[layer_id], 
                                outcrop_z_min[layer_id], 
                                np.sum(irch[row, :] == layer_id))
            top[row, irch[row, :] == layer_id] = slope

    # Smooth transition between zones
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
                    transition_range = np.linspace(outcrop_z_min[layer_id + 1], top[row, end], n)
                    top[row, start:end] = transition_range
  
    return top

def compute_top_slope2b(idomain, length, outcrop_z_min, outcrop_z_max, transition):
    """
    Compute top elevations with smooth transitions between recharge zones using linear slopes for each layer.
    
    This version ensures that the transition happens *within* the zone defined as active in `idomain`
    for each layer. The transition is performed from the end of the current layers domain up to the
    start of the next, without spilling into the next layer's active area.

    Parameters:
    - idomain: 3D array indicating the active layers (shape = nlay x nrow x ncol).
    - length: Total length of the model cross-section (in horizontal units).
    - outcrop_z_min: 1D array of minimum elevations for each layer (shape = nlay).
    - outcrop_z_max: 1D array of maximum elevations for each layer (shape = nlay).
    - transition: Length of the transition zone between adjacent recharge zones.
    
    Returns:
    - top: 2D array of top elevations (shape = nrow x ncol) with transitions.
    """
    import numpy as np

    nlay, nrow, ncol = idomain.shape
    top = np.zeros((nrow, ncol), dtype=float)
    dcol = length / ncol
    transition_cells = int(transition / dcol)

    # Calculate irch (active layers)
    active_layers = np.sum(idomain, axis=0)  # Shape will be (nrow, ncol)
    irch = nlay - active_layers  # Values will range from 0 to nlay-1

    # Assign a sloping top elevation for each active layer
    for layer_id in range(nlay):
        for row in range(nrow):
            slope = np.linspace(outcrop_z_max[layer_id], 
                                outcrop_z_min[layer_id], 
                                np.sum(irch[row, :] == layer_id))
            top[row, irch[row, :] == layer_id] = slope

    # Smooth transition between zones
        # Smooth transition between zones
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
                    transition_range = np.linspace(outcrop_z_min[layer_id+1], top[row, end], n)
                    top[row, start:end] = transition_range
  
    return top

def compute_top_with_transition2b(idomain, length, outcrop_z, transition):
    """
    Compute top elevations with smooth transitions between recharge zones using outcrop_z values.
    Note:
    This version ensures that the transition happens *within* the zone defined as active in `idomain`
    for each layer. The transition is performed from the end of the current layers domain up to the
    start of the next, without spilling into the next layer's active area.

    Parameters:
    - idomain: array indicating the active layers (nlay x nrow x ncol).
    - length: total length of the model cross section (meters).
    - outcrop_z: 1D array of top elevations for each layer (shape = nlay).
    - transition: length of the transition zone (meters).

    Returns:
    - top: 2D array (nrow x ncol) of top elevations with smoothed transitions.
    """
    nlay, nrow, ncol = idomain.shape
    top = np.zeros((nrow, ncol), dtype=float)
    dcol = length / ncol
    transition_cells = int(transition / dcol)

    # Compute which layer outcrops at each surface column
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
                    top[row, start:end] = np.linspace(outcrop_z[layer_id + 1], outcrop_z[layer_id], n)

    return top

def compute_thickness(idomain, base_thicknesses):
    """
    Compute thickness array based on idomain and base_thicknesses.
    Parameters:
    - idomain: 3D array (nlay, nrow, ncol) indicating active (1) and inactive (0) cells.
    - base_thicknesses: 1D array (nlay) of thickness values for each model layer.

    Returns:
    - thickness_array: 3D array (nlay, nrow, ncol) with thicknesses.
      Thickness is set to base_thickness for active cells (idomain == 1),
      and 0 for inactive cells (idomain == 0).
    """
    nlay, nrow, ncol = idomain.shape
    thickness_array = np.zeros_like(idomain, dtype=float)

    # Loop through each layer and assign thickness based on idomain
    for layer in range(nlay):
        thickness_array[layer, :, :] = np.where(idomain[layer, :, :] == 1, base_thicknesses[layer], 0)

    return thickness_array

def compute_thickness_with_transition_a(idomain, length, base_thicknesses, transition):
    """
    Compute the thickness array with smooth transitions from base thickness to zero,
    based on the idomain array.

    Note:
    This version extends the top elevation transition *beyond* the active zones defined in `idomain`
    by the extent specified by the `transition` parameter. This can be useful for certain visualization
    or conceptual model setups, but it does not confine transitions strictly within each layer's domain.

    Parameters:
    - idomain: 3D array (nlay, nrow, ncol) indicating active (1) or inactive (0) cells.
    - base_thicknesses: 1D array (nlay) with the base thicknesses for each model layer.
    - transition: Length of the transition zone.
    
    Returns:
    - thickness_array: 3D array (nlay, nrow, ncol) with smoothed thicknesses.
    """
    
    nlay, nrow, ncol = idomain.shape
    dcol = length/ncol
    transition_cells = int(transition/dcol)
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

def compute_thickness_with_transition_b(idomain, length, base_thicknesses, transition):
    """
    Compute the thickness array with smooth transitions from base thickness to zero,
    based on the idomain array.

    This version ensures that the transition happens *within* the zone defined as active in `idomain`
    for each layer. The transition is performed from the end of the current layers domain up to the
    start of the next, without spilling into the next layer's active area.

    Parameters:
    - idomain: 3D array (nlay, nrow, ncol) indicating active (1) or inactive (0) cells.
    - base_thicknesses: 1D array (nlay) with the base thicknesses for each model layer.
    - transition: Length of the transition zone.
    
    Returns:
    - thickness_array: 3D array (nlay, nrow, ncol) with smoothed thicknesses.
    """
    
    nlay, nrow, ncol = idomain.shape
    dcol = length/ncol
    transition_cells = int(transition/dcol)
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

def compute_bottom(ztop, thickness_array):
    """
    Compute the bottom elevations for each layer based on ztop and thickness_array.
    
    Parameters:
    - ztop: 3D array (nlay, nrow, ncol) representing the top elevations for each layer.
    - thickness_array: 3D array (nlay, nrow, ncol) with the thickness for each layer.
    
    Returns:
    - bottom: 3D array (nlay, nrow, ncol) representing the bottom elevations for each layer.
    """
    
    nlay, nrow, ncol = thickness_array.shape
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
    - thickness_array: 3D array (nlay, nrow, ncol) containing thickness values for each layer.
    - epsilon: Thickness threshold under which cells are deactivated.
    Returns:
    - idomain: 3D array (nlay, nrow, ncol) with 1 (active) where thickness > 0 and 0 (inactive) otherwise.
    """
    # Set idomain to 1 (active) where thickness > epsilon, else 0 (inactive)
    idomain = np.where(thickness_array > epsilon, 1, 0)
    
    return idomain

def calculate_irch(idomain):
    """
    Calculate irch by summing idomain across layers and subtracting 
    from the total number of layers.
    
    Parameters:
    idomain (numpy.ndarray): A 3D numpy array of shape (nlay, nrow, ncol)
                             where nlay is the number of layers, and
                             nrow, ncol are the grid dimensions.

    Returns:
    numpy.ndarray: A 2D numpy array of shape (nrow, ncol) representing irch.
    """
    # Calculate the number of layers
    nlay = idomain.shape[0]
    
    # Sum idomain across the layers
    active_layers = np.sum(idomain, axis=0)
    
    # Calculate irch
    irch = nlay - active_layers
    
    return irch

def compute_recharge(irch, R):
    """
    Compute the recharge array based on irch and layer-specific recharge rates.

    Parameters:
    irch (numpy.ndarray): 2D array of shape (nrow, ncol), containing layer indices (0 to nlay-1).
    R (numpy.ndarray): 1D array of shape (nlay,), containing recharge rates for each layer.

    Returns:
    numpy.ndarray: 2D array of shape (nrow, ncol) with recharge values assigned based on irch.
    """
    # Initialize a recharge array with the same shape as irch
    rch = np.zeros_like(irch, dtype=float)

    # Assign recharge rates based on the layer index in irch
    for layer_idx, recharge_rate in enumerate(R):
        rch[irch == layer_idx] = recharge_rate

    return rch

def create_ztop_array(ztop, zbot):
    """
    Create a 3D start array with the first layer from ztop and subsequent layers from zbot.

    Parameters:
        ztop (ndarray): 2D array of shape (nrow, ncol), representing the top elevation.
        zbot (ndarray): 3D array of shape (nlay, nrow, ncol), representing the bottom elevations.

    Returns:
        ndarray: 3D start array of shape (nlay, nrow, ncol).
    """
    nlay, nrow, ncol = zbot.shape  # Get the dimensions from zbot
    start = np.zeros((nlay, nrow, ncol))  # Initialize the start array
    
    # Assign ztop to the first layer
    start[0, :, :] = ztop
    
    # Assign each subsequent layer from zbot
    for i in range(1, nlay):
        start[i, :, :] = zbot[i - 1, :, :]
    
    return start





