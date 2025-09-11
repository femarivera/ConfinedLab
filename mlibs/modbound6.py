import flopy
import numpy as np

def create_riv_spd(cells, drow, dcol, ztop, thickness_array, k_array, riverbed_thickness=1, river_width=1, a=0.1, b=0.2, conc=None):
    """
    Create river boundary condition parameters for multiple specified cells.

    Parameters:
    - cells (list of tuples): A list of (k, i, j) tuples specifying the layer, row, and column indices of the cells.
    - drow (float): Row length (horizontal discretization) for conductance calculation (river length).
    - dcol (float): Column length (horizontal discretization) for conductance calculation.
    - ztop (3D array): Top elevation of each cell (shape = nlay x nrow x ncol).
    - thickness_array (3D array): Thickness of each cell.
    - k_array (1D array): Hydraulic conductivity for each layer (shape = nlay).
    - riverbed_thickness (float, optional): Thickness of the riverbed (default = 1).
    - river_width (float, optional): Width of the river (default = 1).
    - a (float, optional): Percentage of thickness used to calculate river stage (default = 0.1).
    - b (float, optional): Percentage of thickness used to calculate river bottom (default = 0.2).
    - conc (float or None, optional): River concentration (if applicable). Default is None.

    Returns:
    - riv_spd (dict): Stress period data for the river boundary condition.
    """
    riv_spd = {}
    riv_entries = []

    for k, i, j in cells:
        riv_stage = ztop[k, i, j] - (a * thickness_array[k, i , j])
        riv_bottom = riv_stage - 1  # or ztop[k, i, j] - (b * base_thicknesses[k])
        riv_cond = (k_array[k] * drow * river_width) / (10 * riverbed_thickness)

        if conc is not None:
            riv_entries.append((k, i, j, riv_stage, riv_cond, riv_bottom, conc))
        else:
            riv_entries.append((k, i, j, riv_stage, riv_cond, riv_bottom))

    riv_spd[0] = riv_entries
    return riv_spd

def create_ghb_spd(cells, drow, dcol, ztop, thickness_array, k_array, ghb_distance=1, a=0.1):
    """
    Create ghb boundary condition parameters for multiple specified cells.

    Parameters:
    - cells (list of tuples): A list of (k, i, j) tuples specifying the layer, row, and column indices of the cells.
    - drow (float): Row length (horizontal discretization) for conductance calculation.
    - dcol (float): Column length (horizontal discretization) for conductance calculation.
    - ztop (3D array): Top elevation of each cell (shape = nlay x nrow x ncol).
    - thickness_array (3D array): Thickness of each cell (shape = nlay x nrow x ncol).
    - k_array (1D array): Hydraulic conductivity for each layer (shape = nlay).
    - ghb_distance (float, optional): Distance to ghb (default = 1).
    - a (float, optional): Percentage of thickness used to calculate ghb (default = 0.1).
    

    Returns:
    - ghb_spd (dict): Stress period data for the river boundary condition.
    """
    # Initialize an empty dictionary for the river stress period data
    ghb_spd = {}

    # List to hold all river data entries
    ghb_entries = []

    # Iterate over each cell specified in the input list
    for k, i, j in cells:
        # Compute river stage and bottom for the current cell
        ghb = ztop[k, i, j] - (a * thickness_array[k, i, j])
        
        # Compute river conductance
        ghb_cond = k_array[k] * drow * thickness_array[k, i, j] / ghb_distance

        # Append the river entry for the current cell
        ghb_entries.append((k, i, j, ghb, ghb_cond))

    # Assign the river entries to the first stress period (0)
    ghb_spd[0] = ghb_entries

    return ghb_spd

def create_drn_spd(cells, drow, dcol, ztop, thickness_array, k_array, drn_width=1, a=0.1):
    """
    Create drain boundary condition parameters for multiple specified cells.

    Parameters:
    - cells (list of tuples): A list of (k, i, j) tuples specifying the layer, row, and column indices of the cells.
    - drow (float): Row length (horizontal discretization) for conductance calculation (drn length).
    - river_width (float): Column length (horizontal discretization) for conductance calculation.
    - ztop (3D array): Top elevation of each cell (shape = nlay x nrow x ncol).
    - thickness_array (3D array): Thickness of each cell.
    - k_array (1D array): Hydraulic conductivity for each layer (shape = nlay).
    - a (float, optional): Percentage of thickness used to calculate river stage (default = 0.1).
    - b (float, optional): Percentage of thickness used to calculate river bottom (default = 0.2).

    Returns:
    - drn_spd (dict): Stress period data for the river boundary condition.
    """
    # Initialize an empty dictionary for the river stress period data
    drn_spd = {}

    # List to hold all river data entries
    drn_entries = []

    # Iterate over each cell specified in the input list
    for k, i, j in cells:
        # Compute river stage and bottom for the current cell
        drn_elev = ztop[i, j] - (a * thickness_array[k, i, j])
        #riv_bottom = ztop[k, i, j] - (b * base_thicknesses[k])

        # Compute river conductance
        drn_cond = (k_array[k] * drow * drn_width) / (10)

        # Append the river entry for the current cell
        drn_entries.append((k, i, j, drn_elev, drn_cond))

    # Assign the river entries to the first stress period (0)
    drn_spd[0] = drn_entries

    return drn_spd

def extract_active_cells_row(irch: np.ndarray, idomain: np.ndarray):
    """
    Extract active cell indices (k, i, j) from irch and idomain arrays.
    Parameters:
        irch (np.ndarray): 2D array of shape (1, ncol) with layer indices (0 to nlay-1).
        idomain (np.ndarray): 3D array of shape (nlay, 1, ncol) with values 1 (active) or 0 (inactive).
    Returns:
        List[Tuple[int, int, int]]: List of (k, i, j) indices for active cells.
    """
    nrow = irch.shape[0]
    assert nrow == 1, "irch must have only one row"
    ncol = irch.shape[1]
    i = 0  # only one row
    j_vals = np.arange(ncol-1)
    k_vals = irch[i, j_vals]  # extract layer numbers from irch
    # Check if each (k, 0, j) cell is active
    active_mask = idomain[k_vals, i, j_vals] == 1
    # Build list of active (k, i, j) tuples
    active_cells = [(int(k), i, int(j)) for k, j, active in zip(k_vals, j_vals, active_mask) if active]
    return active_cells

def extract_active_cells(irch: np.ndarray, idomain: np.ndarray):

    import numpy as np

    """
    Extract active cell indices (k, i, j) from irch and idomain arrays.

    Parameters:
        irch (np.ndarray): 2D array of shape (nrow, ncol) with layer indices (0 to nlay-1).
        idomain (np.ndarray): 3D array of shape (nlay, nrow, ncol) with values 1 (active) or 0 (inactive).
    Returns:
        List[Tuple[int, int, int]]: List of (k, i, j) indices for active cells.
    """
    nrow, ncol = irch.shape
    active_cells = []

    for i in range(nrow):
        for j in range(ncol):
            k = int(irch[i, j])  # layer index
            if idomain[k, i, j] == 1:  # check if active
                active_cells.append((k, i, j))

    return active_cells

def extract_active_cells_n_row(irch: np.ndarray, idomain: np.ndarray, n: int):
    """
    Extract active cell indices (k, i, j) from irch and idomain arrays, checking every n-th column.
    Parameters:
        irch (np.ndarray): 2D array of shape (1, ncol) with layer indices (0 to nlay-1).
        idomain (np.ndarray): 3D array of shape (nlay, 1, ncol) with values 1 (active) or 0 (inactive).
        n (int): Step size for column indexing. The function will check the 1st, n-th, 2n-th, etc., columns.
    Returns:
        List[Tuple[int, int, int]]: List of (k, i, j) indices for active cells checked every n-th column.
    """
    nrow = irch.shape[0]
    assert nrow == 1, "irch must have only one row"
    ncol = irch.shape[1]
    i = 0  # only one row
    j_vals = np.arange(0, ncol - 1, n)  # select every n-th column, excluding the last one
    k_vals = irch[i, j_vals]  # extract layer numbers for the selected columns
    # Check if each (k, 0, j) cell is active
    active_mask = idomain[k_vals, i, j_vals] == 1
    # Build list of active (k, i, j) tuples
    active_cells = [(int(k), i, int(j)) for k, j, active in zip(k_vals, j_vals, active_mask) if active]
    return active_cells

def extract_active_cells_n(irch, idomain, n):
    
    import numpy as np  
    """
    Extract active cell indices (k, i, j) from irch and idomain arrays,
    checking every n-th column.

    Parameters:
        irch (np.ndarray): 2D array of shape (nrow, ncol) with layer indices (0 to nlay-1).
        idomain (np.ndarray): 3D array of shape (nlay, nrow, ncol) with values 1 (active) or 0 (inactive).
        n (int): Step size for column indexing. The function will check the 1st, n-th, 2n-th, etc., columns.

    Returns:
        list of (k, i, j) indices for active cells checked every n-th column.
    """
    nrow, ncol = irch.shape
    active_cells = []

    for i in range(nrow):
        j_vals = np.arange(0, ncol - 1, n)  # select every n-th column, excluding the last one
        k_vals = irch[i, j_vals]            # extract layer numbers for selected columns
        active_mask = idomain[k_vals, i, j_vals] == 1
        for k, j, active in zip(k_vals, j_vals, active_mask):
            if active:
                active_cells.append((int(k), i, int(j)))

    return active_cells

def extract_active_cells_range(irch, idomain, row_start, row_end, col_start, col_end):
    import numpy as np
    """
    Extract active cell indices (k, i, j) from irch and idomain arrays,
    within a specified submatrix defined by row and column ranges.

    Parameters:
        irch (np.ndarray): 2D array of shape (nrow, ncol) with layer indices (0 to nlay-1).
        idomain (np.ndarray): 3D array of shape (nlay, nrow, ncol) with values 1 (active) or 0 (inactive).
        row_start (int): Starting row index (inclusive).
        row_end (int): Ending row index (inclusive).
        col_start (int): Starting column index (inclusive).
        col_end (int): Ending column index (inclusive).

    Returns:
        list of (k, i, j) indices for active cells within the specified submatrix.
    """
    nrow, ncol = irch.shape
    # Validate indices
    assert 0 <= row_start <= row_end < nrow, f"Row range {row_start}-{row_end} out of bounds (0–{nrow-1})"
    assert 0 <= col_start <= col_end < ncol, f"Column range {col_start}-{col_end} out of bounds (0–{ncol-1})"

    active_cells = []

    for i in range(row_start, row_end + 1):
        j_vals = np.arange(col_start, col_end + 1)
        k_vals = irch[i, j_vals]
        active_mask = idomain[k_vals, i, j_vals] == 1

        for k, j, active in zip(k_vals, j_vals, active_mask):
            if active:
                active_cells.append((int(k), i, int(j)))

    return active_cells

