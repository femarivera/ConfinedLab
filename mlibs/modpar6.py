import numpy as np
from scipy.fft import fftn, ifftn, fftfreq

def moments_from_arithmetic_mean_variance(arith_mean, arith_var):
    """
    Helper function to convert between log-normal distribution parameters and 
    common summary stats (or knowledge) of hydraulic properties.

    Given arithmetic mean (m) and variance (v) of a lognormal random variable X,
    return geometric mean (GM = exp(mu)) and sill (variance of ln(X) : sigma^2).

    The aritmetic mean generally corresponds to the expected value of K in a heterogeneous aquifer.
    The variance could be expresed as percent variation relative to the mean, for example with a 
    coefficient of variation CV = sqrt(v)/m, then v = (CV*m)^2. 
    """
    import numpy as np

    m = arith_mean
    v = arith_var
    sigma2 = np.log(1.0 + v / m**2)
    mu = np.log(m) - 0.5 * sigma2
    geom_mean = np.exp(mu)
    sill = sigma2
    return geom_mean, sill, mu, sigma2

def moments_from_percentiles(x1, p1, x2, p2):
    """
    Helper function to convert between log-normal distribution parameters and 
    common summary stats (or knowledge) of hydraulic properties.

    Given two percentiles (x1 at p1, x2 at p2) of a lognormal variable,
    return geometric mean (exp(mu)) and sill (sigma^2).
    p1 and p2 within (0,1), e.g. 0.05 and 0.95.
    Very often a given hydraulic property of an aquifer is known as a broad range.
    One could assume that this range represents the 90% confidence interval (5th and 95th percentiles)
    of a log-normal distribution, and use this function to estimate the distribution parameters.
    """
    import numpy as np
    from math import log, sqrt
    from scipy.stats import norm

    z1 = norm.ppf(p1)
    z2 = norm.ppf(p2)
    if z2 == z1:
        raise ValueError("Percentiles must be distinct")
    sigma = (np.log(x2) - np.log(x1)) / (z2 - z1)
    mu = np.log(x1) - z1 * sigma
    geom_mean = np.exp(mu)
    sill = sigma**2
    return geom_mean, sill, mu, sigma

def generate_random_field(shape, variogram_type="exponential",
                          geom_mean=1e-4, sill=1.0, nugget=0.0, range_param=10.0,
                          drow=1.0, dcol=1.0, param_type="K", seed=None):
    """
    Generate a 2D log-normal random field with spatial correlation using a spectral (FFT-based) simulation method.

    This function creates a spatially correlated random field by filtering Gaussian white noise in the frequency domain
    according to a specified variogram model (exponential, gaussian, or spherical). The resulting field is then
    transformed to a log-normal distribution, commonly used for simulating heterogeneous properties such as hydraulic conductivity.

    Args:
        shape (tuple): Shape of the output field (nx, ny).
        variogram_type (str): Type of variogram/covariance model ("exponential", "gaussian", or "spherical").
        geom_mean (float): Geometric mean of the log-normal field.
        sill (float): Sill (variance) of the variogram. The log standard deviation of the field is the square root of the sill.
        nugget (float): Nugget effect (variance at zero distance).
        range_param (float): Correlation length (practical range) in METERS.
        drow (float): Grid spacing in the row direction (meters).
        dcol (float): Grid spacing in the column direction (meters).
        seed (int, optional): Random seed for reproducibility.

    Returns:
        np.ndarray: 2D array of shape (nx, ny) representing the log-normal random field with spatial correlation.
    """
    import numpy as np
    from scipy.fft import fftn, ifftn, fftfreq

    np.random.seed(seed)
    nx, ny = shape

    # Frequency grid (in 1/meters)
    kx = fftfreq(nx, d=drow).reshape(-1, 1)
    ky = fftfreq(ny, d=dcol).reshape(1, -1)
    k = np.sqrt(kx**2 + ky**2)

    # Power spectral density ~ FT of covariance
    if variogram_type == "exponential":
        spectrum = 1.0 / (1.0 + (2*np.pi*k*range_param)**2)**1.5
    elif variogram_type == "gaussian":
        spectrum = np.exp(-(np.pi*k*range_param)**2)
    elif variogram_type == "spherical":
        spectrum = 1.0 / (1.0 + (2*np.pi*k*range_param)**2)**2
    else:
        raise ValueError("Unsupported variogram type")
    spectrum[0, 0] = 1.0  # DC component

    # White noise in Fourier space
    noise = np.random.normal(size=(nx, ny)) + 1j*np.random.normal(size=(nx, ny))

    # Apply spectral filter
    Z_fft = noise * np.sqrt(spectrum)

    # Back to real space (Gaussian field, mean 0, std 1)
    Z = np.real(ifftn(Z_fft))
    Z = (Z - np.mean(Z)) / np.std(Z)

    # Apply sill (variance) and nugget
    Z = (np.sqrt(sill) * Z) + (np.sqrt(nugget) * np.random.normal(size=(nx, ny)))
    # Log-normal transformation
    field = geom_mean * np.exp(Z)

    # Apply parameter-specific constraints
    if param_type.lower() == "sy":
        # Specific yield is bounded between [0, 1], usually << 1
        field = np.clip(field, 0.001, 0.5)

    return field

def stack_fields_to_3D(field_list, nlay, nrow, ncol):
    """
    Stack a list of 2D fields into a 3D array of shape (nlay, nrow, ncol).

    Args:
        field_list (list): List of 2D arrays, each of shape (nrow, ncol).
        nlay (int): Number of layers (should match len(field_list)).
        nrow (int): Number of rows in each field.
        ncol (int): Number of columns in each field.

    Returns:
        np.ndarray: 3D array of shape (nlay, nrow, ncol).

    Raises:
        ValueError: If input dimensions do not match.
    """
    import numpy as np

    if not isinstance(field_list, (list, tuple)):
        raise ValueError("field_list must be a list or tuple of 2D arrays.")
    if len(field_list) != nlay:
        raise ValueError(f"field_list must have length nlay ({nlay}).")
    for i, arr in enumerate(field_list):
        arr = np.asarray(arr)
        if arr.shape != (nrow, ncol):
            raise ValueError(f"Field at index {i} has shape {arr.shape}, expected ({nrow}, {ncol}).")
    arr3d = np.stack([np.asarray(f) for f in field_list], axis=0)
    return arr3d