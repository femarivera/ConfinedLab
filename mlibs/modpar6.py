import numpy as np
from scipy.fft import fftn, ifftn, fftfreq

def generate_random_field(shape, variogram_type="exponential",
                          geom_mean=1e-4, sigma_ln=1.0, range_param=10.0, seed=None):
    """Generate a 2D log-normal random field with spatial correlation."""
    np.random.seed(seed)
    nx, ny = shape

    # Frequency grid
    kx = fftfreq(nx).reshape(-1, 1)
    ky = fftfreq(ny).reshape(1, -1)
    k = np.sqrt(kx**2 + ky**2)

    # Power spectral density
    if variogram_type == "exponential":
        spectrum = 1.0 / (1.0 + (2*np.pi*k*range_param)**2)**1.5
    elif variogram_type == "gaussian":
        spectrum = np.exp(-(np.pi*k*range_param)**2)
    elif variogram_type == "spherical":
        spectrum = 1.0 / (1.0 + (2*np.pi*k*range_param)**2)**2
    else:
        raise ValueError("Unsupported variogram type")
    spectrum[0, 0] = 1.0

    # Gaussian noise in Fourier space
    noise = np.random.normal(size=(nx, ny)) + 1j*np.random.normal(size=(nx, ny))
    Z_fft = noise * np.sqrt(spectrum)

    # Back to real space, standard normal
    Z = np.real(ifftn(Z_fft))
    Z = (Z - np.mean(Z)) / np.std(Z)

    # Log-normal transformation
    field = geom_mean * np.exp(sigma_ln * Z)
    return field

def generate_multilayer_K(shape=(100, 100), seed=42):
    """
    Generate a 5-layer heterogeneous K field for an aquifer–aquitard system.
    Returns array of shape (5, nx, ny).
    """
    layers = []

    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    # 1. Unconfined aquifer
    layers.append(generate_random_field(shape, "exponential",
                                        geom_mean=1e2, sigma_ln=1,
                                        range_param=20, seed=rng.integers(1e6)))

    # 2. Aquitard
    layers.append(generate_random_field(shape, "gaussian",
                                        geom_mean=1e-4, sigma_ln=0.8,
                                        range_param=10, seed=rng.integers(1e6)))

    # 3. Confined aquifer
    layers.append(generate_random_field(shape, "exponential",
                                        geom_mean=10, sigma_ln=0.5,
                                        range_param=25, seed=rng.integers(1e6)))

    # 4. Aquitard
    layers.append(generate_random_field(shape, "gaussian",
                                        geom_mean=1e-4, sigma_ln=0.8,
                                        range_param=8, seed=rng.integers(1e6)))

    # 5. Confined aquifer
    layers.append(generate_random_field(shape, "exponential",
                                        geom_mean=10, sigma_ln=0.5,
                                        range_param=30, seed=rng.integers(1e6)))

    return np.array(layers)

def generate_random_field_param(shape, variogram_type="exponential",
                                geom_mean=1e-4, sigma_ln=1.0, range_param=10.0,
                                param_type="K", seed=None):
    """
    Generate a 2D heterogeneous field (K, Sy, or Ss) with spatial correlation.

    Parameters
    ----------
    shape : tuple
        Grid shape (nx, ny).
    variogram_type : str
        "exponential", "gaussian", or "spherical".
    geom_mean : float
        Geometric mean of the parameter.
        - For K: m/s
        - For Sy: dimensionless
        - For Ss: 1/m
    sigma_ln : float
        Standard deviation of log(parameter).
    range_param : float
        Correlation length (practical range).
    param_type : str
        "K", "Sy", or "Ss".
    seed : int, optional
        Random seed.

    Returns
    -------
    field : ndarray
        2D parameter field.
    """
    np.random.seed(seed)
    nx, ny = shape

    # Frequency grid
    kx = fftfreq(nx).reshape(-1, 1)
    ky = fftfreq(ny).reshape(1, -1)
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

    # Apply filter
    Z_fft = noise * np.sqrt(spectrum)

    # Back to real space
    Z = np.real(ifftn(Z_fft))
    Z = (Z - np.mean(Z)) / np.std(Z)  # standard normal field

    # Transform based on parameter type
    field = geom_mean * np.exp(sigma_ln * Z)

    if param_type.lower() == "sy":
        # Truncate to physical range [0, 1]
        field = np.clip(field, 0.001, 0.5)

    return field
