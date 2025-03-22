"""
Read EPMA maps from XMapTools and calculate P and T.
"""

import h5py
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


def load_biotite_map(map_file: str,
                     element_order: list,
                     # parent: Path = Path(Path.cwd(), "biotite_maps"),
                     filter_range_Si: tuple = (2.0, 3.0),
                     fixed_K: float = 1.0,
                     fixed_Na: float = 0.0):
    """Load biotite compositional maps (EPMA maps) from XmapTools hdf5 export format.
    Biotite compositional map is flattend to a list of compositional vectors. Compositional vectors follow specified element ordering.
    Generates an empty template map array and corresponding masking indices for all "valid" phase pixels.

    Args:
        map_file (str): filename of the biotite compositional map. Must be hdf5.
        element_order (list): Elemnt in the order corresponding to the input vector of the Thermobarometer-ANN.
        parent (Path, optional): Path to parent folder of map_files. Defaults to Path(".", "src", "benchmark_thermobarometer", "biotite_maps").
        filter_range_Si (tuple, optional): Filtering condition for mixing boundary pixels based on SI [apfu] in biotite:
                                           (lower, upper)-condition. Defaults to (2.0, 3.0).
        fixed_K (float, optional): Constant K [apfu] in biotite. Defaults to 1.0.
        fixed_Na (float, optional): Constant Na [apfu] in biotite. Defaults to 0.0.

    Returns:
        biotite (np.ndarray): Array of compositional biotite vectory.
        map_empty (np.ndarray): Map template (filled with zeros) with shape of original compositional map.
        idx_phase_pixel (tuple): Indices (["X-coordinates"], ["Y-coordinates"]) of filtered biotite pixels.
    """

    file_path = map_file
    with h5py.File(file_path, "r") as hdf_file:
        # element_list = list(hdf_file["ElementList"].asstr())
        Al = np.array(hdf_file["CData"]["Al"])
        Ca = np.array(hdf_file["CData"]["Ca"])
        Fe = np.array(hdf_file["CData"]["Fe"])
        K = np.array(hdf_file["CData"]["K"])
        Mg = np.array(hdf_file["CData"]["Mg"])
        Mn = np.array(hdf_file["CData"]["Mn"])
        Na = np.array(hdf_file["CData"]["Na"])
        Si = np.array(hdf_file["CData"]["Si"])
        Ti = np.array(hdf_file["CData"]["Ti"])

    # filtering values to get rid of boundary mixing effects
    # adjust measured data for unacounted compositional effects in the thermodynamic model (e.g. K/Na variation)
    idx_phase_pixel = np.where(np.logical_and(Si >= filter_range_Si[0], Si <= filter_range_Si[1]))

    K[:] = fixed_K
    Na[:] = fixed_Na

    # filter all maps with phase indx
    Al_filtered = Al[idx_phase_pixel]
    Ca_filtered = Ca[idx_phase_pixel]
    Fe_filtered = Fe[idx_phase_pixel]
    K_filtered = K[idx_phase_pixel]
    Mg_filtered = Mg[idx_phase_pixel]
    Mn_filtered = Mn[idx_phase_pixel]
    Na_filtered = Na[idx_phase_pixel]
    Si_filtered = Si[idx_phase_pixel]
    Ti_filtered = Ti[idx_phase_pixel]

    # create an empty temperature map, with the correct shape
    map_shape = np.shape(Si)
    map_empty = np.zeros(map_shape)

    # pack all element vectors into dict, so that they can be acessed in the desired order
    elements_dict = {"Al": Al_filtered,
                     "Ca": Ca_filtered,
                     "Fe": Fe_filtered,
                     "K":  K_filtered,
                     "Mg": Mg_filtered,
                     "Mn": Mn_filtered,
                     "Na": Na_filtered,
                     "Si": Si_filtered,
                     "Ti": Ti_filtered}

    e_in_bt = [elements_dict[key] for key in element_order]

    biotite_map: np.ndarray = np.c_[e_in_bt].T

    return biotite_map, map_empty, idx_phase_pixel


def predict_temperature_map(model: tf.keras.Model, biotite_map: np.ndarray, map_empty: np.ndarray, idx_phase_pixel: tuple, model_type: str = "thermometer"):
    """_summary_

    Args:
        model (tf.keras.Model): _description_
        biotite_map (np.ndarray): _description_
        map_empty (np.ndarray): _description_
        idx_phase_pixel (tuple): _description_
        model_type (str): "thermometer" or "thermobarometer", parameter that tells if models predicts T or P - T.

    Returns:
        _type_: _description_
    """
    if model_type == "thermometer":
        temperature = model.predict(biotite_map).flatten()

    elif model_type == "thermobarometer":
        temperature = model.predict(biotite_map)[:, 1]

    map_empty[idx_phase_pixel] = temperature
    T_map = map_empty

    return T_map


def predict_temperature_Henry2005(biotite_map: np.ndarray, map_empty: np.ndarray, idx_phase_pixel: tuple):
    """Calculate temperature from biotite compositional map using the thermometer of Henry et al. 2005.
    IMPORTANT: This thermomater is only valid for biotite compositions with 22Ox basis. (Adjustement of Fe, Mg, Ti to 22Ox basis is done in the function.)

    Args:
        model (tf.keras.Model): _description_
        biotite_map (np.ndarray): _description_
        map_empty (np.ndarray): _description_

    Returns:
        _type_: _description_
    """

    a = -2.3594
    b = 4.6482e-9
    c = -1.7283

    Fe = biotite_map[:, 0].flatten()
    Mg = biotite_map[:, 1].flatten()
    Ti = biotite_map[:, 2].flatten()
    # adjust to 22Ox basis
    Fe = Fe * 2
    Mg = Mg * 2
    Ti = Ti * 2

    idx_notZero = np.logical_and(np.logical_and(Fe > 0, Mg > 0), Ti > 0)
    Fe = Fe[idx_notZero]
    Mg = Mg[idx_notZero]
    Ti = Ti[idx_notZero]

    X_Mg = Mg / (Mg + Fe)

    temperature = ((np.log(Ti)-a-(c*(X_Mg**3)))/b)**0.333
    temperature = temperature[~np.isnan(temperature)]

    return temperature


def predict_temperature_Wu2015(biotite_map: np.ndarray, map_empty: np.ndarray, idx_phase_pixel: tuple, P_estimate_GPa: float):
    """Calculate temperature from biotite compositional map using the thermometer of Wu et al. 2015.
    IMPORTANT: This thermomater is only valid for biotite compositions with 11Ox basis. (Data should be in 11Ox basis already.)

    Args:
        biotite_map (np.ndarray): _description_
        map_empty (np.ndarray): _description_
        idx_phase_pixel (tuple): _description_
        P_estimate_GPa (float): _description_

    Returns:
        _type_: _description_
    """

    Fe = biotite_map[:, 0].flatten()
    Mg = biotite_map[:, 1].flatten()
    Ti = biotite_map[:, 2].flatten()
    Al = biotite_map[:, 3].flatten()
    Si = biotite_map[:, 4].flatten()

    idx_notZero = np.logical_and(np.logical_and(Fe > 0, Mg > 0), Ti > 0)
    Fe = Fe[idx_notZero]
    Mg = Mg[idx_notZero]
    Ti = Ti[idx_notZero]
    Al = Al[idx_notZero]
    Si = Si[idx_notZero]

    # estimate Al(VI), by assuming Al(IV) = 4 - Si
    Al_VI = Al - (4 - Si)

    if np.any(Al_VI < 0):
        Al_VI[Al_VI < 0] = 0
        print("Al(VI) < 0, set to 0")

    # calculate X_j as j/(Fe+Mg+Al(VI)+Ti), after Wu et al. 2015
    X_Fe = Fe / (Fe + Mg + Al_VI + Ti)
    X_Mg = Mg / (Fe + Mg + Al_VI + Ti)
    X_Ti = Ti / (Fe + Mg + Al_VI + Ti)

    # Wu et al. 2015
    a = 6.313
    b = 0.224
    c = -0.288
    d = -0.449
    e = 0.15

    log_T = a + b * np.log(X_Ti) + c * np.log(X_Fe) + d * np.log(X_Mg) + e * P_estimate_GPa

    temperature = np.exp(log_T)

    return temperature


def plot_map(map: np.ndarray, variable_name: str, cmap: str = "magma", range: tuple | None = None):
    """Plotting function for pressure and temperature maps.

    Args:
        map (np.ndarray): A array of the map (temperature or pressure).
        cmap (str, optional): A matplotlib colormap name. Defaults to "magma".
        range (tuple | None, optional): (lower, upper)-values for color-coding of pressure or temperature.
        If None then the 95% intervall is taken. Defaults to None.
    """
    # auto range calculation:  Exclude outliers (artifact pixels)
    if range is None:
        percentiles = (0.025, 0.975)
        range = np.quantile(map[map != 0], q=percentiles)

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    # mask map with zeros
    map = np.ma.masked_where(map == 0, map)

    map = ax.imshow(map, cmap=cmap, vmin=range[0], vmax=range[1])
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.05)

    ax.set_aspect("equal")
    ax.set_xlabel("x-coordinate")
    ax.set_ylabel("y-coordinate")

    fig.colorbar(map, cax=cax, label=variable_name)

    return fig, ax


def plot_distribution(map: np.ndarray, variable_name: str, bins: int, x_lims: tuple | None = None):
    # filter out zeros and outliers (1% of all values)!
    map = map[map != 0]
    percentile = np.quantile(map, q=(0.005, 0.995))
    map = map[np.logical_and(map >= percentile[0], map <= percentile[1])]

    hist, bin_edges = np.histogram(map, bins=bins)
    # normalse hist with total counts --> probality
    hist = hist / np.size(map)

    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    ax.stairs(hist, bin_edges, fill=True, color="darkgray")
    ax.set_xlabel(variable_name)
    ax.set_ylabel("p(X) in bin")

    if x_lims is not None:
        ax.set_xlim(x_lims)

    mode = (bin_edges[np.argmax(hist)] + bin_edges[np.argmax(hist) + 1]) / 2
    return mode, fig, ax
