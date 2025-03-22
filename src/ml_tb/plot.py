import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
from pathlib import Path
from math import floor

from ml_tb.normalisation import MinMaxScaler

"""
PLOTTING FUNCS: TRAINING CURVES AND PREDICTION VS TRUTH

These plotting functions are used to visualize the training curves and the prediction vs truth of the ANN model.
Used in the notebooks fopr hyperparameter tuning and training of the models.
"""


# from https://github.com/tensorflow/docs/blob/master/tools/tensorflow_docs/plots/__init__.py
def _smooth(values, std):
    """Smooths a list of values by convolving with a Gaussian distribution.

    Assumes equal spacing.

    Args:
      values: A 1D array of values to smooth.
      std: The standard deviation of the Gaussian distribution. The units are
        array elements.

    Returns:
      The smoothed array.
    """
    width = std * 4
    x = np.linspace(-width, width, min(2 * width + 1, len(values)))
    kernel = np.exp(-(x / 5)**2)

    values = np.array(values)
    weights = np.ones_like(values)

    smoothed_values = np.convolve(values, kernel, mode='same')
    smoothed_weights = np.convolve(weights, kernel, mode='same')

    return smoothed_values / smoothed_weights


def plot_training_curve(logfile: str | Path, color: str, label: str, smooth_std=10, log_scale=True, xlims=[None, None, None], ylims=[None, None, None]):
    history = pd.read_csv(Path(logfile))

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))

    axs[0].plot(history['epoch'], _smooth(history['loss'], smooth_std), color=color, label=label)
    axs[0].plot(history['epoch'], _smooth(history['val_loss'], smooth_std), color=color, linestyle='--', label=label + ' (val)')

    axs[1].plot(history['epoch'], _smooth(history['RMSE_T'], smooth_std), color=color, label=label)
    axs[1].plot(history['epoch'], _smooth(history['val_RMSE_T'], smooth_std), color=color, linestyle='--', label=label + ' (val)')

    axs[2].plot(history['epoch'], _smooth(history['RMSE_P'], smooth_std), color=color, label=label)
    axs[2].plot(history['epoch'], _smooth(history['val_RMSE_P'], smooth_std), color=color, linestyle='--', label=label + ' (val)')

    if log_scale:
        axs[0].set_xscale('log')
        axs[1].set_xscale('log')
        axs[2].set_xscale('log')

    for i, xlim in enumerate(xlims):
        if xlim is not None:
            axs[i].set_xlim(xlim)

    for i, ylim in enumerate(ylims):
        if ylim is not None:
            axs[i].set_ylim(ylim)

    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Loss')

    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Temperature RMSE [K]')

    axs[2].set_xlabel('Epoch')
    axs[2].set_ylabel('Pressure RMSE [bar]')

    axs[0].legend()

    fig.tight_layout()

    return fig, axs


def plot_multiple_training_curves(logfiles: list, colors, label: str, smooth_std=10,
                                  log_scale=True, xlims=[None, None, None], ylims=[None, None, None],
                                  T_only=False):

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))

    for logfile, color in zip(logfiles, colors):
        history = pd.read_csv(Path(logfile))

        axs[0].plot(history['epoch'], _smooth(history['loss'], smooth_std), color=color, label=label)
        axs[0].plot(history['epoch'], _smooth(history['val_loss'], smooth_std), color=color, linestyle='--', label=label + ' (val)')

        axs[1].plot(history['epoch'], _smooth(history['RMSE_T'], smooth_std), color=color, label=label)
        axs[1].plot(history['epoch'], _smooth(history['val_RMSE_T'], smooth_std), color=color, linestyle='--', label=label + ' (val)')

        if T_only is False:
            axs[2].plot(history['epoch'], _smooth(history['RMSE_P'], smooth_std), color=color, label=label)
            axs[2].plot(history['epoch'], _smooth(history['val_RMSE_P'], smooth_std), color=color, linestyle='--', label=label + ' (val)')

    if log_scale:
        axs[0].set_xscale('log')
        axs[1].set_xscale('log')
        axs[2].set_xscale('log')

    for i, xlim in enumerate(xlims):
        if xlim is not None:
            axs[i].set_xlim(xlim)

    for i, ylim in enumerate(ylims):
        if ylim is not None:
            axs[i].set_ylim(ylim)

    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Loss')

    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Temperature RMSE [K]')

    axs[2].set_xlabel('Epoch')
    axs[2].set_ylabel('Pressure RMSE [bar]')

    axs[0].legend()

    fig.tight_layout()

    return fig, axs


def prediction_vs_truth(y_true, y_pred, color: str, limits=[None, None]):
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))

    axs[0].plot([1000, 10500], [1000, 10500], color="grey", linestyle="--", lw=1.5)
    axs[0].scatter(y_true[:, 0], y_pred[:, 0], color=color, s=3)

    axs[0].set_aspect('equal')
    axs[0].set_xlabel('True pressure [bar]')
    axs[0].set_ylabel('Predicted pressure [bar]')

    axs[1].plot([380, 920], [380, 920], color="grey", linestyle="--", lw=1.5)
    axs[1].scatter(y_true[:, 1], y_pred[:, 1], color=color, s=3)

    axs[1].set_aspect('equal')
    axs[1].set_xlabel('True temperature [°C]')
    axs[1].set_ylabel('Predicted temperature [°C]')

    for i, lim in enumerate(limits):
        if lim is not None:
            axs[i].set_xlim(lim)
            axs[i].set_ylim(lim)

    fig.tight_layout()

    return fig, axs


def plot_hyperparameter_test(log_files, colors, labels, smooth_std=10, xlims=[None, None, None], ylims=[None, (2, 10), (50, 800)], log_scale=True, legend=True):
    """Plots the results of a hyperparameter test.

    Args:
      log_files: A list of file paths to the log files.
      colors: A list of colors to use for each log file. Create with sns.color_palette().
      labels: A list of labels to use for each log file.
      smooth_std: The standard deviation of the Gaussian distribution used to
        smooth the results. In epoch units.
    """
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))

    for log_file, color, label in zip(log_files, colors, labels):
        history = pd.read_csv(log_file)

        axs[0].plot(history['epoch'], _smooth(history['loss'], smooth_std), color=color, label=label)
        axs[0].plot(history['epoch'], _smooth(history['val_loss'], smooth_std), color=color, linestyle='--', label=label + ' (val)')

        axs[1].plot(history['epoch'], _smooth(history['RMSE_T'], smooth_std), color=color, label=label)
        axs[1].plot(history['epoch'], _smooth(history['val_RMSE_T'], smooth_std), color=color, linestyle='--', label=label + ' (val)')

        axs[2].plot(history['epoch'], _smooth(history['RMSE_P'], smooth_std), color=color, label=label)
        axs[2].plot(history['epoch'], _smooth(history['val_RMSE_P'], smooth_std), color=color, linestyle='--', label=label + ' (val)')

    if log_scale:
        axs[0].set_xscale('log')
        axs[1].set_xscale('log')
        axs[2].set_xscale('log')

    for i, xlim in enumerate(xlims):
        if xlim is not None:
            axs[i].set_xlim(xlim)

    for i, ylim in enumerate(ylims):
        if ylim is not None:
            axs[i].set_ylim(ylim)

    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Loss')

    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Temperature RMSE [K]')

    axs[2].set_xlabel('Epoch')
    axs[2].set_ylabel('Pressure RMSE [bar]')

    if legend:
        axs[0].legend()

    fig.tight_layout()

    return fig, axs


def plot_hyperparameter_test_thermometer(log_files, colors, labels, smooth_std=10, xlims=[None, None], ylims=[None, (2, 10)], log_scale=True, legend=True):
    """Plots the results of a hyperparameter test. FOR THE Ti-XMg THERMOMETER, ONLY RMSE T IS PLOTTED.

    Args:
      log_files: A list of file paths to the log files.
      colors: A list of colors to use for each log file. Create with sns.color_palette().
      labels: A list of labels to use for each log file.
      smooth_std: The standard deviation of the Gaussian distribution used to
        smooth the results. In epoch units.
    """
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))

    for log_file, color, label in zip(log_files, colors, labels):
        history = pd.read_csv(log_file)

        axs[0].plot(history['epoch'], _smooth(history['loss'], smooth_std), color=color, label=label)
        axs[0].plot(history['epoch'], _smooth(history['val_loss'], smooth_std), color=color, linestyle='--', label=label + ' (val)')

        axs[1].plot(history['epoch'], _smooth(history['RMSE_T'], smooth_std), color=color, label=label)
        axs[1].plot(history['epoch'], _smooth(history['val_RMSE_T'], smooth_std), color=color, linestyle='--', label=label + ' (val)')

    if log_scale:
        axs[0].set_xscale('log')
        axs[1].set_xscale('log')

    for i, xlim in enumerate(xlims):
        if xlim is not None:
            axs[i].set_xlim(xlim)

    for i, ylim in enumerate(ylims):
        if ylim is not None:
            axs[i].set_ylim(ylim)

    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Loss')

    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Temperature RMSE [K]')

    if legend:
        axs[0].legend()

    fig.tight_layout()

    return fig, axs


"""
PLOTTING FUNCS: SYSTEMATIC PERFORMANCE ANALYSIS

These plotting functions are used to visualize the performance of the ANN model on the test set and on the sequence test set.
"""

COLOR_BIN_DICT = {"1a":
                  {"Chl": 1,
                   "Bt": 2,
                   "Crd": 4,
                   "Crd-Kfs": 6,
                   "Kfs-And-Crd": 7,
                   "Spl": 9,
                   "Sil/Opx": 10},
                  "1b":
                      {"Chl": 1,
                       "Bt": 2,
                       "Crd": 4,
                       "Crd-Kfs": 6,
                       "Kfs-And-Crd": 7,
                       "Kfs-Sil-Crd-(And)": 8,
                       "Spl": 9,
                       "Grt/Opx": 10},
                  "1c":
                      {"Chl": 1,
                       "Bt": 2,
                       "Crd/And": 3,
                       "Crd-And": 5,
                       "Kfs-And-Crd": 7,
                       "Kfs-Sil-Crd-(And)": 8,
                       "Spl": 9,
                       "Grt/Opx": 10},
                  "2a":
                      {"Chl": 1,
                       "Bt": 2,
                       "Crd/And": 3,
                       "Crd-And": 4,
                       "Sil-(Crd-And)": 6,
                       "Kfs-Sil-Crd-(And)": 8,
                       "Kfs-Crd±Grt": 9,
                       "Opx": 10},
                  "2b":
                      {"Chl": 1,
                       "Bt": 2,
                       "Grt": 3,
                       "And": 4,
                       "Sil-(And)": 6,
                       "Kfs-Sil-(And)": 8,
                       "Kfs-Crd±Grt": 9,
                       "Opx": 10},
                  "2c":
                      {"Chl": 1,
                       "Bt": 2,
                       "Grt": 3,
                       "St": 4,
                       "St-And": 5,
                       "Sil-(St-And)": 6,
                       "Sil": 7,
                       "Kfs-Sil": 8,
                       "Kfs-Crd": 9,
                       "Opx": 10},
                  "3":
                      {"Chl": 1,
                       "Bt": 2,
                       "Grt": 3,
                       "St": 4,
                       "Sil-(St)": 6,
                       "Sil": 7,
                       "Kfs-Sil": 8,
                       "Kfs-Crd": 9},
                  "4a":
                      {"Chl": 1,
                       "Bt": 2,
                       "Grt": 3,
                       "St": 4,
                       "St-Ky": 5,
                       "Sil-(St-Ky)": 6,
                       "Sil": 7,
                       "Kfs-Sil": 8,
                       "Kfs-Sil±Crd": 9},
                  "4a/4b/5":
                      {"Chl": 1,
                       "Bt": 2,
                       "Grt": 3,
                       "St": 4,
                       "St-Ky": 5,
                       "Sil-(St-Ky)": 7,
                       "Kfs-Sil": 8,
                       "Kfs-Sil±Crd": 9},
                  "4b":
                      {"Chl": 1,
                       "Bt": 2,
                       "Grt": 3,
                       "St": 4,
                       "St-Ky": 5,
                       "Ky": 7,
                       "Sil-(Ky)": 8,
                       "Kfs-Sil": 9},
                  "5":
                      {"Chl": 1,
                       "Bt": 2,
                       "Grt": 3,
                       "St": 4,
                       "Ky-(St)": 5,
                       "Ky": 7,
                       "Kfs-Sil": 9}}


def zone_to_number(zone, mas):
    if zone == "Chl":
        return 1
    elif zone == "Bt":
        return 2
    elif zone == "Crd/And" or zone == "Grt" or zone == "±Grt":
        return 3
    elif zone == "Crd" or (zone == "Crd-And" and mas == "1b") or \
            (zone == "St" and (mas == "2c" or mas == "4a" or mas == "4a/4b/5" or mas == "4b" or mas == "5")):
        return 4
    elif (zone == "Crd-And" and mas == "2a") or (zone == "And-St/Crd" and (mas == "2a" or mas == "2b")) or\
            zone == "And" or (zone == "St-Crd-And" and (mas == "2a" or mas == "2b")) or (zone == "St" and (mas == "2b" or mas == "3")):
        return 4.5
    elif ((zone == "Crd-And" or zone == "Crd-Kfs") and mas == "1c") or zone == "St-And" or\
            (zone == "And-St/Crd" and mas == "2c") or (zone == "St-Crd-And" and mas == "2c") or (zone == "St-Ky" and mas == "4a"):
        return 5
    elif (zone == "St-Ky" and mas == "4a/4b/5") or (zone == "St-Ky" and mas == "4b") or (zone == "Ky-(St)" and mas == "5"):
        return 5.5
    elif (zone == "Crd-Kfs" and (mas == "1a" or mas == "1b")) or zone == "Sil-(St-And)" or (zone == "Sil-(And-St/Crd)" and mas == "2c") or\
            (zone == "Sil-(St-Crd-And)" and mas == "2c") or zone == "Sil-(St)" or (zone == "Sil-(St-Ky)" and mas == "4a"):
        return 6
    elif zone == "Sil-(And)" or zone == "Sil-(Crd-And)" or (zone == "Sil-(And-St/Crd)" and (mas == "2a" or mas == "2b")) or\
            (zone == "Sil-(St-Crd-And)" and (mas == "2a" or mas == "2b")):
        return 6.5
    elif (zone == "Kfs-And-Crd" and (mas == "1c" or mas == "1b")) or (zone == "Sil" and (mas == "2c" or mas == "3" or mas == "4a" or mas == "4a/4b/5")) or\
            (zone == "Sil-(St-Ky)" and mas == "4a/4b/5") or zone == "Sil-(Ky)" or (zone == "Ky" and mas == "4b"):
        return 7
    elif (zone == "Kfs-And-Crd" and mas == "1a") or (zone == "Ky" and mas == "5"):
        return 7.5
    elif zone == "Kfs-Sil-Crd-(And)" or zone == "Kfs-Sil-(And)" or zone == "Kfs-And" or\
            (zone == "Kfs-Sil" and (mas == "2c" or mas == "3" or mas == "4a" or mas == "4a/4b/5")) or\
            (zone == "Kfs-Sil-(Ky)" and (mas == "4a" or mas == "4a/4b/5")) or (zone == "Sil" and mas == "4b"):
        return 8
    elif zone == "Spl" or zone == "Kfs-Crd" or zone == "Kfs-Crd±Grt" or zone == "Kfs-Sil±Crd" or\
            (zone == "Kfs-Sil-(Ky)" and mas == "4b") or (zone == "Kfs-Sil" and mas == "4b") or zone == "Kfs-Ky":
        return 9
    elif zone == "Grt/Opx" or zone == "Sil/Opx" or zone == "Opx":
        return 10
    else:
        return 0


def create_sequence_patches(mineral_assemblage_sequence: str, zones: list | None,
                            reference_file: Path = Path("Metapelite-Database-Bt-2024-01-25_PTbins_only.xlsx")):

    # read the reference file
    ref_PT_sequences = pd.read_excel(reference_file, skiprows=1, nrows=11, header=None, index_col=0)
    # change all indices (MAS) to strings
    ref_PT_sequences.index = ref_PT_sequences.index.astype(str)

    p_estimate_center = pd.read_excel(reference_file, skiprows=14, nrows=11, header=None, index_col=0)
    p_estimate_center.index = p_estimate_center.index.astype(str)

    p_estimate_range = pd.read_excel(reference_file, skiprows=28, nrows=11, header=None, index_col=0)
    p_estimate_range.index = p_estimate_range.index.astype(str)

    t_estimate_center = pd.read_excel(reference_file, skiprows=42, nrows=11, header=None, index_col=0)
    t_estimate_center.index = t_estimate_center.index.astype(str)

    t_estimate_range = pd.read_excel(Path(reference_file), skiprows=56, nrows=11, header=None, index_col=0)
    t_estimate_range.index = t_estimate_range.index.astype(str)

    # for none zones, return the whole sequence
    if zones is None:
        zones = ref_PT_sequences.loc[mineral_assemblage_sequence][ref_PT_sequences.loc[mineral_assemblage_sequence].notna()].to_list()

    # create the patches
    patches = []
    for zone in zones:
        # idx_zone = ref_PT_sequences.loc[mineral_assemblage_sequence].index[ref_PT_sequences.loc[mineral_assemblage_sequence] == zone]
        # idx_zone = idx_zone - 1
        idx_zone = floor(zone_to_number(zone, mineral_assemblage_sequence))

        p_center = p_estimate_center.loc[mineral_assemblage_sequence].loc[idx_zone]
        p_range = p_estimate_range.loc[mineral_assemblage_sequence].loc[idx_zone]

        t_center = t_estimate_center.loc[mineral_assemblage_sequence].loc[idx_zone]
        t_range = t_estimate_range.loc[mineral_assemblage_sequence].loc[idx_zone]

        patches.append(Rectangle((t_center - t_range, p_center - p_range), 2*t_range, 2*p_range))

    return patches


def plot_sequence_all_zones(mas: str, colormap_seq: str | list, c_dict: dict = COLOR_BIN_DICT, ax=None):
    if isinstance(colormap_seq, str):
        try:
            colormap_seq = sns.color_palette(colormap_seq, 10)
        except (ValueError, KeyError):
            raise ValueError("Invalid colormap name")

    # create the patches
    patches = create_sequence_patches(mas, None)

    color_idx = list(c_dict[mas].values())

    pc = PatchCollection(patches, alpha=0.8, edgecolor=[colormap_seq[i-1] for i in color_idx], facecolor="None", lw=3)

    if ax is None:
        fig, ax = plt.subplots()
        ax.add_collection(pc)

        return fig, ax

    elif isinstance(ax, plt.Axes):
        ax.add_collection(pc)

        return ax
    else:
        raise ValueError("ax argument must be a matplotlib Axes object")


def plot_sequence_locality(test_data: pd.DataFrame, locality: str, thermobarometer_model, colormap_seq: str | list, c_dict: dict = COLOR_BIN_DICT):
    if isinstance(colormap_seq, str):
        try:
            colormap_seq = sns.color_palette(colormap_seq, 10)
        except (ValueError, KeyError):
            raise ValueError("Invalid colormap name")

    sequence = test_data[test_data["Locality Name"] == locality]
    # get MAS string (if more than one MAS, raise an error)
    if len(sequence["MAS"].unique()) > 1:
        raise ValueError("Multiple sequences in the same locality")
    else:
        mas = sequence["MAS"].unique()[0]

    # get zones
    zones = sequence["Zone"].unique()

    # get the zone colors
    zone_colors_idx = [c_dict[mas][zone] for zone in zones]
    zone_colors = [colormap_seq[i-1] for i in zone_colors_idx]

    # create reference PT patches
    patches = create_sequence_patches(mas, zones)

    pc = PatchCollection(patches, alpha=0.8, edgecolor=zone_colors, facecolor="None", lw=2)

    fig, ax = plt.subplots()
    ax.add_collection(pc)

    # set-up inverse scaling
    inv_scaling_pt = MinMaxScaler(min=[1500, 400], max=[10000, 900], axis=0, invert=True)

    # predict the PT
    for zone, color in zip(zones, zone_colors):
        zone_data = sequence[sequence["Zone"] == zone]

        PT = inv_scaling_pt(thermobarometer_model(zone_data[["Bt-Si", "Bt-Ti", "Bt-Al", "Bt-FeTot", "Bt-Mn", "Bt-Mg"]].values))
        # transform pressure to kbar
        PT = PT.numpy()
        PT[:, 0] = PT[:, 0] / 1000

        ax.scatter(PT[:, 1], PT[:, 0], label=zone, color=color, s=5)

        # calculate the mean and the standard deviation of the PT data
        PT_mean = PT.mean(axis=0)
        PT_std = PT.std(axis=0)

        # plot the mean and the standard deviation
        ax.errorbar(PT_mean[1], PT_mean[0], xerr=PT_std[1], yerr=PT_std[0], c=color, capsize=3)
        ax.scatter(PT_mean[1], PT_mean[0], marker="o", label=zone, color=color, zorder=10)

    return fig, ax


def create_sequence_center_pos(mineral_assemblage_sequence: str, zones: list | None,
                               reference_file: Path = Path("Metapelite-Database-Bt-2024-01-25_PTbins_only.xlsx")):

    # read the reference file
    ref_PT_sequences = pd.read_excel(reference_file, skiprows=1, nrows=11, header=None, index_col=0)
    # change all indices (MAS) to strings
    ref_PT_sequences.index = ref_PT_sequences.index.astype(str)

    p_estimate_center = pd.read_excel(reference_file, skiprows=14, nrows=11, header=None, index_col=0)
    p_estimate_center.index = p_estimate_center.index.astype(str)

    p_estimate_range = pd.read_excel(reference_file, skiprows=28, nrows=11, header=None, index_col=0)
    p_estimate_range.index = p_estimate_range.index.astype(str)

    t_estimate_center = pd.read_excel(reference_file, skiprows=42, nrows=11, header=None, index_col=0)
    t_estimate_center.index = t_estimate_center.index.astype(str)

    t_estimate_range = pd.read_excel(Path(reference_file), skiprows=56, nrows=11, header=None, index_col=0)
    t_estimate_range.index = t_estimate_range.index.astype(str)

    # for none zones, return the whole sequence
    if zones is None:
        zones = ref_PT_sequences.loc[mineral_assemblage_sequence][ref_PT_sequences.loc[mineral_assemblage_sequence].notna()].to_list()

    # create list of tuples with the center position of the zones
    position_in_PT = []

    for zone in zones:
        idx_zone = floor(zone_to_number(zone, mineral_assemblage_sequence))

        p_center = p_estimate_center.loc[mineral_assemblage_sequence].loc[idx_zone]
        t_center = t_estimate_center.loc[mineral_assemblage_sequence].loc[idx_zone]

        position_in_PT.append((t_center, p_center))

    return zones, position_in_PT


def plot_sequence_all_zones_circles(mas: str, colormap_seq: str | list, c_dict: dict = COLOR_BIN_DICT, n_samples: pd.Series | None = None, ax=None):
    if isinstance(colormap_seq, str):
        try:
            colormap_seq = sns.color_palette(colormap_seq, 10)
        except (ValueError, KeyError):
            raise ValueError("Invalid colormap name")

    # create the patches
    zones, position_in_PT = create_sequence_center_pos(mas, None)

    color_idx = list(c_dict[mas].values())

    if n_samples is None:
        circ_size = np.ones(len(position_in_PT))
    else:
        circ_size = np.zeros(len(position_in_PT))
        zone_number = [zone_to_number(zone, mas) for zone in zones]
        for i, zone_nb in enumerate(zone_number):
            # for all zone_nb that exist as keys in n_samples, set circ_size to the value in n_samples
            if zone_nb in n_samples.keys():
                circ_size[i] = n_samples[zone_nb]

    if ax is None:
        fig, ax = plt.subplots()
        ax.scatter([pos[0] for pos in position_in_PT], [pos[1] for pos in position_in_PT],
                   color=[colormap_seq[i-1] for i in color_idx],
                   s=3*circ_size)

        return fig, ax

    elif isinstance(ax, plt.Axes):
        ax.scatter([pos[0] for pos in position_in_PT], [pos[1] for pos in position_in_PT],
                   color=[colormap_seq[i-1] for i in color_idx],
                   s=3*circ_size)

        return ax
    else:
        raise ValueError("ax argument must be a matplotlib Axes object")
