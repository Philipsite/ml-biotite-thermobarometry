import numpy as np
import pandas as pd

from pathlib import Path
from ml_tb.mapping import load_biotite_map, predict_temperature_Henry2005, predict_temperature_Wu2015


def get_PT_map(map_file: Path, model, inv_scaling_pt, element_order: list = ["Si", "Ti", "Al", "Fe", "Mn", "Mg"]):
    biotite_map, map_empty, idx_phase_pixel = load_biotite_map(map_file, element_order)

    return inv_scaling_pt(model(biotite_map))


def get_PT_analyses(data_file: Path, test_id: str, model, inv_scaling_pt, element_order: list = ["Bt-Si", "Bt-Ti", "Bt-Al", "Bt-Fetot", "Bt-Mn", "Bt-Mg"]):
    data = pd.read_excel(data_file)

    analyses = data[data["TEST ID"] == test_id]
    # check if analyses are present
    if len(analyses) == 0:
        raise ValueError("No analyses found for test ID: ", test_id)

    bt_analyses = analyses[element_order].values

    return inv_scaling_pt(model(bt_analyses))


def get_ref_PT_analyses(data_file: Path, test_id: str):
    data = pd.read_excel(data_file)

    analyses = data[data["TEST ID"] == test_id]
    pt_analyses = analyses[["P estimate [bar]", "T estimate [°C]"]].values
    # return only the first analysis, all multi-analyses must have the same ref PT
    return pt_analyses[0]


def PT_median(PT):
    return np.median(PT, axis=0)


def PT_error(PT):
    PT_iqr_25_50 = np.abs(np.percentile(PT, q=25, axis=0) - np.median(PT, axis=0))
    PT_iqr_75_50 = np.abs(np.percentile(PT, q=75, axis=0) - np.median(PT, axis=0))

    P_error = (PT_iqr_25_50[0], PT_iqr_75_50[0])
    T_error = (PT_iqr_25_50[1], PT_iqr_75_50[1])

    return np.expand_dims([P_error, T_error], axis=1)


def T_error(T):
    T_iqr_25_50 = np.abs(np.percentile(T, q=25, axis=0) - np.median(T, axis=0))
    T_iqr_75_50 = np.abs(np.percentile(T, q=75, axis=0) - np.median(T, axis=0))

    T_error = (T_iqr_25_50, T_iqr_75_50)

    return np.expand_dims([T_error], axis=1)


# T estimates for Ti-in-biotite thermometer (TiBt05 + TiBt15)
def get_T_map_Henry05(map_file: Path, element_order: list = ["Fe", "Mg", "Ti"]):
    biotite_map, map_empty, idx_phase_pixel = load_biotite_map(map_file, element_order)

    return predict_temperature_Henry2005(biotite_map, map_empty, idx_phase_pixel)


def get_T_Henry05_analyses(data_file: Path, test_id: str, element_order: list = ["Bt-Fetot", "Bt-Mg", "Bt-Ti"]):
    data = pd.read_excel(data_file)

    analyses = data[data["TEST ID"] == test_id]
    # check if analyses are present
    if len(analyses) == 0:
        raise ValueError("No analyses found for test ID: ", test_id)

    bt_analyses = analyses[element_order].values

    a = -2.3594
    b = 4.6482e-9
    c = -1.7283

    # adjust to 22Ox basis
    Fe = bt_analyses[:, 0].flatten() * 2
    Mg = bt_analyses[:, 1].flatten() * 2
    Ti = bt_analyses[:, 2].flatten() * 2

    idx_notZero = np.logical_and(np.logical_and(Fe > 0, Mg > 0), Ti > 0)
    Fe = Fe[idx_notZero]
    Mg = Mg[idx_notZero]
    Ti = Ti[idx_notZero]

    X_Mg = Mg / (Mg + Fe)

    temperature = ((np.log(Ti)-a-(c*(X_Mg**3)))/b)**0.333
    temperature = temperature[~np.isnan(temperature)]

    return temperature


def get_T_map_Wu15(map_file: Path, P_estimate_GPa: float, element_order: list = ["Fe", "Mg", "Ti", "Al", "Si"]):
    biotite_map, map_empty, idx_phase_pixel = load_biotite_map(map_file, element_order)

    return predict_temperature_Wu2015(biotite_map, map_empty, idx_phase_pixel, P_estimate_GPa)


def get_T_Wu15_analysis(data_file: Path, test_id: str, P_estimate_GPa: np.array, element_order: list = ["Bt-Fetot", "Bt-Mg", "Bt-Ti", "Bt-Al", "Bt-Si"]):
    data = pd.read_excel(data_file)

    analyses = data[data["TEST ID"] == test_id]
    # check if analyses are present
    if len(analyses) == 0:
        raise ValueError("No analyses found for test ID: ", test_id)

    bt_analyses = analyses[element_order].values

    Fe = bt_analyses[:, 0].flatten()
    Mg = bt_analyses[:, 1].flatten()
    Ti = bt_analyses[:, 2].flatten()
    Al = bt_analyses[:, 3].flatten()
    Si = bt_analyses[:, 4].flatten()

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
