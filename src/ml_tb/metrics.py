"""
Custom metrics to evaluate ml models trained on the biotite datasets.
"""
import numpy as np

from keras import backend as K
from scipy.stats import kendalltau

from ml_tb.plot import zone_to_number


def RMSE_denormalised(y_true, y_pred, inv_norm):
    y_true = inv_norm(y_true)
    y_pred = inv_norm(y_pred)

    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=0))


def RMSE_denormalised_P(y_true, y_pred, inv_norm):
    return RMSE_denormalised(y_true, y_pred, inv_norm)[0]


def RMSE_denormalised_T(y_true, y_pred, inv_norm):
    return RMSE_denormalised(y_true, y_pred, inv_norm)[1]


def RMSE_denormalised_temperature_only(y_true, y_pred, inv_norm):
    y_true = inv_norm(y_true)
    y_pred = inv_norm(y_pred)

    return K.sqrt(K.mean(K.square(y_pred - y_true)))


"""
Metrics for the evaluation of sequence data:
"""


def get_composition(data, element_order=["Bt-Si", "Bt-Ti", "Bt-Al", "Bt-FeTot", "Bt-Mn", "Bt-Mg"]):
    """Auxiliary function, used in tau_sequences / k_tau_sequences.

    Args:
        data (pd.DataFrame): A pandas dataframe containing the sequences test dataset loaded from excel.
        element_order (list, optional): Set the elemetn order to be correspondinfg to the models input vector.
        Defaults to ["Bt-Si", "Bt-Ti", "Bt-Al", "Bt-FeTot", "Bt-Mn", "Bt-Mg"].

    Returns:
        array-like: biotite compositional data
    """
    return data[element_order].values


def get_zones_number(data):
    """Auxiliary function, used in tau_sequences / k_tau_sequences.

    Args:
        data (pd.DataFrame): A pandas dataframe containing the sequences test dataset loaded from excel.

    Returns:
        np.array: Unique zone numbers for a given sequence.
    """
    zones = data["Zone"].values
    mas = data["MAS"].values

    return np.array([zone_to_number(zone, ma) for zone, ma in zip(zones, mas)])


def tau_sequences(data, model, inv_scaling_pt, sequences: list):
    tau_temperature_vec = np.empty(shape=(len(sequences), ))
    tau_pressure_vec = np.empty(shape=(len(sequences), ))
    n_points_in_seq_vec = np.empty(shape=(len(sequences), ))

    for i, sequence in enumerate(sequences):
        data_seq = data[data["Locality Name"] == sequence]

        comp_data = get_composition(data_seq)
        zones_data = get_zones_number(data_seq)

        predicted_pt = inv_scaling_pt(model(comp_data))
        # calculate median of the predicted values for each zone
        predictions = np.array([np.mean(predicted_pt[zones_data == zone], axis=0) for zone in np.unique(zones_data)])
        unique_zones_data = np.array([zone for zone in np.unique(zones_data)])

        # save number of zones in sequence for weighting in averaging the tau values for all sequences
        n_points_in_seq = len(np.unique(zones_data))

        tau, _ = kendalltau(predictions[:, 1], unique_zones_data)
        tau_temperature = tau

        tau, _ = kendalltau(predictions[:, 0], unique_zones_data)
        tau_pressure = tau

        tau_temperature_vec[i] = tau_temperature
        tau_pressure_vec[i] = tau_pressure
        n_points_in_seq_vec[i] = n_points_in_seq

    # calculate the weighted average of the tau_temperature_vec and tau_pressure_vec, using the number of points in each sequence as weights
    tau_temperature_weighted = np.average(tau_temperature_vec, weights=n_points_in_seq_vec)
    tau_pressure_weighted = np.average(tau_pressure_vec, weights=n_points_in_seq_vec)

    return tau_pressure_weighted, tau_temperature_weighted


def k_tau_sequences(data, models, inv_scaling_pt, sequences: list, return_p_values=False):
    tau_temperature_matrix = np.empty(shape=(len(sequences), 5))
    tau_pressure_matrix = np.empty(shape=(len(sequences), 5))
    n_points_in_seq_vec = np.empty(shape=(len(sequences), ))

    if return_p_values is True:
        p_temperature_matrix = np.empty(shape=(len(sequences), 5))
        p_pressure_matrix = np.empty(shape=(len(sequences), 5))

    for i, sequence in enumerate(sequences):
        data_seq = data[data["Locality Name"] == sequence]

        comp_data = get_composition(data_seq)
        zones_data = get_zones_number(data_seq)

        kfold_predictions = []
        kfold_zones_data = []

        for model in models:
            predicted_pt = inv_scaling_pt(model(comp_data))
            # calculate median of the predicted values for each zone
            kfold_predictions.append(np.array([np.mean(predicted_pt[zones_data == zone], axis=0) for zone in np.unique(zones_data)]))

            kfold_zones_data.append(np.array([zone for zone in np.unique(zones_data)]))

        kfold_predictions = np.array(kfold_predictions)
        kfold_zones_data = np.array(kfold_zones_data)

        tau_temperature = np.zeros(5)
        tau_pressure = np.zeros(5)
        # save number of zones in sequence for weighting in averaging the tau values for all sequences
        n_points_in_seq = len(np.unique(zones_data))

        if return_p_values is False:
            for j in range(5):
                tau, _ = kendalltau(kfold_predictions[j, :, 1], kfold_zones_data[j])
                tau_temperature[j] = tau

                tau, _ = kendalltau(kfold_predictions[j, :, 0], kfold_zones_data[j])
                tau_pressure[j] = tau

            tau_temperature_matrix[i] = tau_temperature
            tau_pressure_matrix[i] = tau_pressure
            n_points_in_seq_vec[i] = n_points_in_seq

        elif return_p_values is True:
            p_values_temperature = np.zeros(5)
            p_values_pressure = np.zeros(5)

            for j in range(5):
                tau, p_value = kendalltau(kfold_predictions[j, :, 1], kfold_zones_data[j])
                tau_temperature[j] = tau
                p_values_temperature[j] = p_value

                tau, p_value = kendalltau(kfold_predictions[j, :, 0], kfold_zones_data[j])
                tau_pressure[j] = tau
                p_values_pressure[j] = p_value

            tau_temperature_matrix[i] = tau_temperature
            tau_pressure_matrix[i] = tau_pressure
            n_points_in_seq_vec[i] = n_points_in_seq

            p_temperature_matrix[i] = p_values_temperature
            p_pressure_matrix[i] = p_values_pressure

    # calculate the weighted average of the tau_temperature_matrix and tau_pressure_matrix, using the number of points in each sequence as weights
    tau_temperature_weighted = np.average(tau_temperature_matrix, axis=0, weights=n_points_in_seq_vec)
    tau_pressure_weighted = np.average(tau_pressure_matrix, axis=0, weights=n_points_in_seq_vec)

    if return_p_values is False:
        return tau_pressure_weighted, tau_temperature_weighted

    elif return_p_values is True:
        return tau_pressure_weighted, tau_temperature_weighted, p_pressure_matrix, p_temperature_matrix
