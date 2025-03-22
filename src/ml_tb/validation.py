import numpy as np
from pathlib import Path
from keras.models import load_model
from keras import backend as K

"""
VALIDIATION: Module with funcs for validation of models using k-fold cross-validation and sequence data
"""


def load_k_models(model_name, model_dir, metrics, k_models=True):
    if k_models:
        model_files = [model_name + "_0", model_name + "_1", model_name + "_2", model_name + "_3", model_name + "_4"]
    else:
        model_files = np.repeat(model_name, 5)

    models = []
    for model_file in model_files:
        model = load_model(Path("..", model_dir, model_file), compile=False)
        model.compile(optimizer="adam", loss="mean_squared_error", metrics=metrics)
        models.append(model)

    return models


def predict_on_val_set(k_models, val_data, inv_scaler):
    predictions = []
    for model, data in zip(k_models, val_data):
        predictions.append(inv_scaler(model.predict(data)))
    return np.array(predictions)


"""
Validation specific metrics
"""


def RMSE(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))


def kfold_RMSE_PT(y_true, y_pred):
    RMSE_P = []
    RMSE_T = []

    for i in range(len(y_true)):
        RMSE_P.append(RMSE(y_true[i][:, 0], y_pred[i][:, 0]))
        RMSE_T.append(RMSE(y_true[i][:, 1], y_pred[i][:, 1]))

    return RMSE_P, RMSE_T


def kfold_RMSE_T(y_true, y_pred):
    RMSE_T = []

    for i in range(len(y_true)):
        RMSE_T.append(RMSE(y_true[i], y_pred[i]))

    return RMSE_T


# calculate 5 RMSE for 5 ranges of temperature (400-500, 500-600, 600-700, 700-800, 800-900)
def RMSE_temperature_ranges(y_true, y_pred):
    RMSE_T = []

    for i in range(len(y_true)):
        RMSE_T.append([RMSE(y_true[i][np.logical_and(y_true[i] >= 400 + j * 100, y_true[i] < 500 + j * 100)],
                            y_pred[i][np.logical_and(y_true[i] >= 400 + j * 100, y_true[i] < 500 + j * 100)]) for j in range(5)])

    return RMSE_T


def RMSE_ranges(y_true, y_pred):
    RMSE_P = []
    RMSE_T = []

    for i in range(len(y_true)):
        RMSE_P.append([RMSE(y_true[i][np.logical_and(y_true[i][:, 0] >= 1500 + j * 1700, y_true[i][:, 0] < 3200 + j * 1700), 0],
                            y_pred[i][np.logical_and(y_true[i][:, 0] >= 1500 + j * 1700, y_true[i][:, 0] < 3200 + j * 1700), 0]) for j in range(5)])

        RMSE_T.append([RMSE(y_true[i][np.logical_and(y_true[i][:, 1] >= 400 + j * 100, y_true[i][:, 1] < 500 + j * 100), 1],
                            y_pred[i][np.logical_and(y_true[i][:, 1] >= 400 + j * 100, y_true[i][:, 1] < 500 + j * 100), 1]) for j in range(5)])

    return RMSE_P, RMSE_T
