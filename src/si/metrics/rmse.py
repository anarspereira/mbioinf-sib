from cmath import sqrt

import numpy as np


def rmse(y_true: np.ndarray, y_pred: np.ndarray):
    """
    Cálculo do Root Mean Squared error.

    :param y_true: Array de true labels.
    :param y_pred: Array de predicted labels.

    :return: Valor do RMSE.
    """

    rmse_value = sqrt(np.sum((y_true - y_pred) ** 2) / len(y_true))

    return rmse_value