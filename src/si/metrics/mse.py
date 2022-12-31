import numpy as np


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Retorna o Mean Squared Error do modelo num dado dataset.

    :param y_true: np.ndarray
        Labels reais do dataset
    :param y_pred: np.ndarray
        Labels estimadas do dataset

    :return: mse: float
        O Mean Squared Error do modelo entre y_true e y_pred
    """
    return np.sum((y_true - y_pred) ** 2) / (len(y_true) * 2)
