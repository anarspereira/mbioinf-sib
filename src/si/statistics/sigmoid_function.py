import numpy as np


def sigmoid_function(X: np.ndarray) -> np.ndarray:
    """
    Método que calcula a função sigmoid para o input X.

    :param X: Valores de entrada para a função sigmoid.

    :return: Array com probabilidade dos valores serem iguais a 1 (função sigmoid do input).
    """
    return 1/(1 + np.exp(-X))