import numpy as np


def cross_entropy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Método que usa a função cross-entropy para calcular a diferença entre duas distribuições de probabilidades.
    É usada em ML como uma loss function para problemas de classificação, onde o objetivo é prever a probabilidade de
    cada classe para um dado input.

    :param y_true: np.ndarray
        Labels reais
    :param y_pred: np.ndarray
        Labels estimadas

    :return: float
        Diferença entre as duas probabilidades
    """
    return - np.sum(y_true) * np.log(y_pred) / len(y_true)


def cross_entropy_derivative(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Método que usa a função derivativa da cross-entropy, que mede como as loss mudam consoante a probabilidade prevista.
    É usada em ML para calcular o gradiente da loss function, que é usado nos algoritmos de otimização (como o gradient
    descent) para dar update dos parâmetros do modelo e minimizar a loss.

    :param y_true: np.ndarray
        Labels reais
    :param y_pred: np.ndarray
        Labels estimadas

    :return: float
        Valor da derivative of the cross entropy loss function.
    """

    return - (y_true / y_pred) + ((1 - y_true) / (1 - y_pred)) / len(y_true)