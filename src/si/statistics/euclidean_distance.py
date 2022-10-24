import numpy as np


def euclidean_distance(x: np.ndarray, y: np.ndarray): # x: numpy array de 1 amostra; y: np array de n amostras
    """
    Calcula a distância euclideana de um ponto (x) num set de pontos (y).

    distance_y1n = np.sqrt((x1 - y11)^2 + (x2 - y12)^2 + ... + (xn - y1n)^2)
    distance_y2n = np.sqrt((x1 - y12)^2 + (x2 - y22)^2 + ... + (xn - y2n)^2)

    :param x: np.ndarray. Ponto.
    :param y: np.ndarray. Set de pontos.

    :return: np.ndarray da distância euclideana para cada ponto em y.
    """
    return np.sqrt(((x - y) ** 2).sum(axis = 1))