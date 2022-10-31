import numpy as np
import pandas as pd
from data.dataset import Dataset

def read_data_file (filename: str, sep: str = ",", label: bool = False): -> object:
    """
    Método para ler csv e retornar um dataset.

    :param filename: Nome ou caminho do ficheiro csv
    :param sep: Qual o separador entre valores (default: ',')
    :param label: Booleano: True se tem labels; false se não tem labels

    :return: Dataset.
    """
    data = np.genfromtxt(filename, delimiter=sep)
    if label is True:  # se o ficheiro tem label definida
        x = data[:, :-1]  # todas as colunas, exceto a última
        y = data[:, -1]  # todas as linhas da última coluna
    else:  # se o ficheito não tem label definida
        x = data
        y = None  # não tem labels
    return Dataset(x, y)  # cria dataset com os dados

def write_data_file (filename: str, sep: str = ",", label: bool = False) -> object:
    """
    Método para escrever csv.

    :param filename: Nome ou caminho do ficheiro csv
    :param sep: Qual o separador entre valores (default: ',')
    :param label: Booleano: True se tem labels; false se não tem labels

    :return: Objeto com dataset.
    """
    if label is True:  # se temos label definida
        data = np.hstack(dataset.x, dataset.y)  # reune os array na horizontal
        data = np.savetxt(filename, delimiter=sep, data)  # guarda o array num ficheiro
    return data